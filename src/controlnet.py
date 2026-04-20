import io
import zipfile
import logging
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from controlnet_aux import PidiNetDetector
from diffusers import AutoPipelineForInpainting
import traceback

from src.pipeline import get_pipe
from src.transform import TransformParams, apply_transforms

ANNOTATORS_DIR = Path.home() / "sd_annotators"


class AssetRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    input_image: UploadFile

    @classmethod
    def as_form(cls, input_image: UploadFile = File(...)) -> "AssetRequest":
        return cls(input_image=input_image)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ControlNetAssetGenerator:
    """
    A production-grade preprocessor for generating SOTA spatial maps
    (Depth and Crisp Edges) for ControlNet pipelines, and fusing them into masks.
    """

    def __init__(
        self, depth_model="depth-anything/DA3-BASE", edge_model="matched-pidinet-custom"
    ):
        """
        Initializes the pipeline targeting modern Foundation Models and Crisp Edge extractors.
        """
        self.device = self._get_optimal_device()
        logger.info(f"Initializing AssetGenerator on: {self.device.upper()}")

        # 1. Initialize SOTA Depth Model
        try:
            self.depth_pipe = pipeline(
                task="depth-estimation",
                model=depth_model,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info(f"Depth model '{depth_model}' loaded successfully.")
        except ValueError as ve:
            logger.warning(
                f"Bleeding-edge model failed. Falling back to stable SOTA HF weights. Error: {ve}"
            )
            fallback_model = "depth-anything/Depth-Anything-V2-Large-hf"
            self.depth_pipe = pipeline(
                task="depth-estimation", model=fallback_model, device=self.device
            )
            logger.info(f"Depth model '{fallback_model}' loaded successfully.")
        except Exception as e:
            logger.error(f"Critical error loading depth model '{depth_model}': {e}")
            self.depth_pipe = None

        # 2. Initialize SOTA Edge Model (PiDiNet)
        try:
            self.edge_pipe = PidiNetDetector.from_pretrained(
                "lllyasviel/Annotators",
                cache_dir=str(ANNOTATORS_DIR),
            )
            logger.info("PiDiNet Edge model loaded successfully.")
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.warning(f"Could not load custom edge model. Error:\n{error_trace}")
            self.edge_pipe = None

    @staticmethod
    def _get_optimal_device() -> str:
        """Determines the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _extract_masks(
        self, depth_pil: Image.Image, edge_pil: Image.Image
    ) -> dict[str, tuple[Image.Image, Image.Image]]:
        """
        Fuses the depth map and edge map to create precise foreground and background masks.
        Returns a dictionary of 5 different mask variations.
        """
        # Ensure sizes match before processing
        if depth_pil.size != edge_pil.size:
            logger.warning(
                f"Resizing edge map {edge_pil.size} to match depth map {depth_pil.size}"
            )
            edge_pil = edge_pil.resize(depth_pil.size, Image.LANCZOS)

        # Convert PIL images to grayscale OpenCV arrays
        depth_cv = np.array(depth_pil.convert("L"))
        edge_cv = np.array(edge_pil.convert("L"))

        # Calculate base Otsu threshold
        otsu_thresh, _ = cv2.threshold(
            depth_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Define 5 sets of parameters: (thresh_offset, dilate_iters, close_kernel_size)
        variations = [
            {
                "name": "conservative_baseline",
                "thresh_offset": -45,
                "dilate_iters": 5,
                "close_kernel": 15,
            },
            {
                "name": "conservative_depth_push",
                "thresh_offset": -55,
                "dilate_iters": 4,
                "close_kernel": 15,
            },
            {
                "name": "conservative_tight_edge",
                "thresh_offset": -50,
                "dilate_iters": 3,
                "close_kernel": 13,
            },
            {
                "name": "conservative_deep_smooth",
                "thresh_offset": -52,
                "dilate_iters": 4,
                "close_kernel": 19,
            },
            {
                "name": "conservative_surgical",
                "thresh_offset": -48,
                "dilate_iters": 3,
                "close_kernel": 11,
            },
        ]

        results = {}

        for p in variations:
            # 1. Depth Map Thresholding
            thresh_val = np.clip(otsu_thresh + p["thresh_offset"], 0, 255)
            _, depth_mask = cv2.threshold(depth_cv, thresh_val, 255, cv2.THRESH_BINARY)

            # 2. Edge Map Contouring & Filling
            kernel_dilate = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(
                edge_cv, kernel_dilate, iterations=p["dilate_iters"]
            )

            edge_filled = np.zeros_like(edge_cv)
            contours, _ = cv2.findContours(
                edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(edge_filled, contours, -1, (255), thickness=cv2.FILLED)

            # 3. Map Fusion (Logical AND)
            foreground_mask = cv2.bitwise_and(depth_mask, edge_filled)

            # Apply morphological close
            kernel_close = np.ones((p["close_kernel"], p["close_kernel"]), np.uint8)
            foreground_mask = cv2.morphologyEx(
                foreground_mask, cv2.MORPH_CLOSE, kernel_close
            )

            # 4. Soft-edge mask: dilate then Gaussian blur so the
            #    gradient ends at the original hard edge boundary.
            blur_sigma = 7.5
            dilate_radius = int(np.ceil(2 * blur_sigma))
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
            )
            foreground_mask = cv2.dilate(foreground_mask, dilate_kernel, iterations=1)
            foreground_mask = cv2.GaussianBlur(
                foreground_mask, (0, 0), sigmaX=blur_sigma
            )

            # 5. Background Extraction
            background_mask = cv2.bitwise_not(foreground_mask)

            results[p["name"]] = (
                Image.fromarray(foreground_mask),
                Image.fromarray(background_mask),
            )

        return results

    def process(self, input_bytes: bytes) -> bytes:
        """
        Reads an image, generates depth, crisp edges, and extracted masks. Returns a zipped byte stream.
        """
        try:
            logger.info("Loading image from bytes")
            image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        except UnidentifiedImageError:
            logger.error("File is not a valid image")
            return None
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return None

        # State trackers
        depth_image = None
        edge_image = None
        masks = None

        depth_error = None
        edge_error = None
        mask_error = None

        # --- Inference Phase ---

        logger.info("Generating DA3 Depth map...")
        try:
            if self.depth_pipe:
                depth_image = self.depth_pipe(image)["depth"]
            else:
                raise RuntimeError(
                    "Depth pipeline was not initialized due to earlier errors."
                )
        except Exception as e:
            logger.error(f"Inference failed during depth generation: {e}")
            depth_error = str(e)

        logger.info("Generating Crisp Edge map...")
        try:
            if self.edge_pipe:
                edge_image = self.edge_pipe(
                    image,
                    detect_resolution=1024,
                    image_resolution=1024,
                    safe=False,
                )
            else:
                raise RuntimeError("Edge pipeline was not initialized.")
        except Exception as e:
            logger.error(f"Inference failed during edge generation: {e}")
            edge_error = str(e)

        # --- Mask Extraction Phase ---
        if depth_image and edge_image:
            logger.info("Extracting dynamic masks via Depth & Edge fusion...")
            try:
                masks = self._extract_masks(depth_image, edge_image)
            except Exception as e:
                logger.error(f"Mask extraction failed: {e}")
                mask_error = str(e)

        # --- Packaging Phase ---
        try:
            logger.info("Compressing assets into ZIP archive...")
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Save Depth Map
                if depth_image:
                    depth_io = io.BytesIO()
                    depth_image.save(depth_io, format="PNG")
                    zip_file.writestr("da3_depth_map.png", depth_io.getvalue())
                else:
                    zip_file.writestr(
                        "da3_depth_error.txt",
                        f"Failed to generate Depth Map:\n{depth_error}",
                    )

                # Save Edge Map
                if edge_image:
                    edge_io = io.BytesIO()
                    edge_image.save(edge_io, format="PNG")
                    zip_file.writestr("matched_edge_map.png", edge_io.getvalue())
                else:
                    zip_file.writestr(
                        "matched_edge_error.txt",
                        f"Failed to generate Edge Map:\n{edge_error}",
                    )

                # Save Masks into a folder
                if masks:
                    for name, (fg_mask, bg_mask) in masks.items():
                        fg_io = io.BytesIO()
                        fg_mask.save(fg_io, format="PNG")
                        zip_file.writestr(
                            f"masks/{name}_foreground.png", fg_io.getvalue()
                        )

                        bg_io = io.BytesIO()
                        bg_mask.save(bg_io, format="PNG")
                        zip_file.writestr(
                            f"masks/{name}_background.png", bg_io.getvalue()
                        )
                elif mask_error:
                    zip_file.writestr(
                        "masks/mask_error.txt",
                        f"Failed to generate Masks:\n{mask_error}",
                    )

            logger.info("Assets successfully generated and zipped.")
            return zip_buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to package assets: {e}")
            return None


router = APIRouter(prefix="/spatial-assets", tags=["ControlNet Extraction"])


def process_mask(mask_pil: Image.Image) -> Image.Image:
    """
    Takes a binary mask (any mode) and returns a soft-edge processed version:
    1. Gaussian blur to anti-alias the hard binary edge before morphology.
    2. Re-threshold to recover a clean binary mask from the blurred result.
    3. Morphological close to fill small holes and smooth the contour.
    4. Dilate-then-blur (same sigma-aware radius as _extract_masks) to produce
       a feathered gradient that ends at the original hard-edge boundary.
    """
    gray = np.array(mask_pil.convert("L"))

    # 1. Pre-smooth: light Gaussian blur to remove jagged binary edges
    pre_smoothed = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.5)

    # 2. Re-threshold to get a clean binary mask
    _, binary = cv2.threshold(pre_smoothed, 127, 255, cv2.THRESH_BINARY)

    # 3. Morphological close to fill holes and smooth the silhouette contour
    close_kernel = np.ones((13, 13), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # 4. Soft-edge feather: dilate enough to cover the blur falloff, then blur
    blur_sigma = 7.5
    dilate_radius = int(np.ceil(2 * blur_sigma))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
    )
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)
    soft = cv2.GaussianBlur(dilated, (0, 0), sigmaX=blur_sigma)

    return Image.fromarray(soft)


@router.post("/generate", response_class=Response)
async def generate_controlnet_assets(
    request: AssetRequest = Depends(AssetRequest.as_form),
):
    """
    API endpoint to generate a SOTA depth map, edge map, and compositing masks archive.
    """
    generator = ControlNetAssetGenerator()

    img_bytes = await request.input_image.read()

    zip_data = generator.process(img_bytes)

    if zip_data:
        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=controlnet_priors.zip"
            },
        )
    else:
        return Response(
            content='{"success": false, "message": "Failed to generate assets."}',
            media_type="application/json",
            status_code=500,
        )


def _outpaint_fill(
    image: Image.Image,
    void_mask: Image.Image,
    prompt: str,
    model: str = "juggernaut",
    strength: float = 1.0,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
) -> Image.Image:
    """
    Fills void regions of a transformed image using SDXL inpainting.

    Converts the cached base pipeline to an inpainting pipeline via
    from_pipe() (shares weights, no extra VRAM), runs inpainting on the
    void mask, and returns the filled image.
    """
    base_pipe = get_pipe(model)
    inpaint_pipe = AutoPipelineForInpainting.from_pipe(base_pipe)

    rgb_image = image.convert("RGB") if image.mode != "RGB" else image

    # Ensure dimensions are multiples of 8 (required by VAE)
    w, h = rgb_image.size
    w8 = w - (w % 8)
    h8 = h - (h % 8)
    if (w8, h8) != (w, h):
        rgb_image = rgb_image.crop((0, 0, w8, h8))
        void_mask = void_mask.crop((0, 0, w8, h8))

    result = inpaint_pipe(
        prompt=prompt,
        image=rgb_image,
        mask_image=void_mask,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=w8,
        height=h8,
    ).images[0]

    return result


@router.post("/transform", response_class=Response)
async def transform_image(
    input_image: UploadFile = File(...),
    dx: int = Form(0),
    dy: int = Form(0),
    z: float = Form(1.0),
    r: float = Form(0.0),
    prompt: str = Form(None),
    model: str = Form("juggernaut"),
    strength: float = Form(1.0),
):
    """
    Spatially transforms an image (scale, rotate, displace) and optionally
    fills void regions via SDXL outpainting when a prompt is provided.

    Without a prompt, returns the transformed image with black voids.
    """
    img_bytes = await input_image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
    except UnidentifiedImageError:
        return Response(
            content='{"success": false, "message": "File is not a valid image."}',
            media_type="application/json",
            status_code=400,
        )

    params = TransformParams(dx=dx, dy=dy, z=z, r=r)
    transformed, void_mask = apply_transforms(img, params)

    # Outpaint fill: only when caller provides a prompt and image is RGB
    if prompt and transformed.mode == "RGB":
        transformed = _outpaint_fill(
            transformed, void_mask, prompt, model=model, strength=strength
        )

    buf = io.BytesIO()
    transformed.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=transformed.png"},
    )


@router.post("/process-mask", response_class=Response)
async def process_mask_endpoint(mask: UploadFile = File(...)):
    """
    Takes a binary mask image and returns a soft-edge processed PNG.
    Edges are pre-smoothed with a Gaussian blur before morphological processing
    to remove jagged binary artefacts, then feathered with a dilate-blur pass.
    """
    try:
        mask_bytes = await mask.read()
        mask_pil = Image.open(io.BytesIO(mask_bytes))
    except UnidentifiedImageError:
        return Response(
            content='{"success": false, "message": "File is not a valid image."}',
            media_type="application/json",
            status_code=400,
        )

    result = process_mask(mask_pil)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=processed_mask.png"},
    )
