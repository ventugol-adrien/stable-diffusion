import io
import zipfile
import logging
import torch
import base64
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel
from controlnet_aux import PidiNetDetector

ANNOTATORS_DIR = Path.home() / "sd_annotators"


class AssetRequest(BaseModel):
    input_image: str


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ControlNetAssetGenerator:
    """
    A production-grade preprocessor for generating SOTA spatial maps
    (Depth and Crisp Edges) for ControlNet pipelines.
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
            # Point this to the exact path where you curled the .pth file
            model_path = ANNOTATORS_DIR / "pidinet_cg0.11_nc.pth"

            # Load the custom neural annotator
            self.edge_pipe = PidiNetDetector.from_pretrained(
                "lllyasviel/Annotators",
                filename="pidinet_cg0.11_nc.pth",
                cache_dir=ANNOTATORS_DIR,
            )
            logger.info("PiDiNet Edge model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load custom edge model. Error: {e}")
            self.edge_pipe = None

    @staticmethod
    def _get_optimal_device() -> str:
        """Determines the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def process(self, input_bytes: bytes) -> bytes:
        """
        Reads an image, generates depth and crisp edge maps, and returns a zipped byte stream.
        If an asset fails to generate, an error log is placed in the zip instead.
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

        # State trackers for graceful degradation
        depth_image = None
        edge_image = None
        depth_error = None
        edge_error = None

        # --- Inference Phase ---

        # Generate DA3 Depth map
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

        # Generate MatchED Crisp Edge map
        logger.info("Generating Crisp Edge map...")
        try:
            if self.edge_pipe:
                # Assuming custom MatchED pipeline returns dict with mask
                edge_image = self.edge_pipe(image)[0]["mask"]
            else:
                raise RuntimeError(
                    "Edge pipeline was not initialized (missing weights or failed to load)."
                )
        except Exception as e:
            logger.error(f"Inference failed during edge generation: {e}")
            edge_error = str(e)

        # --- Packaging Phase ---

        try:
            logger.info("Compressing assets into ZIP archive...")
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Save Depth Map (or write error file)
                if depth_image:
                    depth_io = io.BytesIO()
                    depth_image.save(depth_io, format="PNG")
                    zip_file.writestr("da3_depth_map.png", depth_io.getvalue())
                else:
                    zip_file.writestr(
                        "da3_depth_error.txt",
                        f"Failed to generate Depth Map:\n{depth_error}",
                    )

                # Save Edge Map (or write error file)
                if edge_image:
                    edge_io = io.BytesIO()
                    edge_image.save(edge_io, format="PNG")
                    zip_file.writestr("matched_edge_map.png", edge_io.getvalue())
                else:
                    zip_file.writestr(
                        "matched_edge_error.txt",
                        f"Failed to generate Edge Map:\n{edge_error}",
                    )

            logger.info("Assets successfully generated and zipped.")
            return zip_buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to package assets: {e}")
            return None


router = APIRouter(prefix="/spatial-assets", tags=["ControlNet Extraction"])


@router.post("/generate", response_class=Response)
def generate_controlnet_assets(request: AssetRequest):
    """
    API endpoint to generate a SOTA depth map and edge map archive.
    """
    generator = ControlNetAssetGenerator()
    raw = request.input_image

    # Strip base64 header if present
    img_bytes = base64.b64decode(raw.split(",")[1] if "," in raw else raw)

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
