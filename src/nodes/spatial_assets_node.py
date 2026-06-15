import io
import gc
import weakref
from pathlib import Path

import torch
from transformers import pipeline
from pydantic import Field
from PIL import Image, UnidentifiedImageError

from controlnet_aux import PidiNetDetector
from src.nodes.base_node import BaseNode
from src.pipeline import register_cleanup_hook
from src.nodes.text2image import Text2ImageInputs

ANNOTATORS_DIR = Path.home() / "sd_annotators"
DEFAULT_EDGE_MODEL = "lllyasviel/Annotators"
_LIVE_SPATIAL_ASSETS_NODES: weakref.WeakSet["SpatialAssetsNode"] = weakref.WeakSet()


def _clear_spatial_assets_models() -> None:
    cleared = 0
    for node in list(_LIVE_SPATIAL_ASSETS_NODES):
        if getattr(node, "depth_pipe", None) is not None:
            node.depth_pipe = None
            cleared += 1
        if getattr(node, "edge_pipe", None) is not None:
            node.edge_pipe = None
            cleared += 1
    if cleared:
        print(f"🧹 cleared {cleared} spatial asset CPU model reference(s)")
    gc.collect()


register_cleanup_hook(_clear_spatial_assets_models)


class SpatialAssetsInputs(Text2ImageInputs):
    depth_model: str = Field(
        "depth-anything/DA3-BASE",
        description="Model to use for depth estimation (e.g., 'Intel/dpt-large-384' or 'depth-anything/Depth-Anything-V2-Large-hf')",
    )
    edge_model: str = Field(
        DEFAULT_EDGE_MODEL,
        description="Model to use for edge detection (e.g., 'lllyasviel/Annotators' for PiDiNet)",
    )


class SpatialAssetsNode(BaseNode):
    output_key = "images"

    def __init__(self, inputs: SpatialAssetsInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "spatial_assets"
        self.images: list[Image.Image] = []
        _LIVE_SPATIAL_ASSETS_NODES.add(self)
        """
        Initializes the pipeline targeting modern Foundation Models and Crisp Edge extractors.
        """
        self.device = self._get_optimal_device()

        # 1. Initialize SOTA Depth Model
        try:
            self.depth_pipe = pipeline(
                task="depth-estimation",
                model=self.params.depth_model,
                device=self.device,
                trust_remote_code=True,
            )
            print(f"Depth model '{self.params.depth_model}' loaded successfully.")
        except ValueError as ve:
            print(
                f"Bleeding-edge model failed. Falling back to stable SOTA HF weights. Error: {ve}"
            )
            fallback_model = "depth-anything/Depth-Anything-V2-Large-hf"
            self.depth_pipe = pipeline(
                task="depth-estimation", model=fallback_model, device=self.device
            )
            print(f"Depth model '{fallback_model}' loaded successfully.")
        except Exception as e:
            print(
                f"Critical error loading depth model '{self.params.depth_model}': {e}"
            )
            self.depth_pipe = None

        # 2. Initialize SOTA Edge Model (PiDiNet)
        self.edge_pipe = self._load_edge_pipe(self.params.edge_model)

    @staticmethod
    def _load_edge_pipe(edge_model: str):
        try:
            edge_pipe = PidiNetDetector.from_pretrained(
                edge_model,
                cache_dir=str(ANNOTATORS_DIR),
            )
            print(f"PiDiNet Edge model '{edge_model}' loaded successfully.")
            return edge_pipe
        except Exception as e:
            if edge_model == DEFAULT_EDGE_MODEL:
                print(f"Could not load PiDiNet edge model '{edge_model}'. Error:\n{e}")
                return None

            print(
                f"Could not load custom edge model '{edge_model}'. Falling back to "
                f"'{DEFAULT_EDGE_MODEL}'. Error:\n{e}"
            )

        try:
            edge_pipe = PidiNetDetector.from_pretrained(
                DEFAULT_EDGE_MODEL,
                cache_dir=str(ANNOTATORS_DIR),
            )
            print(f"PiDiNet Edge model '{DEFAULT_EDGE_MODEL}' loaded successfully.")
            return edge_pipe
        except Exception as fallback_error:
            print(
                f"Could not load fallback PiDiNet edge model '{DEFAULT_EDGE_MODEL}'. "
                f"Error:\n{fallback_error}"
            )
            return None

    @staticmethod
    def _to_pil(image: Image.Image | bytes | bytearray | torch.Tensor) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            elif tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            elif tensor.ndim != 3:
                raise ValueError(
                    "Tensor images must be HxW, CxHxW, HxWxC, or a batched NxCxHxW tensor."
                )

            if tensor.is_floating_point():
                tensor = tensor.float().clamp(0, 1).mul(255).round().byte()
            else:
                tensor = tensor.clamp(0, 255).byte()

            arr = tensor.numpy()
            if arr.shape[-1] == 1:
                return Image.fromarray(arr[..., 0], mode="L").convert("RGB")
            if arr.shape[-1] == 4:
                return Image.fromarray(arr, mode="RGBA").convert("RGB")
            return Image.fromarray(arr, mode="RGB")

        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")

        raise TypeError(f"Unsupported image input type: {type(image).__name__}")

    @staticmethod
    def _resize_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        return image.resize(size, Image.LANCZOS)

    def __call__(
        self,
        images: list[Image.Image | bytes | bytearray] | torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> dict[str, list[Image.Image]]:
        """
        Reads an image, generates depth, crisp edges. Returns Images.
        """
        raw = images if images is not None else self.images
        if isinstance(raw, torch.Tensor):
            if raw.ndim == 4:
                raw_images = [raw[i] for i in range(raw.shape[0])]
            else:
                raw_images = [raw]
        elif isinstance(raw, (Image.Image, bytes, bytearray)):
            raw_images = [raw]
        else:
            raw_images = list(raw)

        if not raw_images:
            raise ValueError("SpatialAssetsNode requires at least one input image.")

        output: list[Image.Image] = []

        # --- Inference Phase ---
        for img in raw_images:
            try:
                print("Loading image")
                image = self._to_pil(img)
            except UnidentifiedImageError:
                print("File is not a valid image")
                raise
            except Exception as e:
                print(f"Error reading image: {e}")
                raise

            target_size = image.size

            print("Generating DA3 Depth map...")
            try:
                if self.depth_pipe:
                    depth_image = self.depth_pipe(image)["depth"]
                    output.append(self._resize_to(depth_image, target_size))

                else:
                    raise RuntimeError(
                        "Depth pipeline was not initialized due to earlier errors."
                    )
            except Exception as e:
                print(f"Inference failed during depth generation: {e}")

            print("Generating Crisp Edge map...")
            try:
                if self.edge_pipe:
                    edge_image = self.edge_pipe(
                        image,
                        detect_resolution=1024,
                        image_resolution=1024,
                        safe=False,
                    )
                    output.append(self._resize_to(edge_image, target_size))
                else:
                    raise RuntimeError("Edge pipeline was not initialized.")
            except Exception as e:
                print(f"Inference failed during edge generation: {e}")
        return {"images": output}

    @staticmethod
    def _get_optimal_device() -> str:
        """Always run on CPU — GPU is reserved for the SD diffusion pipeline."""
        return "cpu"
