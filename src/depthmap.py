import io
import logging
import torch
import base64
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from pathlib import Path
from typing import Union
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel


class DepthMapRequest(BaseModel):
    input_image: str


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DepthMapGenerator:
    """
    A production-grade preprocessor for generating depth maps from images
    using the DPT (MiDaS) model architecture.
    """

    def __init__(self, model_name: str = "Intel/dpt-large"):
        """
        Initializes the pipeline, automatically detecting the best available hardware.

        Args:
            model_name (str): The Hugging Face model ID to use.
        """
        self.device = self._get_optimal_device()
        logger.info(f"Initializing DepthMapGenerator on: {self.device.upper()}")

        try:
            # Initialize the Hugging Face pipeline for depth estimation
            self.pipe = pipeline(
                task="depth-estimation", model=model_name, device=self.device
            )
            logger.info(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Critical error loading model '{model_name}': {e}")
            raise RuntimeError(f"Failed to initialize pipeline: {e}")

    @staticmethod
    def _get_optimal_device() -> str:
        """Determines the best available compute device (Nvidia, Apple Silicon, or CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def process(self, input: bytes) -> str:
        """
        Reads an image, generates a depth map, and returns the image.

        Args:
            input (bytes): The input image in bytes.

        Returns:
            str: Base64 encoded depth map image.
        """

        # 1. Load Image
        try:
            logger.info("Loading image from bytes")
            # Convert to RGB to discard alpha channels which can break the model
            image = Image.open(io.BytesIO(input)).convert("RGB")
        except UnidentifiedImageError:
            logger.error("File is not a valid image")
            return False
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return False

        # 3. Inference
        try:
            logger.info("Generating depth map...")
            # The pipeline returns a dictionary containing a PIL Image under the 'depth' key
            result = self.pipe(image)
            depth_image = result["depth"]
        except Exception as e:
            logger.error(f"Inference failed during depth map generation: {e}")
            return False

        # 4. Save Output
        try:
            buffered = io.BytesIO()
            depth_image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.info("Depth map successfully generated and encoded.")
            return encoded_image
        except Exception as e:
            logger.error(f"Failed to encode depth map: {e}")
            return False


router = APIRouter(prefix="/depthmap", tags=["Depth Map Generation"])


@router.post("/")
def generate_depth_map(request: DepthMapRequest) -> dict:
    """
    API endpoint to generate a depth map from an input image.

    Args:
        input_image (str): Base64 encoded input image.

    Returns:
        dict: A response containing success status and message.
    """
    generator = DepthMapGenerator()
    raw = request.input_image
    img_bytes = base64.b64decode(raw.split(",")[1] if "," in raw else raw)
    depth_map = generator.process(img_bytes)

    if depth_map:
        return Response(content=base64.b64decode(depth_map), media_type="image/png")
    else:
        return {"success": False, "message": "Failed to generate depth map."}
