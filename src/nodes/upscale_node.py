import os
from pathlib import Path
import cv2
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image
import torch
from src.nodes.base_node import BaseNode
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np

CWD = Path(os.getcwd())
UPSCALE_CACHE_DIR = CWD / "caches" / "artifacts" / "realesrgan"
UPSCALE_MODEL_PATH = UPSCALE_CACHE_DIR / "RealESRGAN_x4plus.pth"


class UpscaleInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scale: int = Field(4, description="Upscaling factor (e.g., 2, 4, 8)")
    images: list[Image.Image] = Field(
        default_factory=list, description="List of images to upscale"
    )


class UpscaleNode(BaseNode):
    output_key = "images"

    def __init__(self, inputs: UpscaleInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "upscale"
        self.images: list[Image.Image] = []

        self.upscaler = RealESRGANer(
            scale=4,
            model_path=str(UPSCALE_MODEL_PATH),
            model=RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    @staticmethod
    def _to_pil(image: Image.Image | torch.Tensor) -> Image.Image:
        if isinstance(image, torch.Tensor):
            # diffusers pt output: float32 (C, H, W) in [0, 1]
            arr = (
                (image.float().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            )
            return Image.fromarray(arr, mode="RGB")
        return image

    def __call__(
        self, images: list[Image.Image] | torch.Tensor = None, *args, **kwargs
    ):
        raw = images if images is not None else self.images
        # Handle batch tensor (N, C, H, W) or list
        if isinstance(raw, torch.Tensor):
            raw = [raw[i] for i in range(raw.shape[0])]
        image = self._to_pil(raw[0])
        outscale = self.params.scale
        try:
            cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            upscaled_bgr, _ = self.upscaler.enhance(cv_img, outscale=outscale)
            result = Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"⚠️ Upscaling failed, continuing with original image: {e}")
            result = image
        return {"images": [result]}
