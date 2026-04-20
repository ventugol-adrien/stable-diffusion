import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file

# ---------------------------------------------------------------------------
# torchvision compatibility shim for basicsr
# basicsr.data.degradations imports torchvision.transforms.functional_tensor
# which was removed in torchvision >= 0.15. Inject a compat module before
# any basicsr import so the wildcard import in basicsr.__init__ doesn't crash.
# ---------------------------------------------------------------------------
if "torchvision.transforms.functional_tensor" not in sys.modules:
    try:
        import torchvision.transforms.functional as _tvf

        _compat = types.ModuleType("torchvision.transforms.functional_tensor")
        _compat.rgb_to_grayscale = _tvf.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _compat
    except Exception:
        pass

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

CWD = Path(os.getcwd())
UPSCALE_CACHE_DIR = CWD / "caches" / "artifacts" / "realesrgan"
UPSCALE_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
UPSCALE_MODEL_PATH = UPSCALE_CACHE_DIR / "RealESRGAN_x4plus.pth"

_cached_upscaler: RealESRGANer | None = None


def _ensure_upscale_model() -> Path:
    UPSCALE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if UPSCALE_MODEL_PATH.is_file():
        print(f"📦 Using local cached ESRGAN weights: {UPSCALE_MODEL_PATH}")
        return UPSCALE_MODEL_PATH

    print(f"🌐 Downloading ESRGAN weights (first run): {UPSCALE_MODEL_URL}")
    download_url_to_file(UPSCALE_MODEL_URL, str(UPSCALE_MODEL_PATH), progress=True)
    return UPSCALE_MODEL_PATH


def get_upscaler() -> RealESRGANer:
    global _cached_upscaler
    if _cached_upscaler is not None:
        return _cached_upscaler

    model_path = _ensure_upscale_model()
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    _cached_upscaler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return _cached_upscaler


def upscale_image(image: Image.Image, outscale: int = 2) -> Image.Image:
    """Upscale with Real-ESRGAN in pixel space; returns original image on failure."""
    try:
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        upscaled_bgr, _ = get_upscaler().enhance(cv_img, outscale=outscale)
        return Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"⚠️ Upscaling failed, continuing with original image: {e}")
        return image


def cleanup_upscaler():
    global _cached_upscaler
    if _cached_upscaler is not None:
        del _cached_upscaler
        _cached_upscaler = None
