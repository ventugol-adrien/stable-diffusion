import math
import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Literal

router = APIRouter(prefix="/host", tags=["host"])


class Procedures:
    TEXT2IMAGE = "text2image"
    IMAGE2IMAGE = "image2image"
    CONTROLNET = "controlnet"
    COMPEL = "compel"
    HIRES_FIX = "hires_fix"


class ImageFit(BaseModel):
    procedures: List[str] = Field(
        Procedures.TEXT2IMAGE, description="List of procedures to apply to the image(s)"
    )
    height: int = Field(1024, description="Height of the output image(s)")
    width: int = Field(1024, description="Width of the output image(s)")


class FitRequest(BaseModel):
    images: List[ImageFit] = Field(
        ..., description="List of images to fit with their respective procedures"
    )
    vram_used_gb: float = Field(
        0.0,
        description="Projected VRAM already in use (GB). Use to simulate fit at a different usage level.",
    )


VRAM_INCREASE_PER_1024 = 0.55
BASE_VRAM_REQUIREMENT_GB = 7.45


@router.post("/fit")
async def fit(request: FitRequest):
    images_vram = sum(
        (image.height * image.width) / (1024 * 1024) * VRAM_INCREASE_PER_1024
        for image in request.images
    )
    total_required = BASE_VRAM_REQUIREMENT_GB + images_vram

    total = torch.cuda.get_device_properties(0).total_memory
    total_gb = total / (1024**3)
    available_gb = max(0, total_gb - total_required - request.vram_used_gb)

    batcheable = available_gb >= 0
    max_1024_batch = max(
        0,
        math.floor(
            (total_gb - BASE_VRAM_REQUIREMENT_GB - request.vram_used_gb)
            / VRAM_INCREASE_PER_1024
        ),
    )

    return {
        "base_vram_gb": BASE_VRAM_REQUIREMENT_GB,
        "images_vram_gb": round(images_vram, 2),
        "total_required_gb": round(total_required, 2),
        "available_gb": round(available_gb, 2),
        "batcheable": batcheable,
        "max_1024_batch": max_1024_batch,
    }
