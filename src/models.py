import os
from pydantic import BaseModel


class LoRA(BaseModel):
    name: str
    scale: float = 0.5


class DivergentSpace(BaseModel):
    depthmap: str = None
    depthmap_scale: float = None
    canny_edges: str = None
    edges_scale: float = None
    scale: float = 0.5


class ImageRequest(BaseModel):
    user_input: str
    loras: list[LoRA] = []
    model: str = os.environ.get("DEFAULT_MODEL", "juggernaut")
    strength: float = None
    reference: str = None
    depthmap: str = None
    depth_scales: list[float] = []
    canny_edges: str = None
    edges_scales: list[float] = []
    divergent_spaces: list[DivergentSpace] = []
    ip_adapter_scale: float = None
    ip_adapter_image: str = None
    lightning: bool = False
    image_seed: int = -1
    prompt_seed: int = -1
    batch_size: int = 1
