import os
from pydantic import BaseModel


class LoRA(BaseModel):
    name:str
    scale:float = 0.5


class ImageRequest(BaseModel):
    user_input: str
    loras: list[LoRA] = []
    model: str = os.environ.get("DEFAULT_MODEL", "juggernaut")
    reference_strength: float = None
    reference: bytes = None
    lightning: bool = False
    image_seed: int = -1
    prompt_seed: int = -1
    batch_size: int = 1
