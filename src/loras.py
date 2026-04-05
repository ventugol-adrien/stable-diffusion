from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from pathlib import Path
import os
from .models import LoRA

LORAS_DIR = Path.home() / "sd_loras" 
def add_loras(pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline, loras:list[LoRA]=[]):
    for lora in loras:
        print(f"🔗 Adding LoRA: {lora}")
        pipe.load_lora_weights(LORAS_DIR / f"{lora.name}.safetensors", adapter_name=lora.name)
        pipe.set_adapters(adapter_names=[lora.name], adapter_scales=[lora.scale])
def record_lora_config():
    ...