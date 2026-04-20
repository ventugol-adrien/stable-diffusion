from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from pathlib import Path
import os
from .models import LoRA
from fastapi import APIRouter
from fastapi.responses import JSONResponse

LORAS_DIR = Path.home() / "sd_loras"


def add_loras(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline,
    loras: list[LoRA] = [],
):
    existing = {name for names in pipe.get_list_adapters().values() for name in names}
    requested = {l.name for l in loras}

    # Unload adapters no longer needed
    stale = existing - requested
    for name in stale:
        pipe.delete_adapters([name])

    for lora in loras:
        if lora.name not in existing:
            weight_name = f"{lora.name}.safetensors"
            lora_path = LORAS_DIR / weight_name
            if not lora_path.is_file():
                print(f"⚠️  LoRA file not found, skipping: {lora_path}")
                continue
            print(f"🔗 Adding LoRA: {lora}")
            try:
                # Pass directory + weight_name separately; passing a bare file Path
                # is rejected by some diffusers versions in lora_state_dict.
                pipe.load_lora_weights(
                    str(LORAS_DIR), weight_name=weight_name, adapter_name=lora.name
                )
            except Exception as exc:
                print(f"⚠️  Failed to load LoRA '{lora.name}': {exc}")

    loaded = set(pipe.get_list_adapters().keys())
    active = [(l.name, l.scale) for l in loras if l.name in loaded]
    if active:
        names, weights = zip(*active)
        pipe.set_adapters(adapter_names=list(names), adapter_weights=list(weights))
    elif not loras and existing:
        pipe.set_adapters(adapter_names=[], adapter_weights=[])


def record_lora_config(): ...


router = APIRouter(prefix="/loras", tags=["loras"])


@router.get("/")
async def list_loras():
    return JSONResponse(content=[file.stem for file in LORAS_DIR.glob("*.safetensors")])
