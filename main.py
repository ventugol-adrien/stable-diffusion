import base64
from contextlib import asynccontextmanager
import io, os, json, time
from random import randint
from pathlib import Path
import zipfile
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.models import ImageRequest
from compel import CompelForSDXL
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from src.pipeline import (
    get_pipe,
    get_fast_pipe,
    warmup_pipeline,
    generate_image,
    shutdown,
)
from src.loras import add_loras, record_lora_config
from src.prompt import process_prompt

import torch

MODELS_DIR = Path.home() / "sd_models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load the default pipeline and run a warmup inference to populate
    MIOpen, Triton, and TunableOp runtime caches. This moves the ~250s cold
    start from the first user request to server boot.

    Shutdown: Flush TunableOp cache to disk and release VRAM.
    """
    # --- STARTUP ---

    skip_warmup = os.environ.get("SKIP_PIPELINE_WARMUP", "0") == "1"
    if skip_warmup:
        print("🚀 Lifespan startup: skipping pipeline warmup due to -ollama flag...")
    else:
        print("🚀 Lifespan startup: loading default pipeline + warmup...")
        # Enable TunableOp write-on-exit as a safety net (in addition to explicit flush)
        try:
            if torch.cuda.tunable.is_enabled():
                torch.cuda.tunable.write_file_on_exit(True)
        except Exception:
            pass
        pipe = get_pipe(os.environ.get("DEFAULT_MODEL", "juggernaut"))
        warmup_pipeline(pipe)
        del pipe  # Don't hold a reference — cleanup_resources() must be able to free it
        print("✅ Server ready. First user request will be fast.")

    yield  # --- Server runs here ---

    # --- SHUTDOWN ---
    print("🛑 Lifespan shutdown: flushing caches and releasing VRAM...")
    shutdown()


app = FastAPI(lifespan=lifespan)

origins = json.loads(os.environ.get("ORIGINS", "[]"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins (use ["*"] to allow all)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.post("/generate/image/")
def handle_generate_image(request: ImageRequest):
    breakdown = {}
    start_time = time.monotonic()

    pipe = (
        get_fast_pipe(request.model) if request.lightning else get_pipe(request.model)
    )

    t_to_pipeline = time.monotonic() - start_time
    breakdown["pipeline_load_time"] = t_to_pipeline

    # Track which LoRA combos are used (currently none here)
    if request.loras:
        add_loras(pipe, request.loras)

    t_to_loras = time.monotonic() - t_to_pipeline
    breakdown["lora_load_time"] = t_to_loras

    positive_prompt, negative_prompt = process_prompt(request.user_input)

    compel_proc = CompelForSDXL(pipe=pipe, device="cuda")
    conditioning = compel_proc(positive_prompt, negative_prompt=negative_prompt)

    t_to_prompt = time.monotonic() - t_to_loras
    breakdown["prompt_processing_time"] = t_to_prompt

    init_image = None
    if request.reference:
        print("🖼️ Reference image provided, preparing for img2img generation...")
        init_image = request.reference
        if "," in init_image:
            # Split at the comma and keep only the actual data portion
            init_image = init_image.split(",")[1]
        image_bytes = base64.b64decode(init_image)
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)

    images = generate_image(
        pipe=pipe,
        prompt_embeds=conditioning.embeds,
        pooled_prompt_embeds=conditioning.pooled_embeds,
        negative_prompt_embeds=conditioning.negative_embeds,
        negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
        image=init_image,
        strength=request.strength if request.strength is not None else 0.6,
        num_inference_steps=8 if request.lightning else 30,
        cfg=1.5 if request.lightning else 7.0,
        height=1024,
        width=1024,
        num_images_per_prompt=request.batch_size,
    )
    t_to_generation = time.monotonic() - t_to_prompt
    breakdown["generation_time"] = t_to_generation
    # record_lora_config(request.model, request.loras)

    latency = time.monotonic() - start_time
    throughput = request.batch_size / latency if latency > 0 else 0

    zip_buffer = io.BytesIO()
    metrics = {"latency": latency, "throughput": throughput, "breakdown": breakdown}

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("metrics.json", json.dumps(metrics))
        for i, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            zip_file.writestr(f"image_{i}.png", img_buffer.getvalue())

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=results.zip",
        },
    )


@app.get("/models/")
def get_models():
    """Return a list of available models and their active LoRAs."""
    model_safetensors = [f.stem for f in MODELS_DIR.glob("*.safetensors")]

    return JSONResponse(content=model_safetensors, media_type="application/json")
