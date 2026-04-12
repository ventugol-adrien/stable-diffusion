import base64
from contextlib import asynccontextmanager
import io, os, json, time
from random import randint
from pathlib import Path
import zipfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.models import ImageRequest
from compel import CompelForSDXL
from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
from PIL import Image, ImageOps
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from src.pipeline import (
    get_pipe,
    get_fast_pipe,
    warmup_pipeline,
    generate_image,
    shutdown,
    MODEL_CACHE_DIR,
)
from src.loras import add_loras, record_lora_config, router as loras_router
from src.prompt import process_prompt

from src.controlnet import router as depthmap_router

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

app.include_router(depthmap_router)
app.include_router(loras_router)


@app.post("/generate/image")
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

        if init_image.width < 1024 or init_image.height < 1024:
            upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
            ).to("cuda")
            init_image = upscale_pipe(prompt=positive_prompt, image=init_image).images[
                0
            ]
        init_image = ImageOps.fit(init_image, (1024, 1024), method=Image.LANCZOS)
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)

    # 1. Initialize dynamic lists for Multi-ControlNet
    controlnets = []
    control_images = []
    control_scales = []

    # 2. Process Depth Prior (Xinsir SOTA)
    if request.depthmap:
        print("🕳️ Depth map provided, loading Xinsir Depth ControlNet...")
        depthmap_data = request.depthmap
        if "," in depthmap_data:
            depthmap_data = depthmap_data.split(",")[1]
        depthmap_bytes = base64.b64decode(depthmap_data)
        depthmap_img = Image.open(io.BytesIO(depthmap_bytes)).convert("RGB")
        depthmap_img = ImageOps.fit(depthmap_img, (1024, 1024), method=Image.LANCZOS)
        control_images.append(depthmap_img)
        controlnets.append(
            ControlNetModel.from_pretrained(
                "xinsir/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
            )
        )
        control_scales.append(
            request.depth_scales[0] if request.depth_scales else 0.5
        )  # Default weight for structural depth

    # 3. Process Canny Prior (Xinsir SOTA)
    if request.canny_edges:
        print("✏️ Canny map provided, loading Xinsir Canny ControlNet...")
        canny_data = request.canny_edges
        if "," in canny_data:
            canny_data = canny_data.split(",")[1]
        canny_bytes = base64.b64decode(canny_data)
        canny_img = Image.open(io.BytesIO(canny_bytes)).convert("RGB")
        canny_img = ImageOps.fit(canny_img, (1024, 1024), method=Image.LANCZOS)
        control_images.append(canny_img)
        controlnets.append(
            ControlNetModel.from_pretrained(
                "xinsir/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
            )
        )
        control_scales.append(
            request.edges_scales[0] if request.edges_scales else 0.4
        )  # Default weight for fine edge details

    # 4. Initialize Pipeline if any spatial priors exist
    if controlnets:
        print(f"🚀 Initializing SDXL Pipeline with {len(controlnets)} ControlNet(s)...")

        # The Diffusers pipeline natively accepts a list of ControlNet models
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_CACHE_DIR / request.model,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to("cuda")

        # Note: During the actual generation call (pipe(...)), you MUST pass:
        # image=control_images
        # controlnet_conditioning_scale=control_scales

        # 5. IP-Adapter Integration
        if request.ip_adapter_image and request.ip_adapter_scale:
            print("🧩 IP-Adapter image and scale provided, adding to pipeline...")
            ip_adapter_image_bytes = base64.b64decode(request.ip_adapter_image)
            ip_adapter_image = Image.open(io.BytesIO(ip_adapter_image_bytes)).convert(
                "RGB"
            )
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
            pipe.set_ip_adapter_scale(request.ip_adapter_scale)

    images = generate_image(
        pipe=pipe,
        prompt_embeds=conditioning.embeds,
        pooled_prompt_embeds=conditioning.pooled_embeds,
        negative_prompt_embeds=conditioning.negative_embeds,
        negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
        image=control_images if control_images else None,
        controlnet_conditioning_scale=control_scales if control_scales else None,
        num_inference_steps=8 if request.lightning else 30,
        guidance_scale=1.5 if request.lightning else 7.0,  # Fixed from 'cfg'
        height=1024,
        width=1024,
        num_images_per_prompt=request.batch_size,
        ip_adapter_image=ip_adapter_image if request.ip_adapter_image else None,
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


@app.get("/models")
def get_models():
    """Return a list of available models and their active LoRAs."""
    model_safetensors = [f.stem for f in MODELS_DIR.glob("*.safetensors")]

    return JSONResponse(content=model_safetensors, media_type="application/json")
