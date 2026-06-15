import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# Load .env before anything else so all subsequent os.getenv() calls see the vars.
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# Captured before uvicorn mutates argv — used by /cleanup to execve-restart.
_ORIG_ARGV = sys.argv[:]
from src.nodes.compel_node import CompelInputs, CompelNode
from src.llama import pause_llm
from src.classes import PNGStreamingResponse, ZipStreamingResponse
from typing import AsyncIterable


class _TeeStream:
    """Mirror writes to both the original stream and a file."""

    def __init__(self, original_stream, file_handle):
        self._original_stream = original_stream
        self._file_handle = file_handle

    def write(self, data):
        self._original_stream.write(data)
        self._file_handle.write(data)
        return len(data)

    def flush(self):
        self._original_stream.flush()
        self._file_handle.flush()

    def isatty(self):
        return self._original_stream.isatty()


_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_DIR = Path(os.environ.get("SD_LOG_DIR", _PROJECT_ROOT / "logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ.setdefault("SD_RUN_ID", _RUN_ID)
os.environ.setdefault("SD_LOG_DIR", str(_LOG_DIR))
os.environ.setdefault("SD_GPU_TELEMETRY_JSONL", str(_LOG_DIR / f"perf_{_RUN_ID}.jsonl"))
_STDOUT_LOG_FILE = _LOG_DIR / f"runtime_{_RUN_ID}.stdout.log"
_STDERR_LOG_FILE = _LOG_DIR / f"runtime_{_RUN_ID}.stderr.log"

# Capture everything emitted by Python + native libraries to stdout/stderr.
if os.environ.get("SD_STREAM_LOGS_TO_FILES", "1") == "1":
    _stdout_fh = open(_STDOUT_LOG_FILE, "a", buffering=1)
    _stderr_fh = open(_STDERR_LOG_FILE, "a", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, _stdout_fh)
    sys.stderr = _TeeStream(sys.stderr, _stderr_fh)

# Route HIP and rocBLAS logs to project-local files for post-mortem analysis.
os.environ.setdefault("AMD_LOG_LEVEL_FILE", str(_LOG_DIR / f"hip_{_RUN_ID}.log"))
os.environ.setdefault("ROCBLAS_LOG_PATH", str(_LOG_DIR / f"rocblas_{_RUN_ID}.log"))

# ── ROCm / MIOpen environment overrides ────────────────────────────────────────
# Only applied on ROCm — CUDA systems don't need or want these variables.
# Must be set BEFORE torch or diffusers are imported; uvicorn does not source
# ~/.bashrc so these would be absent from the process environment otherwise.
# src.utils._detect_gpu() uses only subprocess (rocminfo/nvidia-smi), safe here.
from src.utils import (
    is_rocm as _is_rocm,
    rocm_profile as _rocm_profile,
    is_vram_pressure_high,
    stream_image,
    stream_zip,
)

if _is_rocm():
    _profile = _rocm_profile()
    _debug_logs = os.environ.get("SD_ROCM_DEBUG_LOGS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if _profile == "w7800":
        # Let gfx1100 run natively. Clear stale parent-shell overrides from older
        # hardware profiles so ROCm compiles for the card actually installed.
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "12.0.0":
            os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    elif _profile == "compat":
        # Compatibility keeps the prior conservative override available for the
        # older gfx1200-oriented runtime profile.
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.0")
    # MIOPEN_FIND_MODE=2 — skips exhaustive kernel benchmarking; uses the
    #   pre-compiled heuristic DB instead. Benchmarking mode executes micro-kernels
    #   that contain unresolved memory-indexing bugs on the narrow (wf32) wavefront.
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    # FLASH_ATTENTION_TRITON_AMD_ENABLE — routes SDPA through AOTriton, generating
    #   native wf32 instructions instead of the wf64-defaulting CDNA path.
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # PYTORCH_HIP_ALLOC_CONF — expandable_segments is unsupported on ROCm and
    #   causes the allocator to stall during large VAE decode allocations, which
    #   manifests as hipErrorLaunchFailure in the conv upsampler. Hard-override
    #   (not setdefault) so a stale ~/.bashrc value cannot sneak back in.
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    if _debug_logs:
        # Enable ROCm runtime and GEMM logging at moderate verbosity only when
        # explicitly debugging. Leaving these on during perf runs creates a lot
        # of synchronous stderr/file traffic around the hot path.
        # ROCBLAS_LAYER=1 logs API call shapes only; value 2 triggers an exhaustive
        # benchmark sweep of every solution per GEMM, causing minutes of pre-denoise stall.
        os.environ.setdefault("ROCBLAS_LAYER", "1")
        # MIOpen logs go to stderr; stderr is mirrored to project log files above.
        os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "1")
        os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")
        # TORCH_LOGS="+inductor" prints inductor compilation decisions.
        os.environ.setdefault("TORCH_LOGS", "+inductor")
    # ROCBLAS_USE_HIPBLASLT=0 was tested to force Tensile backend for the gfx1200
    # VAE upsampler SIGABRT, but it broke dispatch for standard GEMM shapes in the
    # CLIP text encoder (RuntimeError: Expected iter != ops_.end()).  Do not re-enable.
# ───────────────────────────────────────────────────────────────────────────────

import base64
from contextlib import asynccontextmanager
import io, json, time, asyncio
from random import randint
import zipfile
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.nodes.tiling_node import TilingInputs, TilingNode
from src.nodes.upscale_node import UpscaleInputs, UpscaleNode
from src.nodes.hi_res_node import HiResInputs, HiResNode
from src.nodes.transform_node import TransformInputs, TransformNode
from src.nodes.outpainting_node import OutpaintingInputs, OutpaintingNode
from src.nodes.image2image import Image2ImageNode, Image2ImageInputs
from src.nodes.response_node import ResponseInputs, ResponseNode
from src.nodes.text2image import Text2ImageInputs, Text2ImageNode
from src.nodes.qwen_node import (
    QwenInputs,
    QwenNode,
    replace_negative_prompt_embeds,
)
from src.nodes.spatial_assets_node import SpatialAssetsNode, SpatialAssetsInputs

# from src.nodes.compel_node import CompelInputs, CompelNode
from src.executor import execute_dag
from src.nodes.base_node import BaseNode
from src.models import (
    DAGForm,
    Image2ImageRequest,
    ImageRequest,
    OutpaintRequest,
    SpatialAssetsRequest,
    Text2ImageRequest,
)
from compel import CompelForSDXL
from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
)
from PIL import Image, ImageOps
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from src.pipeline import (
    cleanup_resources,
    get_pipe,
    get_fast_pipe,
    warmup_pipeline,
    generate_image,
    shutdown,
    MODEL_CACHE_DIR,
    log_runtime_diagnostics_once,
)
from src.loras import add_loras, record_lora_config, router as loras_router
from src.prompt import process_prompt

from src.controlnet import router as depthmap_router
from src.host import router as host_router

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

    print(f"📝 Log directory: {_LOG_DIR}")
    print(f"📝 Runtime stdout log: {_STDOUT_LOG_FILE}")
    print(f"📝 Runtime stderr log: {_STDERR_LOG_FILE}")
    print(f"📝 HIP log: {os.environ.get('AMD_LOG_LEVEL_FILE', '<unset>')}")
    print(f"📝 rocBLAS log: {os.environ.get('ROCBLAS_LOG_PATH', '<unset>')}")
    log_runtime_diagnostics_once()

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
    allow_origins=["*"],  # Allows specific origins (use ["*"] to allow all)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(depthmap_router)
app.include_router(loras_router)
app.include_router(host_router)


@app.post("/generate/image")
async def handle_generate_image(
    request: ImageRequest = Depends(ImageRequest.as_form), stream: bool = False
):
    cleanup_resources()
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
        init_image = Image.open(request.reference.file).convert("RGB")
        print(
            f"🖼️ Reference image loaded: size={init_image.size} mode={init_image.mode}"
        )
        init_image = ImageOps.fit(init_image, (1024, 1024), method=Image.LANCZOS)
        print(f"🖼️ Reference image resized to {init_image.size}")
        print(
            f"🔀 Converting pipe ({type(pipe).__name__}) → AutoPipelineForImage2Image..."
        )
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)
        print(f"🔀 Converted to {type(pipe).__name__}")

    # 1. Initialize dynamic lists for Multi-ControlNet
    controlnets = []
    control_images = []
    control_scales = []

    # 2. Process Depth Prior (Xinsir SOTA)
    if request.depthmap:
        print("🕳️ Depth map provided, loading Xinsir Depth ControlNet...")
        depthmap_img = Image.open(request.depthmap.file).convert("RGB")
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
        canny_img = Image.open(request.canny_edges.file).convert("RGB")
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

    # 4. Process Divergent Spaces (Heterogeneous Control Batching)
    has_mask_in_divergent = False
    reference_tensor = None
    mask_tensor = None
    active_mask_strength = 1.0

    if request.divergent_spaces:
        if len(request.divergent_spaces) != request.batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Number of divergent spaces ({len(request.divergent_spaces)}) must match batch size ({request.batch_size}).",
            )
        print(
            f"🌌 Divergent Spaces provided. Pre-computing sparse tensors for batch size {request.batch_size}..."
        )
        import torchvision.transforms.functional as TF

        batch_size = request.batch_size
        has_depth = any(ds.depthmap for ds in request.divergent_spaces)
        has_canny = any(ds.canny_edges for ds in request.divergent_spaces)
        has_mask_in_divergent = any(ds.mask for ds in request.divergent_spaces)

        target_width, target_height = 1024, 1024
        if has_mask_in_divergent:
            for ds in request.divergent_spaces:
                if ds.mask and ds.reference:
                    ref_img = Image.open(ds.reference.file).convert("RGB")
                    target_width = ref_img.width - (ref_img.width % 8)
                    target_height = ref_img.height - (ref_img.height % 8)
                    break

        if has_depth:
            depth_tensor = torch.zeros(
                (batch_size, 3, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            active_depth_scale = 0.5
            for i in range(batch_size):
                space = request.divergent_spaces[i]
                if space.depthmap:
                    img = Image.open(space.depthmap.file).convert("RGB")
                    if has_mask_in_divergent:
                        img = ImageOps.fit(
                            img, (target_width, target_height), method=Image.LANCZOS
                        )
                    else:
                        img = ImageOps.fit(img, (1024, 1024), method=Image.LANCZOS)
                    img_tensor = TF.to_tensor(img).to(
                        device="cuda", dtype=torch.float16
                    )
                    depth_tensor[i] = img_tensor
                    if space.depthmap_scale is not None:
                        active_depth_scale = space.depthmap_scale
            controlnets.append(
                ControlNetModel.from_pretrained(
                    "xinsir/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
                )
            )
            control_images.append(depth_tensor)
            control_scales.append(active_depth_scale)

        if has_canny:
            canny_tensor = torch.zeros(
                (batch_size, 3, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            active_canny_scale = 0.4
            for i in range(batch_size):
                space = request.divergent_spaces[i]
                if space.canny_edges:
                    img = Image.open(space.canny_edges.file).convert("RGB")
                    if has_mask_in_divergent:
                        img = ImageOps.fit(
                            img, (target_width, target_height), method=Image.LANCZOS
                        )
                    else:
                        img = ImageOps.fit(img, (1024, 1024), method=Image.LANCZOS)
                    img_tensor = TF.to_tensor(img).to(
                        device="cuda", dtype=torch.float16
                    )
                    canny_tensor[i] = img_tensor
                    if space.edges_scale is not None:
                        active_canny_scale = space.edges_scale
            controlnets.append(
                ControlNetModel.from_pretrained(
                    "xinsir/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
                )
            )
            control_images.append(canny_tensor)
            control_scales.append(active_canny_scale)

        if has_mask_in_divergent:
            mask_tensor = torch.ones(
                (batch_size, 1, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            reference_tensor = torch.zeros(
                (batch_size, 3, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            for i in range(batch_size):
                space = request.divergent_spaces[i]
                if space.mask and space.reference:
                    mask_img = Image.open(space.mask.file).convert("L")
                    mask_img = ImageOps.fit(
                        mask_img, (target_width, target_height), method=Image.LANCZOS
                    )
                    mask_tensor[i] = TF.to_tensor(mask_img).to(
                        device="cuda", dtype=torch.float16
                    )

                    ref_img = Image.open(space.reference.file).convert("RGB")
                    ref_img = ImageOps.fit(
                        ref_img, (target_width, target_height), method=Image.LANCZOS
                    )
                    reference_tensor[i] = TF.to_tensor(ref_img).to(
                        device="cuda", dtype=torch.float16
                    )

                    if space.mask_strength is not None:
                        active_mask_strength = space.mask_strength

    # 5. Initialize Pipeline if any spatial priors exist
    if controlnets:
        print(f"🚀 Initializing SDXL Pipeline with {len(controlnets)} ControlNet(s)...")

        # The Diffusers pipeline natively accepts a list of ControlNet models
        if has_mask_in_divergent:
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                MODEL_CACHE_DIR / request.model,
                controlnet=controlnets,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                MODEL_CACHE_DIR / request.model,
                controlnet=controlnets,
                torch_dtype=torch.float16,
            ).to("cuda")

        # Note: During the actual generation call (pipe(...)), you MUST pass:
        # image=control_images
        # controlnet_conditioning_scale=control_scales

        # 5. IP-Adapter Integration
        if request.ip_adapter_image and request.ip_adapter_scale:
            print("🧩 IP-Adapter image and scale provided, adding to pipeline...")
            ip_adapter_image = Image.open(request.ip_adapter_image.file).convert("RGB")
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
            pipe.set_ip_adapter_scale(request.ip_adapter_scale)

    # When ControlNets are active with batch_size > 1, the control image tensors
    # already carry the batch dimension. Expand prompt embeds to match and set
    # num_images_per_prompt=1 so diffusers' check_inputs won't reject the mismatch.
    prompt_embeds = conditioning.embeds
    pooled_prompt_embeds = conditioning.pooled_embeds
    negative_prompt_embeds = conditioning.negative_embeds
    negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds
    num_images = request.batch_size
    if control_images and batch_size > 1:
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
            batch_size, 1
        )
        num_images = 1

    gen_kwargs = {
        "pipe": pipe,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "num_inference_steps": 8 if request.lightning else 30,
        "guidance_scale": 1.5 if request.lightning else 7.0,  # Fixed from 'cfg'
        "num_images_per_prompt": num_images,
        "seed": request.image_seed,
    }

    # height/width are text2image-only; img2img derives dimensions from the input image.
    # Inpaint (has_mask_in_divergent) uses tensor inputs that already carry dimensions.
    if init_image is None and not has_mask_in_divergent:
        gen_kwargs["height"] = target_height if "target_height" in locals() else 1024
        gen_kwargs["width"] = target_width if "target_width" in locals() else 1024

    if request.ip_adapter_image:
        gen_kwargs["ip_adapter_image"] = ip_adapter_image

    if controlnets:
        gen_kwargs["controlnet_conditioning_scale"] = control_scales
        gen_kwargs["control_guidance_end_step"] = 0.5

    if has_mask_in_divergent:
        gen_kwargs["image"] = reference_tensor
        gen_kwargs["mask_image"] = mask_tensor
        gen_kwargs["control_image"] = control_images if control_images else None
        gen_kwargs["strength"] = active_mask_strength
    elif init_image is not None:
        gen_kwargs["image"] = init_image
        if request.strength is not None:
            gen_kwargs["strength"] = request.strength
        if control_images:
            gen_kwargs["control_image"] = control_images
    else:
        gen_kwargs["image"] = control_images if control_images else None

    loggable = {
        k: (type(v).__name__ if hasattr(v, "__class__") else v)
        for k, v in gen_kwargs.items()
        if k != "pipe"
    }
    loggable["pipe_type"] = type(gen_kwargs["pipe"]).__name__
    print(f"🚀 generate_image kwargs: {loggable}")

    images = generate_image(**gen_kwargs)

    t_to_generation = time.monotonic() - t_to_prompt
    breakdown["generation_time"] = t_to_generation
    # record_lora_config(request.model, request.loras)

    latency = time.monotonic() - start_time
    throughput = request.batch_size / latency if latency > 0 else 0

    metrics = {"latency": latency, "throughput": throughput, "breakdown": breakdown}

    if stream:
        if len(images) == 1:
            img_buffer = io.BytesIO()
            image: Image.Image = images[0]
            if stream:
                print("Streaming response...")
                return PNGStreamingResponse(
                    stream_image(image),
                    headers={
                        "Content-Disposition": "inline; filename=result.png",
                        "X-Metrics-Latency": str(latency),
                        "X-Metrics-Throughput": str(throughput),
                        "X-Metrics-Breakdown": json.dumps(breakdown),
                    },
                )
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("metrics.json", json.dumps(metrics))
                for i, img in enumerate(images):
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="PNG")
                    zip_file.writestr(f"image_{i}.png", img_buffer.getvalue())
            print("Streaming ZIP response...")
            return ZipStreamingResponse(
                stream_zip(zip_buffer),
                headers={
                    "Content-Disposition": "attachment; filename=results.zip",
                },
            )
    if len(images) == 1:
        img_buffer = io.BytesIO()
        images[0].save(img_buffer, format="PNG")
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=result.png",
                "X-Metrics-Latency": str(latency),
                "X-Metrics-Throughput": str(throughput),
                "X-Metrics-Breakdown": json.dumps(breakdown),
            },
        )

    zip_buffer = io.BytesIO()
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


@app.post("/workflows/")
def execute_workflows(request: DAGForm = Depends(DAGForm.as_form)):
    cleanup_resources()
    model = request.nodes["1"].model
    hires_strength = request.hires_strength

    _HIRES_PROMPT = (
        "masterpiece, ultra-detailed, sharp focus, 8k, photorealistic, "
        "intricate textures, subsurface scattering, fine detail, crisp"
    )
    _HIRES_NEGATIVE = (
        "blurry, soft focus, low resolution, jpeg artifacts, "
        "watermark, oversmoothed, deformed"
    )

    qwen_node = QwenNode(request.nodes["0"])
    # compel_node = CompelNode(request.nodes["0"])
    hires_qwen_node = QwenNode(
        QwenInputs(prompt=_HIRES_PROMPT, negative_prompt=_HIRES_NEGATIVE, model=model)
    )
    # hires_compel_node = CompelNode(
    #     CompelInputs(
    #         prompt=_HIRES_PROMPT,
    #         negative_prompt=_HIRES_NEGATIVE,
    #         model=model,
    #     )
    # )
    if request.init_image is not None:
        image_node = Image2ImageNode(request.nodes["1"])
    else:
        image_node = Text2ImageNode(request.nodes["1"])
    upscale_node = UpscaleNode(UpscaleInputs(scale=4))
    tiling_node = TilingNode(TilingInputs())
    hires_node = HiResNode(HiResInputs(strength=hires_strength, model=model))
    transform_node = TransformNode(TransformInputs())
    response_node = ResponseNode()

    embeds = qwen_node()
    # embeds = compel_node()
    hires_embeds = hires_qwen_node()
    # hires_embeds = hires_compel_node()
    if request.init_image is not None:
        images = image_node(images=[request.init_image], **embeds)
    else:
        images = image_node(**embeds)
    upscaled = upscale_node(**images)
    tiling_plan = tiling_node(**upscaled)
    hires_node.embeds = hires_embeds
    hires_images = hires_node(tiling_outputs=tiling_plan)
    final_images = transform_node(**hires_images)
    return response_node(**final_images)


@app.post("/workflows/outpaint")
async def execute_outpaint_workflow(
    request: OutpaintRequest = Depends(OutpaintRequest.as_form), stream=False
):
    cleanup_resources()
    print("Pausing LLM Inference...")
    await pause_llm()

    # qwen_node = QwenNode(
    #     QwenInputs(
    #         prompt=request.user_input,
    #         negative_prompt=request.negative_input,
    #         model=request.model,
    #     )
    # )
    compel_node = CompelNode(
        CompelInputs(
            prompt=request.user_input,
            negative_prompt=request.negative_input,
            model=request.model,
        )
    )
    transform_node = TransformNode(
        TransformInputs(
            z=request.transform_z or 1.0,
            dx=request.transform_dx or 0,
            dy=request.transform_dy or 0,
            r=request.transform_r or 0.0,
        )
    )
    outpaint_node = OutpaintingNode(
        OutpaintingInputs(
            model=request.model,
            steps=request.steps,
            strength=request.strength if request.strength is not None else 1.0,
        )
    )
    # Aggressively free VRAM before outpainting, which is memory-hungry
    response_node = ResponseNode(ResponseInputs(stream=stream))
    # embeds = qwen_node()
    embeds = compel_node()
    transformed = transform_node(
        fit_resize=True, images=[Image.open(request.reference.file).convert("RGB")]
    )
    outpainted = outpaint_node(
        images=transformed["images"],
        masks=transformed.get("masks"),
        **embeds,
    )
    return response_node(**outpainted)


@app.post("/workflows/inpaint")
async def execute_inpaint_workflow(
    request: OutpaintRequest = Depends(OutpaintRequest.as_form), stream=False
):
    cleanup_resources()
    print("Pausing LLM Inference...")
    await pause_llm()

    # qwen_node = QwenNode(
    #     QwenInputs(
    #         prompt=request.user_input,
    #         negative_prompt=request.negative_input,
    #         model=request.model,
    #     )
    # )
    compel_node = CompelNode(
        CompelInputs(
            prompt=request.user_input,
            negative_prompt=request.negative_input,
            model=request.model,
        )
    )
    transform_node = TransformNode(
        TransformInputs(
            z=request.transform_z or 1.0,
            dx=request.transform_dx or 0,
            dy=request.transform_dy or 0,
            r=request.transform_r or 0.0,
        )
    )
    outpaint_node = OutpaintingNode(
        OutpaintingInputs(
            model=request.model,
            steps=request.steps,
            strength=request.strength if request.strength is not None else 1.0,
        )
    )
    # Aggressively free VRAM before outpainting, which is memory-hungry
    response_node = ResponseNode(ResponseInputs(stream=stream))
    # embeds = qwen_node()
    embeds = compel_node()
    inputs = {}
    with Image.open(request.mask.file) as mask_file:
        mask_image = mask_file.convert("L")
    with Image.open(request.reference.file) as reference_file:
        reference_image = reference_file.convert("RGB")

    transformed = transform_node(
        images=[reference_image],
        masks=[mask_image],
        fill_color="black",
    )
    if request.depth_map and request.depth_map_scale:
        print("Depth map and scale provided, adding to inpaint masked area...")
        with Image.open(request.depth_map.file) as depth_map:
            transformed_depth = transform_node(
                images=[depth_map.convert("RGB")],
                masks=[mask_image],
                fill_color="black",
            )
        inputs["depthmap"] = transformed_depth["images"][0]
        inputs["depthmap_scale"] = request.depth_map_scale
    if request.edges_map and request.edges_map_scale:
        print("Edges map and scale provided, adding to inpaint masked area...")
        with Image.open(request.edges_map.file) as edges_map:
            transformed_edges = transform_node(
                images=[edges_map.convert("RGB")],
                masks=[mask_image],
                fill_color="black",
            )
        inputs["edgesmap"] = transformed_edges["images"][0]
        inputs["edgesmap_scale"] = request.edges_map_scale
    outpainted = outpaint_node(
        images=transformed["images"],
        masks=transformed["masks"],
        **inputs,
        **embeds,
    )
    return response_node(**outpainted)


@app.post("/workflows/img2img")
async def execute_image2image_workflow(
    request: Image2ImageRequest = Depends(Image2ImageRequest.as_form), stream=False
):
    cleanup_resources()
    print("Pausing LLM Inference...")
    await pause_llm()

    # qwen_node = QwenNode(
    #     QwenInputs(
    #         prompt=request.user_input,
    #         negative_prompt=request.negative_input,
    #         model=request.model,
    #     )
    # )
    compel_node = CompelNode(
        CompelInputs(
            prompt=request.user_input,
            negative_prompt=request.negative_input,
            model=request.model,
        )
    )
    img2img_node = Image2ImageNode(
        Image2ImageInputs(
            model=request.model,
            steps=request.steps,
            strength=request.strength if request.strength is not None else 1.0,
        )
    )
    # Aggressively free VRAM before outpainting, which is memory-hungry
    response_node = ResponseNode(ResponseInputs(stream=stream))
    # embeds = qwen_node()
    embeds = compel_node()
    if request.ip_adapter_image and request.ip_adapter_scale:
        print("IP-Adapter image and scale provided, adding to Image2ImageNode...")
        image = img2img_node(
            images=[Image.open(request.reference.file).convert("RGB")],
            ip_adapter_image=Image.open(request.ip_adapter_image.file).convert("RGB"),
            ip_adapter_scale=request.ip_adapter_scale,
            **embeds,
        )
    else:
        print(
            "No IP-Adapter provided, proceeding with standard Image2Image generation..."
        )
        image = img2img_node(
            images=[Image.open(request.reference.file).convert("RGB")], **embeds
        )
    return response_node(**image, stream=stream)


@app.post("/workflows/txt2img")
async def execute_text2image_workflow(
    request: Text2ImageRequest = Depends(Text2ImageRequest.as_form),
    stream=False,
    version: str = "v1",
):
    # cleanup_resources()
    # print("Pausing LLM Inference...")
    # await pause_llm()

    txt2img_node = Text2ImageNode(
        Text2ImageInputs(
            cfg_scale=request.cfg_scale,
            model=request.model,
            steps=request.steps,
        )
    )
    use_depthmap = request.depth_map and request.depth_map_scale
    use_edgesmap = request.edges_map and request.edges_map_scale
    use_controlnet = use_depthmap or use_edgesmap
    use_ip_adapter = request.ip_adapter_image and request.ip_adapter_scale

    # Aggressively free VRAM before outpainting, which is memory-hungry
    response_node = ResponseNode(ResponseInputs(stream=stream))
    embeds = _build_txt2img_conditioning(request, version)
    inputs = {}
    if use_depthmap:
        print(
            "Depth map and scale provided, adding to Text2ImageNode with ControlNet..."
        )
        inputs["depthmap"] = Image.open(request.depth_map.file).convert("RGB")
        inputs["depthmap_scale"] = request.depth_map_scale
    if use_edgesmap:
        print(
            "Edges map and scale provided, adding to Text2ImageNode with ControlNet..."
        )
        inputs["edgesmap"] = Image.open(request.edges_map.file).convert("RGB")
        inputs["edgesmap_scale"] = request.edges_map_scale
    if use_ip_adapter:
        print("IP-Adapter image and scale provided, adding to Text2ImageNode...")
        inputs["ip_adapter_image"] = Image.open(request.ip_adapter_image.file).convert(
            "RGB"
        )
        inputs["ip_adapter_scale"] = request.ip_adapter_scale
    image = txt2img_node(**inputs, **embeds)
    return response_node(**image, stream=stream)


def _build_txt2img_conditioning(request: Text2ImageRequest, version="v1"):
    if request.prompt_embedding == "qwen":
        qwen_node = QwenNode(
            QwenInputs(
                qwen_model_path=os.environ.get(
                    "QWEN_MODEL_PATH",
                    "/home/adrien/my_models/qwen3.5-27b/qwen3.5-27b.gguf",
                ),
                projector_path=f"/home/adrien/sd_projectors/qwen_sdxl_projector_{version}.gguf",
                qwen_normalize_embeddings=request.normalize_embeddings,
                use_input_layernorm=request.use_input_layernorm,
                qwen_n_gpu_layers=0,
                qwen_n_ctx=512,
                prompt=request.user_input,
                negative_prompt=request.negative_input,
                model=request.model,
                qwen_use_cached_negative_prompt_embeds=False,
            )
        )
        compel_node = CompelNode(
            CompelInputs(
                prompt=request.user_input,
                negative_prompt=request.negative_input,
                model=request.model,
            )
        )
        return replace_negative_prompt_embeds(qwen_node(), compel_node())
    else:
        compel_node = CompelNode(
            CompelInputs(
                prompt=request.user_input,
                negative_prompt=request.negative_input,
                model=request.model,
            )
        )
        return compel_node()


@app.post("/workflows/spatial_assets")
async def execute_spatial_assets_workflow(
    request: SpatialAssetsRequest = Depends(SpatialAssetsRequest.as_form),
    stream=True,
):
    cleanup_resources()
    print("Pausing LLM Inference...")
    await pause_llm()

    spatial_assets_node = SpatialAssetsNode(SpatialAssetsInputs())
    response_node = ResponseNode()
    with Image.open(request.image.file) as image:
        assets = spatial_assets_node(images=[image.convert("RGB")])
    return response_node(**assets)


@app.post("/cleanup")
def cleanup():
    cleanup_resources()

    # Python-level cleanup above frees model weights. The CUDA/HIP context
    # itself (hipBLAS workspace, MIOpen kernel cache, Triton AOT blobs) can
    # only be freed when the process exits. We execve-restart with
    # SKIP_PIPELINE_WARMUP=1 so the new process starts with zero GPU
    # allocations until the first inference request.
    def _restart():
        import time

        time.sleep(0.4)  # let the HTTP response flush
        env = os.environ.copy()
        env["SKIP_PIPELINE_WARMUP"] = "1"

        # 1. Safely resolve the absolute path to your .venv python binary
        python_exe = os.path.join(sys.prefix, "bin", "python")

        # 2. Reconstruct the startup args.
        # _ORIG_ARGV[0] is the uvicorn script path (which we drop).
        # We replace it with "-m", "uvicorn" to perfectly match your ExecStart command.
        args = [python_exe, "-m", "uvicorn"] + _ORIG_ARGV[1:]

        os.execve(python_exe, args, env)

    threading.Thread(target=_restart, daemon=False).start()
    return {"status": "restarting"}
