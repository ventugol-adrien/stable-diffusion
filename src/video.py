from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from src.controlnet import ControlNetAssetGenerator
from src.pipeline import cleanup_resources
from PIL import Image
import cv2
import gc
import io
import logging
import time
import zipfile
import tempfile
import shutil
import os
from pathlib import Path
import re
import torch

# Disable xet download backend — crashes with "Background writer channel closed"
os.environ["HF_HUB_DISABLE_XET"] = "1"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/generate", tags=["video"])

# ── Wan 2.2 A14B MoE constants ────────────────────────────────────────────────

_WAN_T2V_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_WAN_I2V_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# ── Distill tuning knobs ──────────────────────────────────────────────────────
_DISTILL_NUM_STEPS = 8  # number of denoising steps in distill mode
_DISTILL_LORA_WEIGHT_HIGH = 1.0  # LoRA weight for high-noise expert (transformer)
_DISTILL_LORA_WEIGHT_LOW = 1.0  # LoRA weight for low-noise expert (transformer_2)
_BOUNDARY_RATIO = 0.5  # MoE expert switch point (0.5 = 50/50 split)

# Evenly-spaced sigma timesteps derived from step count (e.g. 8 -> [1000,875,...,125])
_DISTILL_STEPS = [
    int(1000 - i * (1000 / _DISTILL_NUM_STEPS)) for i in range(_DISTILL_NUM_STEPS)
]

# ── LoRA directory + registry ─────────────────────────────────────────────────────
# Local: loras/i2v/<name>/high.safetensors + low.safetensors
# Registry: fallback for HuggingFace-hosted LoRAs
_LORA_DIR = Path(__file__).resolve().parent.parent / "loras"
_LORA_REGISTRY = {
    "distill": {
        "repo": "lightx2v/Wan2.2-Distill-Loras",
        "t2v_high": "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        "t2v_low": "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        "i2v_high": "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "i2v_low": "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
    },
}

_CACHES_DIR = Path(__file__).resolve().parent.parent / "caches"
_WAN_T2V_LOCAL_CACHE = _CACHES_DIR / "wan-t2v-a14b-4bit"
_WAN_I2V_LOCAL_CACHE = _CACHES_DIR / "wan-i2v-a14b-4bit"

_wan_t2v_pipe = None
_wan_i2v_pipe = None


def _parse_lora_params(form) -> list[dict]:
    """Parse loras.N.name / loras.N.weight_high / loras.N.weight_low from multipart form."""
    lora_fields: dict[int, dict] = {}
    for key in form:
        m = re.match(r"loras\.(\d+)\.(\w+)", key)
        if m:
            idx, field = int(m.group(1)), m.group(2)
            lora_fields.setdefault(idx, {})[field] = form[key]
    loras = []
    for idx in sorted(lora_fields):
        entry = lora_fields[idx]
        if "name" not in entry:
            continue
        loras.append(
            {
                "name": str(entry["name"]),
                "weight_high": float(entry.get("weight_high", 1.0)),
                "weight_low": float(entry.get("weight_low", 1.0)),
            }
        )
    return loras


def _ensure_lora_loaded(pipe, name: str, pipeline_type: str):
    """Load a LoRA adapter pair (high + low noise) by name if not already loaded."""
    adapter_high = f"{name}_high"
    adapter_low = f"{name}_low"

    if name in pipe._loaded_lora_names:
        return adapter_high, adapter_low

    # Try local directory first: loras/<pipeline_type>/<name>/high.safetensors
    local_dir = _LORA_DIR / pipeline_type / name
    high_path = local_dir / "high.safetensors"
    low_path = local_dir / "low.safetensors"

    if high_path.is_file() and low_path.is_file():
        logger.info(f"[WAN] Loading LoRA '{name}' from {local_dir}")
        pipe.load_lora_weights(
            str(local_dir), weight_name="high.safetensors", adapter_name=adapter_high
        )
        pipe.load_lora_weights(
            str(local_dir),
            weight_name="low.safetensors",
            adapter_name=adapter_low,
            load_into_transformer_2=True,
        )
    elif name in _LORA_REGISTRY:
        reg = _LORA_REGISTRY[name]
        logger.info(f"[WAN] Loading LoRA '{name}' from {reg['repo']}")
        pipe.load_lora_weights(
            reg["repo"],
            weight_name=reg[f"{pipeline_type}_high"],
            adapter_name=adapter_high,
        )
        pipe.load_lora_weights(
            reg["repo"],
            weight_name=reg[f"{pipeline_type}_low"],
            adapter_name=adapter_low,
            load_into_transformer_2=True,
        )
    else:
        raise ValueError(f"LoRA '{name}' not found in {local_dir} or registry")

    pipe._loaded_lora_names.add(name)
    logger.info(
        f"[WAN] LoRA '{name}' loaded as adapters [{adapter_high}, {adapter_low}]"
    )
    return adapter_high, adapter_low


def _setup_distill_scheduler(pipe):
    """Create an Euler scheduler with forced distill timesteps."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    import numpy as np

    pipe._base_scheduler = pipe.scheduler

    orig_cfg = pipe.scheduler.config
    euler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=orig_cfg.num_train_timesteps,
        shift=orig_cfg.get("flow_shift", 1.0),
    )

    def _distill_set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
        num_train = self.config.num_train_timesteps
        n = num_inference_steps if num_inference_steps else len(_DISTILL_STEPS)
        steps = [int(1000 - i * (1000 / n)) for i in range(n)]
        sigmas = np.array([t / num_train for t in steps], dtype=np.float32)
        sigmas[0] = min(sigmas[0], 1.0 - 1e-6)
        sigmas = np.concatenate([sigmas, [0.0]])
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = (self.sigmas[:-1] * num_train).to(
            device=device, dtype=torch.int64
        )
        self.num_inference_steps = n
        self._step_index = None
        self._begin_index = None

    euler.set_timesteps = _distill_set_timesteps.__get__(
        euler, FlowMatchEulerDiscreteScheduler
    )
    pipe._distill_scheduler = euler
    pipe.scheduler = euler


def _load_wan_t2v():
    """Load Wan 2.2 T2V A14B MoE pipeline (4-bit NF4 quantized dual transformers)."""
    global _wan_t2v_pipe

    if _wan_t2v_pipe is not None:
        return _wan_t2v_pipe

    from diffusers import AutoencoderKLWan, BitsAndBytesConfig, WanPipeline
    from diffusers.quantizers.pipe_quant_config import PipelineQuantizationConfig

    device = "cuda:0"
    local_cached = (
        _WAN_T2V_LOCAL_CACHE.is_dir()
        and (_WAN_T2V_LOCAL_CACHE / "model_index.json").is_file()
    )

    if local_cached:
        logger.info(
            f"[WAN-T2V] Loading cached 4-bit pipeline from {_WAN_T2V_LOCAL_CACHE}..."
        )
        t0 = time.monotonic()
        pipe = WanPipeline.from_pretrained(
            _WAN_T2V_LOCAL_CACHE, torch_dtype=torch.bfloat16
        )
        logger.info(f"[WAN-T2V] Cached pipeline loaded in {time.monotonic() - t0:.1f}s")
    else:
        bnb_4bit = dict(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": BitsAndBytesConfig(**bnb_4bit),
                "transformer_2": BitsAndBytesConfig(**bnb_4bit),
            }
        )
        logger.info("[WAN-T2V] Loading with 4-bit quantized dual transformers...")
        t0 = time.monotonic()
        vae = AutoencoderKLWan.from_pretrained(
            _WAN_T2V_REPO, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanPipeline.from_pretrained(
            _WAN_T2V_REPO,
            vae=vae,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        logger.info(f"[WAN-T2V] Pipeline loaded in {time.monotonic() - t0:.1f}s")

        logger.info(f"[WAN-T2V] Saving to {_WAN_T2V_LOCAL_CACHE}...")
        t0 = time.monotonic()
        pipe.save_pretrained(_WAN_T2V_LOCAL_CACHE)
        logger.info(f"[WAN-T2V] Saved in {time.monotonic() - t0:.1f}s")

    pipe.to(device)
    pipe.vae.enable_tiling()
    pipe.register_to_config(boundary_ratio=_BOUNDARY_RATIO)

    _setup_distill_scheduler(pipe)
    pipe._loaded_lora_names = set()
    logger.info("[WAN-T2V] Loaded on GPU + distill scheduler (LoRAs loaded on demand)")

    _wan_t2v_pipe = pipe
    return _wan_t2v_pipe


def _load_wan_i2v():
    """Load Wan 2.2 I2V A14B MoE pipeline (4-bit NF4 quantized dual transformers)."""
    global _wan_i2v_pipe

    if _wan_i2v_pipe is not None:
        return _wan_i2v_pipe

    from diffusers import AutoencoderKLWan, BitsAndBytesConfig, WanImageToVideoPipeline
    from diffusers.quantizers.pipe_quant_config import PipelineQuantizationConfig

    device = "cuda:0"
    local_cached = (
        _WAN_I2V_LOCAL_CACHE.is_dir()
        and (_WAN_I2V_LOCAL_CACHE / "model_index.json").is_file()
    )

    if local_cached:
        logger.info(
            f"[WAN-I2V] Loading cached 4-bit pipeline from {_WAN_I2V_LOCAL_CACHE}..."
        )
        t0 = time.monotonic()
        pipe = WanImageToVideoPipeline.from_pretrained(
            _WAN_I2V_LOCAL_CACHE, torch_dtype=torch.bfloat16
        )
        logger.info(f"[WAN-I2V] Cached pipeline loaded in {time.monotonic() - t0:.1f}s")
    else:
        bnb_4bit = dict(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": BitsAndBytesConfig(**bnb_4bit),
                "transformer_2": BitsAndBytesConfig(**bnb_4bit),
            }
        )
        logger.info("[WAN-I2V] Loading with 4-bit quantized dual transformers...")
        t0 = time.monotonic()
        vae = AutoencoderKLWan.from_pretrained(
            _WAN_I2V_REPO, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            _WAN_I2V_REPO,
            vae=vae,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        logger.info(f"[WAN-I2V] Pipeline loaded in {time.monotonic() - t0:.1f}s")

        logger.info(f"[WAN-I2V] Saving to {_WAN_I2V_LOCAL_CACHE}...")
        t0 = time.monotonic()
        pipe.save_pretrained(_WAN_I2V_LOCAL_CACHE)
        logger.info(f"[WAN-I2V] Saved in {time.monotonic() - t0:.1f}s")

    pipe.to(device)
    pipe.vae.enable_tiling()
    pipe.register_to_config(boundary_ratio=_BOUNDARY_RATIO)

    _setup_distill_scheduler(pipe)
    pipe._loaded_lora_names = set()
    logger.info("[WAN-I2V] Loaded on GPU + distill scheduler (LoRAs loaded on demand)")

    _wan_i2v_pipe = pipe
    return _wan_i2v_pipe


def cleanup_wan_resources():
    """Release Wan pipeline VRAM."""
    global _wan_t2v_pipe, _wan_i2v_pipe

    if _wan_t2v_pipe is not None:
        del _wan_t2v_pipe
        _wan_t2v_pipe = None
    if _wan_i2v_pipe is not None:
        del _wan_i2v_pipe
        _wan_i2v_pipe = None

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[WAN] VRAM released")


@router.post("/video", response_class=Response)
async def generate_video(
    request: Request,
    prompt: str = Form(...),
    negative_prompt: str = Form(
        default="worst quality, inconsistent motion, blurry, jittery, distorted"
    ),
    image: UploadFile | None = File(default=None),
    end_image: UploadFile | None = File(default=None),
    lightning: bool = Form(default=True),
    width: int = Form(default=832),
    height: int = Form(default=480),
    num_frames: int = Form(default=81),
    frame_rate: float = Form(default=16.0),
    num_inference_steps: int = Form(default=40),
    guidance_scale: float = Form(default=4.0),
    guidance_scale_2: float = Form(default=3.0),
    boundary_ratio: float = Form(default=_BOUNDARY_RATIO),
    distill_steps: int = Form(default=_DISTILL_NUM_STEPS),
    lora_weight_high: float = Form(default=_DISTILL_LORA_WEIGHT_HIGH),
    lora_weight_low: float = Form(default=_DISTILL_LORA_WEIGHT_LOW),
    seed: int = Form(default=-1),
):
    """Generate a video from a text prompt (and optional starting image) using Wan 2.2."""
    from diffusers.utils import export_to_video

    # Validate inputs
    if width % 16 != 0 or height % 16 != 0:
        raise HTTPException(
            status_code=400, detail="width and height must be divisible by 16"
        )

    logger.info(f"[WAN] Generating video, {width}x{height}, {num_frames} frames")

    # Free SDXL VRAM before loading Wan
    cleanup_resources()

    generator = torch.Generator(device="cpu")
    if seed >= 0:
        generator.manual_seed(seed)
    else:
        seed = generator.seed()
    logger.info(f"[WAN] Using seed: {seed}")

    def _progress_cb(total_steps):
        """Return a callback_on_step_end that logs progress."""

        def _cb(_pipe, step, _timestep, cb_kwargs):
            logger.info(f"[WAN] step {step + 1}/{total_steps}")
            return cb_kwargs

        return _cb

    # Prepare starting image if provided
    input_image = None
    input_last_image = None
    if image is not None:
        image_bytes = await image.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_image = input_image.resize((width, height), Image.LANCZOS)
        logger.info(
            f"[WAN] Using starting image: {image.filename} resized to {width}x{height}"
        )
    if end_image is not None:
        end_bytes = await end_image.read()
        input_last_image = Image.open(io.BytesIO(end_bytes)).convert("RGB")
        input_last_image = input_last_image.resize((width, height), Image.LANCZOS)
        logger.info(
            f"[WAN] Using end image: {end_image.filename} resized to {width}x{height}"
        )

    if input_image is not None:
        # ── Image-to-Video (A14B MoE, 4-bit + distill LoRA) ──────────────
        # Free T2V pipeline VRAM if loaded — can't fit both A14B pipelines
        global _wan_t2v_pipe
        if _wan_t2v_pipe is not None:
            del _wan_t2v_pipe
            _wan_t2v_pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[WAN] Released T2V pipeline to make room for I2V")
        pipe = _load_wan_i2v()
        pipe.register_to_config(boundary_ratio=boundary_ratio)

        # Parse LoRA params from multipart form (loras.N.name / .weight_high / .weight_low)
        form = await request.form()
        loras = _parse_lora_params(form)

        if lightning:
            if not loras:
                loras = [
                    {
                        "name": "distill",
                        "weight_high": lora_weight_high,
                        "weight_low": lora_weight_low,
                    }
                ]
            pipe.scheduler = pipe._distill_scheduler
            n_steps = distill_steps
            cfg = 1.0  # CFG must be disabled for distill LoRA
            cfg2 = 1.0
            lora_desc = "+".join(l["name"] for l in loras)
            mode = f"I2V-distill-{distill_steps}step [{lora_desc}]"
        else:
            pipe.scheduler = pipe._base_scheduler
            n_steps = num_inference_steps
            cfg = guidance_scale
            cfg2 = guidance_scale_2
            mode = "I2V-A14B-base"

        # Activate requested LoRAs (or disable if none)
        if loras:
            adapter_names = []
            adapter_weights = []
            for lora in loras:
                try:
                    high, low = _ensure_lora_loaded(pipe, lora["name"], "i2v")
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                adapter_names.extend([high, low])
                adapter_weights.extend([lora["weight_high"], lora["weight_low"]])
            pipe.set_adapters(adapter_names, adapter_weights)
        else:
            pipe.disable_lora()

        logger.info(f"[WAN] {mode}: generating {width}x{height} @ {n_steps} steps...")
        t0 = time.monotonic()
        output = pipe(
            image=input_image,
            last_image=input_last_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=n_steps,
            guidance_scale=cfg,
            guidance_scale_2=cfg2,
            generator=generator,
            output_type="np",
            return_dict=False,
            callback_on_step_end=_progress_cb(n_steps),
        )
    else:
        # ── Text-to-Video (A14B MoE, 4-bit + distill LoRA) ───────────────
        # Free I2V pipeline VRAM if loaded — can't fit both A14B pipelines
        global _wan_i2v_pipe
        if _wan_i2v_pipe is not None:
            del _wan_i2v_pipe
            _wan_i2v_pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[WAN] Released I2V pipeline to make room for T2V")
        pipe = _load_wan_t2v()
        pipe.register_to_config(boundary_ratio=boundary_ratio)

        # Parse LoRA params from multipart form (loras.N.name / .weight_high / .weight_low)
        form = await request.form()
        loras = _parse_lora_params(form)

        if lightning:
            if not loras:
                loras = [
                    {
                        "name": "distill",
                        "weight_high": lora_weight_high,
                        "weight_low": lora_weight_low,
                    }
                ]
            pipe.scheduler = pipe._distill_scheduler
            n_steps = distill_steps
            cfg = 1.0
            cfg2 = 1.0
            lora_desc = "+".join(l["name"] for l in loras)
            mode = f"T2V-distill-{distill_steps}step [{lora_desc}]"
        else:
            pipe.scheduler = pipe._base_scheduler
            n_steps = num_inference_steps
            cfg = guidance_scale
            cfg2 = guidance_scale_2
            mode = "T2V-A14B-base"

        # Activate requested LoRAs (or disable if none)
        if loras:
            adapter_names = []
            adapter_weights = []
            for lora in loras:
                try:
                    high, low = _ensure_lora_loaded(pipe, lora["name"], "t2v")
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                adapter_names.extend([high, low])
                adapter_weights.extend([lora["weight_high"], lora["weight_low"]])
            pipe.set_adapters(adapter_names, adapter_weights)
        else:
            pipe.disable_lora()

        logger.info(f"[WAN] {mode}: generating {width}x{height} @ {n_steps} steps...")
        t0 = time.monotonic()
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=n_steps,
            guidance_scale=cfg,
            guidance_scale_2=cfg2,
            generator=generator,
            output_type="np",
            return_dict=False,
            callback_on_step_end=_progress_cb(n_steps),
        )

    video_frames = output[0][0]  # (num_frames, H, W, 3) float32 [0,1]
    logger.info(f"[WAN] {mode} done in {time.monotonic() - t0:.1f}s")

    # ── Export to MP4 ─────────────────────────────────────────────────────
    logger.info(f"[WAN] Exporting video to MP4 at {frame_rate} fps")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_to_video(
            list(video_frames), output_video_path=tmp_path, fps=int(frame_rate)
        )
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"[WAN] Video generated: {len(video_bytes) / 1024 / 1024:.1f} MB")

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=wan_video_{seed}.mp4"},
    )


_asset_generator: ControlNetAssetGenerator | None = None


def get_asset_generator() -> ControlNetAssetGenerator:
    global _asset_generator
    if _asset_generator is None:
        _asset_generator = ControlNetAssetGenerator()
    return _asset_generator


@router.post("/frames", response_class=Response)
async def generate_frames(video: UploadFile = File(...), num_frames: int = Form(...)):
    print(f"🎬 Receiving video: {video.filename} ({video.content_type})")

    if num_frames < 1:
        raise HTTPException(status_code=400, detail="num_frames must be >= 1")

    cleanup_resources()
    print("🧹 VRAM cleared for keyframe extraction")

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with temp_video as buffer:
            shutil.copyfileobj(video.file, buffer)
        print(f"✅ Video saved to temporary storage: {temp_video.name}")

        cap = cv2.VideoCapture(temp_video.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise HTTPException(
                status_code=400, detail="Could not read frames from video"
            )

        # Split into num_frames+1 equal segments, pick boundary frames
        indices = [
            int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)
        ]
        print(
            f"🎞️ Extracting {num_frames} keyframes from {total_frames} total (indices: {indices})"
        )

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            frames.append(buf.getvalue())
        cap.release()

        if not frames:
            raise HTTPException(status_code=500, detail="Failed to extract any frames")

        print(f"✅ Extracted {len(frames)} keyframes, generating ControlNet assets...")

        generator = get_asset_generator()
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, frame_bytes in enumerate(frames):
                zf.writestr(f"frame_{i:04d}/frame.png", frame_bytes)
                asset_zip = generator.process(frame_bytes)
                if asset_zip:
                    zf.writestr(f"frame_{i:04d}/assets.zip", asset_zip)
                print(f"  📦 Frame {i+1}/{len(frames)} processed")

        print(f"✅ All frames processed, returning ZIP archive")

        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=frames_assets.zip"},
        )
    finally:
        Path(temp_video.name).unlink(missing_ok=True)
