from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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
import torch

# Disable xet download backend — crashes with "Background writer channel closed"
os.environ["HF_HUB_DISABLE_XET"] = "1"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/generate", tags=["video"])

# ── Wan 2.2 constants ─────────────────────────────────────────────────────────

_WAN_T2V_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_WAN_I2V_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
_DISTILL_LORA_REPO = "lightx2v/Wan2.1-Distill-Loras"
_DISTILL_LORA_I2V = "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors"
_DISTILL_STEPS = [1000, 750, 500, 250]
_CACHES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "caches")
_WAN_T2V_LOCAL_CACHE = os.path.join(_CACHES_DIR, "wan-t2v-1.3b")
_WAN_I2V_LOCAL_CACHE = os.path.join(_CACHES_DIR, "wan-i2v-14b-4bit")

_wan_t2v_pipe = None
_wan_i2v_pipe = None


def _load_wan_t2v():
    """Load Wan 2.1 T2V 1.3B pipeline (bf16, no quantization needed)."""
    global _wan_t2v_pipe

    if _wan_t2v_pipe is not None:
        return _wan_t2v_pipe

    from diffusers import WanPipeline

    device = "cuda:0"
    local_cached = os.path.isdir(_WAN_T2V_LOCAL_CACHE) and os.path.isfile(
        os.path.join(_WAN_T2V_LOCAL_CACHE, "model_index.json")
    )
    src = _WAN_T2V_LOCAL_CACHE if local_cached else _WAN_T2V_REPO

    logger.info(f"[WAN-T2V] Loading from {src}...")
    t0 = time.monotonic()
    pipe = WanPipeline.from_pretrained(src, torch_dtype=torch.bfloat16)
    logger.info(f"[WAN-T2V] Pipeline loaded in {time.monotonic() - t0:.1f}s")

    if not local_cached:
        logger.info(f"[WAN-T2V] Saving to {_WAN_T2V_LOCAL_CACHE}...")
        t0 = time.monotonic()
        pipe.save_pretrained(_WAN_T2V_LOCAL_CACHE)
        logger.info(f"[WAN-T2V] Saved in {time.monotonic() - t0:.1f}s")

    pipe.to(device)
    pipe.vae.enable_tiling()
    logger.info("[WAN-T2V] Loaded on GPU + VAE tiling enabled")

    _wan_t2v_pipe = pipe
    return _wan_t2v_pipe


def _load_wan_i2v():
    """Load Wan 2.1 I2V 14B pipeline (4-bit NF4 quantized transformer)."""
    global _wan_i2v_pipe

    if _wan_i2v_pipe is not None:
        return _wan_i2v_pipe

    from diffusers import BitsAndBytesConfig, WanImageToVideoPipeline
    from diffusers.quantizers.pipe_quant_config import PipelineQuantizationConfig

    device = "cuda:0"
    local_cached = os.path.isdir(_WAN_I2V_LOCAL_CACHE) and os.path.isfile(
        os.path.join(_WAN_I2V_LOCAL_CACHE, "model_index.json")
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
            }
        )
        logger.info("[WAN-I2V] Loading with 4-bit quantized transformer...")
        t0 = time.monotonic()
        pipe = WanImageToVideoPipeline.from_pretrained(
            _WAN_I2V_REPO,
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

    # Load LightX2V 4-step distill LoRA
    logger.info("[WAN-I2V] Loading LightX2V 4-step distill LoRA...")
    t0 = time.monotonic()
    pipe.load_lora_weights(_DISTILL_LORA_REPO, weight_name=_DISTILL_LORA_I2V)
    logger.info(f"[WAN-I2V] LoRA loaded in {time.monotonic() - t0:.1f}s")

    # Use FlowMatchEulerDiscreteScheduler with forced distill timesteps.
    # Euler is the correct solver for distilled models (single-step prediction).
    # UniPC's multi-step state causes both device bugs and wrong results with distill.
    from diffusers import FlowMatchEulerDiscreteScheduler
    import numpy as np

    # Keep the original scheduler for non-distill (full-step) mode
    pipe._base_scheduler = pipe.scheduler

    orig_cfg = pipe.scheduler.config
    euler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=orig_cfg.num_train_timesteps,
        shift=orig_cfg.get("flow_shift", 1.0),
    )

    def _distill_set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
        num_train = self.config.num_train_timesteps
        sigmas = np.array([t / num_train for t in _DISTILL_STEPS], dtype=np.float32)
        # Clamp first sigma below 1.0 to avoid log(0) = -inf
        sigmas[0] = min(sigmas[0], 1.0 - 1e-6)
        sigmas = np.concatenate([sigmas, [0.0]])
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = (self.sigmas[:-1] * num_train).to(
            device=device, dtype=torch.int64
        )
        self.num_inference_steps = len(_DISTILL_STEPS)
        self._step_index = None
        self._begin_index = None

    euler.set_timesteps = _distill_set_timesteps.__get__(
        euler, FlowMatchEulerDiscreteScheduler
    )
    pipe._distill_scheduler = euler
    pipe.scheduler = euler
    logger.info("[WAN-I2V] Loaded on GPU + distill LoRA + Euler 4-step schedule")

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
    num_inference_steps: int = Form(default=30),
    guidance_scale: float = Form(default=5.0),
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
        # ── Image-to-Video (14B, 4-bit + distill LoRA) ────────────────────
        pipe = _load_wan_i2v()

        # Monkey-patch encode_image so that when the pipeline internally calls
        # encode_image([image, last_image]) (batch=2), we flatten to (1, 2*S, D)
        # to match the text embedding batch dim. Without this the transformer's
        # concat on dim=1 fails with mismatched batch sizes.
        if not hasattr(pipe, "_orig_encode_image"):
            pipe._orig_encode_image = pipe.encode_image.__func__

            def _patched_encode_image(self, image, device=None):
                embeds = self._orig_encode_image(self, image, device)
                if embeds.shape[0] > 1:
                    embeds = embeds.reshape(1, -1, embeds.shape[-1])
                return embeds

            pipe.encode_image = _patched_encode_image.__get__(pipe, type(pipe))

        if lightning:
            # Fast 4-step distill LoRA mode
            pipe.enable_lora()
            pipe.scheduler = pipe._distill_scheduler
            n_steps = len(_DISTILL_STEPS)
            cfg = 1.0  # CFG must be disabled for distill LoRA
            mode = "I2V-14B-distill"
        else:
            # Full quality mode — base model weights, original scheduler
            pipe.disable_lora()
            pipe.scheduler = pipe._base_scheduler
            n_steps = num_inference_steps
            cfg = guidance_scale
            mode = "I2V-14B-base"

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
            generator=generator,
            output_type="np",
            return_dict=False,
            callback_on_step_end=_progress_cb(n_steps),
        )
    else:
        # ── Text-to-Video (1.3B, bf16) ───────────────────────────────────
        pipe = _load_wan_t2v()
        mode = "T2V-1.3B"
        logger.info(
            f"[WAN] {mode}: generating {width}x{height} @ {num_inference_steps} steps..."
        )
        t0 = time.monotonic()
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np",
            return_dict=False,
            callback_on_step_end=_progress_cb(num_inference_steps),
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
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

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
        os.unlink(temp_video.name)
