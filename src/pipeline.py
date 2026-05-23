from random import randint
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForImage2Image,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from pathlib import Path
import os, gc, time
import re
import torch
from src.utils import is_rocm, vram_gb, has_vram_gte

_cached_pipe: StableDiffusionXLPipeline | None = None
_cached_fast_pipe: StableDiffusionXLPipeline | None = None
_cached_model_name: str | None = None
_cached_cpu_vae: AutoencoderKL | None = None
_cpu_vae_dtype: torch.dtype = (
    torch.float32
)  # upgraded to bf16 at first load if CPU supports it
_cleanup_hooks: list = (
    []
)  # callbacks registered by other modules (e.g. outpainting_node)
DTYPE = torch.float16
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CWD = Path(os.getcwd())
MODEL_CACHE_DIR = CWD / "caches" / "models"
WARMED_CONFIGS_FILE = CWD / "caches" / "warmed_configs.json"
MODELS_DIR = Path.home() / "sd_models"
_warmed_configs_cache: set[str] | None = None  # in-memory cache of config keys
_runtime_diag_logged = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# pip nvidia-cudnn-cu12 conflicts with the system CUDA 12.8 driver — any call into
# the cuDNN library raises CUDNN_STATUS_NOT_INITIALIZED. Disable all cuDNN backends
# so PyTorch uses its built-in CUDA kernels for conv and SDPA instead.
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_cudnn_sdp(False)


def register_cleanup_hook(fn) -> None:
    """Register a zero-argument callable to be invoked by cleanup_resources().

    Use this to clear module-level caches in other modules that share VRAM
    with the base pipeline (e.g. ControlNet pipe cache in outpainting_node).
    Avoids circular imports: callers import from src.pipeline, not the reverse.
    """
    _cleanup_hooks.append(fn)


def _vram_stats() -> str:
    try:
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        return f"allocated={a:.3f} GB  reserved={r:.3f} GB"
    except Exception:
        return "<unavailable>"


def cleanup_resources():
    """
    Forcefully releases VRAM. Critical for avoiding Linux 6.14 GTT Swap crashes.
    """
    global _cached_pipe, _cached_fast_pipe, _cached_cpu_vae, _cached_model_name

    print(f"🧹 cleanup_resources: start — {_vram_stats()}")

    # Unload IP-Adapter and zero out every VRAM-resident component so their GPU
    # tensors are freed immediately — even if a from_pipe() wrapper, a Compel
    # tokenizer reference, or a Python closure still holds the pipeline object.
    for _pipe in (_cached_pipe, _cached_fast_pipe):
        if _pipe is None:
            continue
        try:
            _pipe.unload_ip_adapter()
        except Exception:
            pass
        for _attr in (
            "unet",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
            "controlnet",
        ):
            try:
                setattr(_pipe, _attr, None)
            except Exception:
                pass

    # Explicitly delete references
    if _cached_pipe is not None:
        del _cached_pipe
        _cached_pipe = None

    if _cached_fast_pipe is not None:
        del _cached_fast_pipe
        _cached_fast_pipe = None

    if _cached_cpu_vae is not None:
        del _cached_cpu_vae
        _cached_cpu_vae = None

    _cached_model_name = None

    # Drop other module caches BEFORE GC so shared CUDA tensors (e.g. UNet/VAE
    # still referenced by _cn_pipe_cache after _cached_pipe is deleted) have
    # their refcount reach zero before gc.collect() + empty_cache() run.
    for hook in _cleanup_hooks:
        try:
            hook()
        except Exception:
            pass

    print(f"🧹 after ref-drop — {_vram_stats()}")

    # Collect Python cycles; on ROCm tensor finalizers run synchronously.
    gc.collect()
    gc.collect()

    print(f"🧹 after gc.collect — {_vram_stats()}")

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    # Collect IPC tensor handles (shared-memory tensors from other processes).
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        # HIP context may already be dead after a kernel crash — ignore
        pass

    print(f"🧹 after empty_cache — {_vram_stats()}")

    # One more GC pass in case CUDA callbacks released additional Python objects.
    gc.collect()

    # ROCm 5.5+ routes hipMalloc/hipFree through a per-process hipMemPool.
    # empty_cache() above calls hipFree, returning blocks to the pool — but the
    # pool holds onto the pages so the same process can reuse them cheaply.
    # Other processes cannot allocate that VRAM until the pool returns the pages
    # to the OS/driver. hipMemPoolTrimTo(defaultPool, 0) forces that return.
    if is_rocm():
        try:
            import ctypes

            _hip = ctypes.CDLL("libamdhip64.so")
            _pool = ctypes.c_void_p(None)
            rc = _hip.hipDeviceGetDefaultMemPool(ctypes.byref(_pool), ctypes.c_int(0))
            print(f"🧹 hipDeviceGetDefaultMemPool rc={rc}  pool={_pool.value!r}")
            if rc == 0 and _pool.value is not None:
                _hip.hipMemPoolTrimTo(_pool, ctypes.c_size_t(0))
                print(f"🧹 after hipMemPoolTrimTo — {_vram_stats()}")
            else:
                print("⚠️ HIP default pool unavailable; skipping trim.")
        except Exception as e:
            print(f"⚠️ HIP pool trim failed: {e}")

    print(f"🧹 cleanup_resources: done — {_vram_stats()}")


def log_runtime_diagnostics_once():
    """Log one-time runtime info that explains long pre-step stalls."""
    global _runtime_diag_logged
    if _runtime_diag_logged:
        return

    print("🧪 Runtime diagnostics:")
    print(
        f"   torch={torch.__version__} cuda={torch.version.cuda} hip={torch.version.hip}"
    )
    print(f"   is_rocm={is_rocm()} vram_gb={vram_gb():.1f}")
    print(
        "   env HSA_OVERRIDE_GFX_VERSION="
        f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION', '<unset>')}"
    )
    print(f"   env MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE', '<unset>')}")
    print(
        "   env PYTORCH_HIP_ALLOC_CONF="
        f"{os.environ.get('PYTORCH_HIP_ALLOC_CONF', '<unset>')}"
    )
    print(
        "   env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="
        f"{os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '<unset>')}"
    )
    _runtime_diag_logged = True


def attach_inference_timing(pipe_kwargs: dict, label: str) -> tuple[dict, float]:
    """
    Attach per-request denoise-step timing logs to a diffusers call.
    """
    t0 = time.monotonic()
    steps = int(pipe_kwargs.get("num_inference_steps") or 0)
    print(f"⏱️  {label}: dispatching pipeline call (steps={steps})")

    if "callback_on_step_end" in pipe_kwargs:
        print(f"⏱️  {label}: callback already provided; pre-step timing unavailable.")
        return pipe_kwargs, t0

    first_step_logged = False

    def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        nonlocal first_step_logged
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        step_str = f"step={step_index + 1}/{max(steps, 1)}"
        if not first_step_logged:
            elapsed = time.monotonic() - t0
            print(
                f"⏱️  {label}: first denoise step reached in {elapsed:.2f}s "
                f"({step_str}) | GPU peak={peak:.2f} GB reserved={reserved:.2f} GB"
            )
            first_step_logged = True
        else:
            print(
                f"⏱️  {label}: {step_str} | GPU peak={peak:.2f} GB reserved={reserved:.2f} GB"
            )
        return callback_kwargs

    pipe_kwargs["callback_on_step_end"] = on_step_end
    pipe_kwargs["callback_on_step_end_tensor_inputs"] = []
    return pipe_kwargs, t0


def finalize_inference_timing(label: str, t0: float):
    elapsed = time.monotonic() - t0
    print(f"⏱️  {label}: pipeline call finished in {elapsed:.2f}s")


def _load_pipeline(model: str) -> StableDiffusionXLPipeline:
    """
    Load an SDXL model. Uses a diffusers-format cache when available
    (from_pretrained is ~3× faster than from_single_file). On first load
    the model is converted and cached automatically.
    """
    cached_dir = MODEL_CACHE_DIR / model

    # FAST PATH: diffusers cache exists (VAE is already madebyollin/sdxl-vae-fp16-fix,
    # baked in when the cache was saved — no need to reload it from HF hub).
    if (cached_dir / "model_index.json").is_file():
        print(f"⚡ Loading from diffusers cache: {cached_dir}")
        t0 = time.monotonic()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cached_dir,
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        print(f"   Loaded in {time.monotonic() - t0:.1f}s (cached, flash_attn)")
        return pipe

    # SLOW PATH: first-time load from single .safetensors file
    target_model_path = Path.home() / "sd_models" / f"{model}.safetensors"

    print(f"📦 Loading FP16-Fixed VAE: {VAE_ID}")
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE)

    print(f"⚡ Loading SDXL Model (single-file) @ {target_model_path}")
    t0 = time.monotonic()
    pipe = StableDiffusionXLPipeline.from_single_file(
        target_model_path,
        vae=vae,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16",
    )
    print(f"   Loaded in {time.monotonic() - t0:.1f}s (flash_attn)")

    # Save as diffusers format for faster future loads
    print(f"💾 Caching as diffusers format: {cached_dir}")
    pipe.save_pretrained(cached_dir)

    return pipe


def get_pipe(model: str = "juggernaut"):
    """
    Initializes the SDXL pipeline with RDNA4-specific optimizations.
    """
    global _cached_pipe, _cached_model_name

    # Return existing pipe if model hasn't changed
    if _cached_pipe is not None and _cached_model_name == model:
        return _cached_pipe

    if _cached_pipe is not None or _cached_fast_pipe is not None:
        print("🔄 Switching pipeline/model. Clearing VRAM...")
        cleanup_resources()

    print(f"🚀 Initializing Optimized Pipeline for L40S (Ada Lovelace)...")

    pipe = _load_pipeline(model)

    # 4. SCHEDULER
    try:
        scheduler_config = dict(pipe.scheduler.config)

        is_vpred = any(k in model.lower() for k in ("vpred", "noob"))
        if is_vpred:
            scheduler_config["prediction_type"] = "v_prediction"
            scheduler_config["rescale_betas_zero_snr"] = True
            scheduler_config["timestep_spacing"] = "trailing"
            print(
                f"  [Scheduler] Configured for v-prediction + zero-SNR for model: {model}"
            )

        if is_vpred:
            pipe.scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        else:
            # Use the model's native scheduler config (epsilon for illustrious).
            # Every v_prediction + zero-SNR combination produced grey output —
            # testing whether the shipped epsilon config produces correct results.
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                scheduler_config, use_karras_sigmas=True
            )
        _diag_keys = ("prediction_type", "rescale_betas_zero_snr", "timestep_spacing")
        _diag = {k: pipe.scheduler.config.get(k, "<absent>") for k in _diag_keys}
        print(
            f"  [Scheduler] {pipe.scheduler.__class__.__name__} for '{model}': {_diag}"
        )
    except TypeError:
        pass

    # 5. MEMORY EFFICIENT ATTENTION
    # FlashAttention 2 is used automatically via PyTorch SDPA (AttnProcessor2_0)
    print("🔥 Pipeline Ready. Using FlashAttention 2 (via PyTorch SDPA).")

    # 6. MEMORY PRESSURE MITIGATIONS
    # On ROCm: attention slicing must never be used — torch.baddbmm triggers an illegal
    # memory access in SlicedAttnProcessor. enable_model_cpu_offload() was tried but
    # its accelerate hooks corrupt the GPU context for the VAE conv upsampler.
    # VAE tiling and slicing are re-enabled: they are not confirmed causes of the hangs.
    # On CUDA with sufficient VRAM these concerns don't apply.
    if is_rocm() and not has_vram_gte(24.0):
        print(
            f"⚠️  ROCm GPU has {vram_gb():.1f} GB VRAM — enabling VAE tiling and slicing."
        )
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        # Raise tile thresholds so a standard 1024x1024 canvas (128x128 latent)
        # is decoded as a single pass. Default tile_latent_min_size=64 (derived
        # from sample_size=512, 4 down-blocks) causes the 128x128 latent to be
        # split into 64x64 chunks, introducing linear-blend seam artifacts that
        # are especially visible at the inpainting void boundary. Tiling only
        # activates when the latent strictly exceeds the threshold, so setting
        # tile_latent_min_size=128 reserves partitioning for outputs >1024x1024.
        pipe.vae.tile_sample_min_size = 1024
        pipe.vae.tile_latent_min_size = 128

    # 7. TRANSFER TO GPU — keep all sub-models resident; no accelerate offload hooks.
    pipe.to("cuda")

    _cached_pipe = pipe
    _cached_model_name = model

    return pipe


def get_fast_pipe(model: str = "juggernaut"): ...
def warmup_pipeline(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline,
    width: int = 1024,
    height: int = 1024,
):
    def run_warmup(run_width: int, run_height: int):
        # Warm UNet kernels, then exercise the decode path used at runtime.
        generator = torch.Generator("cuda").manual_seed(42)
        with torch.no_grad():
            latents = pipe(
                prompt="warmup",
                negative_prompt="",
                num_inference_steps=1,
                guidance_scale=1.0,
                width=run_width,
                height=run_height,
                generator=generator,
                output_type="latent",
            ).images

        if is_rocm():
            # When ROCBLAS_USE_HIPBLASLT=0, rocBLAS routes through Tensile which may
            # have the gfx1200 VAE upsampler kernel.  Test GPU decode; SIGABRT here
            # means Tensile also lacks the kernel — remove the env var to revert.
            if os.environ.get("ROCBLAS_USE_HIPBLASLT") == "0":
                decode_latents_safe(pipe, latents)
            else:
                # gfx1200 (RDNA4) crashes with SIGABRT on GPU VAE decode — always use CPU.
                decode_latents_on_cpu(pipe, latents)
            return

        scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
        with torch.no_grad():
            pipe.vae.decode(latents / scaling, return_dict=False)[0]

    def load_warmup() -> set[str]:
        """Load the set of previously warmed config keys from disk."""
        global _warmed_configs_cache
        if _warmed_configs_cache is not None:
            return _warmed_configs_cache
        try:
            import json

            if WARMED_CONFIGS_FILE.exists():
                with open(WARMED_CONFIGS_FILE, "r") as f:
                    data = json.load(f)
                _warmed_configs_cache = set(data.get("configs", []))
            else:
                _warmed_configs_cache = set()
        except Exception:
            _warmed_configs_cache = set()
        return _warmed_configs_cache

    def save_warmup(key: str):
        """Add a config key to the tracked set and persist to disk."""
        import json

        configs = load_warmup()
        if key in configs:
            return
        configs.add(key)
        WARMED_CONFIGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WARMED_CONFIGS_FILE, "w") as f:
            json.dump({"configs": sorted(configs)}, f, indent=2)
        print(f"💾 Recorded new warmed config: {key}")

    model = _cached_model_name or "juggernaut"
    warmup_key = f"{model}_{width}x{height}_1step_decode"
    legacy_key = f"{model}_{width}x{height}_1step"
    warmed = load_warmup()

    run_width = width
    run_height = height
    selected_key = warmup_key

    if warmup_key in warmed or legacy_key in warmed:
        print(f"⚡ Warmup cache hit: {warmup_key} (running with cached config)")
    else:
        # No exact match: use a cached config for this model if available.
        candidates: list[tuple[int, int, str]] = []
        for key in warmed:
            m = re.match(
                r"^(?P<model>.+)_(?P<w>\d+)x(?P<h>\d+)_1step(?:_decode)?$", key
            )
            if not m:
                continue
            if m.group("model") != model:
                continue
            w = int(m.group("w"))
            h = int(m.group("h"))
            candidates.append((w, h, key))

        if candidates:
            # Prefer the lightest cached size to keep startup fast.
            run_width, run_height, selected_key = min(
                candidates, key=lambda x: x[0] * x[1]
            )
            print(
                "⚡ Using nearest cached warmup config "
                f"{selected_key} instead of {warmup_key}."
            )

    print(
        "🔥 Warming up base pipeline "
        f"({run_width}x{run_height}, 1 step + VAE decode)..."
    )
    t0 = time.monotonic()
    run_warmup(run_width, run_height)
    print(f"   Warmed in {time.monotonic() - t0:.1f}s")
    if run_width == width and run_height == height:
        save_warmup(warmup_key)

    gc.collect()
    torch.cuda.empty_cache()


def generate_image(pipe, **kwargs):
    """
    Safely intercepts integer seeds and converts them to Diffusers-compatible Generators.
    """
    # Extract the custom seed integer, default to random if not provided
    seed = kwargs.pop("seed", -1)
    if seed == -1:
        seed = randint(0, 2**32 - 1)

    print(f"🎲 Generating with seed: {seed}")

    # Diffusers requires a torch.Generator object for deterministic noise
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs["generator"] = generator

    # On ROCm, get latents first so we can measure VRAM at the peak of denoising,
    # then attempt GPU decode (VAE tiling keeps GEMM sizes small enough to avoid the
    # gfx1200 upsampler crash).  Fall back to CPU decode if anything goes wrong.
    force_latent = is_rocm() and kwargs.get("output_type", "pil") == "pil"
    if force_latent:
        kwargs["output_type"] = "latent"

    kwargs, t0 = attach_inference_timing(kwargs, label="generate_image")
    print(
        f"⏱️  generate_image: calling pipe ({type(pipe).__name__}) with keys: {[k for k in kwargs if k != 'callback_on_step_end']}"
    )

    # On ROCm the VAE encoder has the same gfx1200 kernel hang as the decoder.
    # Pre-encode the input image on CPU and pass latents directly so the pipeline
    # skips its internal vae.encode call entirely.
    is_img2img = "image" in kwargs and isinstance(
        kwargs["image"], __import__("PIL").Image.Image
    )
    if is_rocm() and is_img2img:
        kwargs["image"] = encode_image_safe(pipe, kwargs["image"])

    output = pipe(**kwargs).images

    print(f"⏱️  generate_image: pipe() returned, output type={type(output).__name__}")
    finalize_inference_timing("generate_image", t0)

    if force_latent and isinstance(output, torch.Tensor):
        vram_alloc = torch.cuda.memory_allocated()
        vram_reserved = torch.cuda.memory_reserved()
        vram_total = torch.cuda.get_device_properties(0).total_memory
        print(
            f"📊 VRAM at decode: allocated={vram_alloc/1024**3:.2f} GB  "
            f"reserved={vram_reserved/1024**3:.2f} GB  "
            f"total={vram_total/1024**3:.2f} GB  "
            f"free={(vram_total - vram_reserved)/1024**3:.2f} GB"
        )
        t_decode = time.monotonic()
        output = decode_latents_safe(pipe, output)
        print(f"⏱️  VAE decode finished in {time.monotonic() - t_decode:.2f}s")

    return output


def _probe_cpu_bf16() -> bool:
    """
    Probe whether this CPU + PyTorch build supports bf16 arithmetic.
    torch.backends.cpu.is_bf16_supported() mis-detects on some ROCm builds,
    so we do a live conv2d in bf16 and treat any exception as no-support.
    """
    try:
        x = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
        w = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
        torch.nn.functional.conv2d(x, w)
        return True
    except Exception:
        return False


def get_cpu_vae() -> AutoencoderKL:
    """Lazily load and compile a CPU VAE for ROCm-safe latent decoding."""
    global _cached_cpu_vae, _cpu_vae_dtype
    if _cached_cpu_vae is None:
        # is_bf16_supported() mis-detects on some ROCm builds; use a live probe instead.
        api_result = None
        try:
            api_result = torch.backends.cpu.is_bf16_supported()
        except Exception:
            pass
        probe_result = _probe_cpu_bf16()
        use_bf16 = probe_result
        print(
            f"   bf16 probe: api={api_result} live={probe_result} → using bf16={use_bf16}"
        )
        _cpu_vae_dtype = torch.bfloat16 if use_bf16 else torch.float32
        print(f"📦 Loading CPU VAE ({_cpu_vae_dtype}): {VAE_ID}")
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=_cpu_vae_dtype)
        vae.to("cpu").eval()
        # channels_last (NHWC) layout lets the CPU AVX-512 conv2d path run ~2x faster
        # than the default NCHW layout.  Must happen before torch.compile so inductor
        # generates NHWC-aware kernels.
        vae.to(memory_format=torch.channels_last)
        # mode="default" does SIMD vectorisation + loop fusion on CPU.
        # reduce-overhead is CUDA-specific and actively hurts CPU throughput.
        # fullgraph=True ensures inductor sees the whole decoder graph with no breaks.
        t_compile = time.monotonic()
        try:
            _cached_cpu_vae = torch.compile(
                vae, backend="inductor", mode="default", fullgraph=True
            )
            print(
                f"   torch.compile prepared in {time.monotonic() - t_compile:.2f}s "
                f"(JIT compilation happens on first decode call)"
            )
        except Exception as exc:
            print(f"   torch.compile unavailable ({exc}), running eager.")
            _cached_cpu_vae = vae
    return _cached_cpu_vae


def decode_latents_on_gpu(pipe, latents: torch.Tensor) -> list:
    """Attempt GPU VAE decode (fast path). Only safe when Tensile kernels cover the
    VAE upsampler GEMM shapes — requires ROCBLAS_USE_HIPBLASLT=0 on gfx1200."""
    scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
    t_gpu = time.monotonic()
    from PIL import Image

    with torch.inference_mode():
        decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]
    decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
    images = [
        Image.fromarray(
            img.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB"
        )
        for img in decoded
    ]
    print(f"⏱️  GPU VAE decode: {time.monotonic() - t_gpu:.2f}s")
    return images


def encode_image_safe(pipe, image) -> torch.Tensor:
    """
    Encode a PIL image to latents on CPU to avoid the gfx1200 VAE encoder hang.
    Returns a latent tensor on CUDA ready to pass as `image` to img2img pipelines.
    """
    import torchvision.transforms.functional as TF

    cpu_vae = get_cpu_vae()
    scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))

    t_enc = time.monotonic()
    print("📦 CPU VAE encode: encoding input image to latents...")
    with torch.inference_mode():
        img_tensor = (
            TF.to_tensor(image)
            .unsqueeze(0)
            .to("cpu", dtype=_cpu_vae_dtype)
            .to(memory_format=torch.channels_last)
        )
        # VAE expects [-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0
        latents = cpu_vae.encode(img_tensor, return_dict=False)[0].mean * scaling
    latents = latents.to("cuda", dtype=torch.float16)
    print(
        f"⏱️  CPU VAE encode finished in {time.monotonic() - t_enc:.2f}s  shape={tuple(latents.shape)}"
    )
    return latents


def decode_latents_safe(pipe, latents: torch.Tensor):
    """
    Decode latent tensors to PIL images.
    On ROCm with ROCBLAS_USE_HIPBLASLT=0, attempts GPU decode (Tensile backend).
    Falls back to CPU decode otherwise — gfx1200 hipBLASLT path causes SIGABRT.
    """
    if is_rocm() and os.environ.get("ROCBLAS_USE_HIPBLASLT") == "0":
        return decode_latents_on_gpu(pipe, latents)
    return decode_latents_on_cpu(pipe, latents)


def decode_latents_on_cpu(pipe, latents: torch.Tensor):
    """
    Decode latent tensors to PIL images on CPU.
    """
    if not isinstance(latents, torch.Tensor):
        return latents

    cpu_vae = get_cpu_vae()
    scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))

    from PIL import Image

    _is_first_decode = not hasattr(decode_latents_on_cpu, "_compiled_and_warmed")
    if _is_first_decode:
        print("🔨 First CPU VAE decode — torch.compile JIT compilation starting...")
    t_cpu = time.monotonic()
    with torch.inference_mode():
        latents_cpu = (
            latents.detach()
            .to("cpu", dtype=_cpu_vae_dtype)
            .to(memory_format=torch.channels_last)
        ) / scaling
        decoded_batch = cpu_vae.decode(latents_cpu, return_dict=False)[0]
        # bf16 tensors can't be converted to numpy directly — cast to fp32 first.
        decoded_batch = (decoded_batch.float() / 2 + 0.5).clamp(0, 1)
        images = [
            Image.fromarray(
                img.mul(255).round().byte().permute(1, 2, 0).numpy(), mode="RGB"
            )
            for img in decoded_batch
        ]
    elapsed = time.monotonic() - t_cpu
    if _is_first_decode:
        decode_latents_on_cpu._compiled_and_warmed = True
        print(f"⏱️  CPU VAE decode (first, includes JIT compile): {elapsed:.2f}s")
    else:
        print(f"⏱️  CPU VAE decode: {elapsed:.2f}s")

    return images


def shutdown():
    """
    Cleanly releases all VRAM resources. Should be called on application exit.
    """
    print("🛑 Shutting down pipeline and releasing resources...")
    cleanup_resources()
