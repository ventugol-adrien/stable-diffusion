from random import randint
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLPAGPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    ControlNetModel,
)
from pathlib import Path
from urllib.parse import urlparse
import os, sys, gc, time
import numpy as np
import cv2
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir
from huggingface_hub import snapshot_download

_cached_pipe: StableDiffusionXLPipeline | None = None
_cached_fast_pipe: StableDiffusionXLPipeline | None = None
_cached_model_name: str | None = None
_cached_lama: "LaMa | None" = None
DTYPE = torch.float16
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CWD = Path(os.getcwd())
MODEL_CACHE_DIR = CWD / "caches" / "models"
ARTIFACTS_CACHE_DIR = CWD / "caches" / "artifacts"
HF_LOCAL_CACHE_DIR = ARTIFACTS_CACHE_DIR / "huggingface"
CONTROLNET_LOCAL_CACHE_DIR = ARTIFACTS_CACHE_DIR / "controlnet"
IP_ADAPTER_LOCAL_CACHE_DIR = ARTIFACTS_CACHE_DIR / "ip_adapter"
VAE_LOCAL_CACHE_DIR = ARTIFACTS_CACHE_DIR / "vae_fp16_fix"
WARMED_CONFIGS_FILE = CWD / "caches" / "warmed_configs.json"
MODELS_DIR = Path.home() / "sd_models"
_warmed_configs_cache: set[str] | None = None  # in-memory cache of config keys
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# pip nvidia-cudnn-cu12 conflicts with the system CUDA 12.8 driver — any call into
# the cuDNN library raises CUDNN_STATUS_NOT_INITIALIZED. Disable all cuDNN backends
# so PyTorch uses its built-in CUDA kernels for conv and SDPA instead.
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_cudnn_sdp(False)


# ---------------------------------------------------------------------------
# Vendored LaMa inpainting model (Apache-2.0, from simple-lama-inpainting)
# Source: https://github.com/enesmsahin/simple-lama-inpainting
# ---------------------------------------------------------------------------

_LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


def _lama_download(url: str) -> str:
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    cached_file = os.path.join(model_dir, os.path.basename(parts.path))
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=True)
    return cached_file


def _lama_get_image(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("Input must be PIL Image or numpy array")
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:
        img = img[np.newaxis, ...]
    return img.astype(np.float32) / 255


def _lama_pad_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    _, h, w = img.shape
    out_h = h if h % mod == 0 else (h // mod + 1) * mod
    out_w = w if w % mod == 0 else (w // mod + 1) * mod
    return np.pad(img, ((0, 0), (0, out_h - h), (0, out_w - w)), mode="symmetric")


def _lama_prepare(image, mask, device):
    out_image = _lama_get_image(image)
    out_mask = _lama_get_image(mask)
    out_image = _lama_pad_to_modulo(out_image, 8)
    out_mask = _lama_pad_to_modulo(out_mask, 8)
    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)
    out_mask = (out_mask > 0).float()
    return out_image, out_mask


class LaMa:
    def __init__(self, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.environ.get("LAMA_MODEL")
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(f"LaMa model not found: {model_path}")
        if not model_path:
            model_path = _lama_download(_LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image, mask) -> Image.Image:
        image_t, mask_t = _lama_prepare(image, mask, self.device)
        with torch.inference_mode():
            result = self.model(image_t, mask_t)
            out = result[0].permute(1, 2, 0).detach().cpu().numpy()
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Model accessors
# ---------------------------------------------------------------------------


def get_lama() -> LaMa:
    global _cached_lama
    if _cached_lama is None:
        print("🎨 Loading LaMa inpainting model...")
        t0 = time.monotonic()
        _cached_lama = LaMa(device=torch.device("cpu"))
        print(f"   LaMa ready in {time.monotonic() - t0:.1f}s")
    return _cached_lama


def cleanup_resources():
    """
    Forcefully releases VRAM. Critical for avoiding Linux 6.14 GTT Swap crashes.
    """
    global _cached_pipe, _cached_fast_pipe, _cached_lama

    # Unload IP-Adapter first (holds extra GPU tensors outside the main model)
    for pipe in (_cached_pipe, _cached_fast_pipe):
        if pipe is not None:
            try:
                pipe.unload_ip_adapter()
            except Exception:
                pass

    # Explicitly delete references
    if _cached_pipe is not None:
        del _cached_pipe
        _cached_pipe = None

    if _cached_fast_pipe is not None:
        del _cached_fast_pipe
        _cached_fast_pipe = None

    if _cached_lama is not None:
        del _cached_lama
        _cached_lama = None

    # Force Python GC and ROCm cache clear
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 VRAM resources released.")


def _ensure_cache_dirs():
    for p in (
        MODEL_CACHE_DIR,
        ARTIFACTS_CACHE_DIR,
        HF_LOCAL_CACHE_DIR,
        CONTROLNET_LOCAL_CACHE_DIR,
        IP_ADAPTER_LOCAL_CACHE_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def _cache_hf_repo_once(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    _ensure_cache_dirs()
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"📦 Using local cached repo: {local_dir}")
        return local_dir

    print(f"🌐 Caching repo locally (first run): {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    return local_dir


def cache_hf_repo_once(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    return _cache_hf_repo_once(repo_id, local_dir, allow_patterns=allow_patterns)


def get_controlnet_model(model_id: str, cache_name: str) -> ControlNetModel:
    local_dir = CONTROLNET_LOCAL_CACHE_DIR / cache_name
    if (local_dir / "model_index.json").is_file():
        print(f"📦 Using local cached ControlNet: {local_dir}")
        return ControlNetModel.from_pretrained(local_dir, torch_dtype=torch.float16)

    _cache_hf_repo_once(model_id, local_dir)
    return ControlNetModel.from_pretrained(local_dir, torch_dtype=torch.float16)


def load_ip_adapter_local(pipe, variant: str = "general"):
    local_dir = IP_ADAPTER_LOCAL_CACHE_DIR
    _VARIANTS = {
        "general": (
            ["sdxl_models/ip-adapter_sdxl.bin", "**/*.json"],
            "sdxl_models",
            "ip-adapter_sdxl.bin",
        ),
        "face": (
            ["sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", "**/*.json"],
            "sdxl_models",
            "ip-adapter-plus-face_sdxl_vit-h.safetensors",
        ),
    }
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown IP-Adapter variant '{variant}'. Choose 'general' or 'face'."
        )
    allow_patterns, subfolder, weight_name = _VARIANTS[variant]
    weights_path = local_dir / subfolder / weight_name
    if not weights_path.is_file():
        _cache_hf_repo_once(
            "h94/IP-Adapter",
            local_dir,
            allow_patterns=allow_patterns,
        )
    print(f"📦 Using local cached IP-Adapter ({variant}): {weights_path.name}")
    pipe.load_ip_adapter(
        str(local_dir),
        subfolder=subfolder,
        weight_name=weight_name,
    )


def try_enable_pag(pipe, context: str = "pipeline"):
    processors = getattr(pipe, "pag_attn_processors", None)
    if processors:
        print(f"✅ PAG active for {context} ({len(processors)} processor(s)).")
        return True

    if hasattr(pipe, "set_pag_applied_layers") and hasattr(
        pipe, "_set_pag_attn_processor"
    ):
        try:
            pipe.set_pag_applied_layers(["mid"])
            pipe._set_pag_attn_processor(["mid"], do_classifier_free_guidance=True)
            processors = getattr(pipe, "pag_attn_processors", None)
            if processors:
                print(f"✅ PAG enabled for {context} ({len(processors)} processor(s)).")
                return True
        except Exception as e:
            print(f"⚠️ PAG setup failed for {context}: {e}")

    print(f"ℹ️ PAG not active for {context}.")
    return False


def configure_sgm_uniform_scheduler(pipe):
    """
    Configure an SGM-uniform style schedule for refinement.
    Diffusers does not expose a native 'sgm_uniform' scheduler name across all
    pipelines, so we configure DPMSolver++ SDE with trailing timestep spacing.
    """
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=False,
        timestep_spacing="trailing",
    )


def _patch_model_index_for_pag(cached_dir: Path):
    """
    Rewrite _class_name in model_index.json to StableDiffusionXLPAGPipeline so
    that from_pretrained instantiates the correct class regardless of what the
    checkpoint was originally saved as.
    """
    import json as _json

    index_path = cached_dir / "model_index.json"
    if not index_path.is_file():
        return
    with open(index_path) as f:
        data = _json.load(f)
    if data.get("_class_name") != "StableDiffusionXLPAGPipeline":
        data["_class_name"] = "StableDiffusionXLPAGPipeline"
        with open(index_path, "w") as f:
            _json.dump(data, f, indent=2)
        print(f"🔧 Patched model_index.json → StableDiffusionXLPAGPipeline")


def _load_pipeline(model: str) -> StableDiffusionXLPAGPipeline:
    """
    Load an SDXL model. Uses a diffusers-format cache when available
    (from_pretrained is ~3x faster than from_single_file). On first load
    the model is converted and cached automatically.

    PAG strategy:
    - model_index.json is patched to declare StableDiffusionXLPAGPipeline so
      from_pretrained instantiates the right class (it reads _class_name from
      the index, not the calling class, in diffusers 0.37).
    - pag_applied_layers=["mid"] is passed to from_pretrained so __init__
      calls _set_pag_attn_processors at construction time.
    """
    cached_dir = MODEL_CACHE_DIR / model
    _ensure_cache_dirs()

    # SLOW PATH: first-time load from single .safetensors file
    if not (cached_dir / "model_index.json").is_file():
        target_model_path = Path.home() / "sd_models" / f"{model}.safetensors"

        print(f"📦 Loading FP16-Fixed VAE: {VAE_ID}")
        if (VAE_LOCAL_CACHE_DIR / "config.json").is_file():
            print(f"📦 Using local cached VAE: {VAE_LOCAL_CACHE_DIR}")
            vae = AutoencoderKL.from_pretrained(VAE_LOCAL_CACHE_DIR, torch_dtype=DTYPE)
        else:
            vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE)
            print(f"💾 Caching VAE locally: {VAE_LOCAL_CACHE_DIR}")
            vae.save_pretrained(VAE_LOCAL_CACHE_DIR)

        print(f"⚡ Loading SDXL Model (single-file) @ {target_model_path}")
        t0 = time.monotonic()
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            target_model_path,
            vae=vae,
            torch_dtype=DTYPE,
            use_safetensors=True,
            variant="fp16",
        )
        print(f"   Loaded in {time.monotonic() - t0:.1f}s (flash_attn)")

        # Save as plain diffusers format for faster future loads
        print(f"💾 Caching as diffusers format: {cached_dir}")
        base_pipe.save_pretrained(cached_dir)
        del base_pipe
        gc.collect()

    # Ensure model_index.json declares the PAG class before loading.
    _patch_model_index_for_pag(cached_dir)

    print(f"⚡ Loading PAG pipeline from diffusers cache: {cached_dir}")
    t0 = time.monotonic()
    pipe = StableDiffusionXLPAGPipeline.from_pretrained(
        cached_dir,
        torch_dtype=DTYPE,
        use_safetensors=True,
        pag_applied_layers=["mid"],
    )
    print(f"   Loaded in {time.monotonic() - t0:.1f}s (flash_attn)")
    return pipe


def get_pipe(
    model: str = "juggernaut",
    load_ip_adapter: bool = False,
    ip_adapter_variant: str = "general",
):
    """
    Initializes the SDXL pipeline with RDNA4-specific optimizations.

    When load_ip_adapter=True, loads IP-Adapter weights onto the base PAG pipe
    after construction. PAG processors are installed at construction time
    (pag_applied_layers=["mid"]) so the cross-attention layers are free for
    IP-Adapter — no processor key collision occurs.
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
    pipe.enable_freeu(
        s1=0.9,  # Skip connection scaling factor for stage 1
        s2=0.2,  # Skip connection scaling factor for stage 2
        b1=1.3,  # Backbone scaling factor for stage 1
        b2=1.4,  # Backbone scaling factor for stage 2
    )

    # 4. SCHEDULER OPTIMIZATION
    # DPM++ 2M SDE — high quality, good convergence at ~25-30 steps.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
    )
    try_enable_pag(pipe, context="base generation")

    # 5. MEMORY EFFICIENT ATTENTION
    # FlashAttention 2 is used automatically via PyTorch SDPA (AttnProcessor2_0)
    print("🔥 Pipeline Ready. Using FlashAttention 2 (via PyTorch SDPA).")

    # 6. TRANSFER TO GPU
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # Load IP-Adapter AFTER pipe.to("cuda") so weights land on the correct device.
    # PAG self-attention processors (mid block) are already installed at this point;
    # IP-Adapter only touches cross-attention layers — no overlap.
    if load_ip_adapter:
        load_ip_adapter_local(pipe, variant=ip_adapter_variant)
        print(f"🖼️ IP-Adapter ({ip_adapter_variant}) loaded onto base PAG pipe.")

    _cached_pipe = pipe
    _cached_model_name = model

    return pipe


def get_fast_pipe(model: str = "juggernaut"): ...
def warmup_pipeline(
    pipe: StableDiffusionXLPipeline,
    width: int = 1024,
    height: int = 1024,
):
    def run_warmup():
        # Run a single forward pass with dummy data to warm up the model
        generator = torch.Generator("cuda").manual_seed(42)
        with torch.no_grad():
            pipe(
                prompt="warmup",
                negative_prompt="",
                num_inference_steps=1,
                guidance_scale=1.0,
                width=width,
                height=height,
                generator=generator,
                output_type="latent",
            )
        # Also warm up VAE decode (different kernel shapes).
        # Must use autocast: tiling is enabled, and post_quant_conv biases are
        # fp32 even though weights are fp16 — direct decode outside autocast crashes.
        dummy_latent = torch.randn(
            1, 4, height // 8, width // 8, device="cuda", dtype=DTYPE
        )
        with torch.no_grad(), torch.autocast("cuda"):
            pipe.vae.decode(dummy_latent)
        gc.collect()
        torch.cuda.empty_cache()

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
    print(f"🔥 Warming up base pipeline ({width}x{height}, 1 step)...")
    t0 = time.monotonic()
    run_warmup()
    print(f"   Warmed in {time.monotonic() - t0:.1f}s")
    save_warmup(f"{model}_{width}x{height}_1step")

    gc.collect()
    torch.cuda.empty_cache()


def generate_image(pipe, **kwargs):
    """
    Safely intercepts integer seeds and converts them to Diffusers-compatible Generators.
    ip_adapter_scale: if provided, calls pipe.set_ip_adapter_scale() before generation
    so the scale can be changed per-call without reloading weights.
    """
    seed = kwargs.pop("seed", -1)
    ip_adapter_scale = kwargs.pop("ip_adapter_scale", None)
    if seed == -1:
        seed = randint(0, 2**32 - 1)

    print(f"🎲 Generating with seed: {seed}")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs["generator"] = generator

    if getattr(pipe, "pag_attn_processors", None):
        kwargs.setdefault("pag_scale", 3.0)

    if ip_adapter_scale is not None:
        pipe.set_ip_adapter_scale(ip_adapter_scale)

    return pipe(**kwargs).images


def shutdown():
    """
    Cleanly releases all VRAM resources. Should be called on application exit.
    """
    try:
        from src.upscaler import cleanup_upscaler

        cleanup_upscaler()
    except Exception:
        pass
    print("🛑 Shutting down pipeline and releasing resources...")
    cleanup_resources()


def apply_filmic_finish(
    img: Image.Image, grain_intensity: float = 0.020
) -> Image.Image:
    """
    Applies micro-contrast recovery (unsharp mask) and monochromatic luma grain
    to a LANCZOS-downsampled image, simulating physical sensor noise and restoring
    the microscopic edge sharpness destroyed by mathematical averaging.
    """
    from PIL import ImageFilter

    # Strip alpha; process RGB only so noise never contaminates the alpha channel.
    alpha = None
    if img.mode == "RGBA":
        r, g, b, alpha = img.split()
        img = Image.merge("RGB", (r, g, b))
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Unsharp mask: recovers micro-contrast softened by LANCZOS averaging.
    sharpened = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))

    if grain_intensity > 0:
        img_array = np.array(sharpened, dtype=np.float32)
        # Thread-safe RNG (no global state mutation) — intentionally non-deterministic.
        rng = np.random.default_rng()
        noise = rng.normal(
            loc=0,
            scale=255 * grain_intensity,
            size=(img_array.shape[0], img_array.shape[1], 1),
        )
        # Broadcast across all 3 RGB channels equally: monochromatic luma grain.
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(img_array)
    else:
        result = sharpened

    if alpha is not None:
        result.putalpha(alpha)

    return result
