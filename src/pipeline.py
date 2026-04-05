from random import randint
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL
from pathlib import Path
import os, gc, time
import torch

_cached_pipe: StableDiffusionXLPipeline | None = None
_cached_fast_pipe: StableDiffusionXLPipeline | None = None
_cached_model_name: str | None = None
DTYPE = torch.float16
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CWD = Path(os.getcwd())
MODEL_CACHE_DIR = CWD / "caches" / "models"
WARMED_CONFIGS_FILE = CWD / "caches" / "warmed_configs.json"
_warmed_configs_cache: set[str] | None = None  # in-memory cache of config keys

def cleanup_resources():
    """
    Forcefully releases VRAM. Critical for avoiding Linux 6.14 GTT Swap crashes.
    """
    global _cached_pipe, _cached_fast_pipe

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

    # Force Python GC and ROCm cache clear
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 VRAM resources released.")

def _load_pipeline(model: str) -> StableDiffusionXLPipeline:
    """
    Load an SDXL model. Uses a diffusers-format cache when available
    (from_pretrained is ~3× faster than from_single_file). On first load
    the model is converted and cached automatically.
    """
    cached_dir = MODEL_CACHE_DIR / model

    # FAST PATH: diffusers cache exists
    if (cached_dir / "model_index.json").is_file():
        print(f"⚡ Loading from diffusers cache: {cached_dir}")
        t0 = time.monotonic()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cached_dir, torch_dtype=DTYPE, use_safetensors=True
        )
        print(f"   Loaded in {time.monotonic() - t0:.1f}s (cached)")
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
    print(f"   Loaded in {time.monotonic() - t0:.1f}s")

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

    print(f"🚀 Initializing Optimized Pipeline for RDNA4 (gfx1200)...")

    pipe = _load_pipeline(model)
    # 3. CRITICAL VRAM OPTIMIZATIONS
    # VAE Tiling splits the image into chunks for decoding.
    # This keeps peak memory usage low, preventing OOM and avoiding
    # the deadly swap-to-CPU behavior that crashes Linux 6.14 amdkfd.[20]
    print("🧩 Enabling VAE Tiling (Tile Size: 512)...")
    pipe.vae.enable_tiling()

    # Enable Slicing for further memory savings during batch processing
    pipe.vae.enable_slicing()

    # 4. SCHEDULER OPTIMIZATION
    # Euler Ancestral is fast and widely compatible.
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # 5. MEMORY EFFICIENT ATTENTION
    # Since we enabled the Triton flag above, we try to use Scaled Dot Product Attention (SDP)
    # which is PyTorch's native fast attention (Flash Attention equivalent).
    print("🔥 Pipeline Ready. Using Native SDPA (Flash Attention equivalent).")

    # 6. TRANSFER TO GPU
    # We rely on the env vars to handle the allocation strategy.
    pipe.to("cuda")
    # pipe.load_ip_adapter(
    #     "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
    # )
    # pipe.set_ip_adapter_scale(0.8)

    # pipe.unet = torch.compile(pipe.unet, mode="default",dynamic=True)

    # Optional: Compile VAE decode (smaller speedup, but helps)
    # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="default",dynamic=True)

    _cached_pipe = pipe
    _cached_model_name = model

    return pipe
def get_fast_pipe(model: str = "juggernaut"): ...
def warmup_pipeline(pipe : StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline, width:int=1024, height:int=1024):
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
        # Also warm up VAE decode (different kernel shapes)
        dummy_latent = torch.randn(
            1, 4, height // 8, width // 8, device="cuda", dtype=DTYPE
        )
        with torch.no_grad():
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

def generate_image(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline, **kwargs
):
    real_image_seed = kwargs.get("image_seed", -1)
    if real_image_seed == -1:
        real_image_seed = randint(0, 2**32 - 1)
    kwargs["image_seed"] = real_image_seed
    return pipe(**kwargs).images


def shutdown(): ...
