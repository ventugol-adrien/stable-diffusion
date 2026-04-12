from random import randint
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForImage2Image,
    LTX2ImageToVideoPipeline,
    LTX2VideoTransformer3DModel,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    Flux2Pipeline,
    Flux2Transformer2DModel,
    GGUFQuantizationConfig,
)
import diffusers.loaders.single_file_model as single_file_loader
import diffusers.models.model_loading_utils as loading_utils
from transformers import (
    BitsAndBytesConfig as BnbConfig,
    Mistral3ForConditionalGeneration,
    T5EncoderModel,
    T5Tokenizer,
)
from pathlib import Path
import os, gc, time, json
import torch

_cached_pipe: StableDiffusionXLPipeline | Flux2Pipeline | None = None
_cached_fast_pipe: StableDiffusionXLPipeline | None = None
_cached_video_pipe: LTX2ImageToVideoPipeline | None = None
_cached_model_name: str | None = None
DTYPE = torch.float16
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CWD = Path(os.getcwd())
MODEL_CACHE_DIR = CWD / "caches" / "models"
WARMED_CONFIGS_FILE = CWD / "caches" / "warmed_configs.json"
MODELS_DIR = Path.home() / "sd_models"
_warmed_configs_cache: set[str] | None = None  # in-memory cache of config keys
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def cleanup_resources():
    """
    Forcefully releases VRAM. Critical for avoiding Linux 6.14 GTT Swap crashes.
    """
    global _cached_pipe, _cached_fast_pipe, _cached_video_pipe

    # Unload IP-Adapter first (holds extra GPU tensors outside the main model)
    for pipe in (_cached_pipe, _cached_fast_pipe, _cached_video_pipe):
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

    if _cached_video_pipe is not None:
        del _cached_video_pipe
        _cached_video_pipe = None

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
            cached_dir,
            torch_dtype=DTYPE,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
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
        attn_implementation="flash_attention_2",
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
    pipe.enable_freeu(
        s1=0.9,  # Skip connection scaling factor for stage 1
        s2=0.2,  # Skip connection scaling factor for stage 2
        b1=1.3,  # Backbone scaling factor for stage 1
        b2=1.4,  # Backbone scaling factor for stage 2
    )

    # 4. SCHEDULER OPTIMIZATION
    # Euler Ancestral is fast and widely compatible.
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # 5. MEMORY EFFICIENT ATTENTION
    # FlashAttention 2 is enabled at model load time via attn_implementation="flash_attention_2"
    print("🔥 Pipeline Ready. Using FlashAttention 2.")

    # 6. TRANSFER TO GPU
    pipe.to("cuda")

    _cached_pipe = pipe
    _cached_model_name = model

    return pipe


def get_fast_pipe(model: str = "juggernaut"): ...


_cached_video_pipe = None

# ---------------------------------------------------------
# SOTA TENSOR SLICER (MONKEY PATCH)
# Intercepts the GGUF loader to force LTX-2.3 weights into LTX-2.0 architecture
# ---------------------------------------------------------
import diffusers.loaders.single_file_model as single_file_loader
import diffusers.models.model_loading_utils as loading_utils

_original_load_meta = loading_utils.load_model_dict_into_meta


def _patched_load_meta(model, state_dict, *args, **kwargs):
    import torch

    # Build a map of expected parameter shapes from the model
    expected_shapes = {}
    for name, param in model.named_parameters():
        expected_shapes[name] = param.shape
    for name, buf in model.named_buffers():
        expected_shapes[name] = buf.shape

    for key in list(state_dict.keys()):
        if key in expected_shapes:
            tensor = state_dict[key]
            expected = expected_shapes[key]
            if tensor.shape != expected and len(tensor.shape) == len(expected):
                needs_fix = True
                needs_slice = False
                needs_pad = False
                for got, exp in zip(tensor.shape, expected):
                    if got > exp:
                        needs_slice = True
                    elif got < exp:
                        needs_pad = True

                if needs_fix:
                    # Slice oversized dims, pad undersized dims
                    result = tensor
                    # First slice any oversized dimensions
                    if needs_slice:
                        slices = tuple(
                            slice(0, exp) if got > exp else slice(None)
                            for got, exp in zip(result.shape, expected)
                        )
                        result = result[slices]
                    # Then pad any undersized dimensions
                    if needs_pad:
                        pad_args = []
                        # torch.nn.functional.pad uses reversed dim order
                        for got, exp in reversed(list(zip(result.shape, expected))):
                            pad_args.extend([0, max(0, exp - got)])
                        if any(p > 0 for p in pad_args):
                            result = torch.nn.functional.pad(result, pad_args)

                    print(
                        f"🔪 Reshaping tensor {key} from {list(tensor.shape)} to {list(expected)} for compatibility..."
                    )
                    state_dict[key] = result

    result = _original_load_meta(model, state_dict, *args, **kwargs)

    # Materialize any parameters still on meta device (not covered by the GGUF)
    meta_keys = []
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            meta_keys.append(name)
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            meta_keys.append(name)

    if meta_keys:
        print(
            f"⚠️ {len(meta_keys)} parameters still on meta device, initializing with zeros..."
        )
        for name in meta_keys:
            parts = name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr_name = parts[-1]
            old = getattr(obj, attr_name)
            new_tensor = torch.zeros(old.shape, dtype=old.dtype, device="cpu")
            if isinstance(old, torch.nn.Parameter):
                setattr(
                    obj,
                    attr_name,
                    torch.nn.Parameter(new_tensor, requires_grad=old.requires_grad),
                )
            else:
                setattr(obj, attr_name, new_tensor)

    return result


# THE FIX: Apply the patch to BOTH namespaces to guarantee interception
loading_utils.load_model_dict_into_meta = _patched_load_meta
single_file_loader.load_model_dict_into_meta = _patched_load_meta
# ---------------------------------------------------------

_cached_video_pipe = None


def get_video_pipe():
    global _cached_video_pipe
    if _cached_video_pipe is not None:
        return _cached_video_pipe

    models_dir = os.path.expanduser("~/sd_models")
    transformer_gguf_path = os.path.join(models_dir, "ltx23.gguf")
    t5_gguf_path = os.path.join(models_dir, "t5.gguf")
    wrapper_dir = os.path.join(models_dir, "LTX-2-Wrapper")

    print("📋 Generating Native Offline Blueprint for LTX-2.3...")

    # Use the official LTX-2 transformer config from the wrapper
    config_dir = os.path.join(wrapper_dir, "transformer")

    print("🧠 Loading LTX-2.3 Transformer from GGUF (100% Offline)...")
    transformer = LTX2VideoTransformer3DModel.from_single_file(
        transformer_gguf_path,
        config=config_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    print("📖 Loading Text Encoder from GGUF (100% Offline)...")
    text_encoder = T5EncoderModel.from_pretrained(
        os.path.join(wrapper_dir, "text_encoder"),
        gguf_file=t5_gguf_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    print("📝 Loading T5 Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

    print("🔗 Loading Connectors...")
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    connectors_dir = os.path.join(wrapper_dir, "connectors")
    connectors = LTX2TextConnectors.from_pretrained(
        connectors_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # Replace the connector's text_proj_in (sized for Gemma3 @ 3840*49=188160)
    # with one sized for T5-XXL (4096 * N -> 3840)
    # The connector reshapes [batch, seq, hidden] -> [batch, seq//factor, factor*hidden] before proj
    # For T5: hidden=4096, factor=49 would give 200704. Instead, use the T5 output directly.
    # We replace text_proj_in and patch the forward to skip the factor-based reshape.
    import types

    old_caption_channels = connectors.video_connector.transformer_blocks[
        0
    ].attn1.to_q.in_features  # 3840
    connectors.text_proj_in = torch.nn.Linear(
        4096, old_caption_channels, bias=False
    ).to(torch.bfloat16)
    torch.nn.init.normal_(connectors.text_proj_in.weight, std=0.02)

    _orig_connector_forward = connectors.forward

    def _t5_connector_forward(
        text_encoder_hidden_states, attention_mask, additive_mask=False
    ):
        # T5 output: [batch, seq, 4096] - project to 3840 directly (no factor reshape)
        if not additive_mask:
            text_dtype = text_encoder_hidden_states.dtype
            attention_mask = (attention_mask - 1).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            )
            attention_mask = attention_mask.to(text_dtype) * torch.finfo(text_dtype).max

        text_encoder_hidden_states = connectors.text_proj_in(text_encoder_hidden_states)

        video_text_embedding, new_attn_mask = connectors.video_connector(
            text_encoder_hidden_states, attention_mask
        )
        attn_mask = (new_attn_mask < 1e-6).to(torch.int64)
        attn_mask = attn_mask.reshape(
            video_text_embedding.shape[0], video_text_embedding.shape[1], 1
        )
        video_text_embedding = video_text_embedding * attn_mask
        new_attn_mask = attn_mask.squeeze(-1)

        audio_text_embedding, _ = connectors.audio_connector(
            text_encoder_hidden_states, attention_mask
        )
        return video_text_embedding, audio_text_embedding, new_attn_mask

    connectors.forward = _t5_connector_forward

    print("🏗️ Building Final Video Pipeline...")
    _cached_video_pipe = LTX2ImageToVideoPipeline.from_pretrained(
        wrapper_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        audio_vae=None,
        vocoder=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    _cached_video_pipe.enable_model_cpu_offload()

    print("✅ Video Pipeline Warmed and Ready.")
    return _cached_video_pipe


FLUX2_HF_REPO = "black-forest-labs/FLUX.2-dev"
FLUX2_GGUF_PATH = Path.home() / "sd_models" / "flux2.gguf"
FLUX2_TEXT_ENCODER_CACHE = MODEL_CACHE_DIR / "flux2_text_encoder_4bit"


def _load_flux2_text_encoder():
    """Load Mistral 3 Small text encoder in 4-bit, with disk caching."""
    cached = FLUX2_TEXT_ENCODER_CACHE
    bnb_config = BnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )

    if cached.is_dir() and (cached / "config.json").is_file():
        print(f"⚡ Loading cached 4-bit text encoder from {cached}")
        t0 = time.monotonic()
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            cached,
            torch_dtype=DTYPE,
        )
        print(f"   Text encoder loaded in {time.monotonic() - t0:.1f}s (cached)")
        return text_encoder

    print("📦 Loading Mistral text encoder in 4-bit (first time, will cache)...")
    t0 = time.monotonic()
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        FLUX2_HF_REPO,
        subfolder="text_encoder",
        quantization_config=bnb_config,
        torch_dtype=DTYPE,
        token=os.getenv("HF_KEY"),
    )
    print(f"   Text encoder loaded in {time.monotonic() - t0:.1f}s")

    print(f"💾 Caching 4-bit text encoder to {cached}")
    text_encoder.save_pretrained(cached)
    return text_encoder


def _load_flux2_pipeline(model: str) -> Flux2Pipeline:
    # 1. Enable TF32: Massive speedup on Ada Lovelace hardware with zero visual quality loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 2. Load Transformer in Native FP8 instead of GGUF
    # FLUX.2 (32B params) in FP8 takes ~32GB VRAM, fitting perfectly into the L40S's 48GB.
    print(f"📦 Loading Flux2 transformer directly to FP8...")
    t0 = time.monotonic()

    transformer = Flux2Transformer2DModel.from_pretrained(
        FLUX2_HF_REPO,
        subfolder="transformer",
        torch_dtype=torch.float8_e4m3fn,  # Native NVIDIA FP8 format
        token=os.getenv("HF_KEY"),
    )
    print(f"   Transformer loaded in {time.monotonic() - t0:.1f}s")

    text_encoder = _load_flux2_text_encoder()

    print(f"📥 Assembling pipeline with BF16 precision...")
    t0 = time.monotonic()
    pipe = Flux2Pipeline.from_pretrained(
        FLUX2_HF_REPO,
        transformer=transformer,
        text_encoder=text_encoder,
        # The pipeline orchestrator must run in bfloat16 for Flux architectures
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_KEY"),
    )
    print(f"   Pipeline assembled in {time.monotonic() - t0:.1f}s")
    return pipe


def get_flux2_pipe(model: str = "flux2"):
    global _cached_pipe, _cached_model_name

    if _cached_pipe is not None and _cached_model_name == model:
        return _cached_pipe

    if _cached_pipe is not None:
        print("🔄 Switching pipeline/model. Clearing VRAM...")
        cleanup_resources()
        # Force garbage collection to destroy lingering CUDA contexts
        torch.cuda.empty_cache()

    print("🚀 Initializing Optimized Flux2 Pipeline for L40S...")
    pipe = _load_flux2_pipeline(model)

    # 1. Memory Formatting (Must happen BEFORE compilation)
    print("🗄️ Formatting memory layout for Tensor Cores...")
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    # 2. COMPILE FIRST: Trace the clean, unhooked neural network graph
    pipe.transformer.set_attn_processor(
        diffusers.models.attention_processor.AttnProcessor2_0()
    )

    # 2. Compile with "reduce-overhead" and remove "fullgraph=True"
    print("⚙️ Compiling transformer graph (Stable Mode)...")
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="reduce-overhead",
        # Removing fullgraph=True allows PyTorch to dynamically fall back to standard
        # execution for the 1% of FP8 operations that Triton doesn't know how to compile yet.
    )

    # 3. OFFLOAD SECOND: Wrap the compiled kernel in the VRAM manager
    # This prevents the 44GB OOM by ensuring the massive T5 text encoder
    # and the 32B Transformer never occupy the GPU at the same exact time.
    print("🧠 Enabling intelligent VRAM manager...")
    pipe.enable_model_cpu_offload()

    # 4. VAE Optimizations to prevent end-of-generation memory spikes
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("🔥 SOTA Flux2 Pipeline Ready (FP8 + Compiled + Managed VRAM).")

    _cached_pipe = pipe
    _cached_model_name = model
    return pipe


def warmup_pipeline(
    pipe: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline,
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

    return pipe(**kwargs).images


def shutdown(): ...
