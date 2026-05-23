from diffusers import ControlNetModel, DPMSolverMultistepScheduler
import numpy as np
from diffusers import (
    AutoPipelineForInpainting,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
)
from pydantic import Field
from PIL import Image
import torch
from src.pipeline import (
    get_pipe,
    get_cpu_vae,
    decode_latents_safe,
    encode_image_safe,
    attach_inference_timing,
    finalize_inference_timing,
    register_cleanup_hook,
)
from src.nodes.base_node import BaseNode, BaseNodeModel
from src.utils import is_rocm, has_vram_gte

_EMBED_KEYS = {
    "prompt_embeds",
    "pooled_prompt_embeds",
    "negative_prompt_embeds",
    "negative_pooled_prompt_embeds",
}

_CONTROLNET_ID = "xinsir/controlnet-union-sdxl-1.0"
_controlnet_cache: ControlNetModel | None = None
_cn_pipe_cache: dict[
    str,
    StableDiffusionXLControlNetPipeline
    | StableDiffusionXLControlNetInpaintPipeline
    | StableDiffusionXLInpaintPipeline,
] = {}


def _clear_cn_cache() -> None:
    """Drop all cached CN/inpainting pipelines so cleanup_resources() fully frees VRAM.

    The CN pipe shares UNet/VAE references with the base pipeline. Without this,
    those components stay alive on CUDA even after cleanup_resources() deletes
    _cached_pipe, causing OOM on the next get_pipe() call.
    """
    global _cn_pipe_cache, _controlnet_cache
    _cn_pipe_cache.clear()
    _controlnet_cache = None


register_cleanup_hook(_clear_cn_cache)


def _get_cn_pipe(
    model: str,
) -> (
    StableDiffusionXLControlNetPipeline
    | StableDiffusionXLControlNetInpaintPipeline
    | StableDiffusionXLInpaintPipeline
):
    global _controlnet_cache, _cn_pipe_cache

    # Load base pipe first so any model-switch cleanup runs before we check the cache.
    base = get_pipe(model)

    # If the cached pipe was built from a now-stale base (e.g. after a model switch),
    # drop it and rebuild so we don't hold stale UNet/VAE references alive.
    if model in _cn_pipe_cache and _cn_pipe_cache[model].unet is not base.unet:
        del _cn_pipe_cache[model]

    if model not in _cn_pipe_cache:
        is_inpaint_unet = base.unet.config.in_channels == 9

        if has_vram_gte(24.0):
            # High VRAM: load ControlNet and build a ControlNet-guided pipeline.
            if _controlnet_cache is None:
                print(f"📦 Loading ControlNet: {_CONTROLNET_ID}")
                _controlnet_cache = ControlNetModel.from_pretrained(
                    _CONTROLNET_ID, torch_dtype=torch.float16
                ).to("cuda")
            print(f"📦 Building ControlNet pipeline for model: {model}")
            pipeline_cls = (
                StableDiffusionXLControlNetInpaintPipeline
                if is_inpaint_unet
                else StableDiffusionXLControlNetPipeline
            )
            cn_pipe = pipeline_cls(
                vae=base.vae,
                text_encoder=base.text_encoder,
                text_encoder_2=base.text_encoder_2,
                tokenizer=base.tokenizer,
                tokenizer_2=base.tokenizer_2,
                unet=base.unet,
                controlnet=_controlnet_cache,
                scheduler=base.scheduler,
            )
        else:
            # Low VRAM (<24 GB): skip ControlNet entirely, use a plain inpainting
            # pipeline to stay within the VRAM budget.
            print(
                f"📦 Building plain inpainting pipeline for model: {model} (VRAM < 24 GB)"
            )
            cn_pipe = AutoPipelineForInpainting.from_pipe(base)

        if is_rocm() and hasattr(cn_pipe, "_encode_vae_image"):
            # gfx1200 GPU VAE encoder silently produces zeros (hipErrorLaunchFailure).
            # Encode on the dedicated CPU VAE to avoid this. Critically, do NOT convert
            # to PIL first — PIL clamps to uint8 (256 levels), introducing an 8-bit
            # quantization floor in the latents that manifests as persistent grainy noise
            # the U-Net cannot fully remove. Stay in float the entire time.
            # Called by diffusers with image: [B, C, H, W] float16 in [-1, 1] on CUDA.
            def _encode_vae_image_cpu(image, generator=None):
                cpu_vae = get_cpu_vae()  # madebyollin/sdxl-vae-fp16-fix, pre-compiled
                scaling = float(getattr(cn_pipe.vae.config, "scaling_factor", 0.18215))
                # Match the CPU VAE's weight dtype (bf16 or fp32 per CPU capability)
                vae_dtype = next(iter(cpu_vae.parameters())).dtype
                results = []
                for img_t in image:
                    # Unsqueeze to [1, C, H, W], keep full float precision
                    cpu_t = img_t.unsqueeze(0).to("cpu", dtype=vae_dtype).contiguous()
                    with torch.inference_mode():
                        # Use .mean (deterministic) — avoids passing a CUDA generator
                        # to a CPU op, and gives stable conditioning for inpainting
                        latents = cpu_vae.encode(cpu_t, return_dict=False)[0].mean
                        latents = latents * scaling
                    results.append(latents.to(image.device, dtype=torch.float16))
                return torch.cat(results, dim=0)

            cn_pipe._encode_vae_image = _encode_vae_image_cpu

        # Use a stochastic SDE solver for outpainting. The base pipeline uses a
        # deterministic DPM++ 2M ODE which converges to a blurry textural mean when
        # denoising a structureless void. DPM++ 2M SDE injects Gaussian noise at every
        # step, forcing the UNet to resolve micro-textures rather than averaging them.
        # This scheduler is scoped to the cn_pipe cache and does not affect text2image.
        cn_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            cn_pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

        # s2=0.9: Raised from 0.2. The previous value applied an 80% low-pass
        # filter to stage-2 skip connections, annihilating the high-frequency
        # micro-texture that the SDE solver (sde-dpmsolver++) is specifically
        # designed to inject. s2=0.9 is mild dampening — structural integrity
        # is maintained by the SDE's stochastic coherence, not skip suppression.
        cn_pipe.enable_freeu(s1=0.9, s2=0.9, b1=1.3, b2=1.4)
        _cn_pipe_cache[model] = cn_pipe
    return _cn_pipe_cache[model]


class OutpaintingInputs(BaseNodeModel):
    model: str = Field("juggernaut")
    steps: int = Field(50, ge=1, le=150)
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0)
    strength: float = Field(1.0, ge=0.0, le=1.0)
    width: int = Field(1024)
    height: int = Field(1024)
    white_threshold: int = Field(
        245,
        ge=0,
        le=255,
        description="Pixels with all channels >= this value are treated as the fill zone.",
    )
    mask_blur: int = Field(
        10,
        ge=0,
        description="Blur radius applied to the mask edges for a softer blend. 0 disables.",
    )
    composite_original: bool = Field(
        False,
        description="Paste the original pixels back over the preserved region after generation. Ensures pixel-perfect fidelity outside the fill zone at the cost of a visible hard seam.",
    )


def _make_mask(img: Image.Image, threshold: int) -> Image.Image:
    """Return an 'L' mask: 255 where the image is near-white (fill zone), 0 elsewhere."""
    arr = np.array(img.convert("RGB"))
    white = np.all(arr >= threshold, axis=-1)
    return Image.fromarray((white * 255).astype(np.uint8), mode="L")


def _apply_mask_to_image(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Zero out the fill zone (mask=255) to produce the masked conditioning image."""
    img_arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    mask_arr = np.array(mask.convert("L")).astype(np.float32) / 255.0
    masked = img_arr * (1.0 - mask_arr[..., None])
    return Image.fromarray((masked * 255).astype(np.uint8), mode="RGB")


class OutpaintingNode(BaseNode):
    output_key = "images"

    def __init__(self, inputs: OutpaintingInputs = OutpaintingInputs()):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "outpainting"
        self.embeds: dict | None = None
        self.images: list[Image.Image] = []

    def __call__(
        self,
        images: list[Image.Image] | None = None,
        masks: list[Image.Image] | None = None,
        *args,
        **kwargs,
    ) -> dict[str, list[Image.Image]]:
        p = self.params
        raw = images if images is not None else self.images
        init_images = [
            img.convert("RGB").resize((p.width, p.height), Image.LANCZOS) for img in raw
        ]
        # Sharp masks: precise fill-zone map from TransformNode when available,
        # otherwise fall back to white-pixel detection.
        if masks is not None:
            sharp_masks = [
                m.convert("L").resize((p.width, p.height), Image.NEAREST) for m in masks
            ]
        else:
            sharp_masks = [_make_mask(img, p.white_threshold) for img in init_images]
        masks_pipeline = list(sharp_masks)

        pipe = _get_cn_pipe(p.model)

        # Zero out the fill zone in the ControlNet conditioning image so the model
        # sees a neutral void rather than a hard white boundary.
        masked_control_images = [
            _apply_mask_to_image(img, m) for img, m in zip(init_images, masks_pipeline)
        ]

        # Blur the inpainting blend mask AFTER computing the control images so the
        # ControlNet still gets a sharp void boundary while the latent blend uses a
        # soft gradient transition to avoid visible seams.
        if p.mask_blur > 0 and hasattr(pipe, "mask_processor"):
            masks_pipeline = [
                pipe.mask_processor.blur(m, blur_factor=p.mask_blur)
                for m in masks_pipeline
            ]

        force_latent = is_rocm()
        is_cn_pipe = isinstance(
            pipe,
            (
                StableDiffusionXLControlNetPipeline,
                StableDiffusionXLControlNetInpaintPipeline,
            ),
        )
        is_inpaint_pipe = isinstance(
            pipe,
            (
                StableDiffusionXLControlNetInpaintPipeline,
                StableDiffusionXLInpaintPipeline,
            ),
        )
        pipe_kwargs = {
            "width": p.width,
            "height": p.height,
            "num_inference_steps": p.steps,
            "guidance_scale": p.cfg_scale,
            "strength": p.strength,
            "output_type": "latent" if force_latent else "pil",
        }
        if is_cn_pipe:
            pipe_kwargs["controlnet_conditioning_scale"] = 0.85
            pipe_kwargs["control_guidance_start"] = 0.0
            # Truncate ControlNet at 80% of the schedule. Full-schedule guidance
            # (1.0) forces the network to extract structure from pure void noise
            # in early high-variance steps and then locks in those hallucinated
            # priors through final refinement. The last 20% of steps run free,
            # letting the base SDXL U-Net organically blend the seam boundary.
            pipe_kwargs["control_guidance_end"] = 0.8
            # guidance_rescale compensates for exposure blowout caused by CFG on models
            # trained with a v_prediction objective and zero terminal SNR. Applying it
            # to standard epsilon-prediction models (juggernaut, master) crushes the
            # micro-texture variance the UNet needs to generate sharp detail in the void.
            # Only set it when the scheduler was configured for v_prediction.
            # The UNet config.json does not contain prediction_type for SDXL checkpoints;
            # pipeline.get_pipe() injects "prediction_type": "v_prediction" into the
            # scheduler config for vpred/noob/illustrious models — that is the
            # authoritative source.
            is_vpred = (
                pipe.scheduler.config.get("prediction_type", "epsilon")
                == "v_prediction"
            )
            if is_vpred:
                pipe_kwargs["guidance_rescale"] = 0.7
        if is_inpaint_pipe:
            # Inpainting pipeline: image=init (PIL), mask_image=fill zone.
            # On ROCm, _encode_vae_image is monkeypatched at pipeline-build time (see
            # _get_cn_pipe) to encode via CPU VAE, avoiding the gfx1200 GPU encoder hang.
            pipe_kwargs["image"] = init_images
            pipe_kwargs["mask_image"] = masks_pipeline
            if is_cn_pipe:
                # Zeroed-out control image: ControlNet sees existing structure on one
                # side and a neutral void on the other — no white-pixel collision.
                pipe_kwargs["control_image"] = masked_control_images
        else:
            # Non-inpainting ControlNet: image IS the ControlNet conditioning input.
            # Pass the zeroed version so white pixels don't appear as geometry.
            pipe_kwargs["image"] = masked_control_images

        # Merge compel embeds — from self.embeds (set by DAG executor)
        # or passed directly as **kwargs from a workflow.
        if self.embeds is not None:
            pipe_kwargs.update(self.embeds)
        embed_kwargs = {k: v for k, v in kwargs.items() if k in _EMBED_KEYS}
        pipe_kwargs.update(embed_kwargs)

        pipe_kwargs, t0 = attach_inference_timing(pipe_kwargs, label="outpainting")
        output = pipe(**pipe_kwargs).images
        finalize_inference_timing("outpainting", t0)

        if force_latent and isinstance(output, torch.Tensor):
            output = decode_latents_safe(pipe, output)
        if isinstance(output, torch.Tensor):
            output = [
                Image.fromarray(
                    (img.float().clamp(0, 1) * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy(),
                    mode="RGB",
                )
                for img in output
            ]

        if p.composite_original:
            composited = []
            for gen_img, orig_img, mask in zip(output, init_images, sharp_masks):
                result = gen_img.copy()
                inv_mask = Image.fromarray(255 - np.array(mask), mode="L")
                result.paste(orig_img, mask=inv_mask)
                composited.append(result)
            output = composited

        return {"images": output}
