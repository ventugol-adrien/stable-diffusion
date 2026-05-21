from diffusers import AutoPipelineForImage2Image
from pydantic import Field, ConfigDict
from PIL import Image, ImageOps
import torch
from src.nodes.text2image import Text2ImageInputs
from src.pipeline import (
    get_pipe,
    decode_latents_safe,
    encode_image_safe,
    attach_inference_timing,
    finalize_inference_timing,
)
from src.nodes.base_node import BaseNode
from src.utils import is_rocm


class Image2ImageInputs(Text2ImageInputs):
    strength: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Strength for image-to-image generation (0.0 = no change, 1.0 = full transformation)",
    )


class Image2ImageNode(BaseNode):
    output_key = "images"

    def __init__(self, inputs: Image2ImageInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "image2image"
        self.embeds = None
        self.images: list[Image.Image] = []

    def __call__(
        self, images: list[Image.Image] | torch.Tensor = None, *args, **kwargs
    ) -> dict[str, list[Image.Image]]:
        force_latent = is_rocm() and self.params.output_type == "pil"
        raw = images if images is not None else self.images
        if isinstance(raw, torch.Tensor):
            init_images = raw
        else:
            init_images = [
                ImageOps.fit(
                    img, (self.params.width, self.params.height), method=Image.LANCZOS
                )
                for img in raw
            ]
        base_pipe = get_pipe(self.params.model)

        # On ROCm gfx1200 the GPU VAE encoder hangs (hipErrorLaunchFailure).
        # Pre-encode input images on CPU via encode_image_safe so the pipeline
        # receives a [B, 4, H/8, W/8] latent tensor and skips its internal
        # vae.encode() call entirely (diffusers checks shape[1] == 4).
        if is_rocm() and not isinstance(init_images, torch.Tensor):
            encoded = [encode_image_safe(base_pipe, img) for img in init_images]
            init_images = torch.cat(encoded, dim=0)

        pipe_kwargs = {
            "image": init_images,
            "width": self.params.width,
            "height": self.params.height,
            "num_inference_steps": self.params.steps,
            "guidance_scale": self.params.cfg_scale,
            "num_images_per_prompt": self.params.num_images_per_prompt,
            "strength": self.params.strength,
            "output_type": "latent" if force_latent else self.params.output_type,
        }
        if self.embeds is not None:
            pipe_kwargs.update(self.embeds)
        pipe_kwargs.update(kwargs)
        pipe = AutoPipelineForImage2Image.from_pipe(base_pipe)
        pipe_kwargs, t0 = attach_inference_timing(pipe_kwargs, label="image2image")
        output = pipe(**pipe_kwargs).images
        finalize_inference_timing("image2image", t0)
        if isinstance(output, torch.Tensor):
            _m = output.float().mean().item()
            _s = output.float().std().item()
            print(
                f"  [img2img diag] strength={self.params.strength:.2f}  latent mean={_m:.4f}  std={_s:.4f}"
            )
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
        return {"images": output}
