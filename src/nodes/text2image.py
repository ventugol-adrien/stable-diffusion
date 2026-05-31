from typing import Literal
from pathlib import Path
import gc
import torch
import numpy as np
from pydantic import Field, ConfigDict
from PIL import Image
from diffusers import ControlNetUnionModel, StableDiffusionXLControlNetUnionPipeline
from src.pipeline import (
    get_pipe,
    decode_latents_safe,
    attach_inference_timing,
    finalize_inference_timing,
)
from src.nodes.base_node import BaseNode, BaseNodeModel
from src.utils import is_rocm

IP_ADAPTER_DIR = Path("/home/adrien/ip_adapters")

# xinsir/controlnet-union-sdxl-1.0 control type IDs:
#   0 = openpose, 1 = depth, 2 = thick line (scribble/hed/softedge),
#   3 = thin line (canny/lineart/mlsd), 4 = normal, 5 = segment
_UNION_CONTROL_DEPTH = 1
_UNION_CONTROL_EDGES = 3
_CONTROLNET_UNION_ID = "xinsir/controlnet-union-sdxl-1.0"


def _get_union_cn_pipe(model: str) -> StableDiffusionXLControlNetUnionPipeline:
    print(f"📦 Loading ControlNet Union model: {_CONTROLNET_UNION_ID}")
    controlnet = ControlNetUnionModel.from_pretrained(
        _CONTROLNET_UNION_ID, torch_dtype=torch.float16
    ).to("cuda")
    print(f"📦 Building ControlNet Union pipeline for model: {model}")
    base = get_pipe(model)
    return StableDiffusionXLControlNetUnionPipeline(
        vae=base.vae,
        text_encoder=base.text_encoder,
        text_encoder_2=base.text_encoder_2,
        tokenizer=base.tokenizer,
        tokenizer_2=base.tokenizer_2,
        unet=base.unet,
        controlnet=controlnet,
        scheduler=base.scheduler,
    )


class Text2ImageInputs(BaseNodeModel):
    width: int = Field(1024, description="Width of the generated image")
    height: int = Field(1024, description="Height of the generated image")
    steps: int = Field(50, description="Number of steps for image generation")
    cfg_scale: float = Field(7.5, description="CFG scale for image generation")
    model: str = Field("juggernaut", description="Model to use for image generation")
    num_images_per_prompt: int = Field(
        1, description="Number of images to generate per prompt"
    )
    output_type: Literal["pil", "pt"] = Field(
        "pil", description="Output type: 'pil' for PIL images, 'pt' for PyTorch tensors"
    )
    model_config = ConfigDict(extra="allow")


class Text2ImageNode(BaseNode):
    def __init__(self, inputs: Text2ImageInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "text2image"
        self.embeds = None

    def __call__(self, *args, **kwargs) -> dict[str, list[Image.Image]]:
        force_latent = is_rocm() and self.params.output_type == "pil"
        use_ip_adapter = kwargs.get("ip_adapter_image", None) and kwargs.get(
            "ip_adapter_scale", None
        )
        use_depthmap = kwargs.get("depthmap", None) and kwargs.get(
            "depthmap_scale", None
        )
        use_edgesmap = kwargs.get("edgesmap", None) and kwargs.get(
            "edgesmap_scale", None
        )
        use_controlnet = use_depthmap or use_edgesmap
        pipe_kwargs = {
            "width": self.params.width,
            "height": self.params.height,
            "num_inference_steps": self.params.steps,
            "guidance_scale": self.params.cfg_scale,
            "num_images_per_prompt": self.params.num_images_per_prompt,
            "output_type": "latent" if force_latent else self.params.output_type,
        }
        if use_ip_adapter:
            pipe_kwargs["ip_adapter_image"] = kwargs["ip_adapter_image"]
            print(f"Using IP Adapter with scale {kwargs['ip_adapter_scale']}")
        if use_depthmap:
            print(f"Using ControlNet depth with scale {kwargs['depthmap_scale']}")
        if use_edgesmap:
            print(f"Using ControlNet edges with scale {kwargs['edgesmap_scale']}")

        if self.embeds is not None:
            pipe_kwargs.update(self.embeds)
        pipe_kwargs.update(kwargs)

        # Remove non-pipeline kwargs that were only passed for routing decisions.
        _CN_KWARGS = {"depthmap", "depthmap_scale", "edgesmap", "edgesmap_scale"}
        for _k in _CN_KWARGS:
            pipe_kwargs.pop(_k, None)
        # ip_adapter_scale is applied via pipe.set_ip_adapter_scale(), not as a call arg.
        ip_adapter_scale = pipe_kwargs.pop("ip_adapter_scale", None)

        if use_controlnet:
            # Build ordered control_image / control_mode lists for the union model.
            control_images: list = []
            control_modes: list[int] = []
            control_scales: list[float] = []
            if use_depthmap:
                control_images.append(kwargs["depthmap"])
                control_modes.append(_UNION_CONTROL_DEPTH)
                control_scales.append(float(kwargs["depthmap_scale"]))
            if use_edgesmap:
                control_images.append(kwargs["edgesmap"])
                control_modes.append(_UNION_CONTROL_EDGES)
                control_scales.append(float(kwargs["edgesmap_scale"]))

            pipe_kwargs["control_image"] = control_images
            pipe_kwargs["control_mode"] = control_modes
            pipe_kwargs["controlnet_conditioning_scale"] = control_scales

            pipe = _get_union_cn_pipe(self.params.model)
            if use_ip_adapter:
                pipe.load_ip_adapter(
                    IP_ADAPTER_DIR,
                    subfolder="",
                    weight_name="ip_adapter_plus.safetensors",
                    image_encoder_folder="image_encoder",
                )
                pipe.set_ip_adapter_scale(ip_adapter_scale)
        else:
            pipe = get_pipe(self.params.model)
            if use_ip_adapter:
                pipe.load_ip_adapter(
                    IP_ADAPTER_DIR,
                    subfolder="",
                    weight_name="ip_adapter_plus.safetensors",
                    image_encoder_folder="image_encoder",
                )
                pipe.set_ip_adapter_scale(ip_adapter_scale)

        print(pipe.__class__.__name__)
        print(pipe.scheduler.__class__.__name__)
        pipe_kwargs, t0 = attach_inference_timing(pipe_kwargs, label="text2image")
        output = pipe(**pipe_kwargs).images
        finalize_inference_timing("text2image", t0)
        if isinstance(output, torch.Tensor):
            _m = output.float().mean().item()
            _s = output.float().std().item()
            print(f"  [txt2img diag] latent mean={_m:.4f}  std={_s:.4f}")
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
        if use_ip_adapter:
            print("Unloading IP Adapter")
            pipe.unload_ip_adapter()
            if getattr(pipe, "image_encoder", None) is not None:
                pipe.image_encoder = None
            torch.cuda.empty_cache()
        if use_controlnet:
            # The CN pipeline is not cached — drop it and its controlnet weights now
            # so VRAM is freed before the next request rather than waiting for GC.
            pipe.controlnet = None
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        return {"images": output}

    def __enter__(self, *args, **kwds):
        super().__enter__(*args, **kwds)
        if self.is_source():
            pass

    def __exit__(self, *args, **kwds):
        print(f"Exiting context for node: {self} with params: {self.params}")
        # Cleanup resources if needed
        pass
