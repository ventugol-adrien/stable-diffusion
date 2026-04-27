from diffusers import AutoPipelineForImage2Image
from pydantic import Field, ConfigDict
from PIL import Image, ImageOps
import torch
from src.nodes.text2image import Text2ImageInputs
from src.pipeline import get_pipe
from src.nodes.base_node import BaseNode


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
        pipe_kwargs = {
            "image": init_images,
            "width": self.params.width,
            "height": self.params.height,
            "num_inference_steps": self.params.steps,
            "guidance_scale": self.params.cfg_scale,
            "num_images_per_prompt": self.params.num_images_per_prompt,
            "strength": self.params.strength,
        }
        if self.embeds is not None:
            pipe_kwargs.update(self.embeds)
        pipe_kwargs.update(kwargs)
        pipe = AutoPipelineForImage2Image.from_pipe(get_pipe(self.params.model))
        return {"images": pipe(**pipe_kwargs).images}
