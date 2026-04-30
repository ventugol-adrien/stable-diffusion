from typing import Literal
from pydantic import Field, ConfigDict
from PIL import Image
from src.pipeline import get_pipe
from src.nodes.base_node import BaseNode, BaseNodeModel


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
        pipe_kwargs = {
            "width": self.params.width,
            "height": self.params.height,
            "num_inference_steps": self.params.steps,
            "guidance_scale": self.params.cfg_scale,
            "num_images_per_prompt": self.params.num_images_per_prompt,
            "output_type": self.params.output_type,
        }
        if self.embeds is not None:
            pipe_kwargs.update(self.embeds)
        pipe_kwargs.update(kwargs)
        pipe = get_pipe(self.params.model)
        print(pipe.__class__.__name__)
        print(pipe.scheduler.__class__.__name__)
        return {"images": pipe(**pipe_kwargs).images}

    def __enter__(self, *args, **kwds):
        super().__enter__(*args, **kwds)
        if self.is_source():
            pass

    def __exit__(self, *args, **kwds):
        print(f"Exiting context for node: {self} with params: {self.params}")
        # Cleanup resources if needed
        pass
