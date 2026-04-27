from pydantic import Field, ConfigDict
from PIL import Image
from src.pipeline import get_pipe
from src.nodes.base_node import BaseNode, BaseNodeModel


class Text2ImageInputs(BaseNodeModel):
    width: int = Field(1024, description="Width of the generated image")
    height: int = Field(1024, description="Height of the generated image")
    steps: int = Field(30, description="Number of steps for image generation")
    cfg_scale: float = Field(7.5, description="CFG scale for image generation")
    model: str = Field("juggernaut", description="Model to use for image generation")
    model_config = ConfigDict(extra="allow")


class Text2ImageNode(BaseNode):
    def __init__(self, inputs: Text2ImageInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs

    def __call__(self, *args, **kwargs) -> list[Image.Image]:
        # Placeholder for actual image generation logic
        print(f"Generating image with kwargs: {kwargs}")
        pipe = get_pipe()
        if super().is_terminal():
            return pipe(**kwargs).images
        else:
            return pipe(**kwargs).images

    def __enter__(self, *args, **kwds):
        super().__enter__(*args, **kwds)
        if self.is_source():
            pass

    def __exit__(self, *args, **kwds):
        print(f"Exiting context for node: {self} with params: {self.params}")
        # Cleanup resources if needed
        pass
