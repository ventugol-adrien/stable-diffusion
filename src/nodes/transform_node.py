from PIL import Image
from pydantic import Field

from src.nodes.base_node import BaseNode, BaseNodeModel


class TransformInputs(BaseNodeModel):
    width: int = Field(1024)
    height: int = Field(1024)


class TransformNode(BaseNode):
    output_key = "images"

    def __init__(self, inputs: TransformInputs = TransformInputs()):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "transform"
        self.images: list[Image.Image] = []

    def __call__(
        self, images: list[Image.Image] | None = None, *args, **kwargs
    ) -> dict[str, list[Image.Image]]:
        imgs = images if images is not None else self.images
        size = (self.params.width, self.params.height)
        return {
            "images": [img.convert("RGB").resize(size, Image.LANCZOS) for img in imgs]
        }
