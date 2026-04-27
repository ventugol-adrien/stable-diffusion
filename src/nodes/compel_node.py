from pydantic import BaseModel, Field, ConfigDict

from PIL import Image
from src.prompt import process_prompt
from src.pipeline import get_pipe
from src.nodes.base_node import BaseNode, BaseNodeModel
from diffusers import StableDiffusionXLControlNetPipeline
from compel import CompelForSDXL
from torch import Tensor


class CompelInputs(BaseNodeModel):
    prompt: list[str] | str = Field(..., description="Text prompt for image generation")
    negative_prompt: list[str] | str | None = Field(
        None, description="Negative text prompt for image generation"
    )
    model: str = Field("juggernaut", description="Model to use for image generation")
    model_config = ConfigDict(extra="allow")


class PromptEmbeds:
    def __init__(
        self,
        prompt_embeds: Tensor,
        pooled_prompt_embeds: Tensor,
        negative_prompt_embeds: Tensor,
        negative_pooled_prompt_embeds: Tensor,
    ):
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    def keys(self):
        return [
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]

    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, key)


class CompelNode(BaseNode):
    output_key = "embeds"

    def __init__(self, inputs: CompelInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "compel"

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        pipe = get_pipe()
        positive_prompt, negative_prompt = process_prompt(
            self.params.prompt, self.params.negative_prompt, self.params.model
        )
        compel_proc = CompelForSDXL(pipe=pipe, device="cuda")
        conditioning = compel_proc(positive_prompt, negative_prompt=negative_prompt)
        # Placeholder for actual image generation logic
        embeds = {
            "prompt_embeds": conditioning.embeds,
            "pooled_prompt_embeds": conditioning.pooled_embeds,
            "negative_prompt_embeds": conditioning.negative_embeds,
            "negative_pooled_prompt_embeds": conditioning.negative_pooled_embeds,
        }

        return PromptEmbeds(**embeds)

    def __enter__(self, *args, **kwds):
        super().__enter__(*args, **kwds)
        if self.is_source():
            pass

    def __exit__(self, *args, **kwds):
        super().__exit__(*args, **kwds)
        if self.is_terminal():
            pass
