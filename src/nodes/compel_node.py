from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
from src.prompt import process_prompt
from src.pipeline import get_pipe
from src.nodes.base_node import BaseNode, BaseNodeModel
from diffusers import StableDiffusionXLControlNetPipeline
from compel import CompelForSDXL


class CompelInputs(BaseNodeModel):
    prompt: list[str] | str = Field(..., description="Text prompt for image generation")
    model: str = Field("juggernaut", description="Model to use for image generation")
    model_config = ConfigDict(extra="allow")


class CompelOutputs(BaseModel):
    prompt_embeds: list[float] = Field(..., description="Positive prompt embeddings")
    pooled_prompt_embeds: list[float] = Field(
        ..., description="Pooled positive prompt embeddings"
    )
    negative_prompt_embeds: list[float] = Field(
        ..., description="Negative prompt embeddings"
    )
    negative_pooled_prompt_embeds: list[float] = Field(
        ..., description="Pooled negative prompt embeddings"
    )


class CompelNode(BaseNode):
    def __init__(self, inputs: CompelInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "compel"

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        pipe = get_pipe()
        positive_prompt, negative_prompt = process_prompt(
            self.params.prompt, self.params.model
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

        return CompelOutputs(**embeds)

    def __enter__(self, *args, **kwds):
        super().__enter__(*args, **kwds)
        if self.is_source():
            pass

    def __exit__(self, *args, **kwds):
        super().__exit__(*args, **kwds)
        if self.is_terminal():
            pass
