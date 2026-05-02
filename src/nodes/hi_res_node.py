import gc

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from pydantic import Field

from src.nodes.base_node import BaseNode, BaseNodeModel
from src.nodes.tiling_node import TilingOutputs, stitch
from src.pipeline import get_pipe


class HiResInputs(BaseNodeModel):
    strength: float = Field(0.35, ge=0.0, le=1.0)
    steps: int = Field(30)
    cfg_scale: float = Field(5.0)
    model: str = Field("juggernaut")


class HiResNode(BaseNode):
    output_key = "images"

    def __init__(self, inputs: HiResInputs = HiResInputs()):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "hires"
        self.embeds: dict | None = None
        self.tiling_outputs: TilingOutputs | None = None

    def __call__(
        self, tiling_outputs: TilingOutputs | None = None, *args, **kwargs
    ) -> dict[str, list[Image.Image]]:
        plan = tiling_outputs if tiling_outputs is not None else self.tiling_outputs
        if plan is None:
            raise ValueError("HiResNode requires a TilingOutputs input.")

        pipe = AutoPipelineForImage2Image.from_pipe(get_pipe(self.params.model))

        pipe_kwargs = {
            "strength": self.params.strength,
            "num_inference_steps": self.params.steps,
            "guidance_scale": self.params.cfg_scale,
            "num_images_per_prompt": 1,
        }
        if self.embeds is not None:
            pipe_kwargs.update(self.embeds)

        for i, (tile, meta) in enumerate(zip(plan.tiles, plan.tile_meta)):
            print(f"   HiRes tile {i + 1}/{len(plan.tiles)} {meta['box']}")
            generator = torch.Generator(device="cuda").manual_seed(meta["seed"])
            refined = (
                pipe(image=tile, generator=generator, **pipe_kwargs)
                .images[0]
                .convert("RGB")
                .resize(meta["size"], Image.LANCZOS)
            )
            refined_np = np.asarray(refined, dtype=np.float32)
            x0, y0, x1, y1 = meta["box"]
            plan.accum[y0:y1, x0:x1] += refined_np * meta["mask"]
            plan.weights[y0:y1, x0:x1] += meta["mask"]
            del refined, refined_np

        gc.collect()
        torch.cuda.empty_cache()

        return {"images": [stitch(plan)]}
