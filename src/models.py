import io
import os
import base64
import json
from typing import Literal
import httpx
from pydantic import BaseModel, ConfigDict, Field
from fastapi import Request, UploadFile, File
from PIL import Image

from src.nodes.compel_node import CompelInputs, CompelNode
from src.nodes.text2image import Text2ImageInputs, Text2ImageNode
from src.nodes.image2image import Image2ImageInputs, Image2ImageNode
from src.nodes.base_node import BaseNode


class TransformParams(BaseModel):
    dx: int = 0
    dy: int = 0
    z: float = Field(1.0, ge=0.1, le=5.0)
    r: float = Field(0.0, ge=-360.0, le=360.0)


async def _decode_init_image(value) -> Image.Image | None:
    if value is None:
        return None
    if hasattr(value, "read"):  # UploadFile
        data = await value.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(value, str):
        if value.startswith("data:"):
            _, encoded = value.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        async with httpx.AsyncClient() as client:
            resp = await client.get(value, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return None


class DAGForm(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: dict[str, BaseModel] = Field(..., description="List of nodes in the DAG")
    init_image: Image.Image | None = None
    hires_strength: float = 0.35

    @classmethod
    async def as_form(cls, request: Request) -> "DAGForm":
        form_data = await request.form()
        prompt = form_data.get("prompt")
        negative_prompt = form_data.get("negative_prompt")
        model = form_data.get("model", "juggernaut")
        lightning = form_data.get("lightning", "false").lower() == "true"
        width = int(form_data.get("width", 1024))
        height = int(form_data.get("height", 1024))
        steps = int(form_data.get("steps", 50))
        cfg_scale = float(form_data.get("cfg_scale", 7.5))
        num_images_per_prompt = int(form_data.get("batch_size", 1))
        init_image_raw = form_data.get("init_image")
        strength = float(form_data.get("strength", 0.75))
        hires_strength = float(form_data.get("hires_strength", 0.35))
        decoded_init_image = await _decode_init_image(init_image_raw)

        if decoded_init_image is not None:
            node1: Text2ImageInputs | Image2ImageInputs = Image2ImageInputs(
                num_images_per_prompt=num_images_per_prompt,
                model=model,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                strength=strength,
                dependencies=["0"],
                next_nodes=["2"],
            )
        else:
            node1 = Text2ImageInputs(
                num_images_per_prompt=num_images_per_prompt,
                model=model,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                output_type="pt",
                dependencies=["0"],
                next_nodes=["2"],
            )

        return cls(
            init_image=decoded_init_image,
            hires_strength=hires_strength,
            nodes={
                "0": CompelInputs(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model=model,
                    lightning=lightning,
                    next_nodes=["1"],
                ),
                "1": node1,
                "2": Image2ImageInputs(
                    num_images_per_prompt=num_images_per_prompt,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    strength=strength,
                    dependencies=["1"],
                ),
            },
        )


class AssetRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    input_image: UploadFile

    @classmethod
    def as_form(cls, input_image: UploadFile = File(...)) -> "AssetRequest":
        return cls(input_image=input_image)


class LoRA(BaseModel):
    name: str
    scale: float = 0.5


class DivergentSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    depthmap: UploadFile = Field(default=None, alias="depth_map")
    depthmap_scale: float = Field(default=0.6, alias="depth_map_scale")
    canny_edges: UploadFile = None
    edges_scale: float = Field(default=0.2, alias="canny_edges_scale")
    mask: UploadFile = None
    reference: UploadFile = None
    mask_strength: float = Field(default=1.0, alias="strength")
    ip_image: UploadFile = None
    ip_scale: float = None

    # Spatial transform parameters (sent as 0.transform.dx, 0.transform.dy, etc.)
    transform_input_image: UploadFile = None
    transform_dx: int = None
    transform_dy: int = None
    transform_z: float = None
    transform_r: float = None
    transform_strength: float = None


class ImageRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    user_input: str
    loras: list[LoRA] = []
    model: str = os.environ.get("DEFAULT_MODEL", "juggernaut")
    strength: float = None
    reference: UploadFile = None
    depthmap: UploadFile = Field(default=None, alias="depth_map")
    depth_scales: list[float] = []
    canny_edges: UploadFile = None
    edges_scales: list[float] = Field(default=[], alias="canny_edges_scales")
    divergent_spaces: list[DivergentSpace] = []
    ip_adapter_scale: float = Field(default=None, alias="ip_scale")
    ip_adapter_image: UploadFile = Field(default=None, alias="ip_image")
    lightning: bool = False
    resolution: Literal["360p", "480p", "720p", "1080p"] = "480p"
    image_seed: int = -1
    prompt_seed: int = -1
    batch_size: int = 1
    final_strength: float = Field(default=None, ge=0.0, le=1.0)
    grain_intensity: float = Field(default=0.020, ge=0.0, le=0.10)

    @classmethod
    async def as_form(cls, request: Request) -> "ImageRequest":
        form_data = await request.form()
        data = {}
        divergent_spaces = {}
        lora_spaces: dict[int, dict] = {}

        for key, value in form_data.multi_items():
            if value == "":
                continue

            if "." in key:
                parts = key.split(".", 1)
                if parts[0].isdigit():
                    idx = int(parts[0])
                    # Collapse nested dots: "transform.dx" → "transform_dx"
                    field = parts[1].replace(".", "_")
                    if idx not in divergent_spaces:
                        divergent_spaces[idx] = {}

                    if field in divergent_spaces[idx]:
                        if not isinstance(divergent_spaces[idx][field], list):
                            divergent_spaces[idx][field] = [
                                divergent_spaces[idx][field]
                            ]
                        divergent_spaces[idx][field].append(value)
                    else:
                        divergent_spaces[idx][field] = value
                elif parts[0] == "loras" and "." in parts[1]:
                    # loras.<idx>.<field>  e.g. loras.0.name, loras.0.scale
                    idx_str, field = parts[1].split(".", 1)
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        if idx not in lora_spaces:
                            lora_spaces[idx] = {}
                        lora_spaces[idx][field] = value
            else:
                if key == "loras" and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        pass

                if key in ["depth_scales", "edges_scales", "loras"]:
                    if key not in data:
                        data[key] = []
                    if isinstance(value, list):
                        data[key].extend(value)
                    else:
                        data[key].append(value)
                else:
                    if key in data:
                        if not isinstance(data[key], list):
                            data[key] = [data[key]]
                        data[key].append(value)
                    else:
                        data[key] = value

        if divergent_spaces:
            sorted_spaces = [
                divergent_spaces[k] for k in sorted(divergent_spaces.keys())
            ]
            data["divergent_spaces"] = sorted_spaces

        if lora_spaces:
            sorted_loras = [lora_spaces[k] for k in sorted(lora_spaces.keys())]
            existing = data.get("loras", [])
            if not isinstance(existing, list):
                existing = []
            data["loras"] = existing + sorted_loras

        return cls.model_validate(data)
