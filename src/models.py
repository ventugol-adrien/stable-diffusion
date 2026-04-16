import os
import json
from pydantic import BaseModel, ConfigDict, Field
from fastapi import Request, UploadFile


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
    image_seed: int = -1
    prompt_seed: int = -1
    batch_size: int = 1

    @classmethod
    async def as_form(cls, request: Request) -> "ImageRequest":
        form_data = await request.form()
        data = {}
        divergent_spaces = {}

        for key, value in form_data.multi_items():
            if value == "":
                continue

            if "." in key:
                parts = key.split(".", 1)
                if parts[0].isdigit():
                    idx = int(parts[0])
                    field = parts[1]
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

        return cls.model_validate(data)
