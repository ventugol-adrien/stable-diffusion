import math
from random import randint

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from src.nodes.base_node import BaseNode, BaseNodeModel

# ── helpers ───────────────────────────────────────────────────────────────────


def _axis_starts(length: int, tile_size: int, min_overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    tile_count = max(2, math.ceil((length - min_overlap) / (tile_size - min_overlap)))
    max_start = length - tile_size
    starts = [round(i * max_start / (tile_count - 1)) for i in range(tile_count)]
    deduped: list[int] = []
    for s in starts:
        if not deduped or s != deduped[-1]:
            deduped.append(s)
    if deduped[-1] != max_start:
        deduped.append(max_start)
    return deduped


def _axis_weights(
    length: int, feather: int, fade_in: bool, fade_out: bool
) -> np.ndarray:
    w = np.ones(length, dtype=np.float32)
    if feather <= 0 or length <= 1:
        return w
    feather = min(feather, max(1, length // 2))
    ramp = 0.5 * (
        1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, feather, dtype=np.float32))
    )
    if fade_in:
        w[:feather] = np.minimum(w[:feather], ramp)
    if fade_out:
        w[-feather:] = np.minimum(w[-feather:], ramp[::-1])
    return w


def _tile_blend_mask(
    tile_w: int,
    tile_h: int,
    feather: int,
    left: bool,
    right: bool,
    top: bool,
    bottom: bool,
) -> np.ndarray:
    x = _axis_weights(tile_w, feather, left, right)
    y = _axis_weights(tile_h, feather, top, bottom)
    return (y[:, None] * x[None, :])[..., None]  # (H, W, 1)


def _tile_boxes(
    width: int, height: int, tile_size: int, min_overlap: int
) -> list[tuple[int, int, int, int]]:
    xs = _axis_starts(width, tile_size, min_overlap)
    ys = _axis_starts(height, tile_size, min_overlap)
    return [
        (x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height))
        for y0 in ys
        for x0 in xs
    ]


# ── models ───────────────────────────────────────────────────────────────────


class TilingOutputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tiles: list[Image.Image]
    tile_meta: list[dict]
    original: np.ndarray
    accum: np.ndarray
    weights: np.ndarray


# ── node ──────────────────────────────────────────────────────────────────────


class TilingInputs(BaseNodeModel):
    num_tiles: int = Field(4)
    min_overlap: int = Field(256)
    feather: int = Field(256)


class TilingNode(BaseNode):
    output_key = "tiles"

    def __init__(self, inputs: TilingInputs = TilingInputs()):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "tiling"
        self.images: list[Image.Image] = []

    def __call__(
        self, images: list[Image.Image] | None = None, *args, **kwargs
    ) -> TilingOutputs:
        imgs = images if images is not None else self.images
        image = imgs[0].convert("RGB")
        width, height = image.size

        tiles_per_axis = max(1, round(math.sqrt(self.params.num_tiles)))
        tile_size = self.params.min_overlap + math.ceil(
            (max(width, height) - self.params.min_overlap) / tiles_per_axis
        )
        boxes = _tile_boxes(width, height, tile_size, self.params.min_overlap)
        print(f"🧩 {len(boxes)} tile(s) @ {tile_size}px covering {width}x{height}")

        tiles, tile_meta = [], []
        for x0, y0, x1, y1 in boxes:
            tw, th = x1 - x0, y1 - y0
            tiles.append(
                image.crop((x0, y0, x1, y1)).resize((1024, 1024), Image.LANCZOS)
            )
            tile_meta.append(
                {
                    "box": (x0, y0, x1, y1),
                    "size": (tw, th),
                    "seed": randint(0, 2**32 - 1),
                    "mask": _tile_blend_mask(
                        tw,
                        th,
                        self.params.feather,
                        x0 > 0,
                        x1 < width,
                        y0 > 0,
                        y1 < height,
                    ),
                }
            )

        return TilingOutputs(
            tiles=tiles,
            tile_meta=tile_meta,
            original=np.asarray(image, dtype=np.float32),
            accum=np.zeros((height, width, 3), dtype=np.float32),
            weights=np.zeros((height, width, 1), dtype=np.float32),
        )


# ── stitch helper (called by hrfix node after generation) ─────────────────────


def stitch(output: TilingOutputs) -> Image.Image:
    blended = np.where(
        output.weights > 0,
        output.accum / np.maximum(output.weights, 1e-6),
        output.original,
    )
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
