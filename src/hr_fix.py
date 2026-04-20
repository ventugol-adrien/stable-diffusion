import gc
import math
from random import randint

import numpy as np
import torch
from PIL import Image


def _axis_starts(length: int, tile_size: int, min_overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = tile_size - min_overlap
    tile_count = max(2, math.ceil((length - min_overlap) / stride))
    max_start = length - tile_size
    starts = [round(i * max_start / (tile_count - 1)) for i in range(tile_count)]

    deduped = []
    for start in starts:
        if not deduped or start != deduped[-1]:
            deduped.append(start)

    if deduped[-1] != max_start:
        deduped.append(max_start)

    return deduped


def _axis_weights(
    length: int, feather: int, fade_in: bool, fade_out: bool
) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if feather <= 0 or length <= 1:
        return weights

    feather = min(feather, max(1, length // 2))
    t = np.linspace(0.0, 1.0, feather, dtype=np.float32)
    cosine_ramp = 0.5 * (1.0 - np.cos(np.pi * t))
    if fade_in:
        weights[:feather] = np.minimum(weights[:feather], cosine_ramp)
    if fade_out:
        weights[-feather:] = np.minimum(weights[-feather:], cosine_ramp[::-1])
    return weights


def _tile_weight_mask(
    tile_width: int,
    tile_height: int,
    feather: int,
    blend_left: bool,
    blend_right: bool,
    blend_top: bool,
    blend_bottom: bool,
) -> np.ndarray:
    x_weights = _axis_weights(tile_width, feather, blend_left, blend_right)
    y_weights = _axis_weights(tile_height, feather, blend_top, blend_bottom)
    return y_weights[:, None] * x_weights[None, :]


def _iter_tile_boxes(
    width: int,
    height: int,
    tile_size: int,
    min_overlap: int,
) -> list[tuple[int, int, int, int]]:
    x_starts = _axis_starts(width, tile_size, min_overlap)
    y_starts = _axis_starts(height, tile_size, min_overlap)
    return [
        (x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height))
        for y0 in y_starts
        for x0 in x_starts
    ]


def tiled_refine_image(
    image: Image.Image,
    pipe,
    *,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    pag_scale: float | None = None,
    num_tiles: int = 4,
    min_overlap: int = 256,
    feather: int = 256,
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    tiles_per_axis = max(1, round(math.sqrt(num_tiles)))
    tile_size = min_overlap + math.ceil(
        (max(width, height) - min_overlap) / tiles_per_axis
    )
    boxes = _iter_tile_boxes(width, height, tile_size, min_overlap)
    print(f"🧩 Tiled 4x refinement: {len(boxes)} tile(s) covering {width}x{height}")

    original = np.asarray(image, dtype=np.float32)
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)

    _BATCH = 5
    base_call_kwargs = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    if pag_scale is not None:
        base_call_kwargs["pag_scale"] = pag_scale

    total_batches = math.ceil(len(boxes) / _BATCH)
    for batch_idx in range(total_batches):
        batch_boxes = boxes[batch_idx * _BATCH : (batch_idx + 1) * _BATCH]
        first_tile = batch_idx * _BATCH + 1
        print(
            f"   Refining batch {batch_idx + 1}/{total_batches}"
            f" (tiles {first_tile}\u2013{first_tile + len(batch_boxes) - 1} of {len(boxes)})"
        )

        for x0, y0, x1, y1 in batch_boxes:
            tile_w, tile_h = x1 - x0, y1 - y0
            tile = image.crop((x0, y0, x1, y1)).resize((1024, 1024), Image.LANCZOS)
            generator = torch.Generator(device="cuda").manual_seed(
                randint(0, 2**32 - 1)
            )

            refined_tile = (
                pipe(
                    **base_call_kwargs,
                    image=tile,
                    generator=generator,
                )
                .images[0]
                .convert("RGB")
                .resize((tile_w, tile_h), Image.LANCZOS)
            )

            refined_np = np.asarray(refined_tile, dtype=np.float32)
            mask = _tile_weight_mask(
                tile_width=tile_w,
                tile_height=tile_h,
                feather=feather,
                blend_left=x0 > 0,
                blend_right=x1 < width,
                blend_top=y0 > 0,
                blend_bottom=y1 < height,
            )[..., None]

            accum[y0:y1, x0:x1] += refined_np * mask
            weights[y0:y1, x0:x1] += mask
            del tile, refined_tile, refined_np, mask

        gc.collect()
        torch.cuda.empty_cache()

    refined = np.where(weights > 0, accum / np.maximum(weights, 1e-6), original)
    return Image.fromarray(np.clip(refined, 0, 255).astype(np.uint8))
