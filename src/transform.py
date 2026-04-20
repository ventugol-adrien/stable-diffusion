from PIL import Image
from pydantic import BaseModel, Field


class TransformParams(BaseModel):
    dx: int = 0
    dy: int = 0
    z: float = Field(1.0, ge=0.1, le=5.0)
    r: float = Field(0.0, ge=-360.0, le=360.0)


def apply_transforms(
    img: Image.Image, params: TransformParams
) -> tuple[Image.Image, Image.Image]:
    """
    Applies spatial transforms (Scale → Rotate → Displace) to an image.

    Uses RGBA alpha tracking to precisely identify void regions created by
    the transforms, then converts back to the original image mode.

    Returns:
        (transformed_image, void_mask) where void_mask is L-mode with
        white (255) = void, black (0) = content. This matches the diffusers
        inpainting mask convention (white = region to inpaint).
    """
    original_mode = img.mode
    w, h = img.size

    # Convert to RGBA so alpha tracks which pixels are "real" content
    if img.mode == "RGBA":
        work = img.copy()
    elif img.mode == "LA":
        work = img.convert("RGBA")
    else:
        work = img.convert("RGBA")
        # Set full opacity on all original pixels
        work.putalpha(255)

    # --- Scale ---
    if params.z != 1.0:
        new_w = max(1, round(w * params.z))
        new_h = max(1, round(h * params.z))
        scaled = work.resize((new_w, new_h), Image.LANCZOS)

        # Center the scaled image on a transparent canvas of original size
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        canvas.paste(scaled, (paste_x, paste_y))
        work = canvas

    # --- Rotate ---
    if params.r != 0.0:
        work = work.rotate(
            params.r,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0, 0),
        )

    # --- Displace ---
    if params.dx != 0 or params.dy != 0:
        work = work.transform(
            (w, h),
            Image.AFFINE,
            # Inverse matrix: to find source pixel for output (x,y),
            # we look at (x - dx, y - dy) in the source.
            (1, 0, -params.dx, 0, 1, -params.dy),
            resample=Image.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

    # --- Extract void mask from alpha channel ---
    alpha = work.split()[-1]  # A channel
    void_mask = Image.eval(alpha, lambda a: 255 if a == 0 else 0).convert("L")

    # --- Convert back to original mode ---
    if original_mode == "RGBA":
        result = work
    elif original_mode == "LA":
        result = work.convert("LA")
    else:
        result = work.convert(original_mode)

    return result, void_mask
