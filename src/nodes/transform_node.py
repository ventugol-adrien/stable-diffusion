import math

from PIL import Image, ImageOps
from pydantic import Field

from src.nodes.base_node import BaseNode, BaseNodeModel


class TransformInputs(BaseNodeModel):
    width: int = Field(1024)
    height: int = Field(1024)
    dx: int = Field(0)
    dy: int = Field(0)
    z: float = Field(1.0)
    r: float = Field(0.0)


class TransformNode(BaseNode):
    output_key = "images"

    def __init__(self, inputs: TransformInputs = TransformInputs()):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "transform"
        self.images: list[Image.Image] = []

    def _transform(self, img: Image.Image) -> tuple[Image.Image, Image.Image]:
        p = self.params
        w, h = p.width, p.height

        # Step 1: Resize to target dimensions with LANCZOS.
        img = img.convert("RGB").resize((w, h), Image.LANCZOS)

        # Tracker: single-channel, all-white = every pixel is original content.
        # Each step fills newly introduced pixels with 0 (black = fill zone).
        tracker = Image.new("L", (w, h), 255)

        # Step 2: Apply zoom via resize (LANCZOS for image, NEAREST for tracker).
        if p.z != 1.0:
            zw = max(1, round(w * p.z))
            zh = max(1, round(h * p.z))
            img = img.resize((zw, zh), Image.LANCZOS)
            tracker = tracker.resize((zw, zh), Image.NEAREST)

            canvas_img = Image.new("RGB", (w, h), (255, 255, 255))
            canvas_tracker = Image.new("L", (w, h), 0)
            ox, oy = (w - zw) // 2, (h - zh) // 2
            src_x, src_y = max(0, -ox), max(0, -oy)
            dst_x, dst_y = max(0, ox), max(0, oy)
            pw = min(zw - src_x, w - dst_x)
            ph = min(zh - src_y, h - dst_y)
            crop = (src_x, src_y, src_x + pw, src_y + ph)
            canvas_img.paste(img.crop(crop), (dst_x, dst_y))
            canvas_tracker.paste(tracker.crop(crop), (dst_x, dst_y))
            img = canvas_img
            tracker = canvas_tracker

        # Step 3: Apply rotation + translation via affine (BICUBIC — LANCZOS unsupported).
        if p.r != 0.0 or p.dx != 0 or p.dy != 0:
            cx, cy = w / 2, h / 2
            angle = math.radians(p.r)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            a = cos_a
            b = sin_a
            c = cx - (cx + p.dx) * cos_a - (cy + p.dy) * sin_a
            d = -sin_a
            e = cos_a
            f = cy + (cx + p.dx) * sin_a - (cy + p.dy) * cos_a

            img = img.transform(
                (w, h),
                Image.AFFINE,
                (a, b, c, d, e, f),
                resample=Image.BICUBIC,
                fillcolor=(255, 255, 255),
            )
            tracker = tracker.transform(
                (w, h),
                Image.AFFINE,
                (a, b, c, d, e, f),
                resample=Image.NEAREST,
                fillcolor=0,
            )

        # Invert: 255 = fill zone, 0 = original content.
        fill_mask = ImageOps.invert(tracker)
        return img, fill_mask

    def __call__(
        self, images: list[Image.Image] | None = None, *args, **kwargs
    ) -> dict[str, list[Image.Image]]:
        imgs = images if images is not None else self.images
        results = [self._transform(img) for img in imgs]
        return {
            "images": [r[0] for r in results],
            "masks": [r[1] for r in results],
        }
