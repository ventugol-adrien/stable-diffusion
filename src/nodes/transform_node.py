import math
from typing import Literal

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

    def _build_affine(self) -> tuple[float, ...] | None:
        """Return the (a,b,c,d,e,f) affine coefficients, or None if identity."""
        p = self.params
        if p.r == 0.0 and p.dx == 0 and p.dy == 0:
            return None
        cx, cy = p.width / 2, p.height / 2
        angle = math.radians(p.r)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (
            cos_a,
            sin_a,
            cx - (cx + p.dx) * cos_a - (cy + p.dy) * sin_a,
            -sin_a,
            cos_a,
            cy + (cx + p.dx) * sin_a - (cy + p.dy) * cos_a,
        )

    def _apply_fit(
        self, img: Image.Image, mask: Image.Image | None = None
    ) -> tuple[Image.Image, Image.Image, Image.Image | None]:
        """Scale img so its longest edge fits (width, height), center on white canvas.

        Returns (canvas_img, tracker, canvas_mask) where:
          tracker: L image, 255 = original content area, 0 = added fill zone.
          canvas_mask: same convention as input mask, 0 in fill-zone border pixels;
                       None when no mask was supplied.
        """
        w, h = self.params.width, self.params.height
        iw, ih = img.size
        scale = min(w / iw, h / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        ox, oy = (w - nw) // 2, (h - nh) // 2

        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(img.convert("RGB").resize((nw, nh), Image.LANCZOS), (ox, oy))

        tracker = Image.new("L", (w, h), 0)
        tracker.paste(Image.new("L", (nw, nh), 255), (ox, oy))

        if mask is not None:
            canvas_mask = Image.new("L", (w, h), 0)
            canvas_mask.paste(
                mask.convert("L").resize((nw, nh), Image.NEAREST), (ox, oy)
            )
            return canvas, tracker, canvas_mask

        return canvas, tracker, None

    def _transform_white(
        self, img: Image.Image, _fit_tracker: Image.Image | None = None
    ) -> tuple[Image.Image, Image.Image]:
        """Fill new canvas areas with white and produce a fresh fill-zone mask."""
        p = self.params
        w, h = p.width, p.height

        if _fit_tracker is not None:
            # img is already canvas-sized from _apply_fit; tracker marks fill zones
            tracker = _fit_tracker
        else:
            img = img.convert("RGB").resize((w, h), Image.LANCZOS)
            tracker = Image.new("L", (w, h), 255)

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

        affine = self._build_affine()
        if affine is not None:
            img = img.transform(
                (w, h),
                Image.AFFINE,
                affine,
                resample=Image.BICUBIC,
                fillcolor=(255, 255, 255),
            )
            tracker = tracker.transform(
                (w, h),
                Image.AFFINE,
                affine,
                resample=Image.NEAREST,
                fillcolor=0,
            )

        # Invert tracker: 255 = fill zone, 0 = original content.
        return img, ImageOps.invert(tracker)

    def _transform_black(
        self, img: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """Transform only the mask (same geometry as white mode). Image is returned as-is."""
        p = self.params
        w, h = p.width, p.height

        mask = mask.convert("L").resize((w, h), Image.NEAREST)

        if p.z != 1.0:
            zw = max(1, round(w * p.z))
            zh = max(1, round(h * p.z))
            mask = mask.resize((zw, zh), Image.NEAREST)
            canvas_mask = Image.new("L", (w, h), 0)
            ox, oy = (w - zw) // 2, (h - zh) // 2
            src_x, src_y = max(0, -ox), max(0, -oy)
            dst_x, dst_y = max(0, ox), max(0, oy)
            pw = min(zw - src_x, w - dst_x)
            ph = min(zh - src_y, h - dst_y)
            crop = (src_x, src_y, src_x + pw, src_y + ph)
            canvas_mask.paste(mask.crop(crop), (dst_x, dst_y))
            mask = canvas_mask

        affine = self._build_affine()
        if affine is not None:
            mask = mask.transform(
                (w, h),
                Image.AFFINE,
                affine,
                resample=Image.NEAREST,
                fillcolor=0,
            )

        return img, mask

    def __call__(
        self,
        images: list[Image.Image] | None = None,
        masks: list[Image.Image] | None = None,
        fill_color: Literal["white", "black"] = "white",
        fit_resize: bool = False,
        *args,
        **kwargs,
    ) -> dict[str, list[Image.Image]]:
        imgs = images if images is not None else self.images
        fit_trackers: list[Image.Image | None] = [None] * len(imgs)

        if fit_resize:
            prefit = [
                self._apply_fit(img, m)
                for img, m in zip(imgs, masks or [None] * len(imgs))
            ]
            imgs = [canvas for canvas, _, _ in prefit]
            fit_trackers = [tracker for _, tracker, _ in prefit]
            masks = [fm for _, _, fm in prefit] if masks is not None else None

        if fill_color == "black":
            if masks is None or len(masks) != len(imgs):
                raise ValueError(
                    "fill_color='black' requires one existing mask per image"
                )
            results = [self._transform_black(img, m) for img, m in zip(imgs, masks)]
        if fill_color == "white":
            results = [
                self._transform_white(img, _fit_tracker=t)
                for img, t in zip(imgs, fit_trackers)
            ]
        return {
            "images": [r[0] for r in results],
            "masks": [r[1] for r in results],
        }
