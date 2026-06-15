import io
import json
import zipfile

import torch
from fastapi.responses import Response
from PIL import Image
from pydantic import Field, ConfigDict

from src.classes import PNGStreamingResponse, ZipStreamingResponse
from src.utils import stream_image, stream_zip
from src.nodes.base_node import BaseNode, BaseNodeModel


class ResponseInputs(BaseNodeModel):
    media_type: str = Field(
        "image/png", description="Media type for single-image response"
    )
    filename: str = Field("image", description="Base filename (without extension)")
    stream: bool = Field(True, description="Whether to stream the response")
    model_config = ConfigDict(extra="allow")


class ResponseNode(BaseNode):
    def __init__(self, inputs: ResponseInputs = ResponseInputs()):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "response"

    @staticmethod
    def _to_pil(image: Image.Image | torch.Tensor) -> Image.Image:
        if isinstance(image, torch.Tensor):
            arr = (
                (image.float().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            )
            return Image.fromarray(arr, mode="RGB")
        return image

    def __call__(
        self, images: list[Image.Image | torch.Tensor], data: dict = {}, *args, **kwargs
    ) -> Response:
        print(
            f"[ResponseNode] received {len(images)} image(s), types: {[type(img).__name__ for img in images]}"
        )
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                print(
                    f"[ResponseNode] image[{i}] is Tensor shape={tuple(img.shape)} dtype={img.dtype}"
                )
            else:
                print(
                    f"[ResponseNode] image[{i}] is {type(img).__name__} size={getattr(img, 'size', '?')}"
                )
        images = [self._to_pil(img) for img in images]
        print(f"[ResponseNode] after _to_pil: {[type(img).__name__ for img in images]}")
        if self.params.stream:
            if len(images) == 1:
                image: Image.Image = images[0]
                print("Streaming response...")
                return PNGStreamingResponse(
                    stream_image(image),
                    headers={
                        "Content-Disposition": f"inline; filename={self.params.filename}.png",
                        **(
                            {
                                "X-Metrics-Latency": str(data["latency"]),
                                "X-Metrics-Throughput": str(data["throughput"]),
                                "X-Metrics-Breakdown": json.dumps(data["breakdown"]),
                            }
                            if "latency" in data
                            else {}
                        ),
                    },
                )
            else:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    if data:
                        zip_file.writestr("metrics.json", json.dumps(data))
                    for i, img in enumerate(images):
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format="PNG")
                        zip_file.writestr(
                            f"{self.params.filename}_{i}.png", img_buffer.getvalue()
                        )
                print("Streaming ZIP response...")
                return ZipStreamingResponse(
                    stream_zip(zip_buffer),
                    headers={
                        "Content-Disposition": f"attachment; filename={self.params.filename}.zip",
                    },
                )
        if len(images) == 1:
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"inline; filename={self.params.filename}.png"
                },
            )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if data:
                zf.writestr("data.json", json.dumps(data))
            for i, img in enumerate(images):
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(f"{self.params.filename}_{i}.png", img_buf.getvalue())

        return Response(
            content=zip_buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={self.params.filename}.zip"
            },
        )
