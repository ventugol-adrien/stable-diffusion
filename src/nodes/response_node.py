import io
import json
import zipfile

from fastapi.responses import Response
from PIL import Image
from pydantic import Field, ConfigDict

from src.nodes.base_node import BaseNode, BaseNodeModel


class ResponseInputs(BaseNodeModel):
    media_type: str = Field(
        "image/png", description="Media type for single-image response"
    )
    filename: str = Field("image", description="Base filename (without extension)")
    model_config = ConfigDict(extra="allow")


class ResponseNode(BaseNode):
    def __init__(self, inputs: ResponseInputs = ResponseInputs()):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "response"

    def __call__(
        self, images: list[Image.Image], data: dict = {}, *args, **kwargs
    ) -> Response:
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
