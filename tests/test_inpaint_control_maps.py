from __future__ import annotations

import asyncio
import io
import types
import unittest
from unittest import mock

from fastapi import UploadFile
from PIL import Image

import main
from src.models import OutpaintRequest
from src.nodes.outpainting_node import _apply_mask_to_control_map


class _FormData:
    def __init__(self, items):
        self._items = items

    def multi_items(self):
        return self._items


class _Request:
    def __init__(self, items):
        self._items = items

    async def form(self):
        return _FormData(self._items)


class InpaintControlMapTests(unittest.TestCase):
    @staticmethod
    def _upload_image(color: tuple[int, int, int], filename: str) -> UploadFile:
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buf, format="PNG")
        buf.seek(0)
        return UploadFile(buf, filename=filename)

    def test_outpaint_request_parses_depth_and_edge_maps(self) -> None:
        depth_upload = UploadFile(io.BytesIO(b"depth"), filename="depth.png")
        edge_upload = UploadFile(io.BytesIO(b"edge"), filename="edge.png")
        request = asyncio.run(
            OutpaintRequest.as_form(
                _Request(
                    [
                        ("user_input", "repair this"),
                        ("depth_map", depth_upload),
                        ("depth_map_scale", "0.7"),
                        ("edges_map", edge_upload),
                        ("edges_map_scale", "0.3"),
                    ]
                )
            )
        )

        self.assertIs(request.depth_map, depth_upload)
        self.assertEqual(request.depth_map_scale, 0.7)
        self.assertIs(request.edges_map, edge_upload)
        self.assertEqual(request.edges_map_scale, 0.3)

    def test_outpaint_request_leaves_omitted_control_map_scales_unset(self) -> None:
        depth_upload = UploadFile(io.BytesIO(b"depth"), filename="depth.png")
        edge_upload = UploadFile(io.BytesIO(b"edge"), filename="edge.png")
        request = asyncio.run(
            OutpaintRequest.as_form(
                _Request(
                    [
                        ("user_input", "repair this"),
                        ("depth_map", depth_upload),
                        ("edges_map", edge_upload),
                    ]
                )
            )
        )

        self.assertIsNone(request.depth_map_scale)
        self.assertIsNone(request.edges_map_scale)

    def test_control_map_is_kept_only_inside_mask(self) -> None:
        control = Image.new("RGB", (2, 1), (200, 100, 50))
        mask = Image.new("L", (2, 1), 0)
        mask.putpixel((1, 0), 255)

        result = _apply_mask_to_control_map(control, mask, (2, 1))

        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((1, 0)), (200, 100, 50))

    def test_inpaint_workflow_transforms_control_maps_before_handoff(self) -> None:
        transformed_depth = Image.new("RGB", (2, 2), (10, 20, 30))
        transformed_edges = Image.new("RGB", (2, 2), (40, 50, 60))
        transform_outputs = [
            {"images": [Image.new("RGB", (2, 2))], "masks": [Image.new("L", (2, 2))]},
            {"images": [transformed_depth], "masks": [Image.new("L", (2, 2))]},
            {"images": [transformed_edges], "masks": [Image.new("L", (2, 2))]},
        ]
        transform_instance = mock.Mock(side_effect=transform_outputs)
        outpaint_instance = mock.Mock(
            return_value={"images": [Image.new("RGB", (1, 1))]}
        )
        response_instance = mock.Mock(return_value="response")
        request = types.SimpleNamespace(
            user_input="repair this",
            negative_input="",
            model="juggernaut",
            transform_z=None,
            transform_dx=None,
            transform_dy=None,
            transform_r=None,
            steps=1,
            strength=1.0,
            reference=self._upload_image((1, 2, 3), "reference.png"),
            mask=self._upload_image((255, 255, 255), "mask.png"),
            depth_map=self._upload_image((4, 5, 6), "depth.png"),
            depth_map_scale=0.7,
            edges_map=self._upload_image((7, 8, 9), "edges.png"),
            edges_map_scale=0.3,
        )

        with mock.patch("main.cleanup_resources"):
            with mock.patch("main.pause_llm", new=mock.AsyncMock()):
                with (
                    mock.patch(
                        "main.CompelNode", return_value=mock.Mock(return_value={})
                    ),
                    mock.patch("main.TransformNode", return_value=transform_instance),
                    mock.patch("main.OutpaintingNode", return_value=outpaint_instance),
                    mock.patch("main.ResponseNode", return_value=response_instance),
                ):
                    result = asyncio.run(main.execute_inpaint_workflow(request))

        self.assertEqual(result, "response")
        self.assertEqual(transform_instance.call_count, 3)
        outpaint_kwargs = outpaint_instance.call_args.kwargs
        self.assertIs(outpaint_kwargs["depthmap"], transformed_depth)
        self.assertEqual(outpaint_kwargs["depthmap_scale"], 0.7)
        self.assertIs(outpaint_kwargs["edgesmap"], transformed_edges)
        self.assertEqual(outpaint_kwargs["edgesmap_scale"], 0.3)


if __name__ == "__main__":
    unittest.main()
