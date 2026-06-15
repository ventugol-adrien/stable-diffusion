from __future__ import annotations

import types
import unittest
from unittest import mock

from PIL import Image

from src.nodes.spatial_assets_node import (
    DEFAULT_EDGE_MODEL,
    SpatialAssetsInputs,
    SpatialAssetsNode,
    _clear_spatial_assets_models,
)


class _DepthPipe:
    def __call__(self, image: Image.Image) -> dict[str, Image.Image]:
        return {"depth": Image.new("L", (2, 2), 128)}


class _EdgePipe:
    def __call__(self, image: Image.Image, **kwargs) -> Image.Image:
        return Image.new("L", (10, 10), 255)


class SpatialAssetsNodeTests(unittest.TestCase):
    def test_call_uses_images_argument_and_normalizes_asset_sizes(self) -> None:
        node = SpatialAssetsNode.__new__(SpatialAssetsNode)
        node.node_type = "spatial_assets"
        node.images = []
        node.params = types.SimpleNamespace()
        node.depth_pipe = _DepthPipe()
        node.edge_pipe = _EdgePipe()

        image = Image.new("RGB", (4, 6), (10, 20, 30))

        result = node(images=[image])

        self.assertEqual(len(result["images"]), 2)
        self.assertEqual(result["images"][0].size, image.size)
        self.assertEqual(result["images"][1].size, image.size)

    def test_constructor_honors_edge_model(self) -> None:
        with mock.patch(
            "src.nodes.spatial_assets_node.pipeline", return_value=_DepthPipe()
        ):
            with mock.patch(
                "src.nodes.spatial_assets_node.PidiNetDetector.from_pretrained",
                return_value=_EdgePipe(),
            ) as edge_loader:
                SpatialAssetsNode(
                    SpatialAssetsInputs(
                        depth_model="depth-model",
                        edge_model="custom-edge-model",
                    )
                )

        edge_loader.assert_called_once()
        self.assertEqual(edge_loader.call_args.args[0], "custom-edge-model")

    def test_constructor_falls_back_when_custom_edge_model_fails(self) -> None:
        with mock.patch(
            "src.nodes.spatial_assets_node.pipeline", return_value=_DepthPipe()
        ):
            with mock.patch(
                "src.nodes.spatial_assets_node.PidiNetDetector.from_pretrained",
                side_effect=[RuntimeError("missing repo"), _EdgePipe()],
            ) as edge_loader:
                node = SpatialAssetsNode(
                    SpatialAssetsInputs(
                        edge_model="missing-private-edge-model",
                    )
                )

        self.assertIsNotNone(node.edge_pipe)
        self.assertEqual(
            edge_loader.call_args_list[0].args[0], "missing-private-edge-model"
        )
        self.assertEqual(edge_loader.call_args_list[1].args[0], DEFAULT_EDGE_MODEL)

    def test_cleanup_clears_loaded_cpu_models(self) -> None:
        with mock.patch(
            "src.nodes.spatial_assets_node.pipeline", return_value=_DepthPipe()
        ):
            with mock.patch(
                "src.nodes.spatial_assets_node.PidiNetDetector.from_pretrained",
                return_value=_EdgePipe(),
            ):
                node = SpatialAssetsNode(SpatialAssetsInputs())

        self.assertIsNotNone(node.depth_pipe)
        self.assertIsNotNone(node.edge_pipe)

        _clear_spatial_assets_models()

        self.assertIsNone(node.depth_pipe)
        self.assertIsNone(node.edge_pipe)


if __name__ == "__main__":
    unittest.main()
