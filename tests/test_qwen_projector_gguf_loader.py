from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np
import torch
from torch.nn import functional as F

from src.nodes.qwen_node import (
    QwenToSdxlGgufProjector,
    load_projector_from_gguf,
    project_sdxl_qwen_embedding,
)


def _write_projector_gguf(
    gguf_path: Path,
    metadata: dict[str, str | int | bool],
    tensors: dict[str, torch.Tensor],
) -> None:
    writer = gguf.GGUFWriter(str(gguf_path), "projector")
    for key, value in metadata.items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            raise TypeError(f"Unsupported GGUF metadata value for {key}: {value!r}")

    for name, tensor in tensors.items():
        writer.add_tensor(
            name,
            np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32),
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class QwenProjectorGgufLoaderTests(unittest.TestCase):
    def test_loads_state_dict_sdxl_projector(self) -> None:
        torch.manual_seed(1234)
        reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=5,
            prompt_dim=6,
            pooled_dim=7,
            hidden_dim=8,
            prompt_token_dim=3,
            trunk_depth=2,
            residual_trunk=True,
            prompt_head_hidden_dim=9,
            pooled_head_hidden_dim=10,
            prompt_head_second_activation=True,
        ).eval()
        qwen_embedding = torch.randn(4)

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "projector_state_dict.gguf"
            _write_projector_gguf(
                gguf_path,
                {
                    "projector.target_family": "sdxl",
                    "projector.schema_version": 3,
                    "projector.qwen_dim": 4,
                    "projector.hidden_dim": 8,
                    "projector.prompt_seq_len": 5,
                    "projector.prompt_dim": 6,
                    "projector.pooled_dim": 7,
                    "projector.prompt_token_dim": 3,
                    "projector.trunk_depth": 2,
                    "projector.residual_trunk": True,
                    "projector.prompt_head_hidden_dim": 9,
                    "projector.pooled_head_hidden_dim": 10,
                },
                {
                    f"state_dict.{name}": tensor
                    for name, tensor in reference.state_dict().items()
                },
            )

            projector, metadata = load_projector_from_gguf(gguf_path, device="cpu")

        self.assertEqual(metadata["target_family"], "sdxl")
        self.assertEqual(metadata["schema_version"], 3)
        self.assertEqual(metadata["trunk_depth"], 2)
        self.assertTrue(metadata["residual_trunk"])
        self.assertEqual(metadata["prompt_head_hidden_dim"], 9)
        self.assertEqual(metadata["pooled_head_hidden_dim"], 10)
        self.assertFalse(metadata["use_input_layernorm"])
        self.assertFalse(metadata["normalize_input"])
        self.assertTrue(metadata["prompt_head_second_activation"])

        with torch.no_grad():
            expected_prompt, expected_pooled = reference(qwen_embedding)
            actual_prompt, actual_pooled = projector(qwen_embedding)
            scaled_prompt, scaled_pooled = projector(qwen_embedding * 10)

        self.assertTrue(torch.allclose(actual_prompt, expected_prompt))
        self.assertTrue(torch.allclose(actual_pooled, expected_pooled))
        self.assertFalse(torch.allclose(actual_prompt, scaled_prompt))
        self.assertFalse(torch.allclose(actual_pooled, scaled_pooled))

    def test_loads_state_dict_sdxl_projector_with_sequential_layernorm_layout(
        self,
    ) -> None:
        torch.manual_seed(5678)
        reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=5,
            prompt_dim=6,
            pooled_dim=7,
            hidden_dim=8,
            prompt_token_dim=3,
            trunk_depth=1,
            residual_trunk=True,
            prompt_head_hidden_dim=9,
            pooled_head_hidden_dim=10,
            use_input_layernorm=True,
            prompt_head_second_activation=True,
        ).eval()
        qwen_embedding = torch.randn(4)
        legacy_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in reference.state_dict().items():
            if name.startswith("trunk_layers.0.layernorm."):
                legacy_state_dict[
                    name.replace("trunk_layers.0.layernorm.", "input_projection.1.")
                ] = tensor
                continue
            if name.startswith("input_projection.2."):
                legacy_state_dict[
                    name.replace("input_projection.2.", "input_projection.3.")
                ] = tensor
                continue
            legacy_state_dict[name] = tensor

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "projector_state_dict_layernorm.gguf"
            _write_projector_gguf(
                gguf_path,
                {
                    "projector.target_family": "sdxl",
                    "projector.schema_version": 3,
                    "projector.qwen_dim": 4,
                    "projector.hidden_dim": 8,
                    "projector.prompt_seq_len": 5,
                    "projector.prompt_dim": 6,
                    "projector.pooled_dim": 7,
                    "projector.prompt_token_dim": 3,
                    "projector.trunk_depth": 1,
                    "projector.residual_trunk": True,
                    "projector.prompt_head_hidden_dim": 9,
                    "projector.pooled_head_hidden_dim": 10,
                },
                {
                    f"state_dict.{name}": tensor
                    for name, tensor in legacy_state_dict.items()
                },
            )

            projector, metadata = load_projector_from_gguf(gguf_path, device="cpu")

        self.assertEqual(metadata["target_family"], "sdxl")
        self.assertEqual(metadata["schema_version"], 3)
        self.assertTrue(metadata["use_input_layernorm"])

        with torch.no_grad():
            expected_prompt, expected_pooled = reference(qwen_embedding)
            actual_prompt, actual_pooled = projector(qwen_embedding)

        self.assertTrue(torch.allclose(actual_prompt, expected_prompt))
        self.assertTrue(torch.allclose(actual_pooled, expected_pooled))

    def test_loads_state_dict_sdxl_projector_with_input_layernorm_layout(self) -> None:
        torch.manual_seed(6789)
        reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=5,
            prompt_dim=6,
            pooled_dim=7,
            hidden_dim=8,
            prompt_token_dim=3,
            trunk_depth=1,
            residual_trunk=True,
            prompt_head_hidden_dim=9,
            pooled_head_hidden_dim=10,
            use_input_layernorm=True,
            prompt_head_second_activation=True,
        ).eval()
        qwen_embedding = torch.randn(4)
        legacy_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in reference.state_dict().items():
            if name.startswith("trunk_layers.0.layernorm."):
                legacy_state_dict[
                    name.replace("trunk_layers.0.layernorm.", "input_layernorm.")
                ] = tensor
                continue
            legacy_state_dict[name] = tensor

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "projector_state_dict_input_layernorm.gguf"
            _write_projector_gguf(
                gguf_path,
                {
                    "projector.target_family": "sdxl",
                    "projector.schema_version": 3,
                    "projector.qwen_dim": 4,
                    "projector.hidden_dim": 8,
                    "projector.prompt_seq_len": 5,
                    "projector.prompt_dim": 6,
                    "projector.pooled_dim": 7,
                    "projector.prompt_token_dim": 3,
                    "projector.trunk_depth": 1,
                    "projector.residual_trunk": True,
                    "projector.prompt_head_hidden_dim": 9,
                    "projector.pooled_head_hidden_dim": 10,
                },
                {
                    f"state_dict.{name}": tensor
                    for name, tensor in legacy_state_dict.items()
                },
            )

            projector, metadata = load_projector_from_gguf(gguf_path, device="cpu")

        self.assertEqual(metadata["target_family"], "sdxl")
        self.assertEqual(metadata["schema_version"], 3)
        self.assertTrue(metadata["use_input_layernorm"])

        with torch.no_grad():
            expected_prompt, expected_pooled = reference(qwen_embedding)
            actual_prompt, actual_pooled = projector(qwen_embedding)

        self.assertTrue(torch.allclose(actual_prompt, expected_prompt))
        self.assertTrue(torch.allclose(actual_pooled, expected_pooled))

    def test_loads_legacy_sdxl_projector_layout(self) -> None:
        torch.manual_seed(4321)
        trunk_w1 = torch.randn(8, 4)
        trunk_b1 = torch.randn(8)
        trunk_w2 = torch.randn(8, 8)
        trunk_b2 = torch.randn(8)
        prompt_seed_w = torch.randn(15, 8)
        prompt_seed_b = torch.randn(15)
        prompt_proj_w = torch.randn(6, 3)
        prompt_proj_b = torch.randn(6)
        pooled_w = torch.randn(7, 8)
        pooled_b = torch.randn(7)
        qwen_embedding = torch.randn(4)

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "projector_legacy.gguf"
            _write_projector_gguf(
                gguf_path,
                {
                    "projector.target_family": "sdxl",
                    "projector.qwen_dim": 4,
                    "projector.hidden_dim": 8,
                    "projector.prompt_seq_len": 5,
                    "projector.prompt_dim": 6,
                    "projector.pooled_dim": 7,
                    "projector.prompt_token_dim": 3,
                },
                {
                    "trunk_w1": trunk_w1,
                    "trunk_b1": trunk_b1,
                    "trunk_w2": trunk_w2,
                    "trunk_b2": trunk_b2,
                    "prompt_seed_w": prompt_seed_w,
                    "prompt_seed_b": prompt_seed_b,
                    "prompt_proj_w": prompt_proj_w,
                    "prompt_proj_b": prompt_proj_b,
                    "pooled_w": pooled_w,
                    "pooled_b": pooled_b,
                },
            )

            projector, metadata = load_projector_from_gguf(gguf_path, device="cpu")

        self.assertEqual(metadata["target_family"], "sdxl")
        self.assertEqual(metadata["schema_version"], 0)
        self.assertEqual(metadata["trunk_depth"], 0)
        self.assertFalse(metadata["residual_trunk"])
        self.assertEqual(metadata["prompt_head_hidden_dim"], 6)
        self.assertEqual(metadata["pooled_head_hidden_dim"], 8)
        self.assertFalse(metadata["use_input_layernorm"])
        self.assertFalse(metadata["normalize_input"])
        self.assertFalse(metadata["prompt_head_second_activation"])

        with torch.no_grad():
            hidden = F.linear(qwen_embedding, trunk_w1, trunk_b1)
            hidden = F.gelu(hidden)
            hidden = F.linear(hidden, trunk_w2, trunk_b2)
            prompt_seed = F.linear(hidden, prompt_seed_w, prompt_seed_b).view(5, 3)
            expected_prompt = F.linear(
                F.gelu(prompt_seed), prompt_proj_w, prompt_proj_b
            )
            expected_pooled = F.linear(F.gelu(hidden), pooled_w, pooled_b)
            actual_prompt, actual_pooled = projector(qwen_embedding)

        self.assertTrue(torch.allclose(actual_prompt, expected_prompt))
        self.assertTrue(torch.allclose(actual_pooled, expected_pooled))

    def test_loader_can_force_input_layernorm_for_artifacts_without_it(self) -> None:
        torch.manual_seed(2468)
        base_reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=5,
            prompt_dim=6,
            pooled_dim=7,
            hidden_dim=8,
            prompt_token_dim=3,
            trunk_depth=1,
            residual_trunk=True,
            prompt_head_hidden_dim=9,
            pooled_head_hidden_dim=10,
            use_input_layernorm=False,
            prompt_head_second_activation=True,
        ).eval()
        forced_layernorm_reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=5,
            prompt_dim=6,
            pooled_dim=7,
            hidden_dim=8,
            prompt_token_dim=3,
            trunk_depth=1,
            residual_trunk=True,
            prompt_head_hidden_dim=9,
            pooled_head_hidden_dim=10,
            use_input_layernorm=True,
            prompt_head_second_activation=True,
        ).eval()
        forced_state_dict = forced_layernorm_reference.state_dict()
        for name, tensor in base_reference.state_dict().items():
            forced_state_dict[name] = tensor
        forced_layernorm_reference.load_state_dict(forced_state_dict)
        qwen_embedding = torch.randn(4)

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "projector_state_dict_force_layernorm.gguf"
            _write_projector_gguf(
                gguf_path,
                {
                    "projector.target_family": "sdxl",
                    "projector.schema_version": 3,
                    "projector.qwen_dim": 4,
                    "projector.hidden_dim": 8,
                    "projector.prompt_seq_len": 5,
                    "projector.prompt_dim": 6,
                    "projector.pooled_dim": 7,
                    "projector.prompt_token_dim": 3,
                    "projector.trunk_depth": 1,
                    "projector.residual_trunk": True,
                    "projector.prompt_head_hidden_dim": 9,
                    "projector.pooled_head_hidden_dim": 10,
                },
                {
                    f"state_dict.{name}": tensor
                    for name, tensor in base_reference.state_dict().items()
                },
            )

            projector, metadata = load_projector_from_gguf(
                gguf_path,
                device="cpu",
                use_input_layernorm=True,
            )

        self.assertTrue(metadata["use_input_layernorm"])

        with torch.no_grad():
            expected_prompt, expected_pooled = forced_layernorm_reference(
                qwen_embedding
            )
            actual_prompt, actual_pooled = projector(qwen_embedding)

        self.assertTrue(torch.allclose(actual_prompt, expected_prompt))
        self.assertTrue(torch.allclose(actual_pooled, expected_pooled))

    def test_loads_state_dict_sdxl_projector_with_output_calibrators(self) -> None:
        torch.manual_seed(1357)
        reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=5,
            prompt_dim=6,
            pooled_dim=7,
            hidden_dim=8,
            prompt_token_dim=3,
            trunk_depth=2,
            residual_trunk=True,
            prompt_head_hidden_dim=9,
            pooled_head_hidden_dim=10,
            prompt_head_second_activation=True,
            use_output_calibrator=True,
            output_mode="calibrated",
        ).eval()
        with torch.no_grad():
            reference.prompt_output_calibrator.gain.copy_(
                torch.linspace(0.5, 1.5, steps=6)
            )
            reference.prompt_output_calibrator.bias.copy_(
                torch.linspace(-0.3, 0.3, steps=6)
            )
            reference.pooled_output_calibrator.gain.copy_(
                torch.linspace(0.75, 1.25, steps=7)
            )
            reference.pooled_output_calibrator.bias.copy_(
                torch.linspace(-0.2, 0.2, steps=7)
            )
        qwen_embedding = torch.randn(4)

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "projector_state_dict_output_calibrator.gguf"
            _write_projector_gguf(
                gguf_path,
                {
                    "projector.target_family": "sdxl",
                    "projector.schema_version": 3,
                    "projector.qwen_dim": 4,
                    "projector.hidden_dim": 8,
                    "projector.prompt_seq_len": 5,
                    "projector.prompt_dim": 6,
                    "projector.pooled_dim": 7,
                    "projector.prompt_token_dim": 3,
                    "projector.trunk_depth": 2,
                    "projector.residual_trunk": True,
                    "projector.prompt_head_hidden_dim": 9,
                    "projector.pooled_head_hidden_dim": 10,
                    "projector.use_output_calibrator": True,
                    "projector.output_mode": "calibrated",
                },
                {
                    f"state_dict.{name}": tensor
                    for name, tensor in reference.state_dict().items()
                },
            )

            projector, metadata = load_projector_from_gguf(gguf_path, device="cpu")

        identity_calibrator_reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=5,
            prompt_dim=6,
            pooled_dim=7,
            hidden_dim=8,
            prompt_token_dim=3,
            trunk_depth=2,
            residual_trunk=True,
            prompt_head_hidden_dim=9,
            pooled_head_hidden_dim=10,
            prompt_head_second_activation=True,
            use_output_calibrator=True,
            output_mode="calibrated",
        ).eval()
        identity_state_dict = {
            name: tensor.clone() for name, tensor in reference.state_dict().items()
        }
        identity_state_dict["prompt_output_calibrator.gain"] = torch.ones(6)
        identity_state_dict["prompt_output_calibrator.bias"] = torch.zeros(6)
        identity_state_dict["pooled_output_calibrator.gain"] = torch.ones(7)
        identity_state_dict["pooled_output_calibrator.bias"] = torch.zeros(7)
        identity_calibrator_reference.load_state_dict(identity_state_dict)

        self.assertEqual(metadata["target_family"], "sdxl")
        self.assertEqual(metadata["schema_version"], 3)
        self.assertTrue(metadata["use_output_calibrator"])
        self.assertEqual(metadata["output_mode"], "calibrated")

        with torch.no_grad():
            expected_prompt, expected_pooled = reference(qwen_embedding)
            actual_prompt, actual_pooled = projector(qwen_embedding)
            identity_prompt, identity_pooled = identity_calibrator_reference(
                qwen_embedding
            )

        self.assertTrue(torch.allclose(actual_prompt, expected_prompt))
        self.assertTrue(torch.allclose(actual_pooled, expected_pooled))
        self.assertFalse(torch.allclose(actual_prompt, identity_prompt))
        self.assertFalse(torch.allclose(actual_pooled, identity_pooled))

    def test_loads_embedding_standardization_payload_for_schema5_projectors(
        self,
    ) -> None:
        torch.manual_seed(9753)
        reference = QwenToSdxlGgufProjector(
            qwen_dim=4,
            prompt_seq_len=3,
            prompt_dim=2,
            pooled_dim=3,
            hidden_dim=5,
            prompt_token_dim=2,
            trunk_depth=1,
            residual_trunk=True,
            prompt_head_hidden_dim=4,
            pooled_head_hidden_dim=6,
            prompt_head_second_activation=True,
            output_mode="plain",
        ).eval()
        raw_qwen_embedding = torch.tensor([1.5, -0.5, 0.75, 2.0], dtype=torch.float32)
        qwen_mean = torch.tensor([0.5, -1.0, 0.25, 1.5], dtype=torch.float32)
        qwen_std = torch.tensor([2.0, 4.0, 0.5, 0.25], dtype=torch.float32)
        prompt_mean = torch.tensor([10.0, -3.0], dtype=torch.float32)
        prompt_std = torch.tensor([0.5, 2.0], dtype=torch.float32)
        pooled_mean = torch.tensor([4.0, -5.0, 6.0], dtype=torch.float32)
        pooled_std = torch.tensor([1.5, 0.25, 3.0], dtype=torch.float32)

        standardized_qwen_embedding = (raw_qwen_embedding - qwen_mean) / qwen_std
        with torch.no_grad():
            expected_prompt_std, expected_pooled_std = reference(
                standardized_qwen_embedding
            )
        expected_prompt = expected_prompt_std * prompt_std + prompt_mean
        expected_pooled = expected_pooled_std * pooled_std + pooled_mean

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "projector_state_dict_schema5_standardized.gguf"
            _write_projector_gguf(
                gguf_path,
                {
                    "projector.target_family": "sdxl",
                    "projector.schema_version": 5,
                    "projector.qwen_dim": 4,
                    "projector.hidden_dim": 5,
                    "projector.prompt_seq_len": 3,
                    "projector.prompt_dim": 2,
                    "projector.pooled_dim": 3,
                    "projector.prompt_token_dim": 2,
                    "projector.trunk_depth": 1,
                    "projector.residual_trunk": True,
                    "projector.prompt_head_hidden_dim": 4,
                    "projector.pooled_head_hidden_dim": 6,
                    "projector.uses_embedding_standardization": True,
                    "projector.output_mode": "plain",
                },
                {
                    **{
                        f"state_dict.{name}": tensor
                        for name, tensor in reference.state_dict().items()
                    },
                    "standardization.qwen.mean": qwen_mean,
                    "standardization.qwen.std": qwen_std,
                    "standardization.prompt_embeds.mean": prompt_mean,
                    "standardization.prompt_embeds.std": prompt_std,
                    "standardization.pooled_prompt_embeds.mean": pooled_mean,
                    "standardization.pooled_prompt_embeds.std": pooled_std,
                },
            )

            projector, metadata = load_projector_from_gguf(gguf_path, device="cpu")

        self.assertEqual(metadata["schema_version"], 5)
        self.assertTrue(metadata["uses_embedding_standardization"])
        self.assertTrue(metadata["requires_input_standardization"])
        self.assertTrue(metadata["requires_output_denormalization"])

        with torch.no_grad():
            actual_prompt, actual_pooled = project_sdxl_qwen_embedding(
                projector,
                raw_qwen_embedding,
            )

        self.assertTrue(torch.allclose(actual_prompt, expected_prompt))
        self.assertTrue(torch.allclose(actual_pooled, expected_pooled))


if __name__ == "__main__":
    unittest.main()
