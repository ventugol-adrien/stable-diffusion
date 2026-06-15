from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.llama import LlamaTokenEmbeddings
from src.nodes.qwen_node import EmbeddingStandardizationStats, QwenInputs, QwenNode


class _DummyEmbeddingSession:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SessionWithPooledEmbedding:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding

    def embed_text(self, text: str) -> LlamaTokenEmbeddings:
        del text
        return LlamaTokenEmbeddings(
            token_ids=[1, 2, 3],
            embedding=self.embedding,
        )


class _RecordingProjector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.seen: torch.Tensor | None = None

    def forward(self, embedding: torch.Tensor):
        self.seen = embedding.detach().clone()
        prompt = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        pooled = torch.arange(4, dtype=torch.float32)
        return prompt, pooled


class _StandardizingProjector(_RecordingProjector):
    def __init__(self) -> None:
        super().__init__()
        self.uses_embedding_standardization = True
        self.embedding_standardization_stats = EmbeddingStandardizationStats(
            qwen_mean=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            qwen_std=torch.tensor([2.0, 2.0, 4.0, 8.0], dtype=torch.float32),
            target_means={
                "prompt_embeds": torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
                "pooled_prompt_embeds": torch.tensor([40.0, 50.0, 60.0, 70.0], dtype=torch.float32),
            },
            target_stds={
                "prompt_embeds": torch.tensor([0.5, 1.5, 2.0], dtype=torch.float32),
                "pooled_prompt_embeds": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            },
        )

    def forward(self, embedding: torch.Tensor):
        self.seen = embedding.detach().clone()
        prompt = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        )
        pooled = torch.tensor([7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
        return prompt, pooled


class QwenNodeRuntimeTests(unittest.TestCase):
    def test_qwen_node_disables_llama_embedding_normalization_by_default(self) -> None:
        node = QwenNode(
            QwenInputs(
                prompt="test prompt",
                negative_prompt="",
                model="juggernaut",
            )
        )
        metadata = {
            "target_family": "sdxl",
            "qwen_dim": 4,
            "prompt_seq_len": 2,
            "prompt_dim": 3,
            "pooled_dim": 4,
        }
        llama_session_ctor = unittest.mock.Mock(
            return_value=_DummyEmbeddingSession(None)
        )

        call_impl = QwenNode.__call__.__wrapped__.__wrapped__

        with patch("src.nodes.qwen_node.torch.cuda.is_available", return_value=False):
            with patch(
                "src.nodes.qwen_node._normalize_prompt_batch",
                return_value=(["test prompt"], [""]),
            ):
                with patch(
                    "src.nodes.qwen_node.LlamaEmbeddingSession", llama_session_ctor
                ):
                    with patch.object(
                        node,
                        "_resolve_qwen_model_path",
                        return_value="/tmp/fake-qwen.gguf",
                    ):
                        with patch.object(
                            node,
                            "_resolve_projector_path",
                            return_value="/tmp/fake-projector.gguf",
                        ):
                            with patch.object(
                                node,
                                "_load_sdxl_projector",
                                return_value=(object(), metadata),
                            ):
                                with patch.object(
                                    QwenNode,
                                    "_load_negative_prompt_batch",
                                    return_value=(
                                        torch.zeros((1, 2, 3), dtype=torch.float32),
                                        torch.zeros((1, 4), dtype=torch.float32),
                                    ),
                                ):
                                    with patch.object(
                                        QwenNode,
                                        "_project_prompt_batch",
                                        return_value=(
                                            torch.ones((1, 2, 3), dtype=torch.float32),
                                            torch.ones((1, 4), dtype=torch.float32),
                                        ),
                                    ):
                                        call_impl(node)

        llama_config = llama_session_ctor.call_args.args[0]
        self.assertFalse(llama_config.normalize_embeddings)

    def test_qwen_node_can_enable_llama_embedding_normalization(self) -> None:
        node = QwenNode(
            QwenInputs(
                prompt="test prompt",
                negative_prompt="",
                model="juggernaut",
                qwen_normalize_embeddings=True,
            )
        )
        metadata = {
            "target_family": "sdxl",
            "qwen_dim": 4,
            "prompt_seq_len": 2,
            "prompt_dim": 3,
            "pooled_dim": 4,
        }
        llama_session_ctor = unittest.mock.Mock(
            return_value=_DummyEmbeddingSession(None)
        )

        call_impl = QwenNode.__call__.__wrapped__.__wrapped__

        with patch("src.nodes.qwen_node.torch.cuda.is_available", return_value=False):
            with patch(
                "src.nodes.qwen_node._normalize_prompt_batch",
                return_value=(["test prompt"], [""]),
            ):
                with patch(
                    "src.nodes.qwen_node.LlamaEmbeddingSession", llama_session_ctor
                ):
                    with patch.object(
                        node,
                        "_resolve_qwen_model_path",
                        return_value="/tmp/fake-qwen.gguf",
                    ):
                        with patch.object(
                            node,
                            "_resolve_projector_path",
                            return_value="/tmp/fake-projector.gguf",
                        ):
                            with patch.object(
                                node,
                                "_load_sdxl_projector",
                                return_value=(object(), metadata),
                            ):
                                with patch.object(
                                    QwenNode,
                                    "_load_negative_prompt_batch",
                                    return_value=(
                                        torch.zeros((1, 2, 3), dtype=torch.float32),
                                        torch.zeros((1, 4), dtype=torch.float32),
                                    ),
                                ):
                                    with patch.object(
                                        QwenNode,
                                        "_project_prompt_batch",
                                        return_value=(
                                            torch.ones((1, 2, 3), dtype=torch.float32),
                                            torch.ones((1, 4), dtype=torch.float32),
                                        ),
                                    ):
                                        call_impl(node)

        llama_config = llama_session_ctor.call_args.args[0]
        self.assertTrue(llama_config.normalize_embeddings)

    def test_project_prompt_batch_uses_pooled_llama_embedding_vector(self) -> None:
        node = QwenNode(
            QwenInputs(
                prompt="test prompt",
                negative_prompt="",
                model="juggernaut",
            )
        )
        session = _SessionWithPooledEmbedding(
            np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        )
        projector = _RecordingProjector()

        prompt_embeds, pooled_prompt_embeds = node._project_prompt_batch(
            session=session,
            prompts=["test prompt"],
            projector=projector,
            metadata={
                "qwen_dim": 4,
                "prompt_seq_len": 2,
                "prompt_dim": 3,
                "pooled_dim": 4,
            },
            output_device=torch.device("cpu"),
            output_dtype=torch.float32,
        )

        self.assertIsNotNone(projector.seen)
        self.assertEqual(projector.seen.ndim, 1)
        self.assertTrue(
            torch.allclose(
                projector.seen,
                torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
            )
        )
        self.assertEqual(tuple(prompt_embeds.shape), (1, 2, 3))
        self.assertEqual(tuple(pooled_prompt_embeds.shape), (1, 4))

    def test_project_prompt_batch_applies_standardization_and_denormalization(
        self,
    ) -> None:
        node = QwenNode(
            QwenInputs(
                prompt="test prompt",
                negative_prompt="",
                model="juggernaut",
            )
        )
        session = _SessionWithPooledEmbedding(
            np.asarray([3.0, 6.0, 11.0, 20.0], dtype=np.float32)
        )
        projector = _StandardizingProjector()

        prompt_embeds, pooled_prompt_embeds = node._project_prompt_batch(
            session=session,
            prompts=["test prompt"],
            projector=projector,
            metadata={
                "qwen_dim": 4,
                "prompt_seq_len": 2,
                "prompt_dim": 3,
                "pooled_dim": 4,
            },
            output_device=torch.device("cpu"),
            output_dtype=torch.float32,
        )

        self.assertIsNotNone(projector.seen)
        self.assertTrue(
            torch.allclose(
                projector.seen,
                torch.tensor([1.0, 2.0, 2.0, 2.0], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                prompt_embeds.squeeze(0),
                torch.tensor(
                    [[10.5, 23.0, 36.0], [12.0, 27.5, 42.0]],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                pooled_prompt_embeds.squeeze(0),
                torch.tensor([47.0, 66.0, 87.0, 110.0], dtype=torch.float32),
            )
        )

    def test_qwen_node_keeps_conditioning_float32_on_cuda_handoff(self) -> None:
        test_case = self
        node = QwenNode(
            QwenInputs(
                prompt="test prompt",
                negative_prompt="",
                model="juggernaut",
            )
        )
        metadata = {
            "target_family": "sdxl",
            "qwen_dim": 4,
            "prompt_seq_len": 2,
            "prompt_dim": 3,
            "pooled_dim": 4,
        }
        observed: dict[str, torch.device | torch.dtype] = {}

        def fake_load_negative_prompt_batch(
            self,
            negative_prompts,
            output_device,
            output_dtype,
            expected_prompt_seq_len,
            expected_prompt_dim,
            expected_pooled_dim,
        ):
            observed["negative_device"] = output_device
            observed["negative_dtype"] = output_dtype
            test_case.assertEqual(negative_prompts, [""])
            test_case.assertEqual(expected_prompt_seq_len, 2)
            test_case.assertEqual(expected_prompt_dim, 3)
            test_case.assertEqual(expected_pooled_dim, 4)
            return (
                torch.zeros((1, 2, 3), dtype=torch.float32),
                torch.zeros((1, 4), dtype=torch.float32),
            )

        def fake_project_prompt_batch(
            self,
            session,
            prompts,
            projector,
            metadata,
            output_device,
            output_dtype,
        ):
            observed["positive_device"] = output_device
            observed["positive_dtype"] = output_dtype
            test_case.assertEqual(prompts, ["test prompt"])
            test_case.assertEqual(metadata["prompt_seq_len"], 2)
            test_case.assertEqual(metadata["prompt_dim"], 3)
            test_case.assertEqual(metadata["pooled_dim"], 4)
            return (
                torch.ones((1, 2, 3), dtype=torch.float32),
                torch.ones((1, 4), dtype=torch.float32),
            )

        call_impl = QwenNode.__call__.__wrapped__.__wrapped__

        with patch("src.nodes.qwen_node.torch.cuda.is_available", return_value=True):
            with patch(
                "src.nodes.qwen_node._normalize_prompt_batch",
                return_value=(["test prompt"], [""]),
            ):
                with patch(
                    "src.nodes.qwen_node.LlamaEmbeddingSession", _DummyEmbeddingSession
                ):
                    with patch.object(
                        node,
                        "_resolve_qwen_model_path",
                        return_value="/tmp/fake-qwen.gguf",
                    ):
                        with patch.object(
                            node,
                            "_resolve_projector_path",
                            return_value="/tmp/fake-projector.gguf",
                        ):
                            with patch.object(
                                node,
                                "_load_sdxl_projector",
                                return_value=(object(), metadata),
                            ):
                                with patch.object(
                                    QwenNode,
                                    "_load_negative_prompt_batch",
                                    fake_load_negative_prompt_batch,
                                ):
                                    with patch.object(
                                        QwenNode,
                                        "_project_prompt_batch",
                                        fake_project_prompt_batch,
                                    ):
                                        embeds = call_impl(node)

        self.assertEqual(observed["negative_device"], torch.device("cuda"))
        self.assertEqual(observed["positive_device"], torch.device("cuda"))
        self.assertEqual(observed["negative_dtype"], torch.float32)
        self.assertEqual(observed["positive_dtype"], torch.float32)
        self.assertEqual(embeds.prompt_embeds.dtype, torch.float32)
        self.assertEqual(embeds.pooled_prompt_embeds.dtype, torch.float32)
        self.assertEqual(embeds.negative_prompt_embeds.dtype, torch.float32)
        self.assertEqual(embeds.negative_pooled_prompt_embeds.dtype, torch.float32)

    def test_qwen_node_can_skip_cached_negative_prompt_loading(self) -> None:
        test_case = self
        node = QwenNode(
            QwenInputs(
                prompt="test prompt",
                negative_prompt="bad anatomy",
                model="juggernaut",
                qwen_use_cached_negative_prompt_embeds=False,
            )
        )
        metadata = {
            "target_family": "sdxl",
            "qwen_dim": 4,
            "prompt_seq_len": 2,
            "prompt_dim": 3,
            "pooled_dim": 4,
        }

        def fail_load_negative_prompt_batch(*args, **kwargs):
            raise AssertionError("cached negative embeddings should not be loaded")

        def fake_project_prompt_batch(
            self,
            session,
            prompts,
            projector,
            metadata,
            output_device,
            output_dtype,
        ):
            test_case.assertEqual(prompts, ["test prompt"])
            return (
                torch.ones((1, 2, 3), dtype=torch.float32),
                torch.ones((1, 4), dtype=torch.float32),
            )

        call_impl = QwenNode.__call__.__wrapped__.__wrapped__

        with patch("src.nodes.qwen_node.torch.cuda.is_available", return_value=False):
            with patch(
                "src.nodes.qwen_node._normalize_prompt_batch",
                return_value=(["test prompt"], ["bad anatomy"]),
            ):
                with patch(
                    "src.nodes.qwen_node.LlamaEmbeddingSession", _DummyEmbeddingSession
                ):
                    with patch.object(
                        node,
                        "_resolve_qwen_model_path",
                        return_value="/tmp/fake-qwen.gguf",
                    ):
                        with patch.object(
                            node,
                            "_resolve_projector_path",
                            return_value="/tmp/fake-projector.gguf",
                        ):
                            with patch.object(
                                node,
                                "_load_sdxl_projector",
                                return_value=(object(), metadata),
                            ):
                                with patch.object(
                                    QwenNode,
                                    "_load_negative_prompt_batch",
                                    side_effect=fail_load_negative_prompt_batch,
                                ):
                                    with patch.object(
                                        QwenNode,
                                        "_project_prompt_batch",
                                        fake_project_prompt_batch,
                                    ):
                                        embeds = call_impl(node)

        self.assertEqual(tuple(embeds.negative_prompt_embeds.shape), (1, 2, 3))
        self.assertEqual(tuple(embeds.negative_pooled_prompt_embeds.shape), (1, 4))
        self.assertEqual(embeds.negative_prompt_embeds.dtype, torch.float32)
        self.assertEqual(embeds.negative_pooled_prompt_embeds.dtype, torch.float32)
        self.assertTrue(torch.count_nonzero(embeds.negative_prompt_embeds) == 0)
        self.assertTrue(torch.count_nonzero(embeds.negative_pooled_prompt_embeds) == 0)

    def test_qwen_node_forwards_use_input_layernorm_to_projector_loader(self) -> None:
        node = QwenNode(
            QwenInputs(
                prompt="test prompt",
                negative_prompt="",
                model="juggernaut",
                use_input_layernorm=True,
            )
        )

        with patch(
            "src.nodes.qwen_node.load_projector_from_gguf",
            return_value=(object(), {"target_family": "sdxl"}),
        ) as loader:
            projector, metadata = node._load_sdxl_projector(
                "/tmp/fake-projector.gguf",
                target_device=torch.device("cpu"),
            )

        self.assertIs(projector, loader.return_value[0])
        self.assertEqual(metadata["target_family"], "sdxl")
        loader.assert_called_once_with(
            Path("/tmp/fake-projector.gguf"),
            device=torch.device("cpu"),
            use_input_layernorm=True,
        )


if __name__ == "__main__":
    unittest.main()
