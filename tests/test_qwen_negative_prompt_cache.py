from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import torch

from src.nodes.qwen_node import (
    QwenInputs,
    QwenNode,
    load_sdxl_negative_prompt_embeddings,
    save_sdxl_negative_prompt_embeddings,
)


class _DummyLlamaSession:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


class QwenNegativePromptCacheTests(unittest.TestCase):
    def test_negative_cache_round_trip(self) -> None:
        prompt_embeds = torch.arange(77 * 2048, dtype=torch.float16).reshape(77, 2048)
        pooled_embeds = torch.arange(1280, dtype=torch.float16)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = save_sdxl_negative_prompt_embeddings(
                model="juggernaut",
                negative_prompt="bad anatomy",
                negative_prompt_embeds=prompt_embeds,
                negative_pooled_prompt_embeds=pooled_embeds,
                cache_dir=tmpdir,
            )

            self.assertTrue(cache_path.is_file())

            loaded_prompt_embeds, loaded_pooled_embeds = (
                load_sdxl_negative_prompt_embeddings(
                    model="juggernaut",
                    negative_prompt="bad anatomy",
                    target_device="cpu",
                    target_dtype=torch.float32,
                    cache_dir=tmpdir,
                )
            )

        self.assertEqual(tuple(loaded_prompt_embeds.shape), (77, 2048))
        self.assertEqual(tuple(loaded_pooled_embeds.shape), (1280,))
        self.assertEqual(loaded_prompt_embeds.dtype, torch.float32)
        self.assertEqual(loaded_pooled_embeds.dtype, torch.float32)
        self.assertTrue(torch.equal(loaded_prompt_embeds, prompt_embeds.float()))
        self.assertTrue(torch.equal(loaded_pooled_embeds, pooled_embeds.float()))

    def test_qwen_node_avoids_runtime_pipeline_for_cached_negatives(self) -> None:
        node = QwenNode(
            QwenInputs(
                prompt="A beach in Nice, France",
                negative_prompt="bad anatomy",
                model="juggernaut",
                qwen_n_batch=512,
            )
        )

        llama_session_ctor = mock.Mock(return_value=_DummyLlamaSession())

        with (
            mock.patch(
                "src.nodes.qwen_node.torch.cuda.is_available", return_value=False
            ),
            mock.patch.object(
                node, "_resolve_qwen_model_path", return_value="dummy.gguf"
            ),
            mock.patch.object(
                node, "_resolve_projector_path", return_value="dummy.gguf"
            ),
            mock.patch.object(
                node,
                "_load_sdxl_projector",
                return_value=(
                    mock.Mock(),
                    {
                        "target_family": "sdxl",
                        "qwen_dim": 3584,
                        "prompt_seq_len": 77,
                        "prompt_dim": 2048,
                        "pooled_dim": 1280,
                    },
                ),
            ),
            mock.patch.object(
                node,
                "_load_negative_prompt_batch",
                return_value=(
                    torch.zeros((1, 77, 2048), dtype=torch.float32),
                    torch.zeros((1, 1280), dtype=torch.float32),
                ),
            ),
            mock.patch.object(
                node,
                "_project_prompt_batch",
                return_value=(
                    torch.zeros((1, 77, 2048), dtype=torch.float32),
                    torch.zeros((1, 1280), dtype=torch.float32),
                ),
            ),
            mock.patch("src.nodes.qwen_node.LlamaEmbeddingSession", llama_session_ctor),
        ):
            outputs = node()

        llama_config = llama_session_ctor.call_args.args[0]

        self.assertEqual(tuple(outputs.prompt_embeds.shape), (1, 77, 2048))
        self.assertEqual(tuple(outputs.pooled_prompt_embeds.shape), (1, 1280))
        self.assertEqual(tuple(outputs.negative_prompt_embeds.shape), (1, 77, 2048))
        self.assertEqual(tuple(outputs.negative_pooled_prompt_embeds.shape), (1, 1280))
        self.assertEqual(llama_config.n_batch, 128)
        self.assertFalse(llama_config.offload_kqv)
        self.assertFalse(llama_config.op_offload)


if __name__ == "__main__":
    unittest.main()
