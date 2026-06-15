from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.llama import LlamaCppSessionConfig, LlamaEmbeddingSession

MODEL_4B_PATH = Path("/home/adrien/my_models/qwen3.5-4b/qwen3.5-4b.gguf")
MODEL_27B_PATH = Path("/home/adrien/my_models/qwen3.5-27b/qwen3.5-27b.gguf")
TEST_PROMPT = "A bright studio portrait with soft lighting"


class _DummyLlama:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding
        self.normalize_calls: list[bool] = []

    def reset(self) -> None:
        pass

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        del text, add_bos, special
        return [1, 2, 3]

    def embed(self, text: str, normalize: bool, truncate: bool):
        del text, truncate
        self.normalize_calls.append(normalize)
        return self.embedding


class LlamaEmbeddingSessionTests(unittest.TestCase):
    def _assert_cpu_embeddings(self, model_path: Path) -> None:
        with LlamaEmbeddingSession(
            LlamaCppSessionConfig(
                model_path=str(model_path),
                n_ctx=512,
                n_batch=128,
                n_gpu_layers=0,
                offload_kqv=False,
                op_offload=False,
                verbose=False,
            )
        ) as session:
            result = session.embed_text(TEST_PROMPT)

        self.assertGreater(len(result.token_ids), 0)
        self.assertEqual(result.embedding.ndim, 1)
        self.assertGreater(result.hidden_size, 0)
        self.assertEqual(result.embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(result.embedding).all())
        self.assertAlmostEqual(float(np.linalg.norm(result.embedding)), 1.0, places=3)

    def test_embed_text_normalizes_by_default(self) -> None:
        session = LlamaEmbeddingSession(LlamaCppSessionConfig(model_path="dummy.gguf"))
        session._llama = _DummyLlama(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

        result = session.embed_text("test prompt")

        self.assertEqual(result.embedding.ndim, 1)
        self.assertEqual(result.embedding.dtype, np.float32)
        self.assertEqual(session._llama.normalize_calls, [False])
        self.assertAlmostEqual(float(np.linalg.norm(result.embedding)), 1.0, places=6)

    def test_embed_text_can_disable_normalization(self) -> None:
        session = LlamaEmbeddingSession(
            LlamaCppSessionConfig(
                model_path="dummy.gguf",
                normalize_embeddings=False,
            )
        )
        session._llama = _DummyLlama(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

        result = session.embed_text("test prompt")

        self.assertEqual(result.embedding.ndim, 1)
        self.assertEqual(session._llama.normalize_calls, [False])
        self.assertTrue(
            np.array_equal(
                result.embedding, np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
            )
        )

    @unittest.skipUnless(
        MODEL_4B_PATH.is_file(), f"Missing GGUF model at {MODEL_4B_PATH}"
    )
    def test_embed_text_returns_mean_pooled_embeddings_on_cpu_for_4b(self) -> None:
        self._assert_cpu_embeddings(MODEL_4B_PATH)

    @unittest.skipUnless(
        MODEL_27B_PATH.is_file(), f"Missing GGUF model at {MODEL_27B_PATH}"
    )
    def test_embed_text_returns_mean_pooled_embeddings_on_cpu_for_27b(self) -> None:
        self._assert_cpu_embeddings(MODEL_27B_PATH)


if __name__ == "__main__":
    unittest.main()
