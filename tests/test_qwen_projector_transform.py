from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.llama import LlamaCppSessionConfig, LlamaEmbeddingSession
from src.nodes.qwen_node import load_projector_from_gguf

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
MODEL_27B_PATH = Path("/home/adrien/my_models/qwen3.5-27b/qwen3.5-27b.gguf")
TEST_PROMPT = "A bright studio portrait with soft lighting"
PROJECTOR_ENV_NAME = "QWEN_2_SDXL_PROJECTOR_PATH"
DEFAULT_SDXL_MODEL = os.environ.get("DEFAULT_MODEL", "juggernaut")
SDXL_MODEL_CACHE_PATH = REPO_ROOT / "caches" / "models" / DEFAULT_SDXL_MODEL
SDXL_SEQ_LEN = 77
SDXL_HIDDEN_SIZE = 2048
SDXL_POOLED_SIZE = 1280


def _read_env_value(name: str) -> str | None:
    raw_value = os.getenv(name)
    if raw_value:
        return raw_value

    if not ENV_PATH.is_file():
        return None

    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == name:
            return value

    return None


PROJECTOR_PATH_RAW = _read_env_value(PROJECTOR_ENV_NAME)
PROJECTOR_PATH = Path(PROJECTOR_PATH_RAW).expanduser() if PROJECTOR_PATH_RAW else None


def _tensor_scale_stats(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    return {
        "mean_abs": float(flat.abs().mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "max_abs": float(flat.abs().max().item()),
    }


def _scale_to_match(target_rms: float, source_rms: float) -> float:
    return target_rms / max(source_rms, 1e-12)


def _flattened_cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().to(dtype=torch.float32, device="cpu").reshape(1, -1)
    right_flat = right.detach().to(dtype=torch.float32, device="cpu").reshape(1, -1)
    return float(F.cosine_similarity(left_flat, right_flat).item())


def _encode_prompt_with_sdxl_dual_encoders(
    prompt: str,
    model_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path / "tokenizer",
        local_files_only=True,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_path / "tokenizer_2",
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path / "text_encoder",
        local_files_only=True,
    ).eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path / "text_encoder_2",
        local_files_only=True,
    ).eval()

    try:
        with torch.inference_mode():
            tokenized_prompt = tokenizer(
                [prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_prompt_2 = tokenizer_2(
                [prompt],
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            encoder_output = text_encoder(
                tokenized_prompt.input_ids,
                output_hidden_states=True,
            )
            encoder_output_2 = text_encoder_2(
                tokenized_prompt_2.input_ids,
                output_hidden_states=True,
            )

        prompt_embeds = torch.concat(
            [
                encoder_output.hidden_states[-2],
                encoder_output_2.hidden_states[-2],
            ],
            dim=-1,
        ).squeeze(0)
        pooled_prompt_embeds = encoder_output_2[0].squeeze(0)
        return prompt_embeds.to(dtype=torch.float32), pooled_prompt_embeds.to(
            dtype=torch.float32
        )
    finally:
        del text_encoder
        del text_encoder_2


class QwenProjectorTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("LLAMA_CPP_LIB_PATH", None)
        if PROJECTOR_PATH_RAW:
            os.environ[PROJECTOR_ENV_NAME] = PROJECTOR_PATH_RAW

    @unittest.skipUnless(
        MODEL_27B_PATH.is_file(), f"Missing GGUF model at {MODEL_27B_PATH}"
    )
    @unittest.skipUnless(
        bool(PROJECTOR_PATH and PROJECTOR_PATH.is_file()),
        f"Missing projector checkpoint at {PROJECTOR_PATH}",
    )
    def test_env_projector_transforms_cpu_qwen_embedding_for_27b(self) -> None:
        projector, metadata = load_projector_from_gguf(
            str(PROJECTOR_PATH), device="cpu"
        )
        self.assertFalse(bool(metadata.get("normalize_input", False)))

        with LlamaEmbeddingSession(
            LlamaCppSessionConfig(
                model_path=str(MODEL_27B_PATH),
                n_ctx=512,
                n_batch=128,
                n_gpu_layers=0,
                offload_kqv=False,
                op_offload=False,
                verbose=False,
            )
        ) as session:
            qwen_embeddings = session.embed_text(TEST_PROMPT)

        pooled_qwen_embedding = torch.from_numpy(
            np.asarray(qwen_embeddings.embedding, dtype=np.float32)
        )
        self.assertEqual(int(pooled_qwen_embedding.shape[0]), metadata["qwen_dim"])

        with torch.no_grad():
            transformed = projector(pooled_qwen_embedding)

        if metadata["target_family"] == "sdxl":
            prompt_embeds, pooled_prompt_embeds = transformed
            self.assertEqual(
                tuple(prompt_embeds.shape),
                (metadata["prompt_seq_len"], metadata["prompt_dim"]),
            )
            self.assertEqual(
                tuple(pooled_prompt_embeds.shape), (metadata["pooled_dim"],)
            )
            self.assertTrue(torch.isfinite(prompt_embeds).all().item())
            self.assertTrue(torch.isfinite(pooled_prompt_embeds).all().item())
        else:
            self.assertEqual(metadata["target_family"], "sd")
            self.assertEqual(tuple(transformed.shape), (metadata["sd_dim"],))
            self.assertTrue(torch.isfinite(transformed).all().item())

    @unittest.skipUnless(
        MODEL_27B_PATH.is_file(), f"Missing GGUF model at {MODEL_27B_PATH}"
    )
    @unittest.skipUnless(
        bool(PROJECTOR_PATH and PROJECTOR_PATH.is_file()),
        f"Missing projector checkpoint at {PROJECTOR_PATH}",
    )
    def test_env_projector_matches_sdxl_embed_contract_when_target_is_sdxl(
        self,
    ) -> None:
        projector, metadata = load_projector_from_gguf(
            str(PROJECTOR_PATH), device="cpu"
        )
        if metadata["target_family"] != "sdxl":
            self.skipTest(
                f"Projector target_family is {metadata['target_family']}, not sdxl."
            )

        self.assertEqual(metadata["prompt_seq_len"], SDXL_SEQ_LEN)
        self.assertEqual(metadata["prompt_dim"], SDXL_HIDDEN_SIZE)
        self.assertEqual(metadata["pooled_dim"], SDXL_POOLED_SIZE)

        with LlamaEmbeddingSession(
            LlamaCppSessionConfig(
                model_path=str(MODEL_27B_PATH),
                n_ctx=512,
                n_batch=128,
                n_gpu_layers=0,
                offload_kqv=False,
                op_offload=False,
                verbose=False,
            )
        ) as session:
            qwen_embeddings = session.embed_text(TEST_PROMPT)

        pooled_qwen_embedding = torch.from_numpy(
            np.asarray(qwen_embeddings.embedding, dtype=np.float32)
        )

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = projector(pooled_qwen_embedding)

        self.assertEqual(tuple(prompt_embeds.shape), (SDXL_SEQ_LEN, SDXL_HIDDEN_SIZE))
        self.assertEqual(tuple(pooled_prompt_embeds.shape), (SDXL_POOLED_SIZE,))
        self.assertEqual(prompt_embeds.dtype, torch.float32)
        self.assertEqual(pooled_prompt_embeds.dtype, torch.float32)
        self.assertTrue(torch.isfinite(prompt_embeds).all().item())
        self.assertTrue(torch.isfinite(pooled_prompt_embeds).all().item())

    @unittest.skipUnless(
        MODEL_27B_PATH.is_file(), f"Missing GGUF model at {MODEL_27B_PATH}"
    )
    @unittest.skipUnless(
        bool(PROJECTOR_PATH and PROJECTOR_PATH.is_file()),
        f"Missing projector checkpoint at {PROJECTOR_PATH}",
    )
    def test_env_projector_sdxl_outputs_remain_finite_when_cast_to_fp16(self) -> None:
        projector, metadata = load_projector_from_gguf(
            str(PROJECTOR_PATH), device="cpu"
        )
        if metadata["target_family"] != "sdxl":
            self.skipTest(
                f"Projector target_family is {metadata['target_family']}, not sdxl."
            )

        with LlamaEmbeddingSession(
            LlamaCppSessionConfig(
                model_path=str(MODEL_27B_PATH),
                n_ctx=512,
                n_batch=128,
                n_gpu_layers=0,
                offload_kqv=False,
                op_offload=False,
                verbose=False,
            )
        ) as session:
            qwen_embeddings = session.embed_text(TEST_PROMPT)

        pooled_qwen_embedding = torch.from_numpy(
            np.asarray(qwen_embeddings.embedding, dtype=np.float32)
        )

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = projector(pooled_qwen_embedding)

        self.assertTrue(
            torch.isfinite(prompt_embeds.to(dtype=torch.float16)).all().item()
        )
        self.assertTrue(
            torch.isfinite(pooled_prompt_embeds.to(dtype=torch.float16)).all().item()
        )

    @unittest.skipUnless(
        MODEL_27B_PATH.is_file(), f"Missing GGUF model at {MODEL_27B_PATH}"
    )
    @unittest.skipUnless(
        bool(PROJECTOR_PATH and PROJECTOR_PATH.is_file()),
        f"Missing projector checkpoint at {PROJECTOR_PATH}",
    )
    @unittest.skipUnless(
        (SDXL_MODEL_CACHE_PATH / "model_index.json").is_file(),
        f"Missing cached SDXL model at {SDXL_MODEL_CACHE_PATH}",
    )
    def test_env_projector_reports_scale_gap_against_sdxl_dual_encoders(self) -> None:
        projector, metadata = load_projector_from_gguf(
            str(PROJECTOR_PATH), device="cpu"
        )
        if metadata["target_family"] != "sdxl":
            self.skipTest(
                f"Projector target_family is {metadata['target_family']}, not sdxl."
            )

        with LlamaEmbeddingSession(
            LlamaCppSessionConfig(
                model_path=str(MODEL_27B_PATH),
                n_ctx=512,
                n_batch=128,
                n_gpu_layers=0,
                offload_kqv=False,
                op_offload=False,
                verbose=False,
            )
        ) as session:
            qwen_embeddings = session.embed_text(TEST_PROMPT)

        pooled_qwen_embedding = torch.from_numpy(
            np.asarray(qwen_embeddings.embedding, dtype=np.float32)
        )

        with torch.no_grad():
            projector_prompt_embeds, projector_pooled_prompt_embeds = projector(
                pooled_qwen_embedding
            )

        encoder_prompt_embeds, encoder_pooled_prompt_embeds = (
            _encode_prompt_with_sdxl_dual_encoders(
                TEST_PROMPT,
                SDXL_MODEL_CACHE_PATH,
            )
        )
        projector_prompt_embeds = projector_prompt_embeds.to(dtype=torch.float32)
        projector_pooled_prompt_embeds = projector_pooled_prompt_embeds.to(
            dtype=torch.float32
        )

        self.assertEqual(
            tuple(projector_prompt_embeds.shape), tuple(encoder_prompt_embeds.shape)
        )
        self.assertEqual(
            tuple(projector_pooled_prompt_embeds.shape),
            tuple(encoder_pooled_prompt_embeds.shape),
        )
        self.assertTrue(torch.isfinite(projector_prompt_embeds).all().item())
        self.assertTrue(torch.isfinite(projector_pooled_prompt_embeds).all().item())
        self.assertTrue(torch.isfinite(encoder_prompt_embeds).all().item())
        self.assertTrue(torch.isfinite(encoder_pooled_prompt_embeds).all().item())

        projector_prompt_stats = _tensor_scale_stats(projector_prompt_embeds)
        projector_pooled_stats = _tensor_scale_stats(projector_pooled_prompt_embeds)
        encoder_prompt_stats = _tensor_scale_stats(encoder_prompt_embeds)
        encoder_pooled_stats = _tensor_scale_stats(encoder_pooled_prompt_embeds)

        comparison = {
            "projector_path": str(PROJECTOR_PATH),
            "sdxl_model": DEFAULT_SDXL_MODEL,
            "projector_uses_trunk_layernorm": bool(
                metadata.get("use_input_layernorm", False)
            ),
            "prompt_projector": projector_prompt_stats,
            "prompt_sdxl": encoder_prompt_stats,
            "prompt_rms_scale_to_match_sdxl": _scale_to_match(
                encoder_prompt_stats["rms"],
                projector_prompt_stats["rms"],
            ),
            "prompt_max_abs_scale_to_match_sdxl": _scale_to_match(
                encoder_prompt_stats["max_abs"],
                projector_prompt_stats["max_abs"],
            ),
            "prompt_flat_cosine_similarity": _flattened_cosine_similarity(
                projector_prompt_embeds,
                encoder_prompt_embeds,
            ),
            "pooled_projector": projector_pooled_stats,
            "pooled_sdxl": encoder_pooled_stats,
            "pooled_rms_scale_to_match_sdxl": _scale_to_match(
                encoder_pooled_stats["rms"],
                projector_pooled_stats["rms"],
            ),
            "pooled_max_abs_scale_to_match_sdxl": _scale_to_match(
                encoder_pooled_stats["max_abs"],
                projector_pooled_stats["max_abs"],
            ),
            "pooled_cosine_similarity": _flattened_cosine_similarity(
                projector_pooled_prompt_embeds,
                encoder_pooled_prompt_embeds,
            ),
        }

        scalar_values = [
            comparison["prompt_rms_scale_to_match_sdxl"],
            comparison["prompt_max_abs_scale_to_match_sdxl"],
            comparison["prompt_flat_cosine_similarity"],
            comparison["pooled_rms_scale_to_match_sdxl"],
            comparison["pooled_max_abs_scale_to_match_sdxl"],
            comparison["pooled_cosine_similarity"],
        ]
        self.assertTrue(
            all(math.isfinite(value) for value in scalar_values),
            msg=json.dumps(comparison, indent=2, sort_keys=True),
        )
        self.assertGreater(projector_prompt_stats["rms"], 0.0)
        self.assertGreater(projector_pooled_stats["rms"], 0.0)
        self.assertGreater(encoder_prompt_stats["rms"], 0.0)
        self.assertGreater(encoder_pooled_stats["rms"], 0.0)

        print("Projector vs SDXL encoder comparison:")
        print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
