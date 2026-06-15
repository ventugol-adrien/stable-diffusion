from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llama import LlamaCppSessionConfig, LlamaEmbeddingSession
from src.nodes.qwen_node import load_projector_from_gguf, project_sdxl_qwen_embedding

DEFAULT_QWEN_MODEL_PATH = Path(
    os.environ.get(
        "QWEN_GGUF_PATH",
        "/home/adrien/my_models/qwen3.5-27b/qwen3.5-27b.gguf",
    )
)
DEFAULT_SDXL_MODEL = os.environ.get("DEFAULT_MODEL", "juggernaut")
DEFAULT_SDXL_MODEL_PATH = REPO_ROOT / "caches" / "models" / DEFAULT_SDXL_MODEL
DEFAULT_PROMPT = "A bright studio portrait with soft lighting"
NORM_MODES = ("none", "layernorm", "post")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one or more Qwen->SDXL projectors against the cached SDXL dual "
            "text encoders under multiple normalization modes."
        )
    )
    parser.add_argument(
        "--projector",
        action="append",
        required=True,
        help="Path to a GGUF projector artifact. Repeat for multiple projectors.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to evaluate. Repeat to compare across multiple prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional newline-delimited prompt file.",
    )
    parser.add_argument(
        "--norm-mode",
        action="append",
        choices=NORM_MODES,
        default=None,
        help=(
            "Normalization mode to evaluate. Repeat to compare multiple modes. "
            "Defaults to all modes: none, layernorm, post."
        ),
    )
    parser.add_argument(
        "--qwen-model-path",
        type=Path,
        default=DEFAULT_QWEN_MODEL_PATH,
        help="Path to the Qwen GGUF model used to generate pooled embeddings.",
    )
    parser.add_argument(
        "--sdxl-model-path",
        type=Path,
        default=DEFAULT_SDXL_MODEL_PATH,
        help="Path to the cached diffusers SDXL model directory.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=512,
        help="llama.cpp context size used for prompt embedding extraction.",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=128,
        help="llama.cpp batch size used for prompt embedding extraction.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON output without the summary table.",
    )
    return parser.parse_args()


def _load_prompts(prompt_args: list[str] | None, prompt_file: Path | None) -> list[str]:
    prompts: list[str] = []
    if prompt_args:
        prompts.extend(prompt_args)
    if prompt_file is not None:
        prompts.extend(
            line.strip()
            for line in prompt_file.read_text().splitlines()
            if line.strip()
        )
    if not prompts:
        prompts.append(DEFAULT_PROMPT)
    return prompts


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


def _normalize_projector_outputs(
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    norm_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if norm_mode != "post":
        return prompt_embeds, pooled_prompt_embeds

    prompt_embeds = F.layer_norm(
        prompt_embeds,
        normalized_shape=(prompt_embeds.shape[-1],),
    )
    pooled_prompt_embeds = F.layer_norm(
        pooled_prompt_embeds,
        normalized_shape=(pooled_prompt_embeds.shape[-1],),
    )
    return prompt_embeds, pooled_prompt_embeds


def _projector_layernorm_override(norm_mode: str) -> bool:
    return norm_mode == "layernorm"


def _aggregate_numeric_dicts(items: Iterable[dict[str, float]]) -> dict[str, float]:
    item_list = list(items)
    if not item_list:
        return {}
    keys = item_list[0].keys()
    return {
        key: float(sum(item[key] for item in item_list) / len(item_list))
        for key in keys
    }


def _validate_environment(qwen_model_path: Path, sdxl_model_path: Path) -> None:
    if not qwen_model_path.is_file():
        raise FileNotFoundError(f"Missing GGUF model at {qwen_model_path}")
    if not (sdxl_model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"Missing cached SDXL model at {sdxl_model_path}")


def _compare_projector(
    projector_path: Path,
    norm_mode: str,
    prompts: list[str],
    qwen_model_path: Path,
    sdxl_model_path: Path,
    n_ctx: int,
    n_batch: int,
) -> dict[str, object]:
    projector, metadata = load_projector_from_gguf(
        projector_path,
        device="cpu",
        use_input_layernorm=_projector_layernorm_override(norm_mode),
    )
    if metadata["target_family"] != "sdxl":
        raise RuntimeError(
            f"Projector {projector_path} targets {metadata['target_family']}, not sdxl."
        )

    prompt_results: list[dict[str, object]] = []

    with LlamaEmbeddingSession(
        LlamaCppSessionConfig(
            model_path=str(qwen_model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=0,
            offload_kqv=False,
            op_offload=False,
            ignore_env_llama_cpp_lib_path=True,
            verbose=False,
        )
    ) as session:
        for prompt in prompts:
            qwen_embeddings = session.embed_text(prompt)
            pooled_qwen_embedding = torch.from_numpy(
                np.asarray(qwen_embeddings.embedding, dtype=np.float32)
            )

            with torch.inference_mode():
                projector_prompt_embeds, projector_pooled_prompt_embeds = (
                    project_sdxl_qwen_embedding(
                        projector,
                        pooled_qwen_embedding.to(
                            device=next(projector.parameters()).device,
                            dtype=torch.float32,
                        ),
                    )
                )

            projector_prompt_embeds = projector_prompt_embeds.to(dtype=torch.float32)
            projector_pooled_prompt_embeds = projector_pooled_prompt_embeds.to(
                dtype=torch.float32
            )
            projector_prompt_embeds, projector_pooled_prompt_embeds = (
                _normalize_projector_outputs(
                    projector_prompt_embeds,
                    projector_pooled_prompt_embeds,
                    norm_mode,
                )
            )

            encoder_prompt_embeds, encoder_pooled_prompt_embeds = (
                _encode_prompt_with_sdxl_dual_encoders(prompt, sdxl_model_path)
            )

            projector_prompt_stats = _tensor_scale_stats(projector_prompt_embeds)
            projector_pooled_stats = _tensor_scale_stats(projector_pooled_prompt_embeds)
            encoder_prompt_stats = _tensor_scale_stats(encoder_prompt_embeds)
            encoder_pooled_stats = _tensor_scale_stats(encoder_pooled_prompt_embeds)

            prompt_results.append(
                {
                    "prompt": prompt,
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
            )

    summary = {
        "prompt_projector": _aggregate_numeric_dicts(
            item["prompt_projector"] for item in prompt_results
        ),
        "prompt_sdxl": _aggregate_numeric_dicts(
            item["prompt_sdxl"] for item in prompt_results
        ),
        "prompt_rms_scale_to_match_sdxl": float(
            sum(item["prompt_rms_scale_to_match_sdxl"] for item in prompt_results)
            / len(prompt_results)
        ),
        "prompt_max_abs_scale_to_match_sdxl": float(
            sum(item["prompt_max_abs_scale_to_match_sdxl"] for item in prompt_results)
            / len(prompt_results)
        ),
        "prompt_flat_cosine_similarity": float(
            sum(item["prompt_flat_cosine_similarity"] for item in prompt_results)
            / len(prompt_results)
        ),
        "pooled_projector": _aggregate_numeric_dicts(
            item["pooled_projector"] for item in prompt_results
        ),
        "pooled_sdxl": _aggregate_numeric_dicts(
            item["pooled_sdxl"] for item in prompt_results
        ),
        "pooled_rms_scale_to_match_sdxl": float(
            sum(item["pooled_rms_scale_to_match_sdxl"] for item in prompt_results)
            / len(prompt_results)
        ),
        "pooled_max_abs_scale_to_match_sdxl": float(
            sum(item["pooled_max_abs_scale_to_match_sdxl"] for item in prompt_results)
            / len(prompt_results)
        ),
        "pooled_cosine_similarity": float(
            sum(item["pooled_cosine_similarity"] for item in prompt_results)
            / len(prompt_results)
        ),
    }

    scalar_values = [
        summary["prompt_rms_scale_to_match_sdxl"],
        summary["prompt_max_abs_scale_to_match_sdxl"],
        summary["prompt_flat_cosine_similarity"],
        summary["pooled_rms_scale_to_match_sdxl"],
        summary["pooled_max_abs_scale_to_match_sdxl"],
        summary["pooled_cosine_similarity"],
    ]
    if not all(math.isfinite(value) for value in scalar_values):
        raise RuntimeError(
            f"Non-finite comparison metrics for {projector_path} in mode {norm_mode}."
        )

    return {
        "projector_path": str(projector_path),
        "norm_mode": norm_mode,
        "prompt_count": len(prompts),
        "sdxl_model_path": str(sdxl_model_path),
        "qwen_model_path": str(qwen_model_path),
        "projector_uses_trunk_layernorm": bool(
            metadata.get("use_input_layernorm", False)
        ),
        "metadata": metadata,
        "summary": summary,
        "per_prompt": prompt_results,
    }


def _print_summary_table(results: list[dict[str, object]]) -> None:
    headers = [
        "projector",
        "mode",
        "trunk_ln",
        "p_cos",
        "pool_cos",
        "p_rms_scale",
        "pool_rms_scale",
        "p_max_scale",
        "pool_max_scale",
    ]
    rows = [headers]
    for result in results:
        summary = result["summary"]
        rows.append(
            [
                Path(result["projector_path"]).name,
                result["norm_mode"],
                "yes" if result["projector_uses_trunk_layernorm"] else "no",
                f"{summary['prompt_flat_cosine_similarity']:.3f}",
                f"{summary['pooled_cosine_similarity']:.3f}",
                f"{summary['prompt_rms_scale_to_match_sdxl']:.3f}",
                f"{summary['pooled_rms_scale_to_match_sdxl']:.3f}",
                f"{summary['prompt_max_abs_scale_to_match_sdxl']:.3f}",
                f"{summary['pooled_max_abs_scale_to_match_sdxl']:.3f}",
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    for index, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))


def main() -> None:
    args = _parse_args()
    prompts = _load_prompts(args.prompt, args.prompt_file)
    norm_modes = args.norm_mode or list(NORM_MODES)
    qwen_model_path = args.qwen_model_path.expanduser().resolve()
    sdxl_model_path = args.sdxl_model_path.expanduser().resolve()
    projector_paths = [Path(path).expanduser().resolve() for path in args.projector]

    _validate_environment(qwen_model_path, sdxl_model_path)
    for projector_path in projector_paths:
        if not projector_path.is_file():
            raise FileNotFoundError(f"Missing projector checkpoint at {projector_path}")

    results = [
        _compare_projector(
            projector_path=projector_path,
            norm_mode=norm_mode,
            prompts=prompts,
            qwen_model_path=qwen_model_path,
            sdxl_model_path=sdxl_model_path,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
        )
        for projector_path in projector_paths
        for norm_mode in norm_modes
    ]

    if not args.json_only:
        _print_summary_table(results)
        print()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
