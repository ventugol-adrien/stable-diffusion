from __future__ import annotations

import argparse

from compel import CompelForSDXL

from src.nodes.qwen_node import (
    negative_prompt_cache_path,
    save_sdxl_negative_prompt_embeddings,
)
from src.pipeline import get_pipe
from src.prompt import process_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and cache SDXL negative prompt embeddings for QwenNode.",
    )
    parser.add_argument("--model", default="juggernaut")
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Raw negative prompt. When omitted, the model default negative prompt is used.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional override for the negative embedding cache directory.",
    )
    args = parser.parse_args()

    _, normalized_negative_prompt = process_prompt(
        "",
        args.negative_prompt,
        args.model,
    )
    if not normalized_negative_prompt.strip():
        raise ValueError("negative_prompt must be a non-empty string.")

    pipe = get_pipe(args.model)
    compel_proc = CompelForSDXL(pipe=pipe, device="cuda")
    conditioning = compel_proc("", negative_prompt=normalized_negative_prompt)
    cache_path = save_sdxl_negative_prompt_embeddings(
        model=args.model,
        negative_prompt=normalized_negative_prompt,
        negative_prompt_embeds=conditioning.negative_embeds,
        negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
        cache_dir=args.cache_dir,
    )

    print(f"Saved SDXL negative embeddings to: {cache_path}")
    print(f"Normalized negative prompt: {normalized_negative_prompt}")
    print(
        "Expected cache path for QwenNode lookups: "
        f"{negative_prompt_cache_path(args.model, normalized_negative_prompt, args.cache_dir)}"
    )


if __name__ == "__main__":
    main()
