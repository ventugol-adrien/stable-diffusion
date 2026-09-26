from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / ".local/share/models/sdxl/illustrious.safetensors"
LORAS_DIR = Path.home() / "sd_loras"


def _resolve_lora_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if path.parent == Path("."):
        name = path.name if path.suffix else f"{path.name}.safetensors"
        path = LORAS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"LoRA file not found: {value}")
    return path


def _load_loras(pipe, loras: list[str], scales: list[float]) -> None:
    if not loras:
        return

    adapter_names = []
    for index, value in enumerate(loras):
        path = _resolve_lora_path(value)
        adapter_name = f"cli_{index}_{path.stem}"
        pipe.load_lora_weights(path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
    pipe.set_adapters(adapter_names=adapter_names, adapter_weights=scales)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--lora", action="append", default=[], metavar="NAME_OR_PATH"
    )
    parser.add_argument(
        "--scale", action="append", type=float, default=[], metavar="VALUE"
    )
    parser.add_argument("--output", required=True, type=Path)


def _run(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.cfg < 0:
        raise ValueError("--cfg cannot be negative")
    if args.width < 8 or args.width % 8 or args.height < 8 or args.height % 8:
        raise ValueError("--width and --height must be positive multiples of 8")
    if args.scale and len(args.scale) != len(args.lora):
        raise ValueError("provide one --scale for each --lora")
    if not args.scale:
        args.scale = [0.5] * len(args.lora)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    from src.nodes.text2image import Text2ImageInputs, Text2ImageNode
    from src.pipeline import cleanup_resources, get_pipe

    try:
        cleanup_resources()
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"SDXL checkpoint not found: {MODEL_PATH}")

        pipe = get_pipe(str(MODEL_PATH))
        _load_loras(pipe, args.lora, args.scale)
        node = Text2ImageNode(
            Text2ImageInputs(
                model=str(MODEL_PATH),
                steps=args.steps,
                cfg_scale=args.cfg,
                width=args.width,
                height=args.height,
            )
        )
        images = node(prompt=args.prompt, negative_prompt=args.negative_prompt)["images"]
        images[0].save(args.output)
        print(f"Saved {args.output}")
    finally:
        cleanup_resources()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_image",
        description="Generate an image from a text prompt.",
    )
    parser.epilog = (
        "Canonical order: --steps --cfg --width --height --prompt "
        "--negative-prompt [--lora NAME --scale VALUE ...] --output"
    )
    _add_arguments(parser)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())