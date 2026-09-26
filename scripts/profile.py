from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {name: ROOT / "profiles" / name for name in ("cuda", "rocm")}


def _profile_path(name: str) -> Path:
    path = PROFILES[name]
    if not (path / "pyproject.toml").is_file():
        raise SystemExit(f"Profile is missing pyproject.toml: {path}")
    return path


def _run_uv(profile: str, arguments: list[str], *, with_pytest: bool = False) -> int:
    command = ["uv", "run", "--project", str(_profile_path(profile))]
    if with_pytest:
        command.extend(["--with", "pytest==8.4.2"])
    command.extend(arguments)
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", str(ROOT))
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run commands in a GPU profile.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("sync", "test", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("profile", choices=sorted(PROFILES))
        if command == "run":
            subparser.add_argument("arguments", nargs=argparse.REMAINDER)
        elif command == "test":
            subparser.add_argument("arguments", nargs=argparse.REMAINDER)

    arguments = parser.parse_args()
    profile = arguments.profile
    if arguments.command == "sync":
        return subprocess.run(
            ["uv", "sync", "--project", str(_profile_path(profile))],
            cwd=ROOT,
        ).returncode
    if arguments.command == "test":
        test_arguments = arguments.arguments or ["tests"]
        return _run_uv(profile, ["pytest", *test_arguments], with_pytest=True)

    run_arguments = arguments.arguments or [
        "uvicorn",
        "main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    return _run_uv(profile, run_arguments)


if __name__ == "__main__":
    sys.exit(main())
