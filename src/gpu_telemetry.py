from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

_WARNED_BACKENDS: set[str] = set()
_JSONL_LOCK = threading.Lock()


def telemetry_enabled() -> bool:
    value = os.environ.get("SD_GPU_TELEMETRY") or os.environ.get("SD_PERF_TELEMETRY")
    return str(value or "0").strip().lower() in {"1", "true", "yes", "on"}


def telemetry_interval_seconds() -> float:
    raw = os.environ.get("SD_GPU_TELEMETRY_INTERVAL")
    if raw is None:
        raw_ms = os.environ.get("SD_PERF_SAMPLE_INTERVAL_MS")
        if raw_ms is not None:
            try:
                return max(float(raw_ms) / 1000.0, 0.05)
            except ValueError:
                return 0.5
        return 0.5
    try:
        return max(float(raw), 0.05)
    except ValueError:
        return 0.5


def telemetry_jsonl_path() -> Path:
    configured = os.environ.get("SD_GPU_TELEMETRY_JSONL") or os.environ.get(
        "SD_PERF_TELEMETRY_JSONL"
    )
    if configured:
        return Path(configured).expanduser()
    run_id = os.environ.get("SD_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
    return Path(os.environ.get("SD_LOG_DIR", "logs")) / f"perf_{run_id}.jsonl"


def telemetry_backend_name() -> str:
    if not telemetry_enabled():
        return "disabled"
    for command in ("amd-smi", "rocm-smi"):
        try:
            result = subprocess.run(
                [command, "--help"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return command
        except Exception:
            pass
    return "torch-only"


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_BACKENDS:
        return
    _WARNED_BACKENDS.add(key)
    print(message)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _flatten_json(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_json(item, child_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_json(item, child_prefix))
    else:
        numeric = _numeric(value)
        if numeric is not None:
            flattened[prefix.lower()] = numeric
    return flattened


def _first_matching(flattened: dict[str, float], *needles: str) -> float | None:
    lowered_needles = tuple(needle.lower() for needle in needles)
    for key, value in flattened.items():
        if all(needle in key for needle in lowered_needles):
            return value
    return None


def _collect_torch_sample() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        total = float(torch.cuda.get_device_properties(0).total_memory)
        allocated = float(torch.cuda.memory_allocated())
        reserved = float(torch.cuda.memory_reserved())
        max_allocated = float(torch.cuda.max_memory_allocated())
        max_reserved = float(torch.cuda.max_memory_reserved())
        return {
            "torch_allocated_gb": allocated / 1024**3,
            "torch_reserved_gb": reserved / 1024**3,
            "torch_max_allocated_gb": max_allocated / 1024**3,
            "torch_max_reserved_gb": max_reserved / 1024**3,
            "torch_total_vram_gb": total / 1024**3,
            "torch_reserved_percent": (reserved / total * 100.0) if total > 0 else 0.0,
        }
    except Exception:
        return {}


def _collect_rocm_smi_sample() -> dict[str, float]:
    try:
        result = subprocess.run(
            [
                "rocm-smi",
                "--showuse",
                "--showmemuse",
                "--showtemp",
                "--showpower",
                "--showclocks",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except FileNotFoundError:
        _warn_once(
            "rocm-smi-missing",
            "[perf] rocm-smi not found; GPU telemetry will use PyTorch memory only.",
        )
        return {}
    except Exception as exc:
        _warn_once("rocm-smi-error", f"[perf] rocm-smi sampling failed: {exc}")
        return {}

    if result.returncode != 0 or not result.stdout.strip():
        _warn_once(
            "rocm-smi-nonzero",
            "[perf] rocm-smi returned no JSON; GPU telemetry will use PyTorch memory only.",
        )
        return {}

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        _warn_once(
            "rocm-smi-json",
            "[perf] rocm-smi returned malformed JSON; GPU telemetry will use PyTorch memory only.",
        )
        return {}

    flattened = _flatten_json(payload)
    sample: dict[str, float] = {}
    mappings = {
        "gpu_util_percent": (("gpu", "use"), ("gpu", "busy"), ("use", "%")),
        "vram_used_percent": (
            ("vram", "%"),
            ("vram", "use"),
            ("memory", "%"),
            ("memory", "use"),
        ),
        "power_w": (("power",), ("socket", "power"), ("average", "power")),
        "temperature_c": (("temp",), ("temperature",)),
        "sclk_mhz": (("sclk",), ("socclk",)),
        "mclk_mhz": (("mclk",), ("memclk",)),
    }
    for output_key, choices in mappings.items():
        for choice in choices:
            value = _first_matching(flattened, *choice)
            if value is not None:
                sample[output_key] = value
                break
    return sample


def _summarize(samples: list[dict[str, float]]) -> dict[str, float | int | bool]:
    summary: dict[str, float | int | bool] = {"sample_count": len(samples)}
    keys = sorted({key for sample in samples for key in sample})
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        if not values:
            continue
        summary[f"{key}_avg"] = mean(values)
        summary[f"{key}_max"] = max(values)
        summary[f"{key}_min"] = min(values)

    util_avg = summary.get("gpu_util_percent_avg")
    vram_max = summary.get("vram_used_percent_max")
    torch_reserved_max = summary.get("torch_reserved_percent_max")
    if isinstance(util_avg, (int, float)):
        summary["utilization_avg_below_80"] = util_avg < 80.0
    if isinstance(vram_max, (int, float)):
        summary["vram_peak_above_90"] = vram_max > 90.0
    elif isinstance(torch_reserved_max, (int, float)):
        summary["vram_peak_above_90"] = torch_reserved_max > 90.0
    return summary


class GpuTelemetrySampler:
    def __init__(self, label: str, metadata: dict[str, Any] | None = None):
        self.label = label
        self.metadata = dict(metadata or {})
        self.enabled = telemetry_enabled()
        self.interval = telemetry_interval_seconds()
        self.started_at = 0.0
        self.ended_at = 0.0
        self.samples: list[dict[str, float]] = []
        self.summary: dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuTelemetrySampler":
        self.started_at = time.monotonic()
        if not self.enabled:
            return self
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(error=repr(exc) if exc is not None else None)
        return False

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._sample_once()

    def _sample_once(self) -> None:
        sample = {"t": time.monotonic()}
        sample.update(_collect_torch_sample())
        sample.update(_collect_rocm_smi_sample())
        self.samples.append(sample)

    def finish(
        self, extra: dict[str, Any] | None = None, error: str | None = None
    ) -> dict[str, Any]:
        if self.ended_at:
            return self.summary
        self.ended_at = time.monotonic()
        if self.enabled:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=max(self.interval * 2.0, 0.2))
            self._sample_once()
        elapsed = max(self.ended_at - self.started_at, 0.0)
        summary = _summarize(self.samples) if self.samples else {"sample_count": 0}
        summary["elapsed_seconds"] = elapsed
        self.summary = summary
        if not self.enabled:
            return summary

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": os.environ.get("SD_RUN_ID"),
            "label": self.label,
            "metadata": self.metadata,
            "summary": summary,
        }
        if extra:
            record["extra"] = extra
        if error:
            record["error"] = error
        if os.environ.get("SD_GPU_TELEMETRY_TRACE_SAMPLES", "0") == "1":
            record["samples"] = self.samples
        _write_record(record)
        _print_summary(self.label, summary)
        return summary


def _write_record(record: dict[str, Any]) -> None:
    path = telemetry_jsonl_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _JSONL_LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        _warn_once("jsonl-write", f"[perf] failed to write telemetry JSONL: {exc}")


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    util = summary.get("gpu_util_percent_avg")
    util_max = summary.get("gpu_util_percent_max")
    vram = summary.get("vram_used_percent_max") or summary.get(
        "torch_reserved_percent_max"
    )
    power = summary.get("power_w_avg")
    elapsed = summary.get("elapsed_seconds", 0.0)
    parts = [f"[perf] {label}: elapsed={float(elapsed):.2f}s"]
    if isinstance(util, (int, float)):
        parts.append(f"gpu_util_avg={util:.1f}%")
    if isinstance(util_max, (int, float)):
        parts.append(f"gpu_util_max={util_max:.1f}%")
    if isinstance(vram, (int, float)):
        parts.append(f"vram_peak={vram:.1f}%")
    if isinstance(power, (int, float)):
        parts.append(f"power_avg={power:.1f}W")
    print("  ".join(parts))


def start_gpu_telemetry(
    label: str, metadata: dict[str, Any] | None = None
) -> GpuTelemetrySampler:
    sampler = GpuTelemetrySampler(label, metadata)
    sampler.__enter__()
    return sampler


def telemetry_status() -> dict[str, str | float | bool]:
    return {
        "enabled": telemetry_enabled(),
        "interval_seconds": telemetry_interval_seconds(),
        "backend": telemetry_backend_name(),
        "jsonl_path": str(telemetry_jsonl_path()),
    }
