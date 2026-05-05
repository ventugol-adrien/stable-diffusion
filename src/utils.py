import platform
import subprocess
import json
from pydantic import BaseModel, Field
from typing import Literal


class GPUConfig(BaseModel):
    platform: Literal["cuda", "rocm"] = Field(
        ..., description="GPU platform ('cuda' or 'rocm')"
    )
    name: str = Field(..., description="GPU marketing name")
    vram_gb: float = Field(..., description="Total VRAM in GB")


class HostConfig(BaseModel):
    os: Literal["linux", "windows"] = Field(..., description="Host operating system")
    gpu: GPUConfig


def _detect_gpu() -> GPUConfig:
    # NVIDIA
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            parts = line.split(",", 1)
            name = parts[0].strip()
            vram_gb = round(float(parts[1].strip()) / 1024, 2)
            return GPUConfig(platform="cuda", name=name, vram_gb=vram_gb)
    except Exception:
        pass

    # AMD (ROCm)
    try:
        rinfo = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)
        gpu_names: list[str] = []
        if rinfo.returncode == 0:
            current_name: str | None = None
            for line in rinfo.stdout.splitlines():
                s = line.strip()
                if s.startswith("Agent ") and s[6:].strip().isdigit():
                    current_name = None
                elif s.startswith("Marketing Name:"):
                    current_name = s.split(":", 1)[1].strip() or None
                elif (
                    s.startswith("Device Type:")
                    and s.split(":", 1)[1].strip() == "GPU"
                    and current_name
                ):
                    gpu_names.append(current_name)

        rsmi = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        vram_gb = 0.0
        if rsmi.returncode == 0:
            data = json.loads(rsmi.stdout)
            for key, val in data.items():
                if key.startswith("card"):
                    vram_bytes = int(val.get("VRAM Total Memory (B)", 0))
                    vram_gb = round(vram_bytes / (1024**3), 2)
                    break

        name = gpu_names[0] if gpu_names else "Unknown AMD GPU"
        return GPUConfig(platform="rocm", name=name, vram_gb=vram_gb)
    except Exception:
        pass

    return GPUConfig(platform="rocm", name="Unknown GPU", vram_gb=0.0)


def _detect_os() -> Literal["linux", "windows"]:
    system = platform.system().lower()
    if "windows" in system:
        return "windows"
    return "linux"


def _build_host_config() -> HostConfig:
    return HostConfig(os=_detect_os(), gpu=_detect_gpu())


HOST_CONFIG: HostConfig = _build_host_config()


def is_linux() -> bool:
    return HOST_CONFIG.os == "linux"


def is_windows() -> bool:
    return HOST_CONFIG.os == "windows"


def is_cuda() -> bool:
    return HOST_CONFIG.gpu.platform == "cuda"


def is_rocm() -> bool:
    return HOST_CONFIG.gpu.platform == "rocm"


def vram_gb() -> float:
    return HOST_CONFIG.gpu.vram_gb


def has_vram_gte(gb: float) -> bool:
    return HOST_CONFIG.gpu.vram_gb >= gb


def vram_pressure() -> float:
    """Return current VRAM usage as a fraction (0.0–1.0) of total VRAM."""
    try:
        import torch

        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        if total > 0:
            return allocated / total
    except Exception:
        pass
    return 0.0


def is_vram_pressure_high(threshold: float = 0.70) -> bool:
    """Return True when current VRAM allocation is >= threshold fraction of total."""
    return vram_pressure() >= threshold
