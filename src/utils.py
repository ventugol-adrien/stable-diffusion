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
    import sys

    # Fast path: if torch is already in sys.modules, query the hardware directly.
    # This is more reliable than subprocess and avoids spawning child processes.
    # When called before torch is imported (e.g. from main.py before env vars are
    # set) torch won't be present yet, so we fall through to the subprocess path.
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram = round(props.total_memory / (1024**3), 2)
                plat: Literal["cuda", "rocm"] = (
                    "rocm" if torch.version.hip is not None else "cuda"
                )
                return GPUConfig(platform=plat, name=name, vram_gb=vram)
        except Exception:
            pass

    # Slow path: subprocess detection when torch isn't loaded yet.
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


async def stream_image(image, chunk_size: int = 1024):
    import asyncio
    import io
    import sys

    buffer = io.BytesIO()
    await asyncio.to_thread(image.save, buffer, format="PNG", interlace=True)
    total_size = buffer.getbuffer().nbytes
    buffer.seek(0)
    bytes_sent = 0
    while chunk := buffer.read(chunk_size):
        bytes_sent += len(chunk)
        percent = (bytes_sent / total_size) * 100
        bar_length = 30
        filled_len = int(bar_length * percent // 100)
        bar = "█" * filled_len + "-" * (bar_length - filled_len)
        sys.stdout.write(
            f"\rStreaming: [{bar}] {percent:.1f}% ({bytes_sent}/{total_size} bytes)"
        )
        sys.stdout.flush()
        yield chunk
        await asyncio.sleep(0)
    print("\nStreaming complete.")


async def stream_zip(zip_buffer, chunk_size: int = 1024):
    import asyncio
    import sys

    total_size = zip_buffer.getbuffer().nbytes
    zip_buffer.seek(0)
    bytes_sent = 0
    while chunk := zip_buffer.read(chunk_size):
        bytes_sent += len(chunk)
        percent = (bytes_sent / total_size) * 100
        bar_length = 30
        filled_len = int(bar_length * percent // 100)
        bar = "█" * filled_len + "-" * (bar_length - filled_len)
        sys.stdout.write(
            f"\rStreaming ZIP: [{bar}] {percent:.1f}% ({bytes_sent}/{total_size} bytes)"
        )
        sys.stdout.flush()
        yield chunk
        await asyncio.sleep(0)
    print("\nStreaming ZIP complete.")
