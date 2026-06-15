from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import gpu_telemetry


class _Completed:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class GpuTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        gpu_telemetry._WARNED_BACKENDS.clear()

    def test_summarize_calculates_average_max_and_flags(self) -> None:
        summary = gpu_telemetry._summarize(
            [
                {"gpu_util_percent": 60.0, "vram_used_percent": 70.0},
                {"gpu_util_percent": 90.0, "vram_used_percent": 95.0},
            ]
        )

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["gpu_util_percent_avg"], 75.0)
        self.assertEqual(summary["gpu_util_percent_max"], 90.0)
        self.assertEqual(summary["vram_used_percent_max"], 95.0)
        self.assertTrue(summary["utilization_avg_below_80"])
        self.assertTrue(summary["vram_peak_above_90"])

    def test_collect_rocm_smi_sample_parses_json_payload(self) -> None:
        payload = {
            "card0": {
                "GPU use (%)": "87%",
                "GPU Memory Allocated (VRAM%)": "61%",
                "Temperature (Sensor edge) (C)": "54.0c",
                "Average Graphics Package Power (W)": "198.5W",
                "sclk clock level": "2450Mhz",
                "mclk clock level": "1249Mhz",
            }
        }

        with mock.patch(
            "src.gpu_telemetry.subprocess.run",
            return_value=_Completed(json.dumps(payload)),
        ):
            sample = gpu_telemetry._collect_rocm_smi_sample()

        self.assertEqual(sample["gpu_util_percent"], 87.0)
        self.assertEqual(sample["vram_used_percent"], 61.0)
        self.assertEqual(sample["temperature_c"], 54.0)
        self.assertEqual(sample["power_w"], 198.5)
        self.assertEqual(sample["sclk_mhz"], 2450.0)
        self.assertEqual(sample["mclk_mhz"], 1249.0)

    def test_collect_rocm_smi_sample_fails_open_when_missing(self) -> None:
        with mock.patch(
            "src.gpu_telemetry.subprocess.run", side_effect=FileNotFoundError()
        ):
            sample = gpu_telemetry._collect_rocm_smi_sample()

        self.assertEqual(sample, {})

    def test_sampler_writes_jsonl_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "perf.jsonl"
            with (
                mock.patch.dict(
                    "os.environ",
                    {
                        "SD_GPU_TELEMETRY": "1",
                        "SD_GPU_TELEMETRY_INTERVAL": "10",
                        "SD_GPU_TELEMETRY_JSONL": str(jsonl_path),
                        "SD_RUN_ID": "test-run",
                    },
                    clear=False,
                ),
                mock.patch(
                    "src.gpu_telemetry._collect_torch_sample",
                    return_value={"torch_reserved_percent": 42.0},
                ),
                mock.patch(
                    "src.gpu_telemetry._collect_rocm_smi_sample",
                    return_value={"gpu_util_percent": 88.0, "power_w": 200.0},
                ),
            ):
                sampler = gpu_telemetry.start_gpu_telemetry(
                    "unit_test", {"model": "juggernaut"}
                )
                summary = sampler.finish(extra={"elapsed_seconds": 1.25})

            self.assertTrue(jsonl_path.is_file())
            record = json.loads(jsonl_path.read_text().strip())
            self.assertEqual(record["run_id"], "test-run")
            self.assertEqual(record["label"], "unit_test")
            self.assertEqual(record["metadata"]["model"], "juggernaut")
            self.assertEqual(record["summary"]["gpu_util_percent_avg"], 88.0)
            self.assertEqual(summary["power_w_avg"], 200.0)


if __name__ == "__main__":
    unittest.main()
