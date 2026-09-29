"""GPU schema and file-backed catalog. Hardware estimates are offline utilities.

V1 serving is paced by configured aggregate rates, not these peak specifications.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from kvbench.core.catalog import BUILTIN_CATALOG, read_catalog


@dataclass(frozen=True)
class GPUProfile:
    name: str
    bf16_tflops: float
    hbm_bandwidth_tb_s: float
    hbm_capacity_gb: int
    fp16_tflops: float | None = None
    nvlink_bandwidth_gb_s: float | None = None
    pcie_bandwidth_gb_s: float = 64.0
    tdp_watts: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("GPU name must be nonempty text")
        if self.fp16_tflops is None:
            object.__setattr__(self, "fp16_tflops", self.bf16_tflops)
        for field in ("bf16_tflops", "fp16_tflops", "hbm_bandwidth_tb_s",
                      "hbm_capacity_gb", "nvlink_bandwidth_gb_s", "pcie_bandwidth_gb_s"):
            value = getattr(self, field)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{field} must be finite and positive")

    @property
    def hbm_bandwidth_gb_s(self) -> float:
        return self.hbm_bandwidth_tb_s * 1000

    @property
    def arithmetic_intensity_threshold(self) -> float:
        return self.bf16_tflops * 1000 / self.hbm_bandwidth_gb_s


def load_gpu_profiles(path: Path = BUILTIN_CATALOG) -> dict[str, GPUProfile]:
    return {name: GPUProfile(**spec) for name, spec in read_catalog(path).get("gpus", {}).items()}


GPU_PROFILES = load_gpu_profiles()


def get_gpu_profile(name: str) -> GPUProfile:
    if name not in GPU_PROFILES:
        raise KeyError(f"Unknown GPU profile: {name}. Available profiles: {', '.join(list_gpu_profiles())}")
    return GPU_PROFILES[name]


def list_gpu_profiles() -> list[str]:
    return sorted(GPU_PROFILES)


def get_gpu_profile_info(name: str) -> dict[str, Any]:
    p = get_gpu_profile(name)
    return {**asdict(p), "hbm_bandwidth_gb_s": p.hbm_bandwidth_gb_s,
            "arithmetic_intensity_threshold": p.arithmetic_intensity_threshold}
