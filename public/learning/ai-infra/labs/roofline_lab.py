"""Simple Roofline classifier for AI operators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    peak_tflops: float
    bandwidth_gbs: float


@dataclass(frozen=True)
class Operator:
    name: str
    flops: float
    bytes_moved: float

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_moved


def roofline(operator: Operator, hardware: Hardware) -> tuple[float, str]:
    memory_bound_gflops = operator.arithmetic_intensity * hardware.bandwidth_gbs
    compute_bound_gflops = hardware.peak_tflops * 1000
    attainable = min(memory_bound_gflops, compute_bound_gflops)
    bottleneck = "memory" if memory_bound_gflops < compute_bound_gflops else "compute"
    return attainable, bottleneck


def main() -> None:
    hardware = Hardware(peak_tflops=100.0, bandwidth_gbs=3000.0)
    operators = [
        Operator("vector_add", flops=1e9, bytes_moved=12e9),
        Operator("softmax_like", flops=8e9, bytes_moved=16e9),
        Operator("large_gemm", flops=140e9, bytes_moved=0.1e9),
    ]
    for operator in operators:
        attainable, bottleneck = roofline(operator, hardware)
        print(
            f"{operator.name:14s} AI={operator.arithmetic_intensity:8.2f} "
            f"limit={attainable/1000:8.2f} TFLOP/s bottleneck={bottleneck}"
        )


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Replace ideal bytes with measured profiler bytes.
# 2. Add an achieved-performance input and report percent of Roofline.
# 3. Model a fused vector_add + activation that avoids an intermediate write.
# 4. Explain why a high-AI operator may still miss the compute roof.
