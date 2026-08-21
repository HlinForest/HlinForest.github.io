"""Interactive-free KV-cache capacity calculator with reproducible scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheScenario:
    layers: int
    batch: int
    sequence: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int

    @property
    def bytes(self) -> int:
        return (
            2
            * self.layers
            * self.batch
            * self.sequence
            * self.kv_heads
            * self.head_dim
            * self.bytes_per_element
        )

    @property
    def gib(self) -> float:
        return self.bytes / 1024**3


def capacity(
    gpu_gib: float,
    weights_gib: float,
    workspace_gib: float,
    per_request_gib: float,
    safety_fraction: float = 0.9,
) -> int:
    usable = gpu_gib * safety_fraction - weights_gib - workspace_gib
    return max(0, int(usable // per_request_gib))


def main() -> None:
    base = dict(layers=40, batch=1, sequence=8192, head_dim=128, bytes_per_element=2)
    for label, heads in (("MHA", 32), ("GQA", 8), ("MQA", 1)):
        scenario = CacheScenario(kv_heads=heads, **base)
        print(f"{label}: {scenario.gib:.3f} GiB per request")

    gqa = CacheScenario(kv_heads=8, **base)
    requests = capacity(
        gpu_gib=80,
        weights_gib=30,
        workspace_gib=6,
        per_request_gib=gqa.gib,
    )
    print("idealized GQA request capacity:", requests)
    print("PASS: capacity computed with explicit safety headroom")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Sweep sequence lengths from 1K to 128K.
# 2. Compare BF16 with an 8-bit cache.
# 3. Add block rounding and internal-fragmentation overhead.
# 4. Explain why admission control needs more than this idealized division.
