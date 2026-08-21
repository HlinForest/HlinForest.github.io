"""Collective semantics, Ring communication volume, and alpha-beta model."""

from __future__ import annotations

import math
from dataclasses import dataclass


def broadcast(values: list[list[int]], root: int) -> list[list[int]]:
    return [values[root].copy() for _ in values]


def allgather(values: list[list[int]]) -> list[list[int]]:
    gathered = [item for rank in values for item in rank]
    return [gathered.copy() for _ in values]


def allreduce_sum(values: list[list[int]]) -> list[list[int]]:
    width = len(values[0])
    reduced = [sum(rank[index] for rank in values) for index in range(width)]
    return [reduced.copy() for _ in values]


def reduce_scatter_sum(values: list[list[int]]) -> list[list[int]]:
    reduced = allreduce_sum(values)[0]
    if len(reduced) % len(values) != 0:
        raise ValueError("output width must divide world size")
    chunk = len(reduced) // len(values)
    return [reduced[rank * chunk : (rank + 1) * chunk] for rank in range(len(values))]


def ring_bytes_per_rank(message_bytes: int, world_size: int) -> float:
    return 2 * (world_size - 1) / world_size * message_bytes


@dataclass(frozen=True)
class Link:
    latency_us: float
    bandwidth_gbs: float


def time_us(steps: int, bytes_sent: float, link: Link) -> float:
    transfer_us = bytes_sent / (link.bandwidth_gbs * 1e9) * 1e6
    return steps * link.latency_us + transfer_us


def compare_algorithms(message_bytes: int, world_size: int, link: Link) -> None:
    ring_steps = 2 * (world_size - 1)
    ring_bytes = ring_bytes_per_rank(message_bytes, world_size)
    tree_steps = 2 * math.ceil(math.log2(world_size))
    # Simplified teaching model: each tree level transfers the full message.
    tree_bytes = 2 * math.ceil(math.log2(world_size)) * message_bytes
    print(
        f"message={message_bytes:,} ring={time_us(ring_steps, ring_bytes, link):.2f}us "
        f"tree={time_us(tree_steps, tree_bytes, link):.2f}us"
    )


def main() -> None:
    values = [[1, 2, 3, 4], [10, 20, 30, 40]]
    assert broadcast(values, root=0) == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert allgather([[1, 2], [3, 4]]) == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert allreduce_sum(values) == [[11, 22, 33, 44], [11, 22, 33, 44]]
    assert reduce_scatter_sum(values) == [[11, 22], [33, 44]]

    link = Link(latency_us=2.0, bandwidth_gbs=50.0)
    for size in (1024, 1024**2, 1024**3):
        compare_algorithms(size, world_size=8, link=link)
    print("PASS: collective semantics checks hold")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Add Reduce and AllToAll.
# 2. Print every chunk movement for a four-rank Ring AllReduce.
# 3. Sweep world size and find where latency dominates.
# 4. Explain why the simplified Tree byte model is not a universal NCCL model.
