"""Stable Softmax, cross entropy, tolerances, and reduction-order lab."""

from __future__ import annotations

import math
import struct


def naive_softmax(values: list[float]) -> list[float]:
    exps = [math.exp(value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def stable_softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def cross_entropy_from_logits(logits: list[float], target: int) -> float:
    return logsumexp(logits) - logits[target]


def close(actual: float, expected: float, atol: float = 1e-7, rtol: float = 1e-6) -> bool:
    return abs(actual - expected) <= atol + rtol * abs(expected)


def fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def fp32_sum(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total = fp32(total + value)
    return total


def main() -> None:
    logits = [1000.0, 1001.0, 999.0]
    try:
        naive_softmax(logits)
    except OverflowError:
        print("Expected: naive Softmax overflowed")

    probabilities = stable_softmax(logits)
    assert close(sum(probabilities), 1.0)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    print("stable probabilities:", probabilities)
    print("cross entropy(target=1):", cross_entropy_from_logits(logits, 1))

    values = [1e8, 1.0, -1e8]
    forward = fp32_sum(values)
    reordered = fp32_sum([1e8, -1e8, 1.0])
    print("fp32 forward order:", forward)
    print("fp32 reordered:", reordered)
    assert forward != reordered
    print("PASS: stability and non-associativity demonstrated")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Add temperature to stable_softmax and compare entropy.
# 2. Derive and numerically check d(cross_entropy)/d(logits) = p - one_hot.
# 3. Implement pairwise summation and compare it with fp32_sum.
# 4. Add extreme negative logits and verify no division by zero.
