"""Pure-Python tensor shape, stride, GEMM, and arithmetic-intensity lab."""

from __future__ import annotations

from dataclasses import dataclass


Shape = tuple[int, ...]
Matrix = list[list[float]]


def contiguous_stride(shape: Shape) -> Shape:
    stride: list[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def broadcast_shape(left: Shape, right: Shape) -> Shape:
    output: list[int] = []
    max_rank = max(len(left), len(right))
    padded_left = (1,) * (max_rank - len(left)) + left
    padded_right = (1,) * (max_rank - len(right)) + right
    for a, b in zip(padded_left, padded_right):
        if a != b and a != 1 and b != 1:
            raise ValueError(f"cannot broadcast {left} and {right}")
        output.append(max(a, b))
    return tuple(output)


def matmul_shape(left: Shape, right: Shape) -> Shape:
    if len(left) < 2 or len(right) < 2 or left[-1] != right[-2]:
        raise ValueError(f"invalid matmul: {left} @ {right}")
    batch = broadcast_shape(left[:-2], right[:-2])
    return batch + (left[-2], right[-1])


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("invalid matrix shapes")
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def tiled_matmul(a: Matrix, b: Matrix, tile: int) -> Matrix:
    m, k, n = len(a), len(b), len(b[0])
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i0 in range(0, m, tile):
        for j0 in range(0, n, tile):
            for k0 in range(0, k, tile):
                for i in range(i0, min(i0 + tile, m)):
                    for j in range(j0, min(j0 + tile, n)):
                        for p in range(k0, min(k0 + tile, k)):
                            out[i][j] += a[i][p] * b[p][j]
    return out


@dataclass(frozen=True)
class GemmLedger:
    flops: int
    minimum_bytes: int

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.minimum_bytes


def gemm_ledger(m: int, n: int, k: int, bytes_per_element: int) -> GemmLedger:
    flops = 2 * m * n * k
    minimum_bytes = (m * k + k * n + m * n) * bytes_per_element
    return GemmLedger(flops, minimum_bytes)


def main() -> None:
    assert contiguous_stride((2, 3, 4)) == (12, 4, 1)
    assert broadcast_shape((8, 1, 64), (1, 32, 1)) == (8, 32, 64)
    assert matmul_shape((2, 8, 16, 64), (2, 8, 64, 16)) == (2, 8, 16, 16)

    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    assert matmul(a, b) == tiled_matmul(a, b, tile=1)
    assert matmul(a, b) == tiled_matmul(a, b, tile=2)

    ledger = gemm_ledger(4096, 4096, 4096, bytes_per_element=2)
    print("FLOPs:", f"{ledger.flops:,}")
    print("minimum bytes:", f"{ledger.minimum_bytes:,}")
    print("ideal arithmetic intensity:", f"{ledger.arithmetic_intensity:.2f} FLOPs/byte")
    print("PASS: shape and tiled-matmul checks hold")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Add a linear-offset function and verify a transposed view's stride.
# 2. Add an invalid broadcast case and confirm that it fails.
# 3. Extend the ledger with C read + C write for beta != 0 GEMM.
# 4. Explain why the ideal arithmetic intensity is not the measured value.
