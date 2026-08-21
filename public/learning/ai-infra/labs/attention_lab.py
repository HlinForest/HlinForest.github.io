"""Single-head causal attention implemented with the Python standard library."""

from __future__ import annotations

import math

Matrix = list[list[float]]


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    if len(left[0]) != len(right):
        raise ValueError("incompatible matrix shapes")
    return [
        [sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0]))]
        for i in range(len(left))
    ]


def stable_softmax(row: list[float]) -> list[float]:
    finite = [value for value in row if math.isfinite(value)]
    maximum = max(finite)
    exps = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def causal_attention(q: Matrix, k: Matrix, v: Matrix) -> tuple[Matrix, Matrix]:
    sequence, head_dim = len(q), len(q[0])
    scores = matmul(q, transpose(k))
    scale = math.sqrt(head_dim)
    for query in range(sequence):
        for key in range(sequence):
            scores[query][key] = scores[query][key] / scale if key <= query else -math.inf
    weights = [stable_softmax(row) for row in scores]
    output = matmul(weights, v)
    return weights, output


def main() -> None:
    q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    v = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    weights, output = causal_attention(q, k, v)

    for query, row in enumerate(weights):
        assert math.isclose(sum(row), 1.0)
        assert all(row[key] == 0.0 for key in range(query + 1, len(row)))
    print("weights:")
    for row in weights:
        print([round(value, 4) for value in row])
    print("output:")
    for row in output:
        print([round(value, 4) for value in row])
    print("PASS: causal mask and probability checks hold")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Remove the sqrt(head_dim) scaling and compare the entropy of each row.
# 2. Implement batched heads using an outer list over heads.
# 3. Count score-matrix elements as sequence length doubles.
# 4. Add a padding mask and distinguish it from the causal mask.
