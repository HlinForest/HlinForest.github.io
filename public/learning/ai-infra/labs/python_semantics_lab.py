"""Executable demonstrations for Python object semantics and concurrency choices."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter, sleep


def collect_bad(value: int, result: list[int] = []) -> list[int]:
    result.append(value)
    return result


def collect(value: int, result: list[int] | None = None) -> list[int]:
    if result is None:
        result = []
    result.append(value)
    return result


def demonstrate_binding() -> None:
    data = [1, 2]
    alias = data
    alias.append(3)
    assert data == [1, 2, 3] and alias is data

    shallow = data.copy()
    assert shallow == data and shallow is not data

    nested = [[1], [2]]
    nested_shallow = nested.copy()
    nested_deep = copy.deepcopy(nested)
    nested_shallow[0].append(9)
    assert nested == [[1, 9], [2]]
    assert nested_deep == [[1], [2]]


def demonstrate_defaults_and_closures() -> None:
    first = collect_bad(1)
    second = collect_bad(2)
    assert first is second and second[-2:] == [1, 2]
    assert collect(1) == [1] and collect(2) == [2]

    bad = [lambda: i for i in range(3)]
    good = [lambda i=i: i for i in range(3)]
    assert [fn() for fn in bad] == [2, 2, 2]
    assert [fn() for fn in good] == [0, 1, 2]


@dataclass(frozen=True)
class Timing:
    mode: str
    seconds: float


def io_task(delay: float) -> float:
    sleep(delay)
    return delay


def cpu_task(limit: int) -> int:
    return sum(value * value for value in range(limit))


def timed_pool(mode: str) -> Timing:
    start = perf_counter()
    if mode == "thread-io":
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(io_task, [0.05] * 4))
    elif mode == "process-cpu":
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(cpu_task, [250_000] * 2))
    else:
        raise ValueError(mode)
    return Timing(mode, perf_counter() - start)


def main() -> None:
    demonstrate_binding()
    demonstrate_defaults_and_closures()
    for result in (timed_pool("thread-io"), timed_pool("process-cpu")):
        print(f"{result.mode}: {result.seconds:.4f}s")
    print("PASS: object semantics assertions hold")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Rewrite collect_bad so calls do not share state.
# 2. Explain why deepcopy is not a universal fix for model or tensor state.
# 3. Add a sequential baseline before judging either pool.
# 4. Increase/decrease task granularity and explain when overhead dominates.
