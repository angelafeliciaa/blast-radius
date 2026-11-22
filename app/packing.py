"""Split selected tests across at most 8 workers to minimise busiest load + 2 s per worker."""

from __future__ import annotations

WORKER_COST_MS = 2000
MAX_WORKERS = 8


def duration_ms(seconds: float) -> int:
    import math

    return math.floor(seconds * 1000 + 0.5)


def _lpt(items: list[tuple[int, int]], workers: int) -> tuple[int, list[list[int]]]:
    loads = [0] * workers
    bins: list[list[int]] = [[] for _ in range(workers)]
    for test, ms in items:
        lightest = min(range(workers), key=loads.__getitem__)
        bins[lightest].append(test)
        loads[lightest] += ms
    return max(loads), bins


def pack(durations: dict[int, int]) -> tuple[int, list[list[int]]]:
    """Returns (cost in ms, workers). An empty selection costs nothing."""
    if not durations:
        return 0, []
    items = sorted(durations.items(), key=lambda kv: (-kv[1], kv[0]))
    best: tuple[int, list[list[int]]] | None = None
    for workers in range(1, min(MAX_WORKERS, len(items)) + 1):
        load, bins = _lpt(items, workers)
        bins = [b for b in bins if b]
        cost = load + WORKER_COST_MS * len(bins)
        if best is None or cost < best[0]:
            best = (cost, bins)
    assert best is not None
    return best
