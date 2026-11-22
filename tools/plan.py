"""Cost policy for one commit: which candidates to run, when to stop, how to pack.

Scoring (SCORING.md): commit cost = busiest worker's load + 2 s per worker;
a failing test left out = +300 s; the selector's own CPU is added 1:1.

Two consequences drive everything here:

* After one candidate has failed, a worker is paid for anyway. Including the
  rest unrun costs at most their recorded time; running them costs the same in
  CPU and the failing ones are then paid twice. So: stop at the first failure.
* Before any failure, running is a bet that the commit breaks nothing. It only
  pays while the remaining candidates are short. `Planner` makes that call
  after every test.

Usage:

    p = Planner({num: ms, ...})            # candidates the maps say may fail
    while (t := p.next()) is not None:
        p.report(t, failed=run_test(t), cpu_ms=measured)
    workers = p.workers()                  # [] when nothing needs to run

`pack` is usable on its own to split any set of tests across workers.
"""

from __future__ import annotations

WORKER_MS = 2000
MAX_WORKERS = 8


def _lpt(items: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    loads = [0] * k
    bins: list[list[int]] = [[] for _ in range(k)]
    for num, ms in items:
        j = loads.index(min(loads))
        loads[j] += ms
        bins[j].append(num)
    return max(loads), bins


def pack(tests: dict[int, int]) -> tuple[int, list[list[int]]]:
    """Split tests ({number: ms}) across 1..8 workers at the lowest cost.

    Returns (cost_ms, workers). An empty input costs 0 with no workers.
    """
    if not tests:
        return 0, []
    items = sorted(tests.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(ms for _, ms in items)
    biggest = items[0][1]
    best: tuple[int, list[list[int]]] | None = None
    for k in range(1, min(MAX_WORKERS, len(items)) + 1):
        # No split into k workers can beat this bound, so stop once it loses.
        bound = max(biggest, -(-total // k)) + WORKER_MS * k
        if best is not None and bound >= best[0]:
            continue
        load, bins = _lpt(items, k)
        cost = load + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, [b for b in bins if b])
    assert best is not None
    return best


def pack_cost(tests: dict[int, int]) -> int:
    return pack(tests)[0]


class Planner:
    """Decides, test by test, whether to keep running or include the rest.

    cpu_ratio   expected CPU ms per recorded ms when a candidate is run here.
    p_fail      prior that a commit with candidates breaks at least one.
    early       when a commit does break something, the share of the run
                (by time) that is spent before the first failure shows.
    must        tests included without running, whatever happens (for example
                a big test that a cheap check already proved failing).
    """

    def __init__(
        self,
        candidates: dict[int, int],
        *,
        cpu_ratio: float = 1.0,
        p_fail: float = 0.5,
        early: float = 0.25,
        must: dict[int, int] | None = None,
        order: list[int] | None = None,
    ) -> None:
        self.must = dict(must or {})
        self.ms = {n: ms for n, ms in candidates.items() if n not in self.must}
        self.cpu_ratio = cpu_ratio
        self.p0 = p_fail
        self.early = early
        # Cheapest first unless the caller knows which are likeliest to fail.
        self.todo = list(order) if order else sorted(self.ms, key=lambda n: (self.ms[n], n))
        self.todo = [n for n in self.todo if n in self.ms]
        self.total = sum(self.ms.values())
        self.failed: set[int] = set()
        self.passed: set[int] = set()
        self.stopped = False
        self.cpu_ms = 0.0

    def _posterior(self, remaining: int) -> float:
        """P(some remaining candidate fails | everything run so far passed)."""
        if not self.total:
            return 0.0
        share = remaining / self.total
        p = self.p0 * share
        return p / (p + (1 - self.p0)) if p + (1 - self.p0) > 0 else 1.0

    def next(self) -> int | None:
        """The next test to run, or None when running should stop."""
        if self.stopped or not self.todo:
            return None
        if self.failed or self.must:
            # A worker is already paid for: include the rest unrun.
            self.stopped = True
            return None
        rest = {n: self.ms[n] for n in self.todo}
        remaining = sum(rest.values())
        q = self._posterior(remaining)
        run = self.cpu_ratio * remaining * (1 - q + q * self.early) + q * pack_cost(rest)
        include = pack_cost(rest)
        if run >= include:
            self.stopped = True
            return None
        return self.todo[0]

    def report(self, num: int, failed: bool, cpu_ms: float = 0.0) -> None:
        self.todo.remove(num)
        self.cpu_ms += cpu_ms
        (self.failed if failed else self.passed).add(num)

    def selected(self) -> dict[int, int]:
        """Every test that must be in the answer: failed, forced, or unrun."""
        out = dict(self.must)
        if not (self.failed or self.must or self.todo):
            return out
        for n in self.failed:
            out[n] = self.ms[n]
        if self.failed or self.must or self.stopped:
            for n in self.todo:
                out[n] = self.ms[n]
        return out

    def workers(self) -> list[list[int]]:
        return pack(self.selected())[1]


def _selftest() -> None:
    assert pack({}) == (0, [])
    assert pack({1: 500}) == (2500, [[1]])
    # 21.454 s + 4 s of small tests: two workers beat one.
    cost, w = pack({1: 21454, 2: 2000, 3: 2000})
    assert cost == 21454 + 4000 and len(w) == 2, (cost, w)
    # 46 s of 1 s tests: include-all is far below 46 s.
    cost, w = pack({i: 1000 for i in range(46)})
    assert cost <= 20000, cost
    assert sorted(n for b in w for n in b) == list(range(46))

    # Nothing fails: all cheap candidates get run, answer is empty.
    p = Planner({1: 10, 2: 20, 3: 30})
    ran = []
    while (t := p.next()) is not None:
        ran.append(t)
        p.report(t, failed=False)
    assert ran == [1, 2, 3] and p.workers() == []

    # First failure stops the run; the unrun rest is included, passed is not.
    p = Planner({1: 10, 2: 20, 3: 30})
    p.report(p.next(), failed=False)
    p.report(p.next(), failed=True)
    assert p.next() is None
    assert sorted(p.selected()) == [2, 3]

    # Too much to run: include everything without running any.
    p = Planner({i: 1000 for i in range(46)})
    assert p.next() is None
    assert len(p.selected()) == 46

    # A forced test means nothing else is worth running.
    p = Planner({1: 10, 2: 20}, must={9: 21454})
    assert p.next() is None and sorted(p.selected()) == [1, 2, 9]
    print("plan.py ok")


if __name__ == "__main__":
    _selftest()
