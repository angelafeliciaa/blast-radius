"""Blast Radius selector: for each patch, print which sqlglot tests to run and how to split them.

Reads /data/tests.json and /data/commits/*.patch, writes one JSON line per commit
to stdout. Everything else goes to stderr.
"""

from __future__ import annotations

import gc
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import candidates  # noqa: E402
import guard  # noqa: E402
import packing  # noqa: E402
import scaled  # noqa: E402
import tpcmemo  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", "/data"))
CHECKOUT = os.environ.get("BR_CHECKOUT", "/opt/sqlglot")
MAP_PATH = Path(os.environ.get("BR_MAP", str(HERE / "map.pkl.gz")))
PACKED_MAP = HERE / "map.bin"  # the same map, rewritten by app/packmap.py when the image is built


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_tests() -> list[dict]:
    if TESTS:
        return TESTS[0]
    data = json.loads((DATA / "tests.json").read_text("utf-8"))
    tests = data["tests"]
    for index, test in enumerate(tests):
        if test.get("number") != index:
            raise ValueError("tests.json numbers are not 0..n-1 in order")
    TESTS.append(tests)
    return tests


TESTS: list = []  # tests.json, read once (the guard reads it before it forks the worker)

# Set when this process is the guard's worker (app/guard.py): the pipe its answer lines and
# progress go to, the commits it must give every test without looking, and when the run began.
CHANNEL: int | None = None
SKIP: frozenset = frozenset()
STARTED: float | None = None


class working:
    """Tells the guard which commit this process is busy with, so that one commit that takes
    too long costs only that commit (app/guard.py)."""

    depth = 0

    def __init__(self, commit: str):
        self.commit = commit

    def __enter__(self):
        working.depth += 1
        if CHANNEL is not None and working.depth == 1:
            guard.send(CHANNEL, "P", json.dumps(self.commit))

    def __exit__(self, *exc):
        working.depth -= 1
        if CHANNEL is not None and working.depth == 0:
            guard.send(CHANNEL, "Q")
        return False


class Plan:
    """What we know about one commit, step by step."""

    def __init__(self, commit: str, text: str):
        self.commit = commit
        self.text = text
        self.verdict = None
        self.selected: set[int] | None = None
        self.notes: list[str] = []
        self.splits: dict[str, dict[str, list[int]]] = {}  # test -> fixture items its check run keeps
        self.check: dict = {}  # options of its check run (runtime.child_unittest)


STATE_PATH = HERE / "state.pkl"


def run_diffs(plans: list[Plan], analyzer, budget_end: float) -> None:
    """Fingerprint patched sqlglot for every plan that has import-time changes."""
    import runtime

    todo = [plan for plan in plans if plan.verdict is not None and plan.verdict.needs_diff]
    if not todo:
        return
    if not STATE_PATH.exists():
        log("selector: no fingerprint state; import-time changes use the static guess")
        return
    try:  # once here, so the children need not each load them
        import fingerprint  # noqa: F401

        runtime.load_state(str(STATE_PATH))
        runtime.preload()
    except Exception as error:
        log(f"selector: fingerprint state not preloaded: {error!r}")
    if SHARE:
        try:
            todo = shared_diffs(todo, analyzer, budget_end)
        except Exception as error:
            log(f"selector: shared diff failed: {error!r}")
    running: list[tuple[Plan, runtime.Child, str]] = []
    while todo or running:
        while todo and len(running) < WORKERS and time.time() < budget_end:
            plan = todo.pop(0)
            try:
                root = runtime.make_overlay(f"diff-{plan.commit}", plan.verdict.new_texts)
            except OSError as error:
                plan.notes.append(f"overlay failed {error!r}")
                fallback(plan)
                continue
            out = os.path.join(runtime.SCRATCH, f"diff-{plan.commit}.pkl")
            deadline = min(time.time() + 30, budget_end + 15)
            args = (root, str(STATE_PATH), set(plan.verdict.patched_modules))
            running.append((plan, runtime.Child(plan.commit, runtime.child_diff, args, deadline, out), root))
        if not running:
            for plan in todo:
                plan.notes.append("no time left for the diff")
                fallback(plan)
            todo.clear()
            break
        time.sleep(0.02)
        still = []
        for plan, child, root in running:
            if not child.poll():
                still.append((plan, child, root))
                continue
            runtime.remove_overlay(root)
            result = child.result
            if child.status != "ok" or not isinstance(result, dict):
                plan.notes.append(f"diff {child.status}")
                fallback(plan)
                continue
            apply_diff(plan, result, analyzer)
        running = still


def apply_diff(plan: Plan, result: dict, analyzer) -> None:
    """Turn what a fingerprint diff reported into the plan's tests."""
    with working(plan.commit):
        _apply_diff(plan, result, analyzer)


def _apply_diff(plan: Plan, result: dict, analyzer) -> None:
    # A module the patch adds or deletes has no users of its own in the checkout: whoever
    # imports it fails or changes and is reported under its own name.
    gone = added_or_deleted(plan.verdict.new_texts, analyzer)
    failed = [error.split(":", 1)[0] for error in result["errors"]]
    failed = [module for module in failed if not owned_by(module, gone)]
    changed = {key for key in result["changed"] if not owned_by(key[1], gone)}
    tests, everything, notes = analyzer.diff_tests(changed, failed, plan.verdict.arg_flips)
    plan.notes.extend(notes)
    if everything:
        plan.verdict.everything = True
    else:
        plan.verdict.tests |= tests
    plan.verdict.needs_diff = False
    shown = sorted(k for k in changed if k[0] in ("c", "m"))
    if shown and len(shown) <= 6:
        plan.notes.append("changed " + ", ".join(f"{k[1].rsplit('.', 1)[-1]}.{k[2]}" for k in shown))


SHARE = os.environ.get("BR_SHARE", "1") != "0"


def shared_diffs(todo: list[Plan], analyzer, budget_end: float) -> list[Plan]:
    """Diff the commits that only rewrite modules of the package off one import of the pinned
    sqlglot (runtime.diff_server). Returns the plans that still need a child of their own."""
    import runtime

    jobs = []
    for plan in todo:
        triggers = runtime.diff_triggers(plan.verdict.new_texts)
        if triggers:
            jobs.append({
                "commit": plan.commit, "new_texts": plan.verdict.new_texts, "triggers": triggers,
                "modules": set(plan.verdict.patched_modules),
                "out": os.path.join(runtime.SCRATCH, f"diff-{plan.commit}.pkl"),
            })
    if len(jobs) < 2:  # one commit alone imports just as much either way
        return todo
    out = os.path.join(runtime.SCRATCH, "diff-server.pkl")
    args = (jobs, str(STATE_PATH), budget_end)
    server = runtime.Child("diff-server", runtime.diff_server, args, budget_end + 20, out)
    while not server.poll():
        time.sleep(0.01)
    if server.status != "ok":
        log(f"selector: shared diff ended with {server.status}")
    outcomes = server.result if isinstance(server.result, dict) else {}
    outs = {job["commit"]: job["out"] for job in jobs}
    left = []
    for plan in todo:
        status = outcomes.get(plan.commit)
        result = runtime.take(outs[plan.commit]) if plan.commit in outs else None
        if status == "ok" and isinstance(result, dict):
            apply_diff(plan, result, analyzer)
        elif status == "timeout":
            plan.notes.append("diff timeout")
            fallback(plan)
        else:
            left.append(plan)
    runtime.remove_overlay(os.path.join(runtime.SCRATCH, runtime.TREE))
    return left


def added_or_deleted(new_texts: dict, analyzer) -> set[str]:
    found = set()
    for path, text in new_texts.items():
        if path.startswith("sqlglot/") and path.endswith(".py"):
            module = analyze.module_name(path)
            if text is None or analyzer.module_path(module) is None:
                found.add(module)
    return found


def owned_by(name: str, modules: set[str]) -> bool:
    return any(name == module or name.startswith(module + ".") for module in modules)


def run_import_first(plans: list[Plan], budget_end: float) -> None:
    """For every commit with an import-time change in sqlglot/, import each patched module
    first in a fresh child; if that fails, the run's import order breaks too: every test."""
    import runtime

    todo = [plan for plan in plans if plan.verdict is not None and not plan.verdict.everything
            and (plan.verdict.patched_modules or plan.verdict.new_test_modules)]
    if SHARE:
        try:
            todo = shared_import_first(todo, budget_end)
        except Exception as error:
            log(f"selector: shared import-first check failed: {error!r}")
    running = []
    while todo or running:
        while todo and len(running) < WORKERS:
            plan = todo.pop(0)
            modules = first_modules(plan.verdict)
            try:
                root = runtime.make_overlay(f"first-{plan.commit}", plan.verdict.new_texts)
            except OSError as error:
                plan.verdict.everything = True
                plan.notes.append(f"import-first overlay failed {error!r}: every test")
                continue
            out = os.path.join(runtime.SCRATCH, f"first-{plan.commit}.pkl")
            deadline = min(time.time() + 20, budget_end + 20)
            running.append((plan, runtime.Child(plan.commit, runtime.child_import_first, (root, modules), deadline, out), root))
        time.sleep(0.01)
        still = []
        for plan, child, root in running:
            if not child.poll():
                still.append((plan, child, root))
                continue
            runtime.remove_overlay(root)
            result = child.result
            if child.status != "ok" or not isinstance(result, dict):
                plan.verdict.everything = True
                plan.notes.append(f"import-first check {child.status}: every test")
            elif result["errors"]:
                plan.verdict.everything = True
                plan.notes.append(f"ALL: imported first, {result['errors'][0][:100]}")
        running = still


def first_modules(verdict) -> list[str]:
    """What the import-first check imports after `import sqlglot`: the patched modules of the
    package, then the new test modules (pytest imports each while collecting)."""
    return sorted(m for m in verdict.patched_modules if m.startswith("sqlglot")) + sorted(verdict.new_test_modules)


def shared_import_first(todo: list[Plan], budget_end: float) -> list[Plan]:
    """Run the import-first checks that can start from one `import sqlglot` of the pinned
    checkout (runtime.first_server). Returns the plans that still need a child of their own."""
    import runtime

    if len(todo) < 2:  # one commit alone imports just as much either way
        return todo
    jobs = [{"commit": plan.commit, "new_texts": plan.verdict.new_texts, "modules": first_modules(plan.verdict)}
            for plan in todo]
    out = os.path.join(runtime.SCRATCH, "first-server.pkl")
    server = runtime.Child("first-server", runtime.first_server, (jobs, budget_end), budget_end + 25, out)
    while not server.poll():
        time.sleep(0.01)
    if server.status != "ok":
        log(f"selector: shared import-first check ended with {server.status}")
    outcomes = server.result if isinstance(server.result, dict) else {}
    left = []
    for plan in todo:
        outcome = outcomes.get(plan.commit)
        if outcome is None:
            left.append(plan)
            continue
        status, errors = outcome
        if status != "ok":
            plan.verdict.everything = True
            plan.notes.append(f"import-first check {status}: every test")
        elif errors:
            plan.verdict.everything = True
            plan.notes.append(f"ALL: imported first, {errors[0][:100]}")
    runtime.remove_overlay(os.path.join(runtime.SCRATCH, runtime.TREE))
    return left


def fallback(plan: Plan) -> None:
    with working(plan.commit):
        _fallback(plan)


def _fallback(plan: Plan) -> None:
    verdict = plan.verdict
    verdict.resolve_fallback()
    if verdict.fallback_all:
        verdict.everything = True
    else:
        verdict.tests |= verdict.fallback
    verdict.needs_diff = False


WORKERS = int(os.environ.get("BR_WORKERS", "2"))
# When running the candidates is expected to pay: estimated CPU against the simulated seconds
# it can save. The CPU figures were measured in the entry image on an x86 box: a check run's
# fork, patched copy and test-module imports ~0.2 s; duckdb + pandas 0.4 s (once, in the
# parent, when two or more runs need them). Environment overrides are for measuring other
# settings on a patch set.
VERIFY_BASE = float(os.environ.get("BR_VERIFY_BASE", "0.2"))
VERIFY_HEAVY = float(os.environ.get("BR_VERIFY_HEAVY", "0.4"))
# 0.25 (a run must save ~4x its CPU, from the scorer's ~4.5x box CPU) looked better on the
# proxy set but scored worse on the practice set: d721bc4 155.3 s simulated + 7.3 s CPU against
# 132.0 + 14.5 at 0.5 (29fdd21). The practice set gains far more from check runs.
VERIFY_RATIO = float(os.environ.get("BR_VERIFY_RATIO", "0.5"))
VERIFY_MAX = float(os.environ.get("BR_VERIFY_MAX", "16.0"))
# readers of a changed fixture mostly fail (all 11 of the sample's do): only worth a run when
# the reader is expensive to keep
VERIFY_DATA_RATIO = float(os.environ.get("BR_VERIFY_DATA_RATIO", "0.15"))
SPLIT = os.environ.get("BR_SPLIT", "1") == "1"  # check fixture-driven tests on part of their fixtures
# A check run of only the longest candidates: they set the busiest worker, and the short ones
# cost CPU without moving it. Such a run starts from a test of PARTIAL_START_MS or more (on 1,048
# sample-drawn proxy commits the runs led by a shorter test saved 1.8 s in all for their CPU),
# goes on through tests of PARTIAL_MIN_MS or more and holds at most PARTIAL_MAX tests.
PARTIAL = os.environ.get("BR_PARTIAL", "1") == "1"
PARTIAL_START_MS = int(os.environ.get("BR_PARTIAL_START_MS", "5000"))
PARTIAL_MIN_MS = int(os.environ.get("BR_PARTIAL_MIN_MS", "1000"))
PARTIAL_MAX = int(os.environ.get("BR_PARTIAL_MAX", "12"))
HEAVY_TEST_MODULES = ("tests/test_optimizer.py", "tests/test_executor.py")  # import duckdb and pandas


ITEMS: list = []  # [items.Items or None], loaded when a plan first asks


def item_splits(plan: Plan, the_map, analyzer) -> dict[str, dict[str, list[int]]]:
    """For the fixture-driven candidates of a plan: the fixture items a check run must keep
    (see app/items.py). Empty when the patch is more than function-body statements and data
    files. A test whose every changed fixture yields the same items as before, with no code
    change at all, cannot tell the difference: it is dropped here without a run."""
    verdict = plan.verdict
    if not SPLIT or analyzer is None or not verdict.split_ok or verdict.needs_diff or verdict.everything:
        return {}
    if not ITEMS:
        import items

        ITEMS.append(items.Items.load(CHECKOUT))
    known = ITEMS[0]
    if known is None:
        return {}
    data: dict[str, tuple[str, str]] = {}
    for path in verdict.data_paths:
        old, new = analyzer.source(path), verdict.new_texts.get(path)
        if old is None or new is None:
            return {}
        data[path] = (old, new)
    readers = {path: the_map.file_readers(path) for path in data}
    splits: dict[str, dict[str, list[int]]] = {}
    unchanged: set[int] = set()
    for index in verdict.tests:
        nodeid = the_map.tests[index]
        if nodeid not in known.tests or index in verdict.source_readers:
            continue
        mine = {path: texts for path, texts in data.items() if index in readers[path]}
        keep = known.split(nodeid, verdict.split_lines, mine)
        if keep is None:
            continue
        if not verdict.split_lines and mine and not any(keep.values()):
            unchanged.add(index)
            continue
        splits[nodeid] = keep
    if unchanged:
        verdict.tests -= unchanged
        verdict.data_tests -= unchanged
        plan.notes.append(f"{len(unchanged)} readers get the same fixture items as before")
    return splits


def verification_plan(plan: Plan, the_map, number_by_index: dict[int, int], seconds: dict[int, float], analyzer=None,
                      heavy_cost: float | None = None):
    """Which candidate tests to run now, and whether that is expected to pay off.

    Returns (nodeids to run, map indices kept regardless, estimated CPU) or None.
    heavy_cost: CPU to count for importing duckdb and pandas (0 when the parent has them).
    """
    verdict = plan.verdict
    if verdict is None or verdict.everything or verdict.needs_diff or not verdict.tests:
        return None
    if verdict.no_verify:
        return None
    if any(path not in verdict.new_texts for path in verdict.paths):
        return None  # the patched copy would miss part of the patch: its tests prove nothing
    plan.splits = item_splits(plan, the_map, analyzer)
    index_of_split = {index for index in verdict.tests if the_map.tests[index] in plan.splits}
    other = [path for path in verdict.paths
             if not (path.endswith(".py") and (path.startswith("sqlglot/") or path.startswith("tests/")))]
    if any(not path.startswith("tests/fixtures/") for path in other):
        # files outside the package and the test modules are reached through paths a test may
        # resolve past the copy's symlinks (Path(__file__).resolve()), back to the original
        return None
    # A changed fixture is seen as patched by the tests that load it through tests/helpers,
    # which builds its path from the copy's own location: those are the tests with a split.
    # Any other reader of it stays selected without a run, and so do tests picked only for
    # reading a changed data file when their run cannot be cut down (they mostly fail).
    kept = (set(verdict.data_tests) - index_of_split) | (verdict.unverified & verdict.tests)
    for path in other:
        kept |= the_map.file_readers(path) - index_of_split
    run = set(verdict.tests) - kept
    if not run:
        return None
    cpu = the_map.data.get("cpu", {})
    modules: set[str] = set()
    setup_by_class: dict[str, float] = {}
    estimate = VERIFY_BASE  # fork, pytest start, importing patched sqlglot
    each: dict[str, float] = {}
    worth: dict[str, int] = {}
    stand_ins: dict[str, list] = {}
    for index in run:
        nodeid = the_map.tests[index]
        setup, call = cpu.get(nodeid, (0.5, 1.0))
        recorded = seconds.get(number_by_index.get(index, -1), call)
        if nodeid in plan.splits:
            setup, part = ITEMS[0].cpu(nodeid, plan.splits[nodeid])
            each[nodeid] = min(1.3 * part, max(recorded, 1.3 * call))
        elif scaled.usable(nodeid, verdict.new_texts):
            each[nodeid] = scaled.SCALED[nodeid][2]
            stand_ins[nodeid] = scaled.SCALED[nodeid][1]
        else:
            each[nodeid] = max(recorded, 1.3 * call)
        estimate += each[nodeid]
        worth[nodeid] = packing.duration_ms(seconds.get(number_by_index.get(index, -1), 0.0))
        cls = nodeid.rsplit("::", 1)[0]
        setup_by_class[cls] = max(setup_by_class.get(cls, 0.0), 1.3 * setup)
        modules.add(nodeid.split("::", 1)[0])
    estimate += sum(setup_by_class.values()) + 0.1 * len(modules)
    if modules & set(HEAVY_TEST_MODULES):
        estimate += VERIFY_HEAVY if heavy_cost is None else heavy_cost  # duckdb + pandas imports
    durations = {number_by_index[i]: packing.duration_ms(seconds[number_by_index[i]])
                 for i in verdict.tests if i in number_by_index}
    whole = packing.pack(durations)[0] / 1000
    rest = packing.pack({n: d for n, d in durations.items()
                         if n in {number_by_index.get(i) for i in kept}})[0] / 1000
    saving = whole - rest
    # tests picked only for reading a changed data file mostly fail: running them pays less often
    ratio = VERIFY_DATA_RATIO if run <= verdict.data_tests else VERIFY_RATIO
    chosen = sorted(each, key=lambda nodeid: (-worth[nodeid], nodeid))
    pays = estimate <= ratio * saving and estimate <= VERIFY_MAX
    part = longest_that_pay(chosen, each, worth, setup_by_class, durations, number_by_index, the_map, whole,
                            ratio, heavy_cost) if PARTIAL else None
    if part is not None and pays and ratio * saving - estimate >= part[2]:
        part = None  # every candidate: expected to pay by more than the longest ones alone
    if part is None and not pays:
        plan.splits = {}
        return None
    if part is not None:
        chosen, estimate = part[0], part[1]
        plan.splits = {nodeid: keep for nodeid, keep in plan.splits.items() if nodeid in chosen}
        note = f"check run of the {len(chosen)} longest of {len(each)} candidates"
        if note not in plan.notes:
            plan.notes.append(note)
    plan.check = {"restrict": plan.splits, "worth": worth, "cpu": each, "ratio": ratio,
                  "scaled": {nodeid: edits for nodeid, edits in stand_ins.items() if nodeid in chosen},
                  "tpcmemo": tpcmemo.usable(verdict.new_texts)}
    # the most expensive tests first: whether they pass decides most of what the run can save
    return chosen, kept, estimate


def longest_that_pay(ordered: list[str], each: dict, worth: dict, setup_by_class: dict, durations: dict,
                     number_by_index: dict, the_map, whole: float, ratio: float, heavy_cost: float | None):
    """The first k of `ordered` (candidates, longest first) whose run is expected to pay by the
    most: (those tests, estimated CPU, ratio x saving - CPU), or None when no such run pays."""
    if not ordered or worth[ordered[0]] < PARTIAL_START_MS:
        return None
    index_of = {the_map.tests[index]: index for index in number_by_index}
    best = None
    estimate = VERIFY_BASE
    classes: set[str] = set()
    modules: set[str] = set()
    left = dict(durations)
    for count, nodeid in enumerate(ordered[:PARTIAL_MAX], 1):
        if worth[nodeid] < (PARTIAL_START_MS if count == 1 else PARTIAL_MIN_MS):
            break
        estimate += each[nodeid]
        cls, module = nodeid.rsplit("::", 1)[0], nodeid.split("::", 1)[0]
        if cls not in classes:
            classes.add(cls)
            estimate += setup_by_class[cls]
        if module not in modules:
            modules.add(module)
            estimate += 0.1
            if module in HEAVY_TEST_MODULES and not modules & (set(HEAVY_TEST_MODULES) - {module}):
                estimate += VERIFY_HEAVY if heavy_cost is None else heavy_cost
        left.pop(number_by_index.get(index_of.get(nodeid, -1)), None)
        if estimate > VERIFY_MAX:
            break
        saving = whole - packing.pack(left)[0] / 1000
        if estimate <= ratio * saving and (best is None or ratio * saving - estimate > best[0]):
            best = (ratio * saving - estimate, count, estimate)
    if best is None or best[1] == len(ordered):
        return None
    return ordered[: best[1]], best[2], best[0]


def run_verifications(plans: list[Plan], the_map, number_by_index, seconds, budget_end: float, analyzer=None) -> None:
    import runtime

    def decide(heavy_cost: float | None) -> list[tuple]:
        found = []
        for plan in plans:
            with working(plan.commit):
                decision = verification_plan(plan, the_map, number_by_index, seconds, analyzer, heavy_cost)
            if decision is not None:
                found.append((plan, *decision))
        return found

    def heavy(found: list[tuple]) -> int:
        return sum(1 for job in found if any(n.split("::", 1)[0] in HEAVY_TEST_MODULES for n in job[1]))

    # duckdb and pandas are imported once, here, when two or more runs need them: decide as if
    # they were free, and count them in full only when a single run would import them alone
    jobs = decide(0.0)
    if heavy(jobs) < 2:
        jobs = decide(None)
    if not jobs:
        return
    import unittest  # noqa: F401  # loaded once here instead of in every child

    runtime.preload()

    if heavy(jobs) >= 2:
        try:
            import duckdb  # noqa: F401
            import pandas.testing  # noqa: F401
        except Exception as error:
            log(f"selector: preload failed {error!r}")

    def imports_mock(job: tuple) -> bool:
        plan = job[0]
        for path in {nodeid.split("::", 1)[0] for nodeid in job[1]}:
            text = plan.verdict.new_texts.get(path) if path in plan.verdict.new_texts else None
            if text is None and analyzer is not None:
                text = analyzer.source(path)
            if isinstance(text, str) and ("unittest.mock" in text or ("from unittest import" in text and "mock" in text)):
                return True
        return False

    # unittest.mock brings asyncio with it: 0.05 s of imports in every run whose test modules
    # use it (test_optimizer, test_executor, test_parser...), once here when two or more do
    if sum(1 for job in jobs if imports_mock(job)) >= 2:
        try:
            import unittest.mock  # noqa: F401
        except Exception as error:
            log(f"selector: preload failed {error!r}")
    cpu = the_map.data.get("cpu", {})
    index_of = {nodeid: i for i, nodeid in enumerate(the_map.tests)}

    def limits_for(nodeids, plan: Plan) -> dict[str, float]:
        # a stand-in gets the time of what it runs, not of the test it stands for
        stood_in = plan.check.get("scaled") or {}
        return {n: scaled.SCALED[n][3] if n in stood_in else 2.0 + 4.0 * sum(cpu.get(n, (0.5, 1.0))) for n in nodeids}

    def finish(plan: Plan, nodeids, results: dict, status: str, elapsed: float) -> None:
        passed = {index_of[n] for n in nodeids if results.get(n) == "passed"}
        if any(key.startswith("COLLECT:") for key in results):
            passed = set()
        effect = next((key for key in results if key.startswith("PROCESS:")), None)
        if effect is not None:
            # the patched code changed process-wide state while the tests ran: every later test
            # of the real run sees it
            plan.verdict.everything = True
            plan.notes.append(f"ALL: a check run changed process-wide state {effect[8:120]}")
        plan.verdict.tests -= passed
        cut = f", {len(plan.splits)} on part of their fixtures" if plan.splits else ""
        left = sum(1 for n in nodeids if n not in results)
        cut += f", {left} not run" if left else ""
        plan.notes.append(f"verified {len(nodeids)}{cut} ({status}, {elapsed:.1f}s): {len(nodeids) - len(passed)} kept")

    if SHARE:
        try:
            jobs = shared_verifications(jobs, limits_for, finish, budget_end)
        except Exception as error:
            log(f"selector: shared verification failed: {error!r}")
    running = []
    while jobs or running:
        while jobs and len(running) < WORKERS and time.time() + jobs[0][3] < budget_end:
            plan, nodeids, kept, estimate = jobs.pop(0)
            try:
                root = runtime.make_overlay(f"verify-{plan.commit}", plan.verdict.new_texts)
            except OSError as error:
                plan.notes.append(f"overlay failed {error!r}")
                continue
            limits = limits_for(nodeids, plan)
            out = os.path.join(runtime.SCRATCH, f"verify-{plan.commit}.pkl")
            deadline = min(time.time() + 3 * estimate + 5, budget_end + 10)
            child = runtime.Child(plan.commit, runtime.child_unittest, (root, nodeids, limits, plan.check), deadline, out)
            running.append((plan, nodeids, kept, child, root, out))
        if not running:
            for plan, *_ in jobs:
                plan.notes.append("no time left to verify")
            break
        time.sleep(0.02)
        still = []
        for item in running:
            plan, nodeids, kept, child, root, out = item
            if not child.poll():
                still.append(item)
                continue
            results = runtime.collect_results(child, out)
            runtime.remove_overlay(root)
            finish(plan, nodeids, results, child.status, time.time() - child.started)
        running = still


def shared_verifications(jobs: list[tuple], limits_for, finish, budget_end: float) -> list[tuple]:
    """Run the verifications that can start from one `import sqlglot` of the pinned checkout
    (runtime.verify_server). Returns the jobs that still need a child of their own."""
    import runtime

    def imports_sqlglot(plan: Plan, nodeids) -> bool:
        # the server has sqlglot imported before any test module loads; every test module
        # of the job must be one that imports it anyway
        for path in {nodeid.split("::", 1)[0] for nodeid in nodeids}:
            if path in plan.verdict.new_texts:
                text = plan.verdict.new_texts[path]
            else:
                try:
                    with open(os.path.join(CHECKOUT, path), encoding="utf-8") as handle:
                        text = handle.read()
                except (OSError, UnicodeDecodeError):
                    text = None
            if not isinstance(text, str) or "sqlglot" not in text:
                return False
        return True

    shared = []
    for plan, nodeids, kept, estimate in jobs:
        if imports_sqlglot(plan, nodeids):
            shared.append({
                "commit": plan.commit, "new_texts": plan.verdict.new_texts, "nodeids": nodeids,
                "limits": limits_for(nodeids, plan), "estimate": estimate, "options": plan.check,
                "out": os.path.join(runtime.SCRATCH, f"verify-{plan.commit}.pkl"),
            })
    if len(shared) < 2:  # one run alone imports just as much either way
        return jobs
    out = os.path.join(runtime.SCRATCH, "verify-server.pkl")
    server = runtime.Child("verify-server", runtime.verify_server, (shared, budget_end), budget_end + 20, out)
    while not server.poll():
        time.sleep(0.01)
    if server.status != "ok":
        log(f"selector: shared verification ended with {server.status}")
    outcomes = server.result if isinstance(server.result, dict) else {}
    left = []
    for job in jobs:
        plan, nodeids = job[0], job[1]
        if plan.commit in outcomes:
            finish(plan, nodeids, *outcomes[plan.commit])
        else:
            left.append(job)
    runtime.remove_overlay(os.path.join(runtime.SCRATCH, runtime.TREE))
    return left


def main() -> int:
    started = STARTED if STARTED is not None else time.time()
    if os.environ.get("BR_GC") != "1":
        gc.disable()  # a short run that keeps nearly everything it builds: collections only cost time
    tests = load_tests()
    number_by_id = {test["id"]: test["number"] for test in tests}
    duration = {test["number"]: packing.duration_ms(test["seconds"]) for test in tests}
    commits = sorted((DATA / "commits").glob("*.patch"))
    everything = {test["number"] for test in tests}
    # The run is allowed about 6 s per commit (3 min for 30); keep well inside it, and inside the
    # guard's hard deadline (0.7 of the limit) when there are many commits.
    budget = float(os.environ.get("BR_BUDGET_PER_COMMIT", "2.5")) * max(len(commits), 1)
    budget_end = started + min(budget, 0.45 * guard.time_limit(len(commits)))
    # static analysis of a commit that starts later than this: every test
    analysis_end = started + 0.6 * (budget_end - started)

    the_map = None
    try:
        the_map = analyze.Map(analyze.load_map(MAP_PATH, None if "BR_MAP" in os.environ else PACKED_MAP))
    except Exception as error:  # no map: fall back to running everything
        log(f"selector: cannot load map {MAP_PATH}: {error!r}")
    analyzer = candidates.Analyzer(the_map, CHECKOUT) if the_map is not None else None

    to_number: dict[int, int] = {}
    unknown: set[int] = set(everything)
    if the_map is not None:
        for index, nodeid in enumerate(the_map.tests):
            number = number_by_id.get(nodeid)
            if number is not None:
                to_number[index] = number
                unknown.discard(number)
    if unknown:
        log(f"selector: {len(unknown)} tests are not in the map; they join every non-empty selection")

    plans = []
    for patch in commits:
        raw = patch.read_bytes()
        try:
            plans.append(Plan(patch.stem, raw.decode("utf-8")))
        except UnicodeDecodeError:  # bytes the analysis cannot place: every test (see analyze_patch)
            plans.append(Plan(patch.stem, None))
    for plan in plans:
        if plan.commit in SKIP:
            plan.notes.append("held the worker too long: every test")
            continue
        if time.time() > analysis_end:
            plan.notes.append("no time left to analyse: every test")
            continue
        try:
            if analyzer is None:
                raise RuntimeError("no map")
            with working(plan.commit):
                plan.verdict = analyzer.analyze_patch(plan.text)
        except Exception as error:
            plan.verdict = None
            plan.notes.append(f"error {error!r}")

    if analyzer is not None and os.environ.get("BR_NO_RUNTIME") != "1":
        try:
            run_diffs(plans, analyzer, budget_end)
        except Exception as error:
            log(f"selector: diff stage failed: {error!r}")
        try:
            run_import_first(plans, budget_end)
        except Exception as error:
            log(f"selector: import-first stage failed: {error!r}")
            for plan in plans:  # unchecked: nothing narrower than every test is safe
                if plan.verdict is not None and (plan.verdict.patched_modules or plan.verdict.new_test_modules) \
                        and not plan.verdict.everything:
                    plan.verdict.everything = True
        if os.environ.get("BR_NO_VERIFY") != "1":
            try:
                seconds = {test["number"]: test["seconds"] for test in tests}
                run_verifications(plans, the_map, to_number, seconds, budget_end, analyzer)
            except Exception as error:
                log(f"selector: verification stage failed: {error!r}")

    lines = []
    for plan in plans:
        with working(plan.commit):
            verdict = plan.verdict
            if verdict is None or verdict.everything:
                selected = set(everything)
            else:
                if verdict.needs_diff:
                    fallback(plan)
                if verdict.everything:
                    selected = set(everything)
                else:
                    selected = {to_number[i] for i in verdict.tests if i in to_number}
                    if selected and unknown:
                        selected |= unknown
                    for label, indices in sorted(getattr(verdict.tests, "why", {}).items()):
                        numbers = sorted(to_number[i] for i in indices & verdict.tests if i in to_number)
                        if numbers:
                            log(f"explain: {plan.commit}\t{label}\t{','.join(map(str, numbers))}")
            cost, workers = packing.pack({n: duration[n] for n in selected})
            notes = "; ".join((verdict.notes if verdict else []) + plan.notes)[:400]
            log(f"selector: {plan.commit}: {len(selected)} tests, {cost / 1000:.1f}s | {notes}")
            lines.append(json.dumps({"commit": plan.commit, "workers": workers}, separators=(",", ":")))
        if CHANNEL is not None:
            guard.send(CHANNEL, "L", lines[-1])
    log(f"selector: {len(commits)} commits in {time.time() - started:.1f}s")
    if CHANNEL is not None:
        guard.send(CHANNEL, "E")
        return 0
    global ANSWERED
    ANSWERED = True
    sys.stdout.write("".join(line + "\n" for line in lines))
    sys.stdout.flush()
    return 0


ANSWERED = False


def emergency() -> int:
    """Something went wrong before any answer was printed: run every test on every commit."""
    tests = load_tests()
    duration = {test["number"]: packing.duration_ms(test["seconds"]) for test in tests}
    _, workers = packing.pack(duration)
    lines = [
        json.dumps({"commit": patch.stem, "workers": workers}, separators=(",", ":"))
        for patch in sorted((DATA / "commits").glob("*.patch"))
    ]
    sys.stdout.write("".join(line + "\n" for line in lines))
    sys.stdout.flush()
    return 0


def work(channel: int, skip: frozenset) -> int:
    """The guard's worker (app/guard.py): the whole selector, answers sent to the guard."""
    global CHANNEL, SKIP
    CHANNEL, SKIP = channel, skip
    if skip:  # what a killed worker left in /tmp
        import shutil

        import runtime

        shutil.rmtree(runtime.SCRATCH, ignore_errors=True)
    return main()


def entry() -> int:
    global STARTED
    STARTED = time.time()
    if os.environ.get("BR_GUARD", "1") != "0":
        try:
            tests = load_tests()
            commits = [patch.stem for patch in sorted((DATA / "commits").glob("*.patch"))]
            durations = {test["number"]: packing.duration_ms(test["seconds"]) for test in tests}
        except Exception as error:
            log(f"selector: no guard ({error!r})")
        else:
            code = guard.supervise(work, commits, durations, STARTED)
            if code is not None:
                return code
    try:
        return main()
    except Exception as error:
        log(f"selector: failed ({error!r})")
        return 1 if ANSWERED else emergency()


if __name__ == "__main__":
    sys.exit(entry())
