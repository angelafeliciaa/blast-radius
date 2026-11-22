"""Cheap stand-ins for the five slowest tests.

Running these tests costs their full time in CPU, and including them blindly
costs the same in simulated seconds. Each check below answers "does this test
fail on the patched tree?" in a fraction of that, by exercising the same code
paths at a smaller size and comparing against values recorded from the pinned
commit (maps/big.json.gz, built by tools/build_big_map.py).

Call these in a process where the PATCHED sqlglot is what `import sqlglot`
gives, and `root` (the checkout holding tests/) is on sys.path.

Every check returns one of:
    "fail"     the real test fails: include it.
    "pass"     the real test passes: leave it out.
    "unknown"  not decided within the budget, or the patch touches something
               the recorded values do not cover: treat as a normal candidate.

    verdict = CHECKS[test_id](root=root, changed=changed, budget_s=3.0)

`changed` maps a path such as "sqlglot/parser.py" to the pinned-file line
numbers the patch touches. None means "assume everything is touched".
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import pickle
import signal
import time

FAIL, PASS, UNKNOWN = "fail", "pass", "unknown"

MAX_NODES = "tests/test_parser.py::TestParser::test_max_nodes"
GEN_MEMORY = "tests/test_generator.py::TestGenerator::test_generator_memory"
EXEC_TPCDS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
EXEC_TPCH = "tests/test_executor.py::TestExecutor::test_execute_tpch"
OPT_TPCH = "tests/test_executor.py::TestExecutor::test_optimized_tpch"

_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "big.json.gz")
_map = None


def load_map(path: str | None = None) -> dict:
    global _map
    if _map is None:
        with gzip.open(path or _MAP_PATH, "rt") as fh:
            _map = json.load(fh)
    return _map


def _h(x) -> str:
    return hashlib.sha1(repr(x).encode()).hexdigest()[:16]


class _Timeout(Exception):
    pass


class _limit:
    """Raise _Timeout in the block after `seconds` of wall time (main thread only)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.old = None

    def _fire(self, *_):
        raise _Timeout()

    def __enter__(self):
        try:
            self.old = signal.signal(signal.SIGALRM, self._fire)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        except ValueError:
            self.old = None
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.old)
        return False


# --- tests/test_parser.py::TestParser::test_max_nodes (28.8 s recorded) -------
#
# The real test parses 10000 columns of "x=x=...=x" (depth 100) twice. A parse
# of 20 such columns runs the same 698 lines of sqlglot (measured with
# coverage at the pinned commit), so it gives the same answer for any patch
# whose effect does not depend on the node count itself.


def check_max_nodes(root: str | None = None, changed=None, budget_s: float = 3.0) -> str:
    try:
        from sqlglot import exp, parse_one
        from sqlglot.errors import ParseError

        with _limit(max(budget_s, 2.0)):
            if not isinstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select):
                return FAIL
            cols = ",".join(f"col{i}" for i in range(500))
            if not isinstance(parse_one(f"SELECT {cols} FROM t", max_nodes=1_000_000), exp.Select):
                return FAIL

            sql = "SELECT " + ",".join(["x" + "=x" * 100] * 20)  # about 4000 nodes
            for limit, needle in ((2000, "Maximum number of AST nodes"), (100, "(100)")):
                try:
                    parse_one(sql, max_nodes=limit)
                    return FAIL
                except ParseError as e:
                    if needle not in str(e):
                        return FAIL

            try:
                parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
                return FAIL
            except ParseError:
                pass
            if not isinstance(parse_one("SELECT 1, 2, 3"), exp.Select):
                return FAIL
    except BaseException:
        return FAIL
    return PASS


# --- tests/test_generator.py::TestGenerator::test_generator_memory (6.7 s) ----
#
# The real test builds 1000 generators per dialect and fails on 1024 leaked
# bytes, about one byte a generator. At the pinned commit nothing leaks (0
# bytes at 40 generators), so any growth here is a leak.


def check_generator_memory(root: str | None = None, changed=None, budget_s: float = 3.0) -> str:
    import tracemalloc

    n = 30
    try:
        from sqlglot.dialects.dialect import Dialect, Dialects

        for name in Dialects:
            dialect = Dialect.get_or_raise(name)
            for _ in range(3):
                dialect.generator()
            gc.collect()
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                for _ in range(n):
                    dialect.generator()
                gc.collect()
                after = tracemalloc.get_traced_memory()[0]
            finally:
                tracemalloc.stop()
            if (after - before) * 1000 >= 1024 * n:
                return FAIL
    except BaseException:
        return FAIL
    return PASS


# --- TPC executor tests ------------------------------------------------------
#
# test_execute_tpcds (21.5 s): for each of 99 queries the parent parses it and
# sends it to a worker process; 57 are executed by sqlglot's Python executor
# and compared with DuckDB's answer to transpile(sql, "duckdb").
# The map records, per query at the pinned commit: which sqlglot lines it runs,
# the transpiled text's hash and the result's hash. A query whose lines the
# patch does not touch cannot change. The others are re-run here, cheapest
# first, and any changed hash or exception means the test fails.
#
# test_execute_tpch (3.9 s): no TPC-H query is marked for execution, so the
# test only parses the 22 queries and round-trips them through pickle.
#
# test_optimized_tpch (3.5 s): DuckDB runs transpile(sql) and
# transpile(optimized). Unchanged transpiled text means an unchanged result.

_FIX = {
    "tpcds": ("optimizer/tpc-ds/", "optimizer/tpc-ds/tpc-ds.sql"),
    "tpch": ("optimizer/tpc-h/", "optimizer/tpc-h/tpc-h.sql"),
}
_tables: dict[str, dict] = {"tpcds": {}, "tpch": {}}


def _table(suite: str, name: str):
    """One TPC table, loaded the way TestExecutor.setUpClass loads it."""
    cache = _tables[suite]
    if name not in cache:
        import ast
        import csv

        from sqlglot.executor.table import Table
        from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, TPCH_SCHEMA

        schema = TPCDS_SCHEMA if suite == "tpcds" else TPCH_SCHEMA
        path = f"{FIXTURES_DIR}/{_FIX[suite][0]}{name}.csv.gz"
        reader = csv.reader(gzip.open(path, "rt", newline=""), delimiter="|")
        next(reader)
        rows: list = []
        ctypes: list = []
        for row in reader:
            if not ctypes:
                for v in row:
                    try:
                        ctypes.append(type(ast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        ctypes.append(str)
            rows.append(
                tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row))
            )
        cache[name] = Table(columns=schema[name], rows=rows)
    return cache[name]


def _ser(v) -> str:
    from sqlglot import exp
    from sqlglot.planner import Step

    if isinstance(v, exp.Expr):
        types = ",".join(str(n.type) if n.type is not None else "" for n in v.walk())
        return f"E({v!r};{types})"
    if isinstance(v, Step):
        return f"<{type(v).__name__} {v.name}>"
    if isinstance(v, dict):
        return "{" + ",".join(f"{_ser(k)}:{_ser(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_ser(x) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "S" + ",".join(sorted(_ser(x) for x in v))
    return repr(v)


def plan_fingerprint(optimized, plan) -> str:
    """Everything the executor reads: the optimized tree and every plan step."""
    parts = [_ser(optimized)]
    seen: dict[int, int] = {}
    stack = [plan.root]
    while stack:
        step = stack.pop()
        if id(step) in seen:
            continue
        seen[id(step)] = len(seen)
        fields = []
        for k in sorted(vars(step)):
            fields.append(f"{k}={_ser(getattr(step, k))}")
        parts.append(f"{type(step).__name__}({';'.join(fields)})")
        stack.extend(sorted(step.dependencies, key=lambda d: (d.name or "", type(d).__name__)))
    return _h("\n".join(parts))


def plan_query(suite: str, expression):
    """The part of sqlglot.executor.execute before any row is read."""
    from sqlglot.executor.table import Table, ensure_tables
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from sqlglot import find_tables
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA

    raw = TPCDS_SCHEMA if suite == "tpcds" else TPCH_SCHEMA
    names = [t.name for t in find_tables(expression)]
    ensure_tables({n: Table(columns=raw[n], rows=[]) for n in names})
    schema = ensure_schema(raw)
    optimized = optimize(expression, schema, leave_tables_isolated=True)
    plan = Plan(optimized)
    return names, plan, plan_fingerprint(optimized, plan)


def run_plan(suite: str, names, plan):
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.executor.table import ensure_tables

    tables = ensure_tables({n: _table(suite, n) for n in names})
    out = PythonExecutor(tables=tables).execute(plan)
    out = pickle.loads(pickle.dumps(out))
    return _h((out.columns, out.rows))


def _hits(per_line: dict, nums, slot: int) -> set[int]:
    hit: set[int] = set()
    for n in nums:
        entry = per_line.get(str(n))
        if entry:
            hit.update(entry[slot])
    return hit


def affected(suite: str, changed) -> tuple[list[int], set[int]] | None:
    """(queries whose recorded lines the patch touches, those touched while
    rows are being executed). None = assume all of them."""
    if changed is None:
        return None
    lines = load_map()[suite]["lines"]
    plan: set[int] = set()
    execs: set[int] = set()
    for path, nums in changed.items():
        per = lines.get(path)
        if per:
            plan |= _hits(per, nums, 0)
            execs |= _hits(per, nums, 1)
    return sorted(plan | execs), execs


def _tpc(suite: str, changed, budget_s: float, executes: bool, optimized: bool) -> str:
    m = load_map()[suite]
    try:
        from sqlglot import parse_one, transpile
        from tests.helpers import load_sql_fixture_pairs

        pairs = list(load_sql_fixture_pairs(_FIX[suite][1]))
    except BaseException:
        return FAIL
    recorded = m["q"]
    if len(pairs) != len(recorded):
        return UNKNOWN
    # A query whose text the patch changed has no recorded answer.
    for i, (_, sql, _opt) in enumerate(pairs):
        if _h(sql) != recorded[i]["sql"]:
            return UNKNOWN

    hit = affected(suite, changed)
    if hit is None:
        todo, in_exec = list(range(len(recorded))), None
    else:
        todo, in_exec = hit

    # Pass 1, cheap, every affected query: parse, transpile, plan. A query whose
    # plan is unchanged and whose execution does not run a touched line is done.
    start = time.process_time()
    heavy = []
    for i in todo:
        rec = recorded[i]
        meta, sql, opt = pairs[i]
        if time.process_time() - start > budget_s:
            return UNKNOWN
        try:
            with _limit(5.0):
                expression = parse_one(sql)
                if executes:  # the parent pickles every query for the pool
                    expression = pickle.loads(pickle.dumps(expression))
                runs = executes and bool(meta.get("execute"))
                if (runs or optimized) and _h(transpile(sql, write="duckdb")[0]) != rec["duck"]:
                    return FAIL
                if optimized and _h(transpile(opt, write="duckdb")[0]) != rec["duck_opt"]:
                    return FAIL
                if runs:
                    names, plan, fp = plan_query(suite, expression)
                    same = rec.get("plan") is not None and fp == rec["plan"]
                    if not same or in_exec is None or i in in_exec:
                        heavy.append((rec["ms"], i, names, plan))
        except BaseException:
            return FAIL

    # Pass 2: really execute what is left, cheapest first.
    heavy.sort(key=lambda x: x[:2])
    for ms, i, names, plan in heavy:
        if time.process_time() - start > budget_s:
            return UNKNOWN
        try:
            with _limit(max(2.0, 10 * ms / 1000)):
                if run_plan(suite, names, plan) != recorded[i]["res"]:
                    return FAIL
        except BaseException:
            return FAIL
    return PASS


def check_execute_tpcds(root: str | None = None, changed=None, budget_s: float = 3.0) -> str:
    return _tpc("tpcds", changed, budget_s, executes=True, optimized=False)


def check_execute_tpch(root: str | None = None, changed=None, budget_s: float = 3.0) -> str:
    return _tpc("tpch", changed, budget_s, executes=True, optimized=False)


def check_optimized_tpch(root: str | None = None, changed=None, budget_s: float = 3.0) -> str:
    return _tpc("tpch", changed, budget_s, executes=False, optimized=True)


CHECKS = {
    MAX_NODES: check_max_nodes,
    GEN_MEMORY: check_generator_memory,
    EXEC_TPCDS: check_execute_tpcds,
    EXEC_TPCH: check_execute_tpch,
    OPT_TPCH: check_optimized_tpch,
}
