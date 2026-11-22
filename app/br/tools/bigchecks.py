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

# CPU seconds here per recorded second of the map (BR_SPEED overrides).
SPEED = float(os.environ.get("BR_SPEED", "1.0"))

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
        for suite in _map.values():
            suite["setup"] = {k: set(v) for k, v in suite["setup"].items()}
    return _map


def _h(x) -> str:
    return hashlib.sha1(repr(x).encode()).hexdigest()[:16]


class _Timeout(Exception):
    pass


class _limit:
    """Raise _Timeout in the block after `seconds` of this process's CPU time.

    CPU time, not wall time: on a busy two-core box a wall clock would fail
    healthy code. Main thread only; elsewhere it does nothing.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.old = None

    def _fire(self, *_):
        raise _Timeout()

    def __enter__(self):
        try:
            self.old = signal.signal(signal.SIGPROF, self._fire)
            signal.setitimer(signal.ITIMER_PROF, self.seconds)
        except ValueError:
            self.old = None
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            signal.setitimer(signal.ITIMER_PROF, 0)
            signal.signal(signal.SIGPROF, self.old)
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

        with _limit(10.0):
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
        extra = []
        for n in v.walk():
            t = n.type
            meta = getattr(n, "_meta", None)
            extra.append(f"{t if t is not None else ''}|{sorted(meta.items()) if meta else ''}|{n.comments or ''}")
        return f"E({v!r};{';'.join(extra)})"
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


class _Stop(BaseException):
    pass


def exec_query(suite: str, expression, full: bool, on_exec=None):
    """Call the real sqlglot.executor.execute the way the test's worker does.

    Returns (plan fingerprint, result hash). With full=False it stops where
    execute() hands the plan to PythonExecutor, before any row is read, and the
    result hash is None. The fingerprint is None when execute() never got there.
    """
    import importlib

    from sqlglot import find_tables
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA

    ex = importlib.import_module("sqlglot.executor")
    schema = TPCDS_SCHEMA if suite == "tpcds" else TPCH_SCHEMA
    tables = {}
    for t in find_tables(expression):
        tables[t.name] = _table(suite, t.name)

    real = ex.PythonExecutor
    box: dict = {}

    class Shim:
        def __init__(self, *a, **kw):
            self.a, self.kw = a, kw

        def execute(self, plan):
            box["fp"] = plan_fingerprint(plan.expression, plan)
            if not full:
                raise _Stop()
            if on_exec:
                on_exec()
            return real(*self.a, **self.kw).execute(plan)

    ex.PythonExecutor = Shim
    try:
        out = ex.execute(expression, schema=schema, tables=tables)
    except _Stop:
        return box["fp"], None
    finally:
        ex.PythonExecutor = real
    out = pickle.loads(pickle.dumps(out))
    return box.get("fp"), _h((out.columns, out.rows))


def _hits(per_line: dict, nums, slot: int) -> set[int]:
    hit: set[int] = set()
    for n in nums:
        entry = per_line.get(str(n))
        if entry:
            hit.update(entry[slot])
    return hit


# Files whose import-time state the executor's row-processing phase cannot
# read. At the pinned commit that phase runs lines of 15 files only (the
# probe's exec phase over all queries: executor/, planner, generator,
# generators/python, dialects/dialect, expressions/, helper, schema, trie), and
# from the families below those files import nothing but functions. A function
# reads its module's state only by running, and none of these modules runs a
# line in that phase. So an import-time edit here can change a query's outcome
# only through its plan or its transpiled SQL, which are compared for every
# query.
_FRONT_PREFIX = ("sqlglot/optimizer/",)
_FRONT_DIALECT = ("sqlglot/parsers/", "sqlglot/dialects/", "sqlglot/generators/")
_NOT_FRONT = {"__init__.py", "dialect.py", "base.py", "python.py"}


def front_only(path: str) -> bool:
    if path.startswith(_FRONT_PREFIX):
        return True
    return path.startswith(_FRONT_DIALECT) and path.rsplit("/", 1)[-1] not in _NOT_FRONT


def affected(suite: str, changed) -> tuple[list[int], set[int]] | None:
    """(queries to check, those whose row processing must really be re-run).
    None = every query, fully.

    `changed` maps a file to the pinned line numbers the patch touches, or to
    None when the edit is in code that runs at import. An import-time edit (or
    a touched line that ran during import or table loading) reaches every
    query: all of them fully, unless the file is front_only, when comparing
    every query's plan and transpiled SQL is enough.
    """
    if changed is None:
        return None
    m = load_map()[suite]
    lines, setup = m["lines"], m["setup"]
    plan: set[int] = set()
    execs: set[int] = set()
    every = False
    for path, nums in changed.items():
        if path.startswith("#"):
            continue
        early = setup.get(path)
        if nums is None or (early and not early.isdisjoint(nums)):
            if not front_only(path) or path in m["exec_files"]:
                return None
            every = True
            if nums is None:
                continue
        per = lines.get(path)
        if per:
            plan |= _hits(per, nums, 0)
            execs |= _hits(per, nums, 1)
    if every:
        plan = set(range(len(m["q"])))
    return sorted(plan | execs), execs


_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "tpc_rules.pkl.gz")
_rules = None


def load_rules() -> dict:
    """{suite: {query: [pickled input of each optimizer rule]}} from the pinned commit."""
    global _rules
    if _rules is None:
        with gzip.open(_RULES_PATH, "rb") as fh:
            _rules = pickle.load(fh)
    return _rules


def rule_pairs(suite: str, changed) -> set[tuple[int, int]] | None:
    """(query, optimizer rule index) pairs to replay, when every touched line
    runs only inside optimizer rules: never at import, never in the rest of
    planning (parse, transpile, Plan), never while rows are processed.
    None when that does not hold, or the map has no rule stages."""
    if changed is None or "#cached" in changed or not os.path.exists(_RULES_PATH):
        return None
    m = load_map()[suite]
    lines, setup = m["lines"], m["setup"]
    pairs: set[tuple[int, int]] = set()
    for path, nums in changed.items():
        if nums is None:
            return None
        early = setup.get(path)
        if early and not early.isdisjoint(nums):
            return None
        per = lines.get(path)
        if not per:
            continue
        for n in nums:
            entry = per.get(str(n))
            if not entry:
                continue
            if entry[1] or entry[2]:
                return None
            for k, qs in entry[3].items():
                pairs.update((q, int(k)) for q in qs)
    return pairs


def replay_rules(suite: str, pairs) -> set[int]:
    """Run each (query, rule) on the pinned input of that rule with the patched
    code. Returns the queries where some rule's output differs from the pinned
    one (or the rule raised). A rule whose output is identical leaves everything
    after it identical. Raises when the stored inputs cannot be used."""
    import importlib
    import inspect

    from sqlglot.schema import ensure_schema
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA

    rules = importlib.import_module("sqlglot.optimizer.optimizer").RULES
    recorded = load_map()[suite]["q"]
    blobs = load_rules()[suite]
    by_q: dict[int, list[int]] = {}
    for q, k in pairs:
        by_q.setdefault(q, []).append(k)
    differ: set[int] = set()
    for q, ks in sorted(by_q.items()):
        rec = recorded[q].get("rules")
        if not rec or q not in blobs or len(rules) != len(rec):
            differ.add(q)
            continue
        schema = ensure_schema(TPCDS_SCHEMA if suite == "tpcds" else TPCH_SCHEMA, dialect=None)
        possible = {"db": None, "catalog": None, "schema": schema, "dialect": None, "sql": None,
                    "isolate_tables": True, "quote_identifiers": False, "leave_tables_isolated": True}
        for k in sorted(ks):
            if rec[k] is None:  # not reproducible in isolation at the pinned commit
                differ.add(q)
                break
            tree = pickle.loads(blobs[q][k])
            try:
                with _limit(10.0):
                    rule = rules[k]
                    kw = {a: possible[a] for a in inspect.getfullargspec(rule).args if a in possible}
                    same = _h(_ser(rule(tree, **kw))) == rec[k]
            except BaseException:
                same = False
            if not same:
                differ.add(q)
                break
    return differ


def tpc_estimate(suite: str, changed, executes: bool = True) -> float:
    """Recorded CPU seconds the check would need for this patch (no sqlglot import)."""
    recorded = load_map()[suite]["q"]
    pairs = rule_pairs(suite, changed) if executes else None
    if pairs is not None:
        ms = sum((recorded[q].get("rule_ms") or [0] * (k + 1))[k] for q, k in pairs)
        # unpickle + rule + fingerprint, plus loading the stored inputs
        return (3 * ms / 1000 + 0.004 * len(pairs)) * SPEED + (0.15 if pairs else 0.0)
    hit = affected(suite, changed)
    todo, in_exec = (range(len(recorded)), None) if hit is None else hit
    total = 0.0
    for i in todo:
        rec = recorded[i]
        if executes and "res" in rec:
            total += rec.get("plan_ms", 0)
            if in_exec is None or i in in_exec:
                total += rec["ms"]
        else:
            total += 15  # parse and transpile only
    # Loading the tables the queries read: up to 1.1 s for all of them.
    return total / 1000 * SPEED + (min(1.1, 0.15 * len(list(todo))) if executes and todo else 0.0)


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

    stages = rule_pairs(suite, changed) if executes else None
    if stages is not None:
        try:
            differ = replay_rules(suite, stages)
        except BaseException:
            differ = None
        if differ is not None:
            if not differ:
                return PASS
            # Only these queries can have changed; plan (and if needed run) them.
            todo, in_exec = sorted(differ), set()

    def runs(i):
        return executes and bool(pairs[i][0].get("execute"))

    def must_exec(i):
        return runs(i) and (in_exec is None or i in in_exec)

    # Give up before spending anything when the work cannot fit the budget.
    # The recorded times are CPU seconds on the build machine; SPEED scales
    # them to this one and is re-learned from the queries as they run.
    speed = SPEED
    plan_rec = sum(recorded[i].get("plan_ms", 0) for i in todo if runs(i)) / 1000
    exec_rec = sum(recorded[i]["ms"] for i in todo if must_exec(i)) / 1000
    if (plan_rec + exec_rec) * speed > budget_s:
        return UNKNOWN

    start = time.process_time()
    done_rec = 0.0
    # Cheapest first, so a failing patch is caught early.
    todo.sort(key=lambda i: (recorded[i].get("plan_ms", 0) + (recorded[i]["ms"] if must_exec(i) else 0), i))
    left = plan_rec + exec_rec
    for i in todo:
        rec = recorded[i]
        meta, sql, opt = pairs[i]
        spent = time.process_time() - start
        if done_rec > 0.5:
            speed = max(spent / done_rec, 0.2)
        if spent + left * speed > budget_s:
            return UNKNOWN
        try:
            with _limit(max(10.0, 20 * rec.get("ms", 0) / 1000)):
                expression = parse_one(sql)
                if executes:  # the parent pickles every query for the pool
                    expression = pickle.loads(pickle.dumps(expression))
                if (runs(i) or optimized) and _h(transpile(sql, write="duckdb")[0]) != rec["duck"]:
                    return FAIL
                if optimized and _h(transpile(opt, write="duckdb")[0]) != rec["duck_opt"]:
                    return FAIL
                if runs(i):
                    full = must_exec(i)
                    fp, res = exec_query(suite, expression, full=full)
                    if not full and (fp is None or fp != rec.get("plan")):
                        # The plan changed: only running it tells whether the result did.
                        left += rec["ms"] / 1000
                        if spent + left * speed > budget_s:
                            return UNKNOWN
                        fp, res = exec_query(suite, expression, full=True)
                        full = True
                        done_rec += rec["ms"] / 1000
                        left -= rec["ms"] / 1000
                    if full and res != rec["res"]:
                        return FAIL
                    cost = rec.get("plan_ms", 0) + (rec["ms"] if must_exec(i) else 0)
                    done_rec += cost / 1000
                    left -= cost / 1000
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
