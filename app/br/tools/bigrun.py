"""Decide the five slowest tests for one patch without running them.

The selector calls `verdicts(patch_text, test_ids)` for the slow tests its maps
picked. This builds a patched copy of the sqlglot package under /tmp (symlinks
plus the patched files), runs tools/bigchecks.py against it in a fresh process,
and returns {test_id: "fail" | "pass" | "unknown"}.

    from bigrun import verdicts, BIG
    v = verdicts(patch_text, [t for t in selected_ids if t in BIG])
    drop  = {t for t, x in v.items() if x == "pass"}     # proven not to fail
    # "fail" and "unknown" stay selected.

A patch this module does not understand (touches anything other than existing
.py files under sqlglot/, or does not apply cleanly) gets "unknown" for every
test, so the caller's own selection stands.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bigchecks  # noqa: E402

BIG = set(bigchecks.CHECKS)

# CPU seconds a check may spend before giving up with "unknown". Including the
# test blindly costs its recorded time, so the budget stays well under that.
BUDGET = {
    bigchecks.MAX_NODES: 3.0,
    bigchecks.GEN_MEMORY: 5.0,
    bigchecks.EXEC_TPCH: 3.0,
    bigchecks.OPT_TPCH: 3.0,
    bigchecks.EXEC_TPCDS: float(os.environ.get("BR_TPCDS_BUDGET", "12")),
}
# Cheapest first; the child stops being useful once sqlglot fails to import.
ORDER = [
    bigchecks.MAX_NODES,
    bigchecks.EXEC_TPCH,
    bigchecks.OPT_TPCH,
    bigchecks.GEN_MEMORY,
    bigchecks.EXEC_TPCDS,
]

# The scorer charges about this many seconds per CPU second measured where
# the maps were built (BR_CPU_FACTOR). A check only runs when the simulated
# seconds it is expected to save exceed what its CPU will be charged.
CPU_FACTOR = float(os.environ.get("BR_CPU_FACTOR", "4.5"))
P_PASS = 0.6  # share of selected slow tests that turn out to pass
RECORDED_S = {
    bigchecks.MAX_NODES: 28.817,
    bigchecks.EXEC_TPCDS: 21.454,
    bigchecks.GEN_MEMORY: 6.686,
    bigchecks.EXEC_TPCH: 3.863,
    bigchecks.OPT_TPCH: 3.537,
}
START_S = 0.35  # process start and importing sqlglot
FIXED_S = {
    bigchecks.MAX_NODES: 0.15,
    bigchecks.GEN_MEMORY: 1.0,
}


def plan_checks(wanted, changed, saving=None):
    """Which checks pay for themselves, with the CPU budget each may use.

    `saving` maps a test to the simulated seconds its removal saves (default:
    its recorded seconds, an upper bound).
    """
    chosen = {}
    cost = 0.0
    gain = 0.0
    for t in wanted:
        worth = P_PASS * (saving or RECORDED_S).get(t, RECORDED_S[t])
        if t in FIXED_S:
            est = FIXED_S[t]
        else:
            suite = "tpcds" if t == bigchecks.EXEC_TPCDS else "tpch"
            est = bigchecks.tpc_estimate(suite, changed, executes=t != bigchecks.OPT_TPCH)
        if CPU_FACTOR * est < worth:
            chosen[t] = min(BUDGET[t], worth / CPU_FACTOR)
            cost += est
            gain += worth
    if gain <= CPU_FACTOR * (START_S + cost):
        return {}
    return chosen


_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class Unsupported(Exception):
    pass


def parse_patch(text: str) -> dict[str, list[tuple[int, list[str]]]]:
    """{old path: [(old start line, hunk body lines)]}. Raises Unsupported."""
    files: dict[str, list[tuple[int, list[str]]]] = {}
    lines = text.split("\n")
    cur = None
    old = None
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("diff --git"):
            cur = old = None
        elif line.startswith(("new file", "deleted file", "rename ", "Binary ", "GIT binary")):
            raise Unsupported(line.split()[0])
        elif line.startswith("--- "):
            old = line[4:].split("\t")[0]
        elif line.startswith("+++ ") and old is not None:
            new = line[4:].split("\t")[0]
            if not (old.startswith("a/") and new.startswith("b/")) or old[2:] != new[2:]:
                raise Unsupported("added, removed or renamed file")
            cur = files.setdefault(old[2:], [])
        elif line.startswith("@@"):
            m = _HUNK.match(line)
            if not m or cur is None:
                raise Unsupported("hunk header")
            need_old = int(m.group(2) or 1)
            need_new = int(m.group(4) or 1)
            body: list[str] = []
            while (need_old > 0 or need_new > 0) and i < len(lines):
                b = lines[i]
                i += 1
                tag = b[:1]
                if tag == "\\":
                    continue
                if tag == "-":
                    need_old -= 1
                elif tag == "+":
                    need_new -= 1
                elif tag in (" ", ""):
                    b = b or " "
                    need_old -= 1
                    need_new -= 1
                else:
                    raise Unsupported("hunk body")
                body.append(b)
            if need_old or need_new:
                raise Unsupported("short hunk")
            cur.append((int(m.group(1)), body))
    return files


def apply(old_lines: list[str], hunks) -> tuple[list[str], set[int]]:
    """Apply hunks to the file's lines. Returns (new lines, touched OLD line numbers).

    An inserted line touches the old lines on both sides of it.
    """
    out: list[str] = []
    touched: set[int] = set()
    pos = 0  # index into old_lines
    for start, body in hunks:
        at = max(start - 1, 0)
        if at < pos:
            raise Unsupported("overlapping hunks")
        out.extend(old_lines[pos:at])
        pos = at
        for b in body:
            tag, rest = b[:1], b[1:]
            if tag == "\\":
                continue
            if tag == " ":
                if pos >= len(old_lines) or old_lines[pos] != rest:
                    raise Unsupported("context mismatch")
                out.append(rest)
                pos += 1
            elif tag == "-":
                if pos >= len(old_lines) or old_lines[pos] != rest:
                    raise Unsupported("removal mismatch")
                touched.add(pos + 1)
                pos += 1
            else:
                out.append(rest)
                touched.update((max(pos, 1), min(pos + 1, max(len(old_lines), 1))))
    out.extend(old_lines[pos:])
    return out, touched


_PLAIN_DECORATORS = {"staticmethod", "classmethod", "property"}


def memoized(source: str, touched: set[int]) -> bool:
    """True when a touched line is inside a function with a decorator that may
    cache its result (anything but staticmethod/classmethod/property). A cached
    result is reused later without running the function's lines, so a line map
    cannot say where its effect ends."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.decorator_list:
            if any(f.lineno <= line <= f.end_lineno for line in touched):
                for d in f.decorator_list:
                    name = d.id if isinstance(d, ast.Name) else None
                    if name not in _PLAIN_DECORATORS:
                        return True
    return False


def statement_lines(source: str, touched: set[int]) -> set[int] | None:
    """Widen touched lines to the statements holding them.

    Coverage is recorded per executed line, and one statement can span several.
    Returns None when a touched line is not inside a function body: such code
    runs at import, so no per-query line map can bound its effect.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    funcs = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    out: set[int] = set()
    for line in touched:
        inner = None
        for f in funcs:
            body_start = f.body[0].lineno
            if body_start <= line <= f.end_lineno and (
                inner is None or f.lineno >= inner.lineno
            ):
                inner = f
        if inner is None:
            return None
        best = None
        for n in ast.walk(inner):
            if isinstance(n, ast.stmt) and n is not inner and n.lineno <= line <= n.end_lineno:
                if best is None or (n.end_lineno - n.lineno) < (best.end_lineno - best.lineno):
                    best = n
        if best is None:  # blank line or comment between statements
            out.add(line)
        else:
            out.update(range(best.lineno, best.end_lineno + 1))
    return out


# Fixture files the five slow tests never open: the TPC ones are under
# optimizer/tpc-h/ and optimizer/tpc-ds/, and nothing else is read by them.
_OTHER_FIXTURES = "tests/fixtures/"
_TPC_FIXTURES = ("tests/fixtures/optimizer/tpc-h/", "tests/fixtures/optimizer/tpc-ds/")


def _irrelevant(rel: str) -> bool:
    return (
        rel.startswith(_OTHER_FIXTURES)
        and not rel.startswith(_TPC_FIXTURES)
        and not rel.endswith(".py")
    )


def default_only(old: str, new: str) -> set[int] | None:
    """When the only difference is constant default values in def headers,
    the body lines of those functions; otherwise None.

    A default is read when the function is called without that argument, and a
    call runs the body, so the edit reaches exactly what runs those lines.
    """
    try:
        a, b = ast.parse(old), ast.parse(new)
    except SyntaxError:
        return None
    fa = [n for n in ast.walk(a) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    fb = [n for n in ast.walk(b) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(fa) != len(fb):
        return None
    lines: set[int] = set()
    for x, y in zip(fa, fb):
        dx = x.args.defaults + [d for d in x.args.kw_defaults if d is not None]
        dy = y.args.defaults + [d for d in y.args.kw_defaults if d is not None]
        if len(dx) != len(dy):
            return None
        differs = False
        for u, v in zip(dx, dy):
            if ast.dump(u) != ast.dump(v):
                if not (isinstance(u, ast.Constant) and isinstance(v, ast.Constant)):
                    return None
                differs = True
        if differs:
            if x.decorator_list:
                return None  # a decorator may read the defaults when the module loads
            lines.update(range(x.body[0].lineno, x.end_lineno + 1))
    if not lines:
        return None
    # Everything else must be identical: blank the constant defaults and compare.
    for tree in (a, b):
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n.args.defaults = [ast.Constant(0) if isinstance(d, ast.Constant) else d for d in n.args.defaults]
                n.args.kw_defaults = [
                    ast.Constant(0) if isinstance(d, ast.Constant) else d for d in n.args.kw_defaults
                ]
    return lines if ast.dump(a) == ast.dump(b) else None


def _same_code(old: str, new: str) -> bool:
    try:
        return ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
    except SyntaxError:
        return False


def _overlay(src: str, dest: str, new_files: dict[str, str]) -> None:
    """dest/sqlglot mirrors src/sqlglot through symlinks, except patched files."""
    real_dirs = {""}
    for rel in new_files:
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:i]))
    for d in sorted(real_dirs):
        if not d:
            continue
        os.makedirs(os.path.join(dest, d), exist_ok=True)
        for name in os.listdir(os.path.join(src, d)):
            rel = f"{d}/{name}"
            if rel in real_dirs or rel in new_files or name == "__pycache__":
                continue
            os.symlink(os.path.join(src, d, name), os.path.join(dest, d, name))
    for rel, text in new_files.items():
        with open(os.path.join(dest, rel), "w") as fh:
            fh.write(text)


_CHILD = r"""
import json, os, sys, time
tools, root, jobs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, tools)
out = {}
try:
    import bigchecks
    import sqlglot  # the patched one, first on PYTHONPATH
    for tid, changed, budget in jobs:
        out[tid] = bigchecks.CHECKS[tid](root=root, changed=changed, budget_s=budget)
        sys.stdout.write(json.dumps({tid: out[tid]}) + "\n"); sys.stdout.flush()
except BaseException:
    for tid, _, _ in jobs:
        if tid not in out:
            sys.stdout.write(json.dumps({tid: "fail"}) + "\n")
"""


def verdicts(
    patch_text: str,
    test_ids,
    src: str = "/sqlglot",
    tmp: str | None = None,
    log=None,
    saving: dict[str, float] | None = None,
) -> dict[str, str]:
    """{test_id: "fail" | "pass" | "unknown"} for the slow tests in test_ids."""
    wanted = [t for t in ORDER if t in set(test_ids)]
    result = {t: bigchecks.UNKNOWN for t in wanted}
    if not wanted:
        return result
    try:
        files = parse_patch(patch_text)
        if not files:
            raise Unsupported("no files")
        new_files: dict[str, str] = {}
        changed: dict[str, list[int] | None] = {}
        cached = False
        for rel, hunks in files.items():
            if _irrelevant(rel):
                continue  # a file none of the five tests reads
            if not (rel.startswith("sqlglot/") and rel.endswith(".py")):
                raise Unsupported(f"outside sqlglot/*.py: {rel}")
            old_text = open(os.path.join(src, rel)).read()
            old_lines = old_text.split("\n")
            new_lines, touched = apply(old_lines, hunks)
            new_files[rel] = "\n".join(new_lines)
            if _same_code(old_text, new_files[rel]):
                continue  # comments or layout only: nothing sqlglot runs changed
            lines = statement_lines(old_text, touched)
            if lines is None:
                lines = default_only(old_text, new_files[rel])
            if memoized(old_text, touched):
                cached = True
            # None: the edit is in code that runs at import.
            changed[rel] = None if lines is None else sorted(lines)
    except (Unsupported, OSError, UnicodeDecodeError) as e:
        if log:
            log("bigrun: unsupported patch:", e)
        return result
    if not changed:
        return {t: bigchecks.PASS for t in wanted}
    if cached:
        changed["#cached"] = []  # tells the checks not to reason stage by stage

    budgets = plan_checks(wanted, changed, saving)
    if log and len(budgets) < len(wanted):
        log("bigrun: not worth checking:", [t.split("::")[-1] for t in wanted if t not in budgets])
    wanted = [t for t in wanted if t in budgets]
    if not wanted:
        return result

    work = tempfile.mkdtemp(prefix="brbig", dir=tmp)
    try:
        _overlay(src, work, new_files)
        jobs = [[t, changed, budgets[t]] for t in wanted]
        env = dict(os.environ, PYTHONPATH=f"{work}{os.pathsep}{src}", SKIP_INTEGRATION="0",
                   PYTHONDONTWRITEBYTECODE="1")
        limit = 20 + 3 * sum(budgets.values())
        start = time.time()
        try:
            p = subprocess.run(
                [sys.executable, "-c", _CHILD, HERE, src, json.dumps(jobs)],
                env=env, capture_output=True, text=True, timeout=limit, cwd=work,
            )
            stdout = p.stdout
            hung = False
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            hung = True
        got: dict[str, str] = {}
        for line in stdout.splitlines():
            try:
                got.update(json.loads(line))
            except ValueError:
                pass
        for i, t in enumerate(wanted):
            if t in got:
                result[t] = got[t]
            elif hung:
                # The check that was running hung: so would the real test.
                result[t] = bigchecks.FAIL if all(w in got for w in wanted[:i]) else bigchecks.UNKNOWN
        if log:
            log("bigrun:", {t.split("::")[-1]: v for t, v in result.items()},
                f"{time.time() - start:.2f}s wall",
                "import-time" if any(v is None for v in changed.values())
                else sum(len(v) for v in changed.values()))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


if __name__ == "__main__":
    # python tools/bigrun.py <sqlglot_root> <patch> [test id ...]
    root, patch = sys.argv[1:3]
    ids = sys.argv[3:] or sorted(BIG)
    t = time.time()
    print(json.dumps(verdicts(open(patch).read(), ids, src=root, log=lambda *a: print(*a, file=sys.stderr))))
    print(f"{time.time() - t:.2f}s", file=sys.stderr)
