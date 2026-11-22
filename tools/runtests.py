"""Run a commit's cheap candidate tests against the patched tree.

    from runtests import run_candidates
    r = run_candidates(patch_text, {test_id: recorded_ms, ...})
    r["select"]   test ids that must stay in the answer
    r["drop"]     test ids proven to pass: remove them
    r["all"]      True when sqlglot itself no longer imports: select every test

How: a symlink copy of the checkout with the patched files is made under /tmp,
and one fresh Python process runs the candidates there with unittest, cheapest
first. tools/plan.py decides after every test whether to go on. It stops at
the first failure (the rest is then included unrun, which costs no more than
running it) or when running the rest would cost more than including it.

Do not pass the five slow tests here; tools/bigrun.py decides those.
Anything this module cannot handle returns every candidate in "select", which
is what the caller had before calling.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bigrun import Unsupported, apply, parse_patch  # noqa: E402
from plan import Planner, pack_cost  # noqa: E402

# Expected CPU seconds here per recorded second, and the prior that a commit
# with candidates breaks one of them. Tune from labeled data.
CPU_RATIO = float(os.environ.get("BR_CPU_RATIO", "0.9"))
P_FAIL = float(os.environ.get("BR_P_FAIL", "0.5"))
# A patch that only edits files under tests/ (a fixture, an expected output)
# nearly always breaks the test that reads it: running first just pays twice.
P_FAIL_TESTS_ONLY = 0.95
# CPU seconds to start the process and import sqlglot and the test module.
STARTUP_S = float(os.environ.get("BR_STARTUP", "0.5"))
# Test modules that import pandas and duckdb cost this much more to load.
HEAVY_MODULES = ("tests/test_optimizer.py", "tests/test_executor.py")
HEAVY_STARTUP_S = 1.3


def overlay(src: str, dest: str, new_files: dict[str, str]) -> None:
    """dest mirrors src through symlinks, except for the patched files."""
    real_dirs = {""}
    for rel in new_files:
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:i]))
    for d in sorted(real_dirs):
        os.makedirs(os.path.join(dest, d), exist_ok=True)
        for name in os.listdir(os.path.join(src, d)):
            rel = f"{d}/{name}" if d else name
            if rel in real_dirs or rel in new_files or name == "__pycache__":
                continue
            os.symlink(os.path.join(src, d, name), os.path.join(dest, d, name))
    for rel, text in new_files.items():
        with open(os.path.join(dest, rel), "w") as fh:
            fh.write(text)


def patched_files(patch_text: str, src: str) -> dict[str, str]:
    files = parse_patch(patch_text)
    if not files:
        raise Unsupported("no files")
    out = {}
    for rel, hunks in files.items():
        if rel.startswith("/") or ".." in rel.split("/"):
            raise Unsupported("path")
        old = open(os.path.join(src, rel)).read()
        out[rel] = "\n".join(apply(old.split("\n"), hunks)[0])
    return out


_CHILD = r"""
import json, os, pickle, signal, sys, time, unittest
tools, jobs, opts = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
sys.path.insert(0, os.getcwd())
sys.path.insert(1, tools)
from plan import Planner

def say(**kw):
    sys.stdout.write(json.dumps(kw) + "\n"); sys.stdout.flush()

try:
    import sqlglot
except BaseException as e:
    say(broken=repr(e)[:200]); sys.exit(0)

# Tests that fan work out to a spawned process pool get an in-process pool:
# same functions, same patched code, none of the cost of re-importing sqlglot
# in every worker (and no dependence on process limits).
import concurrent.futures as _cf

class InlinePool:
    def __init__(self, *a, initializer=None, initargs=(), **kw):
        if initializer:
            initializer(*initargs)
    def submit(self, fn, *a, **kw):
        f = _cf.Future()
        try:
            # A real pool pickles the call and its result; a patch can break that.
            a, kw = pickle.loads(pickle.dumps((a, kw)))
            f.set_result(pickle.loads(pickle.dumps(fn(*a, **kw))))
        except BaseException as e:
            if isinstance(e, Slow):
                raise
            f.set_exception(e)
        return f
    def map(self, fn, *its, **kw):
        return [fn(*x) for x in zip(*its)]
    def shutdown(self, *a, **kw):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

class Slow(BaseException):
    pass

_cf.ProcessPoolExecutor = InlinePool
import concurrent.futures.process as _cfp
_cfp.ProcessPoolExecutor = InlinePool

ms = {tid: m for tid, m in jobs}
planner = Planner({i: m for i, (_, m) in enumerate(jobs)}, cpu_ratio=opts["ratio"],
                  p_fail=opts["p_fail"], order=list(range(len(jobs))))
loader = unittest.defaultTestLoader

def fire(*_):
    raise Slow()

class Result(unittest.TestResult):
    def bad(self):
        return len(self.failures) + len(self.errors) + len(self.unexpectedSuccesses)

result = Result()
signal.signal(signal.SIGPROF, fire)
bad_modules = {}
for i, (tid, m) in enumerate(jobs):
    if not opts["run_all"] and planner.next() is None:
        break
    path, _, rest = tid.partition("::")
    module = path[:-3].replace("/", ".")
    t0 = time.process_time()
    failed = False
    if module in bad_modules:
        failed = True
    else:
        try:
            suite = loader.loadTestsFromName(module + "." + rest.replace("::", "."))
            if loader.errors:
                loader.errors.clear(); raise ImportError(tid)
            before = result.bad()
            signal.setitimer(signal.ITIMER_PROF, max(opts["min_limit"], opts["factor"] * m / 1000))
            try:
                suite.run(result)
            except Slow:
                failed = True
            finally:
                signal.setitimer(signal.ITIMER_PROF, 0)
            failed = failed or result.bad() != before or result.testsRun == 0
        except BaseException as e:
            signal.setitimer(signal.ITIMER_PROF, 0)
            failed = True
            if module not in sys.modules:
                bad_modules[module] = 1
    cpu = time.process_time() - t0
    planner.report(i, failed, cpu * 1000)
    say(id=tid, failed=failed, cpu=round(cpu, 4))
say(done=True)
"""


def run_candidates(
    patch_text: str,
    candidates: dict[str, int],
    src: str = "/sqlglot",
    tmp: str | None = None,
    log=None,
    run_all: bool = False,
) -> dict:
    ids = sorted(candidates, key=lambda t: (candidates[t], t))
    keep_all = {"select": list(ids), "drop": [], "all": False, "cpu": 0.0, "ran": 0}
    if not ids:
        return keep_all
    # Not worth starting a process when the policy would include everything anyway.
    try:
        new_files = patched_files(patch_text, src)
    except (Unsupported, OSError, UnicodeDecodeError) as e:
        if log:
            log("runtests: unsupported patch:", e)
        return keep_all
    p_fail = P_FAIL_TESTS_ONLY if all(f.startswith("tests/") for f in new_files) else P_FAIL
    by_num = {i: candidates[t] for i, t in enumerate(ids)}
    include = pack_cost(by_num)
    total_s = sum(candidates.values()) / 1000
    # Expected cost of running: startup, the tests, and (when one fails) the
    # answer we would have printed anyway.
    startup = STARTUP_S + (HEAVY_STARTUP_S if any(t.startswith(HEAVY_MODULES) for t in ids) else 0)
    run = 1000 * (startup + CPU_RATIO * total_s * (1 - p_fail + p_fail * 0.25)) + p_fail * include
    if not run_all and (run >= include or Planner(by_num, cpu_ratio=CPU_RATIO, p_fail=p_fail).next() is None):
        return keep_all

    # Keep one class's tests together so setUpClass runs once, cheapest class first.
    first: dict[str, int] = {}
    for pos, t in enumerate(ids):
        first.setdefault(t.rsplit("::", 1)[0], pos)
    ids.sort(key=lambda t: (first[t.rsplit("::", 1)[0]], candidates[t], t))

    work = tempfile.mkdtemp(prefix="brrun", dir=tmp)
    start = time.time()
    try:
        overlay(src, work, new_files)
        env = dict(os.environ, PYTHONPATH=work, SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")
        opts = {"ratio": CPU_RATIO, "p_fail": p_fail, "run_all": run_all, "min_limit": 5.0, "factor": 10.0}
        total = sum(candidates.values()) / 1000
        try:
            p = subprocess.run(
                [sys.executable, "-c", _CHILD, HERE, json.dumps([[t, candidates[t]] for t in ids]),
                 json.dumps(opts)],
                env=env, capture_output=True, text=True, cwd=work, timeout=30 + 10 * total,
            )
            stdout = p.stdout
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    failed: list[str] = []
    passed: list[str] = []
    cpu = 0.0
    done = False
    for line in stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "broken" in msg:
            if log:
                log("runtests: sqlglot does not import:", msg["broken"])
            return {"select": list(ids), "drop": [], "all": True, "cpu": 0.0, "ran": 0}
        if msg.get("done"):
            done = True
        elif "id" in msg:
            (failed if msg["failed"] else passed).append(msg["id"])
            cpu += msg["cpu"]
    seen = set(failed) | set(passed)
    unrun = [t for t in ids if t not in seen]
    if not done and unrun:
        # The process died or hung inside the next test: that test fails.
        failed.append(unrun.pop(0))
    select = failed + unrun if (failed or unrun) else []
    if log:
        log(f"runtests: ran {len(seen)}/{len(ids)} failed {len(failed)} unrun {len(unrun)} "
            f"cpu {cpu:.2f}s wall {time.time() - start:.2f}s")
    return {"select": select, "drop": passed, "all": False, "cpu": cpu, "ran": len(seen),
            "failed": failed}


if __name__ == "__main__":
    # python tools/runtests.py <sqlglot_root> <patch> <tests.json> <test id> ...
    root, patch, tests = sys.argv[1:4]
    ms = {t["id"]: round(t["seconds"] * 1000) for t in json.load(open(tests))["tests"]}
    print(json.dumps(run_candidates(open(patch).read(), {t: ms[t] for t in sys.argv[4:]}, src=root,
                                    log=lambda *a: print(*a, file=sys.stderr))))
