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
from plan import Planner  # noqa: E402

# Expected CPU seconds here per recorded second, and the prior that a commit
# with candidates breaks one of them. Tune from labeled data.
CPU_RATIO = float(os.environ.get("BR_CPU_RATIO", "0.6"))
P_FAIL = float(os.environ.get("BR_P_FAIL", "0.5"))


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
import json, os, signal, sys, time, unittest
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

ms = {tid: m for tid, m in jobs}
planner = Planner({i: m for i, (_, m) in enumerate(jobs)}, cpu_ratio=opts["ratio"],
                  p_fail=opts["p_fail"], order=list(range(len(jobs))))
loader = unittest.defaultTestLoader

class Slow(Exception):
    pass

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
    probe = Planner({i: candidates[t] for i, t in enumerate(ids)}, cpu_ratio=CPU_RATIO, p_fail=P_FAIL)
    if not run_all and probe.next() is None:
        return keep_all
    try:
        new_files = patched_files(patch_text, src)
    except (Unsupported, OSError, UnicodeDecodeError) as e:
        if log:
            log("runtests: unsupported patch:", e)
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
        opts = {"ratio": CPU_RATIO, "p_fail": P_FAIL, "run_all": run_all, "min_limit": 5.0, "factor": 10.0}
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
