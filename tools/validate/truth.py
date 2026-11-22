"""Ground truth for validation patches: apply each to a copy of the checkout and run
sqlglot's whole suite the way the labels were made. Runs inside the dev image.

Usage: python truth.py <commits_dir> <tests.json> <out answers.json> [parallel]
"""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/app")
import patchfile  # noqa: E402

CHECKOUT = "/opt/sqlglot"


def run_one(patch_path, slot):
    work = f"/tmp/w{slot}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    shutil.rmtree(work)
    shutil.copytree(CHECKOUT, work, symlinks=True)
    text = open(patch_path).read()
    for fp in patchfile.parse(text):
        old = open(os.path.join(work, fp.old_path)).read() if fp.old_path else None
        new = patchfile.apply(old, fp)
        target = os.path.join(work, fp.path)
        if new is None:
            os.unlink(target)
        else:
            with open(target, "w") as handle:
                handle.write(new)
    out = f"/tmp/fail{slot}.json"
    env = dict(os.environ, PYTHONPATH=work + ":/tools", SKIP_INTEGRATION="0", FAILREC_OUT=out,
               PYTHONDONTWRITEBYTECODE="1")
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "failrec",
             "--timeout=60", "tests"],
            cwd=work, env=env, capture_output=True, text=True, timeout=600,
        )
        data = json.load(open(out))
        os.unlink(out)
    except Exception as error:  # timeouts and crashes are recorded, not hidden
        tail = ""
        try:
            tail = (proc.stdout[-800:] + proc.stderr[-800:])
        except Exception:
            pass
        data = {"failed": [], "collect": [], "exit": -1, "error": repr(error) + " " + tail}
    data["seconds"] = round(time.time() - started, 1)
    return data


def main():
    commits_dir, tests_json, out_path = sys.argv[1:4]
    parallel = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    tests = json.load(open(tests_json))["tests"]
    number = {t["id"]: t["number"] for t in tests}
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t["number"])
    patches = sorted(p for p in os.listdir(commits_dir) if p.endswith(".patch"))
    kinds = {}
    kinds_path = os.path.join(commits_dir, "kinds.txt")
    if os.path.exists(kinds_path):
        kinds = dict(line.split() for line in open(kinds_path) if line.strip())
    answers = {}

    def job(item):
        index, name = item
        return name, run_one(os.path.join(commits_dir, name), index % parallel)

    # one slot per worker thread: index % parallel is unique among running jobs only if
    # jobs are handed out in order, so give each thread its own slot instead
    from queue import Queue
    slots = Queue()
    for slot in range(parallel):
        slots.put(slot)

    def job2(name):
        slot = slots.get()
        try:
            return name, run_one(os.path.join(commits_dir, name), slot)
        finally:
            slots.put(slot)

    from concurrent.futures import as_completed

    with ThreadPoolExecutor(parallel) as pool:
        futures = [pool.submit(job2, name) for name in patches]
        for future in as_completed(futures):
            name, data = future.result()
            failing = set()
            for nodeid in data["failed"]:
                if nodeid in number:
                    failing.add(number[nodeid])
            for nodeid in data["collect"]:
                failing.update(by_file.get(nodeid, []))
            if data["exit"] == -1:  # the suite never finished: count every test as failing
                failing = set(number.values())
            commit = name[: -len(".patch")]
            answers[commit] = {"failing": sorted(failing), "kind": kinds.get(commit, "?")}
            print(f"{commit}: {len(failing)} failing, exit {data['exit']}, {data['seconds']}s {data.get('error', '')}",
                  file=sys.stderr, flush=True)
    with open(out_path, "w") as handle:
        json.dump({"commits": answers}, handle, indent=1)


if __name__ == "__main__":
    main()
