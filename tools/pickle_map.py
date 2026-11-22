"""Lines of the pinned checkout that run while process-pool data is pickled or unpickled.

    PYTHONHASHSEED=0 python tools/pickle_map.py /opt/sqlglot app/pickle_lines.pkl.gz

sqlglot expressions cross process pools through serde: unpickling builds each node with
its class's constructor and links it with set/append. When an edit makes that raise in the
parent (a worker's result) or a worker (its arguments), concurrent.futures marks the whole
pool broken, and every later test that shares the pool (TestOptimizer's cls.pool) fails
without running the edited line. Coverage and the in-process check runs cannot see that, so
the selector treats an edit to any line recorded here as reaching every test of a class that
keeps a pool across its tests.

Runs every test module that uses a process pool, with the pool replaced by an in-process
stand-in that round-trips through pickle (as mapgen_plugin does), and records the lines
executed inside those round trips only.
"""

import concurrent.futures
import concurrent.futures.process
import gzip
import os
import pickle
import sys
import unittest


def main(checkout: str, out: str) -> int:
    root = os.path.realpath(checkout) + os.sep
    os.chdir(root)
    sys.path.insert(0, root)
    os.environ["SKIP_INTEGRATION"] = "0"

    mon = sys.monitoring
    tool = 4
    mon.use_tool_id(tool, "picklemap")
    seen: set = set()

    def on_line(code, line):
        if code.co_filename.startswith(root):
            seen.add((code.co_filename[len(root):], line))
        return mon.DISABLE

    def on_start(code, offset):
        if code.co_filename.startswith(root):
            seen.add((code.co_filename[len(root):], code.co_firstlineno))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.register_callback(tool, mon.events.PY_START, on_start)
    round_trips = [0]

    def trip(value):
        mon.set_events(tool, mon.events.LINE | mon.events.PY_START)
        try:
            return pickle.loads(pickle.dumps(value))
        finally:
            mon.set_events(tool, 0)
            round_trips[0] += 1

    class Recording:
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
            if initializer is not None:
                initializer(*trip(initargs))

        def submit(self, fn, /, *args, **kwargs):
            future = concurrent.futures.Future()
            try:
                fn, args, kwargs = trip((fn, args, kwargs))
                future.set_result(trip(fn(*args, **kwargs)))
            except BaseException as error:  # noqa: BLE001
                future.set_exception(error)
            return future

        def map(self, fn, *iterables, timeout=None, chunksize=1):
            return [self.submit(fn, *args).result() for args in zip(*iterables)]

        def shutdown(self, wait=True, *, cancel_futures=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    concurrent.futures.ProcessPoolExecutor = Recording
    concurrent.futures.process.ProcessPoolExecutor = Recording

    modules = []
    for folder, _, files in os.walk(os.path.join(root, "tests")):
        for name in sorted(files):
            if name.startswith("test_") and name.endswith(".py"):
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
                if "ProcessPoolExecutor" in text or "multiprocessing" in text:
                    modules.append(os.path.relpath(path, root)[:-3].replace(os.sep, "."))
    if not modules:
        print("no test module uses a process pool", file=sys.stderr)
        return 1
    suite = unittest.defaultTestLoader.loadTestsFromNames(sorted(modules))
    result = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
    if result.errors or result.failures:
        for test, trace in (result.errors + result.failures)[:5]:
            print(f"{test}: {trace.splitlines()[-1]}", file=sys.stderr)
        return 1
    if not round_trips[0]:
        print("no pool round trip ran", file=sys.stderr)
        return 1
    lines: dict[str, list[int]] = {}
    for path, line in seen:
        lines.setdefault(path, []).append(line)
    data = {"modules": sorted(modules), "lines": {path: sorted(found) for path, found in lines.items()}}
    with gzip.open(out, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(modules)} modules, {result.testsRun} tests, {round_trips[0]} round trips; "
          f"{len(seen)} lines in {len(lines)} files")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
