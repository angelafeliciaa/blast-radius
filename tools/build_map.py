"""Build maps/map.json.gz from per-worker traces written by linetrace.py.

usage: python tools/build_map.py <trace_dir> <out.json.gz>
"""

import glob
import gzip
import json
import os
import sys


def main() -> None:
    trace_dir, out = sys.argv[1:3]
    tests: list[str] = []
    line_tests: dict[str, dict[int, list[int]]] = {}
    opened: dict[str, set[int]] = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "trace_*.json"))):
        d = json.load(open(path))
        for test, files in d["lines"].items():
            t = len(tests)
            tests.append(test)
            for f, ls in files.items():
                per = line_tests.setdefault(f, {})
                for ln in ls:
                    per.setdefault(ln, []).append(t)
        index = {name: i for i, name in enumerate(tests)}
        for test, files in d["opened"].items():
            for f in files:
                opened.setdefault(f, set()).add(index[test])

    sets: dict[tuple, int] = {}
    set_list: list[list[int]] = []
    lines: dict[str, dict[str, int]] = {}
    for f, per in line_tests.items():
        out_per = {}
        for ln, ts in per.items():
            key = tuple(sorted(set(ts)))
            if key not in sets:
                sets[key] = len(set_list)
                set_list.append(list(key))
            out_per[str(ln)] = sets[key]
        lines[f] = out_per

    data = {"tests": tests, "sets": set_list, "lines": lines, "opened": {k: sorted(v) for k, v in opened.items()}}
    with gzip.open(out, "wt") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(len(tests), "tests", len(set_list), "sets", os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
