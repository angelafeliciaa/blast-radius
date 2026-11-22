"""Build maps/big.json.gz for tools/bigchecks.py from tools/tpc_probe.py runs.

usage: python tools/build_big_map.py <out.json.gz> <cov.json> <plain.json> [<plain2.json> ...]

<cov.json>    probe run with COV=1: the line map.
<plain*.json> probe runs without coverage (ideally with different
              PYTHONHASHSEED): honest timings. A plan hash that differs
              between any two runs is dropped, so that query is always executed.
A result hash that differs between runs aborts the build.
"""

import gzip
import json
import os
import sys


def main() -> None:
    out, cov_path, *plain = sys.argv[1:]
    cov = json.load(open(cov_path))
    runs = [json.load(open(p)) for p in plain]
    data = {}
    for suite in ("tpcds", "tpch"):
        qs = [dict(q) for q in runs[0][suite]["q"]]
        unstable = 0
        for other in runs[1:] + [cov]:
            for q, o in zip(qs, other[suite]["q"]):
                for key in ("sql", "duck", "duck_opt", "res"):
                    assert q.get(key) == o.get(key), (suite, key)
                if q.get("plan") != o.get("plan"):
                    q["plan"] = None
        unstable = sum(1 for q in qs if "res" in q and q.get("plan") is None)
        data[suite] = {"q": qs, "lines": cov[suite]["lines"]}
        print(suite, len(qs), "queries,", unstable, "with an unstable plan hash")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with gzip.open(out, "wt") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
