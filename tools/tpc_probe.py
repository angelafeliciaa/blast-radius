"""Record, per TPC query at the pinned commit, what tools/bigchecks.py compares
against: hashes of the query text, its DuckDB transpilation, its plan and its
result, and (with COV=1) which sqlglot lines it runs while planning and while
executing.

usage: [COV=1] python tools/tpc_probe.py <sqlglot_root> <out.json>
"""
import json, os, pickle, sys, time

root, out_path = sys.argv[1:3]
os.environ["SKIP_INTEGRATION"] = "0"
sys.path.insert(0, root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
COV = os.environ.get("COV") == "1"
if COV:
    import coverage

    cov = coverage.Coverage(data_file="/tmp/tpc.cov", source=[root + "/sqlglot"], config_file=False)
    cov.start()

    def ctx(name):
        cov.switch_context(name)
else:
    def ctx(name):
        pass

import bigchecks as bc
from sqlglot import parse_one, transpile
from tests.helpers import load_sql_fixture_pairs

out = {}
for suite in ("tpch", "tpcds"):
    qs = []
    for i, (meta, sql, optimized) in enumerate(load_sql_fixture_pairs(bc._FIX[suite][1])):
        ctx(f"{suite}:{i}:p")
        q = {"sql": bc._h(sql), "duck": bc._h(transpile(sql, write="duckdb")[0])}
        if suite == "tpch":
            q["duck_opt"] = bc._h(transpile(optimized, write="duckdb")[0])
        e = pickle.loads(pickle.dumps(parse_one(sql)))
        if meta.get("execute"):
            names, plan, q["plan"] = bc.plan_query(suite, e)
            for n in names:
                bc._table(suite, n)
            ctx(f"{suite}:{i}:x")
            t = time.process_time()
            q["res"] = bc.run_plan(suite, names, plan)
            q["ms"] = round((time.process_time() - t) * 1000)
        ctx("")
        qs.append(q)
    out[suite] = {"q": qs}

if COV:
    cov.stop()
    data = cov.get_data()
    marker = "/sqlglot/sqlglot/"
    for suite in out:
        out[suite]["lines"] = {}
    for f in data.measured_files():
        rel = f[f.index(marker) + len(marker) - len("sqlglot/"):]
        for ln, ctxs in data.contexts_by_lineno(f).items():
            for c in ctxs:
                if not c:
                    continue
                suite, i, phase = c.split(":")
                slot = out[suite]["lines"].setdefault(rel, {}).setdefault(str(ln), [[], []])
                slot[0 if phase == "p" else 1].append(int(i))
    for suite in out:
        for per in out[suite]["lines"].values():
            for slot in per.values():
                slot[0].sort(); slot[1].sort()
json.dump(out, open(out_path, "w"), separators=(",", ":"))
for suite in out:
    qs = out[suite]["q"]
    print(suite, len(qs), "queries", sum("res" in q for q in qs), "executed", sum(q.get("ms", 0) for q in qs), "ms", file=sys.stderr)
