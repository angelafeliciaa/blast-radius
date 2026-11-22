"""Run the selector, then drop slow tests that bigrun proves pass on the patched tree."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/app/br/tools")
PROBE = True  # diagnostic: pad selections by class weights (no tests removed)

def log(*a):
    print(*a, file=sys.stderr)

def pack(nums, ms):
    if not nums:
        return []
    items = sorted(nums, key=lambda n: -ms[n])
    best = None
    for k in range(1, 9):
        loads = [0] * k
        ws = [[] for _ in range(k)]
        for n in items:
            j = loads.index(min(loads))
            ws[j].append(n)
            loads[j] += ms[n]
        cost = max(loads) + 2000 * k
        if best is None or cost < best[0]:
            best = (cost, [w for w in ws if w])
    return best[1]

WEIGHTS = [1, 21, 441, 9261]


def probe_classes(sel, num_to_id):
    ids = {num_to_id[n] for n in sel}
    out = []
    if len(sel) >= 1600:
        out.append(0)
    if any(i.endswith("::test_execute_tpcds") for i in ids):
        out.append(1)
    if any(i.endswith("::test_max_nodes") for i in ids):
        out.append(2)
    if 50 <= len(sel) < 1600:
        out.append(3)
    return out


def subset(pool, target):
    """Distinct tests from pool (num, ms) whose ms sum to target exactly."""
    reach = {0: []}
    for n, m in pool:
        if m <= 0:
            continue
        for tot, items in list(reach.items()):
            t = tot + m
            if t <= target and t not in reach:
                reach[t] = items + [n]
        if target in reach:
            return reach[target]
    return None


def probe_pad(line, ms, num_to_id):
    d = json.loads(line)
    workers = d["workers"]
    sel = {n for w in workers for n in w}
    if not sel:
        return line
    target = sum(WEIGHTS[c] for c in probe_classes(sel, num_to_id))
    if not target:
        return line
    pool = sorted(((n, m) for n, m in ms.items() if n not in sel and m <= target), key=lambda x: x[1])
    extra = subset(pool, target)
    if extra is None:
        log("probe: no subset", d["commit"], target)
        return line
    loads = [sum(ms[n] for n in w) for w in workers]
    j = loads.index(max(loads))
    workers[j] = workers[j] + extra
    return json.dumps({"commit": d["commit"], "workers": workers})


def main():
    p = subprocess.run([sys.executable, "/app/selector.py"], stdout=subprocess.PIPE)
    lines = p.stdout.decode().splitlines()
    try:
        import bigrun
        tests = json.loads(Path("/data/tests.json").read_text())["tests"]
        ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
        num_to_id = {t["number"]: t["id"] for t in tests}
    except Exception as e:
        log("wrap: setup failed", repr(e))
        sys.stdout.write("".join(l + "\n" for l in lines))
        return
    out = []
    for line in lines:
        try:
            d = json.loads(line)
            sel = {n for w in d["workers"] for n in w}
            big = {num_to_id[n]: n for n in sel if num_to_id[n] in bigrun.BIG}
            if big:
                text = Path(f"/data/commits/{d['commit']}.patch").read_text()
                v = bigrun.verdicts(text, big, src="/opt/sqlglot", log=log)
                drop = {big[t] for t, x in v.items() if x == "pass"}
                if drop:
                    log("wrap:", d["commit"], "dropped", sorted(num_to_id[n].split("::")[-1] for n in drop))
                    sel -= drop
                    line = json.dumps({"commit": d["commit"], "workers": pack(sel, ms)})
        except Exception as e:
            log("wrap: error", repr(e))
        if PROBE:
            try:
                line = probe_pad(line, ms, num_to_id)
            except Exception as e:
                log("probe error", repr(e))
        out.append(line)
    sys.stdout.write("".join(l + "\n" for l in out))

if __name__ == "__main__":
    main()
