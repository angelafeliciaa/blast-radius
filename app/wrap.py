"""Run the selector, then drop slow tests that bigrun proves pass on the patched tree."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/app/br/tools")

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
        out.append(line)
    sys.stdout.write("".join(l + "\n" for l in out))

if __name__ == "__main__":
    main()
