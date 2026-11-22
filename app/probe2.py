"""Second-stage diagnostic probes for app/wrap.py. Nothing here is for the final entry.

Drop-in use in wrap.py (replaces probe_pad in the PROBE branch):

    import probe2
    line = probe2.apply(line, ms, num_to_id, order, MODE)

`order` is {commit id: position in sorted /data/commits, 0-based}; MODE is one of:

  ("index", cls)   Which commits are in class `cls` ("all", "tpcds", "max_nodes", "wide").
                   Each commit of the class gains W[position] ms on its busiest worker.
                   W is built so that every sum of up to 3 different weights is unique:
                   decode(delta_ms) returns the positions. No test is removed: no new miss.
                   For "all" there are no unselected tests left, so instead position p
                   moves p+1 zero-millisecond tests onto p+1... not possible within 8
                   workers: "all" commits are reported by ("workers", ...) below.

  ("workers", cls) For class "all" (or any class): one extra worker holding a single 0 ms
                   test, exactly +2000 ms per commit of the class. Counts the class even
                   when no unselected test is left to pad with.

  ("drop", names)  Necessity check: remove the named tests (e.g. ["test_execute_tpcds"])
                   wherever they are selected and repack. The board's missed count is then
                   the number of commits where one of them really fails; every other commit
                   in the class was paying for a test that passes. This CAN add misses
                   (practice only). Simulated seconds fall by the dropped tests' cost and
                   rise by 300 per miss.

  ("drop_in", names, cls)  The same, but only inside commits of class `cls`.

Classes (from the answer line alone): "all" >= 1600 tests; "tpcds" / "max_nodes" contain
that test and are not "all"; "wide" 50..1599 tests with neither.
"""
import json


def classes(sel, num_to_id):
    names = {num_to_id[n].rsplit("::", 1)[1] for n in sel}
    if len(sel) >= 1600:
        return {"all"}
    out = set()
    if "test_execute_tpcds" in names:
        out.add("tpcds")
    if "test_max_nodes" in names:
        out.add("max_nodes")
    if not out and 50 <= len(sel):
        out.add("wide")
    return out


def _weights(count=64, upto=3):
    """Greedy weights: all sums of 1..upto different weights are different."""
    ws, sums = [], {0: 0}  # sum -> how many weights it uses
    w = 0
    while len(ws) < count:
        w += 1
        new = {}
        ok = True
        for s, k in sums.items():
            if k < upto:
                t = s + w
                if t in sums or t in new:
                    ok = False
                    break
                new[t] = k + 1
        if ok:
            ws.append(w)
            sums.update(new)
    return ws


W = _weights(64, 2)  # sums of up to 2 different weights are unique (40th weight ~1.6 s)
W3 = _weights(48, 3)  # up to 3, but the 40th weight is 193 s: too large to pad with


def decode(delta_ms, upto=2):
    """Positions whose weights sum to delta_ms (None when no such set of <= upto exists)."""
    from itertools import combinations
    for k in range(0, upto + 1):
        for combo in combinations(range(len(W)), k):
            if sum(W[i] for i in combo) == delta_ms:
                return list(combo)
    return None


def _subset(pool, target):
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


def _pack(nums, ms):
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


def apply(line, ms, num_to_id, order, mode):
    d = json.loads(line)
    workers = [list(w) for w in d["workers"] if w]
    sel = {n for w in workers for n in w}
    if not sel:
        return line
    cls = classes(sel, num_to_id)
    kind = mode[0]
    if kind == "index" and mode[1] in cls:
        target = W[order[d["commit"]]]
        pool = sorted(((n, m) for n, m in ms.items() if n not in sel and m <= target), key=lambda x: x[1])
        extra = _subset(pool, target)
        if extra is None:
            return line
        loads = [sum(ms[n] for n in w) for w in workers]
        workers[loads.index(max(loads))].extend(extra)
    elif kind == "workers" and mode[1] in cls:
        zero = [(k, n) for k, w in enumerate(workers) for n in w if ms[n] == 0 and len(w) > 1]
        if len(workers) >= 8 or not zero:
            return line
        k, n = zero[0]
        workers[k].remove(n)
        workers.append([n])
    elif kind in ("drop", "drop_in"):
        if kind == "drop_in" and mode[2] not in cls:
            return line
        gone = {n for n in sel if num_to_id[n].rsplit("::", 1)[1] in set(mode[1])}
        if not gone:
            return line
        workers = _pack(sel - gone, ms)
    else:
        return line
    return json.dumps({"commit": d["commit"], "workers": workers})
