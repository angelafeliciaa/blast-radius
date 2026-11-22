"""Per-commit cost against the floor (the real failing tests, packed the same way) with the
selector's notes, plus the total excess by the kind of rule that produced the selection.

Usage: python excess.py <set dir with tests.json answers.json> <output.jsonl> <stderr log> [top]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
import packing  # noqa: E402

folder, output, err = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
top = int(sys.argv[4]) if len(sys.argv) > 4 else 15
tests = json.loads((folder / "tests.json").read_text())["tests"]
dur = {t["number"]: packing.duration_ms(t["seconds"]) for t in tests}
answers = json.loads((folder / "answers.json").read_text())["commits"]
notes = {}
for line in err.read_text().splitlines():
    m = re.match(r"selector: (\S+): (\d+) tests, [\d.]+s \| (.*)", line)
    if m:
        notes[m.group(1)] = m.group(3)


def rule(note):
    if "ALL:" in note:
        return "all: " + note.split("ALL:", 1)[1].split(":")[-1].strip()[:40]
    kinds = []
    for key, label in (("data file", "data file"), ("diff:", "import-time diff"), ("runtime", "line coverage"),
                       ("deleted module", "deleted module"), ("header of", "def header"), ("test module level", "test module level"),
                       ("readers", "file readers")):
        if key in note:
            kinds.append(label)
    return "+".join(kinds[:2]) or "other"


rows, by_rule = [], defaultdict(float)
for line in output.read_text().splitlines():
    o = json.loads(line)
    sel = {x for w in o["workers"] for x in w}
    cost = (max(sum(dur[x] for x in w) for w in o["workers"]) + 2000 * len(o["workers"])) if o["workers"] else 0
    failing = set(answers[o["commit"]]["failing"])
    floor = packing.pack({n: dur[n] for n in failing})[0]
    miss = not failing <= sel
    note = notes.get(o["commit"], "")
    excess = (cost - floor) / 1000
    rows.append((excess, o["commit"], cost / 1000, floor / 1000, len(sel), len(failing), miss, note))
    by_rule[rule(note)] += excess
rows.sort(reverse=True)
for r in rows[:top]:
    print(f"{r[1]:28s} excess {r[0]:6.2f} cost {r[2]:6.2f} floor {r[3]:6.2f} sel {r[4]:4d} fail {r[5]:4d}{' MISS' if r[6] else ''} | {r[7][:150]}")
print(f"\ntotal cost {sum(r[2] for r in rows):.1f} floor {sum(r[3] for r in rows):.1f} excess {sum(r[0] for r in rows):.1f} misses {sum(r[6] for r in rows)}")
for name, value in sorted(by_rule.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {value:7.1f}  {name}")
