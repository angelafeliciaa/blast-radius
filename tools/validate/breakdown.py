"""Per-commit cost of an output against the ideal (exactly the failing tests).

Usage: python breakdown.py <set dir with tests.json answers.json output.jsonl> [run.err]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
import packing  # noqa: E402

folder = Path(sys.argv[1])
tests = json.loads((folder / "tests.json").read_text())["tests"]
dur = {t["number"]: packing.duration_ms(t["seconds"]) for t in tests}
answers = json.loads((folder / "answers.json").read_text())["commits"]
notes = {}
err = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "run.err"
if err.exists():
    for line in err.read_text().splitlines():
        if line.startswith("selector: ") and ": " in line[10:]:
            commit, _, rest = line[10:].partition(": ")
            notes[commit] = rest[:150]
rows = []
for line in (folder / "output.jsonl").read_text().splitlines():
    o = json.loads(line)
    sel = {x for w in o["workers"] for x in w}
    cost = (max(sum(dur[x] for x in w) for w in o["workers"]) + 2000 * len(o["workers"])) if o["workers"] else 0
    failing = set(answers[o["commit"]]["failing"])
    ideal = packing.pack({n: dur[n] for n in failing})[0]
    miss = "MISS" if not failing <= sel else ""
    rows.append((cost - ideal, o["commit"], answers[o["commit"]]["kind"], cost / 1000, ideal / 1000, len(sel), len(failing), miss))
for row in sorted(rows, reverse=True):
    print(f"{row[1]} {row[2]:8s} cost {row[3]:7.2f} ideal {row[4]:7.2f} sel {row[5]:4d} fail {row[6]:4d} {row[7]} | {notes.get(row[1], '')}")
print(f"total {sum(r[3] for r in rows):.1f} ideal {sum(r[4] for r in rows):.1f} misses {sum(1 for r in rows if r[7])}")
