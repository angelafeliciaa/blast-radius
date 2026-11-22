"""Turn the raw pickle of `MAPGEN_MODE=items` into app/items.pkl.gz, the per-item map the
selector uses to run a fixture-driven test on only the fixture items a change can reach.

Usage: python build_items.py <out.pkl.gz> <map_items.pkl> <map.pkl.gz>

Per test (nodeid): the CPU of its setup and of its body with no items; per fixture file it
loads: the loader, how many items, each item's own CPU and, per sqlglot line, a bit mask of
the items whose one-item run reached it and the no-item run did not; the lines the no-item
run reached ("base": the test's own assertions and set-up; an item may or may not reach them
too, the runs cannot tell); and the lines the test reaches (in this pass or in any pass of
the main map) that neither the no-item run nor any one-item run reached. A change to a base
or an unexplained line cannot be split by item.
A test is left out when any of its cut-down runs failed or it reads one file through both loaders.
"""

import gzip
import pickle
import sys


def by_path(lines) -> dict:
    found: dict[str, set[int]] = {}
    for path, line in lines:
        found.setdefault(path, set()).add(line)
    return {path: frozenset(numbers) for path, numbers in found.items()}


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: build_items.py <out.pkl.gz> <map_items.pkl> <map.pkl.gz>")
    out, raw, main_map = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(raw, "rb") as handle:
        data = pickle.load(handle)["items"]
    with gzip.open(main_map, "rb") as handle:
        the_map = pickle.load(handle)
    wanted = {nodeid: index for index, nodeid in enumerate(the_map["tests"]) if nodeid in data}
    by_index = {index: nodeid for nodeid, index in wanted.items()}
    main_lines: dict[str, set] = {nodeid: set() for nodeid in wanted}
    members = [set(group) & set(by_index) for group in the_map["sets"]]
    for path, lines in the_map["lines"].items():
        if not path.startswith("sqlglot/"):
            continue
        for line, number in lines.items():
            for index in members[number]:
                main_lines[by_index[index]].add((path, line))
    tests = {}
    skipped = []
    for nodeid, entry in sorted(data.items()):
        names = [filename for _, filename in entry["loads"]]
        if not entry.get("ok") or len(set(names)) != len(names):
            skipped.append(nodeid)
            continue
        files = {}
        for (loader, filename), info in entry["files"].items():
            lines: dict[str, dict[int, int]] = {}
            for index, touched in info["items"].items():
                for path, line in touched:
                    by_line = lines.setdefault(path, {})
                    by_line[line] = by_line.get(line, 0) | (1 << index)
            if info["dependent"] and len(info["items"]) != info["n"]:
                skipped.append(nodeid)
                files = None
                break
            files[filename] = {
                "loader": loader,
                "n": info["n"],
                "dependent": info["dependent"],
                "cpu": [info["cpu"].get(index, 0.0) for index in range(info["n"])] if info["dependent"] else [],
                "lines": lines,
            }
        if files is None:
            continue
        if nodeid not in main_lines:
            skipped.append(nodeid)  # a test the main map does not know
            continue
        explained = set(entry["base_lines"])
        for info in entry["files"].values():
            for touched in info["items"].values():
                explained |= touched
        unexplained: dict[str, list[int]] = {}
        for path, line in set(entry.get("unexplained", ())) | (main_lines[nodeid] - explained):
            unexplained.setdefault(path, []).append(line)
        tests[nodeid] = {
            "setup": round(entry["base_cpu"].get("setup", 0.0), 4),
            "base": round(entry["base_cpu"].get("call", 0.0), 4),
            "files": files,
            "unexplained": {path: frozenset(lines) for path, lines in unexplained.items()},
            "base_lines": by_path(entry["base_lines"]),
            "raw": sorted(entry.get("raw", ())),
        }
    with gzip.open(out, "wb", compresslevel=9) as handle:
        pickle.dump({"tests": tests}, handle, protocol=4)
    dependent = sum(1 for t in tests.values() if any(f["dependent"] for f in t["files"].values()))
    print(f"{len(tests)} fixture-driven tests ({dependent} split by item), {len(skipped)} left out: {skipped[:8]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
