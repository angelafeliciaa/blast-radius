"""Merge raw mapgen pickles into the compact map the selector loads at run time.

Usage: python build_map.py <out.pickle.gz> <raw.pkl> [<raw.pkl> ...]

Every raw pickle comes from one run of sqlglot's suite on the pinned checkout
with tools/mapgen_plugin.py (or, for map_collect.pkl, one import of each test module). Their per-test data is unioned, so a test depends
on a line, file or class attribute if any run saw it touch it.
"""

import gzip
import pickle
import sys
from collections import defaultdict


def test_key(context):
    kind, nodeid = context
    if kind in ("test", "setup", "call", "teardown"):
        return nodeid
    return None


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: build_map.py <out> <raw.pkl>...")
    out, raws = sys.argv[1], sys.argv[2:]
    tests: dict[str, int] = {}
    per_test_lines = defaultdict(set)
    per_test_files = defaultdict(set)
    per_test_attrs = defaultdict(set)
    per_test_keys = defaultdict(set)
    keyed_attrs = set()
    import_lines = defaultdict(set)
    import_attrs = set()
    cpu: dict[str, tuple] = {}
    classes = {}
    class_attrs = {}
    import_codes = set()
    profile: dict[str, set] = {}
    failed = {}
    collected = []
    for raw in raws:
        with open(raw, "rb") as handle:
            data = pickle.load(handle)
        classes.update(data["classes"])
        class_attrs.update(data.get("class_attrs", {}))
        import_codes |= set(data.get("import_codes", ()))
        for site, differed in data.get("profile", {}).items():
            profile.setdefault(site, set()).update(differed)
        failed.update(data["outcomes"])
        if data.get("keyed"):
            keyed_attrs |= set(data["keyed"])
        if raw.endswith("map_fork.pkl"):  # one test per process: setup includes setUpClass
            for nodeid, phases in data["cpu"].items():
                cpu[nodeid] = (round(phases.get("setup", 0.0), 4), round(phases.get("call", 0.0), 4))
        for context, found in data["results"].items():
            nodeid = test_key(context)
            if context[0] == "collect" and context[1]:
                collected.append((context[1], found))  # one test module's import, given to its tests below
                continue
            if nodeid is None:
                for path, line in found["lines"]:
                    import_lines[path].add(line)
                import_attrs |= found["attrs"]
                continue
            index = tests.setdefault(nodeid, len(tests))
            per_test_lines[index] |= found["lines"]
            files = found["files"]
            if raw.endswith("map_fresh.pkl"):
                # each test imports its own module there: pytest's assertion rewriting reads the
                # module's source, which is not a data dependency
                files = {f for f in files if not (f.endswith(".py") and f.startswith("tests/")) and "__pycache__" not in f}
            per_test_files[index] |= files
            per_test_attrs[index] |= found["attrs"]
            per_test_keys[index] |= found.get("keys", set())

    # What importing a test module touched (MAPGEN_MODE=collect, one fork per module) is
    # something each of its tests needs: if that import breaks, they all fail.
    by_file = defaultdict(list)
    for nodeid, index in tests.items():
        by_file[nodeid.split("::", 1)[0]].append(index)
    for path, found in collected:
        # importing reads the modules' own source (assertion rewriting): not a data dependency
        files = {f for f in found["files"] if not f.endswith(".py") and "__pycache__" not in f}
        for index in by_file.get(path, ()):
            per_test_lines[index] |= found["lines"]
            per_test_files[index] |= files
            per_test_attrs[index] |= found["attrs"]
            per_test_keys[index] |= found.get("keys", set())

    # Class level setup (setUpClass) runs in the first test of a class during a
    # normal run; give every test of the class the setup and teardown lines.
    by_class = defaultdict(list)
    for nodeid, index in tests.items():
        by_class[nodeid.rsplit("::", 1)[0]].append(index)

    sets: dict[tuple, int] = {}

    def intern(members) -> int:
        key = tuple(sorted(members))
        return sets.setdefault(key, len(sets))

    line_tests = defaultdict(lambda: defaultdict(set))
    for index, lines in per_test_lines.items():
        for path, line in lines:
            line_tests[path][line].add(index)
    file_tests = defaultdict(set)
    for index, files in per_test_files.items():
        for path in files:
            file_tests[path].add(index)
    attr_tests = defaultdict(set)
    for index, attrs in per_test_attrs.items():
        for key in attrs:
            attr_tests[key].add(index)

    key_tests = defaultdict(lambda: defaultdict(set))
    for index, keys in per_test_keys.items():
        for qual, attr, name in keys:
            key_tests[(qual, attr)][name].add(index)

    compact = {
        "tests": [nodeid for nodeid, _ in sorted(tests.items(), key=lambda kv: kv[1])],
        "lines": {
            path: {line: intern(members) for line, members in lines.items()}
            for path, lines in line_tests.items()
        },
        "files": {path: intern(members) for path, members in file_tests.items()},
        "attrs": {key: intern(members) for key, members in attr_tests.items()},
        "keyreaders": {
            pair: {name: intern(members) for name, members in by_key.items()}
            for pair, by_key in key_tests.items()
        },
        "import_lines": {path: sorted(lines) for path, lines in import_lines.items()},
        "import_attrs": sorted(import_attrs),
        "classes": classes,
        "class_attrs": class_attrs,
        "import_codes": sorted(import_codes),
        "profile": {site: sorted(differed) for site, differed in profile.items()},
        "cpu": cpu,  # nodeid -> (setup seconds, call seconds) on the map machine
        "failed_in_mapgen": failed,
    }
    compact["sets"] = [None] * len(sets)
    for key, number in sets.items():
        compact["sets"][number] = key
    with gzip.open(out, "wb", compresslevel=9) as handle:
        pickle.dump(compact, handle, protocol=4)
    print(
        f"{len(tests)} tests, {len(sets)} distinct sets, "
        f"{sum(len(v) for v in compact['lines'].values())} covered lines, "
        f"{len(compact['attrs'])} class attributes, {len(compact['files'])} files",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
