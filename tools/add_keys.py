"""Add the key-level records of extra mapgen runs to an existing map, leaving the rest of it as is.

Usage: python add_keys.py <map.pkl.gz> <out.pkl.gz> <raw.pkl> [<raw.pkl> ...]

Keys are merged the way build_map.py merges them: a test's own records, plus what
importing its test module recorded (MAPGEN_MODE=collect), filed under
keyreaders[(class, attribute)][key].
"""

import gzip
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_map import test_key  # noqa: E402


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: add_keys.py <map.pkl.gz> <out.pkl.gz> <raw.pkl>...")
    source, out, raws = sys.argv[1], sys.argv[2], sys.argv[3:]
    with gzip.open(source, "rb") as handle:
        the_map = pickle.load(handle)
    index = {nodeid: number for number, nodeid in enumerate(the_map["tests"])}
    by_file = defaultdict(list)
    for nodeid, number in index.items():
        by_file[nodeid.split("::", 1)[0]].append(number)
    per_test = defaultdict(set)
    for raw in raws:
        with open(raw, "rb") as handle:
            data = pickle.load(handle)
        for context, found in data["results"].items():
            keys = found.get("keys") or set()
            if context[0] == "collect" and context[1]:
                targets = by_file.get(context[1], [])
            else:
                nodeid = test_key(context)
                targets = [index[nodeid]] if nodeid in index else []
            for number in targets:
                per_test[number] |= keys
    key_tests = defaultdict(lambda: defaultdict(set))
    for number, keys in per_test.items():
        for qual, attr, name in keys:
            key_tests[(qual, attr)][name].add(number)
    sets = {tuple(members): number for number, members in enumerate(the_map["sets"])}
    added = 0
    for pair, by_name in key_tests.items():
        entry = the_map["keyreaders"].setdefault(pair, {})
        for name, members in by_name.items():
            if name in entry:
                members = members | set(the_map["sets"][entry[name]])
            key = tuple(sorted(members))
            if key not in sets:
                sets[key] = len(the_map["sets"])
                the_map["sets"].append(key)
            entry[name] = sets[key]
            added += 1
    with gzip.open(out, "wb", compresslevel=9) as handle:
        pickle.dump(the_map, handle, protocol=4)
    print(f"add_keys: {added} key records over {len(key_tests)} attributes from {len(raws)} runs", file=sys.stderr)


if __name__ == "__main__":
    main()
