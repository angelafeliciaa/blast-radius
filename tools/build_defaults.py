"""Turn the default-reliance passes (tools/gen_raw_maps.sh ONLY=defaults and ONLY=defaults_seq:
MAPGEN_DEFAULTS=1, one fork per test and all tests in one process) into app/defaults.pkl.gz.

Usage: python build_defaults.py <out.pkl.gz> <map.pkl.gz> <map_defaults.pkl> <map_defaults_seq.pkl>

A def is keyed (path, first line incl. decorators, name). For every def in sqlglot/ with a
constant default value the file holds, as bit masks over the main map's test order:
  called[def]            tests that passed in the fork pass and called the def there
  used[def + (param,)]   tests in which, in either pass, some call of the def started with
                         the parameter holding its default constant (left out, or passed equal)
A test in called[def] and not in used[def, param] passed that parameter itself, with another
value, in every call it made: a new default value of it changes nothing that test runs.
"""

import gzip
import pickle
import sys


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    out_path, map_path, fork_path, seq_path = sys.argv[1:5]
    with gzip.open(map_path, "rb") as handle:
        tests = pickle.load(handle)["tests"]
    index = {nodeid: i for i, nodeid in enumerate(tests)}
    called: dict[tuple, int] = {}
    used: dict[tuple, int] = {}
    counts = []
    for path, is_fork in ((fork_path, True), (seq_path, False)):
        with open(path, "rb") as handle:
            raw = pickle.load(handle)
        seen = 0
        for key, slot in raw["results"].items():
            if key[0] != "test" or key[1] not in index:
                continue
            if "called" not in slot or "defaults" not in slot:
                print(f"{path}: no default-reliance data for {key[1]} (pass run without MAPGEN_DEFAULTS=1?)", file=sys.stderr)
                return 1
            seen += 1
            bit = 1 << index[key[1]]
            for entry in slot["defaults"]:
                used[entry] = used.get(entry, 0) | bit
            if is_fork and key[1] not in raw["outcomes"]:
                for entry in slot["called"]:
                    called[entry] = called.get(entry, 0) | bit
        counts.append(seen)
        if seen < 0.99 * len(tests):
            print(f"{path}: only {seen} of {len(tests)} tests recorded", file=sys.stderr)
            return 1
    with gzip.open(out_path, "wb") as handle:
        pickle.dump({"tests": tests, "called": called, "used": used}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    spare = sum(1 for entry, mask in used.items() if called.get(entry[:3], 0) & ~mask)
    print(f"{out_path}: {len(called)} defs called, {len(used)} (def, parameter) pairs used at their default, "
          f"{spare} of them with a caller that never does; tests recorded: fork {counts[0]}, seq {counts[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
