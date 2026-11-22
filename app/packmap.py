"""Rewrite the map for a fast start: python packmap.py <map.pkl.gz> <out>

Run while the image is built. The map as committed is one gzipped pickle whose sets of
test numbers (over a million ints) are all unpickled at every start. The packed file
keeps the same data in pieces that load on demand (see analyze.load_map):

  sets        one buffer of 16-bit test numbers plus the offset where each set starts
  lines       one pickle per source file
  keyreaders  one pickle per (class, attribute)
  the rest    one pickle per section

Every piece is read back and compared with the original before the file is kept.
"""

from __future__ import annotations

import array
import gzip
import pickle
import sys

PER_KEY = ("lines", "keyreaders")  # sections stored one pickle per key


def pack(data: dict) -> dict:
    numbers = array.array("H")
    offsets = array.array("I", [0])
    for members in data["sets"]:
        numbers.extend(members)  # raises if a test number does not fit in 16 bits
        offsets.append(len(numbers))
    packed = {
        "sets": (numbers.tobytes(), offsets.tobytes()),
        "sections": {
            name: pickle.dumps(value, protocol=4)
            for name, value in data.items()
            if name != "sets" and name not in PER_KEY
        },
    }
    for name in PER_KEY:
        packed[name] = {key: pickle.dumps(value, protocol=4) for key, value in data.get(name, {}).items()}
    return packed


def pack_words(checkout: str, out: str) -> None:
    """The identifier index of the checkout's Python files (candidates.Analyzer.word_index),
    stored for candidates.Words: the selector then looks words up instead of reading and
    splitting every file again in each run that needs one."""
    import os

    import candidates

    checkout = os.path.abspath(checkout)
    index = candidates.Analyzer(None, checkout).build_word_index()
    paths: list[str] = []
    numbers: dict[str, int] = {}
    letters: dict[str, dict[str, bytes]] = {}
    for word, lines in index.items():
        pairs = array.array("I")
        for path, line in lines:
            if path not in numbers:
                numbers[path] = len(paths)
                paths.append(path)
            pairs.append(numbers[path])
            pairs.append(line)
        letters.setdefault(word[:1], {})[word] = pairs.tobytes()
    with open(out, "wb") as handle:
        pickle.dump({"checkout": checkout, "paths": paths,
                     "letters": {letter: pickle.dumps(words, protocol=4) for letter, words in letters.items()}},
                    handle, protocol=4)
    stored = candidates.load_words(out, checkout)
    if stored is None or any(stored.get(word) != lines for word, lines in index.items()) or stored.get("no such word!", 7) != 7:
        sys.exit("packmap: stored words differ")
    print(f"packmap: {len(index)} words in {len(paths)} files", file=sys.stderr)


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--words":
        pack_words(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) != 3:
        sys.exit("usage: packmap.py <map.pkl.gz> <out> | packmap.py --words <checkout> <out>")
    source, out = sys.argv[1], sys.argv[2]
    with gzip.open(source, "rb") as handle:
        data = pickle.load(handle)
    with open(out, "wb") as handle:
        pickle.dump(pack(data), handle, protocol=4)

    import analyze

    loaded = analyze.load_packed(out)
    if set(loaded) != set(data):
        sys.exit("packmap: sections differ")
    for name, value in data.items():
        if name == "sets":
            if len(loaded["sets"]) != len(value) or any(
                tuple(loaded["sets"][number]) != tuple(members) for number, members in enumerate(value)
            ):
                sys.exit("packmap: sets differ")
        elif dict(loaded[name]) != value if name in PER_KEY else loaded[name] != value:
            sys.exit(f"packmap: {name} differs")
    print(f"packmap: {len(data['sets'])} sets, {len(data)} sections", file=sys.stderr)


if __name__ == "__main__":
    main()
