"""Generate random mutation patches against the pinned sqlglot checkout, in the
style of the sample commits (operator swaps, constant flips, comment lines,
fixture edits, and combinations). For local validation only.

Usage: python mutate.py <checkout> <out_dir> <count> [seed]
"""

import difflib
import os
import random
import re
import sys

COMMENTS = [
    "# Keep in sync with the generator",
    "# Cheap check first",
    "# Kept for backwards compatibility",
    "# Intentionally not cached",
    "# Mirrors the behavior of the reference implementation",
    "# TODO: revisit",
]
SWAPS = [
    (r" == ", " != "), (r" != ", " == "), (r" < ", " <= "), (r" <= ", " < "), (r" > ", " >= "),
    (r" >= ", " > "), (r" is not ", " != "), (r" is None", " == None"), (r" is ", " == "),
    (r" and ", " or "), (r" or ", " and "), (r" not in ", " in "), (r" in ", " not in "),
    (r"\bTrue\b", "False"), (r"\bFalse\b", "True"), (r"if not ", "if "), (r"return not ", "return "),
    (r"\[0\]", "[1]"), (r"\[1\]", "[0]"), (r"\[-1\]", "[0]"), (r"\b1\b", "2"), (r"\b0\b", "1"), (r"\b2\b", "3"),
    (r" \+ 1\b", " - 1"), (r" - 1\b", " + 1"), (r"copy=False", "copy=True"), (r"copy=True", "copy=False"),
]


def diff(path, old, new):
    lines = list(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                      f"a/{path}", f"b/{path}", n=3))
    if not lines:
        return ""
    out = [f"diff --git a/{path} b/{path}\n", "index 0000000..1111111 100644\n"]
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def py_files(root, top):
    found = []
    for folder, _, files in os.walk(os.path.join(root, top)):
        for name in files:
            if name.endswith(".py"):
                found.append(os.path.relpath(os.path.join(folder, name), root))
    return sorted(found)


def code_mutation(rng, root, files):
    for _ in range(200):
        path = rng.choice(files)
        text = open(os.path.join(root, path)).read()
        lines = text.split("\n")
        index = rng.randrange(len(lines))
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(('"""', "'''")):
            continue
        options = [(a, b) for a, b in SWAPS if re.search(a, line)]
        if not options:
            continue
        a, b = rng.choice(options)
        matches = list(re.finditer(a, line))
        m = rng.choice(matches)
        new_line = line[: m.start()] + re.sub(a, b, m.group(0)) + line[m.end():]
        if new_line == line:
            continue
        lines[index] = new_line
        new_text = "\n".join(lines)
        try:
            compile(new_text, path, "exec")
        except SyntaxError:
            continue
        return path, text, new_text
    raise RuntimeError("no mutation found")


def comment_mutation(rng, root, files):
    path = rng.choice(files)
    text = open(os.path.join(root, path)).read()
    lines = text.split("\n")
    if len(lines) < 5:
        return comment_mutation(rng, root, files)
    for _ in range(100):
        index = rng.randrange(1, len(lines))
        nxt = lines[index]
        if not nxt.strip():
            continue
        indent = nxt[: len(nxt) - len(nxt.lstrip())]
        new_lines = lines[:index] + [indent + rng.choice(COMMENTS)] + lines[index:]
        new_text = "\n".join(new_lines)
        try:
            compile(new_text, path, "exec")
        except SyntaxError:
            continue
        return path, text, new_text
    raise RuntimeError("no comment spot")


def fixture_mutation(rng, root):
    folder = os.path.join(root, "tests/fixtures/optimizer")
    names = sorted(n for n in os.listdir(folder) if n.endswith(".sql"))
    path = "tests/fixtures/optimizer/" + rng.choice(names)
    text = open(os.path.join(root, path)).read()
    lines = text.split("\n")
    for _ in range(100):
        index = rng.randrange(len(lines))
        line = lines[index]
        if not line.strip() or line.startswith("#") or line.startswith("--"):
            continue
        style = rng.random()
        if style < 0.15:
            new_lines = lines[:index] + ["-- " + rng.choice(COMMENTS)[2:]] + lines[index:]
        elif style < 0.25:
            new_lines = lines[:index] + [rng.choice(COMMENTS)] + lines[index:]
        else:
            words = line.split(" ")
            pos = rng.randrange(len(words) + 1)
            token = rng.choice(["42", "x", "0", "NULL", "1"])
            words.insert(pos, token)
            new_lines = lines[:index] + [" ".join(words)] + lines[index + 1:]
        return path, text, "\n".join(new_lines)
    raise RuntimeError("no fixture spot")


def main():
    root, out_dir, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    rng = random.Random(seed)
    files = py_files(root, "sqlglot")
    os.makedirs(out_dir, exist_ok=True)
    kinds = {}
    for number in range(count):
        roll = rng.random()
        parts = []
        if roll < 0.55:
            kind = "code"
            parts.append(code_mutation(rng, root, files))
        elif roll < 0.7:
            kind = "comment"
            parts.append(comment_mutation(rng, root, files))
        elif roll < 0.85:
            kind = "fixture"
            parts.append(fixture_mutation(rng, root))
        else:
            kind = "multi"
            parts.append(comment_mutation(rng, root, files))
            parts.append(code_mutation(rng, root, files) if rng.random() < 0.5 else comment_mutation(rng, root, files))
            parts.append(fixture_mutation(rng, root))
        merged = {}
        for path, old, new in parts:
            if path in merged:
                continue
            merged[path] = (old, new)
        patch = "".join(diff(p, o, n) for p, (o, n) in sorted(merged.items()))
        name = f"m{seed:02d}{number:03d}"
        with open(os.path.join(out_dir, name + ".patch"), "w") as handle:
            handle.write(patch)
        kinds[name] = kind
    with open(os.path.join(out_dir, "kinds.txt"), "w") as handle:
        for name, kind in kinds.items():
            handle.write(f"{name} {kind}\n")


if __name__ == "__main__":
    main()
