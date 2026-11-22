"""Turn the comparison profile of one map pass (MAPGEN_PROFILE=1, one fork per test) into
app/compare.pkl.gz: per comparison site, which tests' runs reached its statement and in
which tests each sibling operator would have given a different answer.

Usage: python build_profile.py <out.pkl.gz> <map_profile.pkl> <map.pkl.gz> <checkout>

A site is "path:line:col" of a single-operator comparison inside a function. Its statement is
the innermost statement that holds it, the span the selector looks up when that comparison
changes. Tests are bit positions in the order of the main map's test list.
"""

import ast
import gzip
import os
import pickle
import sys


def statement_spans(source: str) -> dict[tuple[int, int], tuple[int, int]]:
    """(line, col) of every single-operator comparison inside a function -> the (first, last)
    lines the selector looks up when it changes (app/analyze.py locate): its innermost
    statement inside the innermost function, or the body of a lambda that holds it directly."""
    tree = ast.parse(source)
    found: dict[tuple[int, int], tuple[int, int]] = {}

    def walk(node, scope, stmt):
        for name, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if not isinstance(child, ast.AST):
                    continue
                here_scope, here_stmt = scope, stmt
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and name == "body":
                    here_scope, here_stmt = node, None  # inside this function's body
                if isinstance(child, ast.stmt):
                    here_stmt = child
                if isinstance(child, ast.Compare) and len(child.ops) == 1 and here_scope is not None:
                    if here_stmt is not None:
                        found[(child.lineno, child.col_offset)] = (here_stmt.lineno, here_stmt.end_lineno)
                    elif isinstance(here_scope, ast.Lambda):
                        found[(child.lineno, child.col_offset)] = (here_scope.body.lineno, here_scope.body.end_lineno)
                walk(child, here_scope, here_stmt)

    walk(tree, None, None)
    return found


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit("usage: build_profile.py <out.pkl.gz> <map_profile.pkl> <map.pkl.gz> <checkout>")
    out, raw, main_map, checkout = sys.argv[1:5]
    with open(raw, "rb") as handle:
        data = pickle.load(handle)
    with gzip.open(main_map, "rb") as handle:
        tests = pickle.load(handle)["tests"]
    by_test = data.get("profile_tests")
    if not by_test:
        sys.exit("the raw pickle has no per-test profile (run the profile pass with this plugin)")
    position = {nodeid: index for index, nodeid in enumerate(tests)}
    lines_of: dict[tuple[str, int], int] = {}  # (path, line) -> mask of tests that ran it in this pass
    passed = 0
    for context, found in data["results"].items():
        if context[0] != "test" or context[1] not in position or context[1] not in by_test:
            continue
        if context[1] in data.get("outcomes", {}):
            continue  # a test that did not pass in this pass: its profile says nothing
        passed |= 1 << position[context[1]]
        bit = 1 << position[context[1]]
        for key in found["lines"]:
            lines_of[key] = lines_of.get(key, 0) | bit
    spans: dict[str, dict] = {}
    sites: dict[str, tuple] = {}
    differ: dict[str, dict[str, int]] = {}
    for nodeid, profile in by_test.items():
        if nodeid not in position or not passed >> position[nodeid] & 1:
            continue
        bit = 1 << position[nodeid]
        for site, differed in profile.items():
            entry = differ.setdefault(site, {})
            for alt in differed:
                entry[alt] = entry.get(alt, 0) | bit
    skipped = 0
    for site, by_alt in differ.items():
        path, line, col = site.rsplit(":", 2)
        if path not in spans:
            try:
                with open(os.path.join(checkout, path), encoding="utf-8") as handle:
                    spans[path] = statement_spans(handle.read())
            except (OSError, SyntaxError):
                spans[path] = {}
        span = spans[path].get((int(line), int(col)))
        if span is None:
            skipped += 1
            continue
        reach = 0
        for number in range(span[0], span[1] + 1):
            reach |= lines_of.get((path, number), 0)
        sites[site] = (reach, by_alt)
    with gzip.open(out, "wb", compresslevel=9) as handle:
        pickle.dump({"tests": tests, "passed": passed, "sites": sites}, handle, protocol=4)
    print(f"{len(sites)} comparison sites ({skipped} outside functions or not found), "
          f"{bin(passed).count('1')} of {len(tests)} tests profiled", file=sys.stderr)


if __name__ == "__main__":
    main()
