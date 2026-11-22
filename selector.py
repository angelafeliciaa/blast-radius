"""Blast Radius selector.

For each patch: work out which statements changed (AST diff, so comment-only
edits fall out), map them to tests through a per-test line-coverage map built
at the pinned sqlglot commit, then split the chosen tests over workers.
"""

import ast
import difflib
import gzip
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools"))
import bigrun  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", "/data"))
SRC = Path(os.environ.get("BR_SRC", "/sqlglot"))
MAP = Path(os.environ.get("BR_MAP", Path(__file__).parent / "maps" / "map.json.gz"))


def log(*a):
    print(*a, file=sys.stderr)


# ---------------------------------------------------------------- patches


def parse_patch(text: str):
    """Return {path: (old_path, hunks)}; hunks are lists of (tag, line) with
    the old start line."""
    files = {}
    cur = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)", ln)
            cur = {"old": m.group(1), "new": m.group(2), "hunks": [], "new_file": False, "deleted": False}
            files[m.group(2)] = cur
        elif cur is not None and ln.startswith("new file mode"):
            cur["new_file"] = True
        elif cur is not None and ln.startswith("deleted file mode"):
            cur["deleted"] = True
        elif cur is not None and ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            hunk = {"old_start": int(m.group(1)), "lines": []}
            cur["hunks"].append(hunk)
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "diff --git ")):
                l2 = lines[i]
                if l2.startswith("\\"):
                    pass
                elif l2[:1] in (" ", "-", "+"):
                    hunk["lines"].append((l2[0], l2[1:]))
                elif l2 == "":
                    # could be an empty context line with trailing space stripped,
                    # or the end of the patch
                    if i + 1 < len(lines) and lines[i + 1][:1] in (" ", "-", "+", "\\"):
                        hunk["lines"].append((" ", ""))
                    else:
                        pass
                i += 1
            continue
        i += 1
    return files


def apply_hunks(old_lines: list[str], hunks) -> tuple[list[str], set[int], list[int]]:
    """Apply hunks; return new lines, set of removed old line numbers (1-based),
    and list of insertion points (old line number before which lines were
    inserted)."""
    new = []
    removed: set[int] = set()
    inserts: list[int] = []
    pos = 1
    for h in hunks:
        start = h["old_start"]
        if start == 0:
            start = 1
        while pos < start:
            new.append(old_lines[pos - 1])
            pos += 1
        for tag, text in h["lines"]:
            if tag == " ":
                new.append(old_lines[pos - 1] if pos - 1 < len(old_lines) else text)
                pos += 1
            elif tag == "-":
                removed.add(pos)
                pos += 1
            else:
                new.append(text)
                inserts.append(pos)
    while pos <= len(old_lines):
        new.append(old_lines[pos - 1])
        pos += 1
    return new, removed, inserts


# ---------------------------------------------------------------- AST diff


def dump(node) -> str:
    return ast.dump(node, annotate_fields=False, include_attributes=False)


BODY_FIELDS = ("body", "orelse", "finalbody", "handlers")


def end(node) -> int:
    return getattr(node, "end_lineno", node.lineno) or node.lineno


class Change:
    def __init__(self, old_node, new_node, scope, parent_old, prev_old, next_old):
        self.old = old_node
        self.new = new_node
        self.scope = scope  # "func" or "module" or "class:<name>"
        self.parent_old = parent_old
        self.prev_old = prev_old
        self.next_old = next_old


def diff_bodies(old_body, new_body, scope, parent_old, out):
    od = [dump(n) for n in old_body]
    nd = [dump(n) for n in new_body]
    sm = difflib.SequenceMatcher(a=od, b=nd, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        olds = old_body[i1:i2]
        news = new_body[j1:j2]
        prev_old = old_body[i1 - 1] if i1 > 0 else None
        next_old = old_body[i2] if i2 < len(old_body) else None
        if tag == "replace" and len(olds) == len(news):
            for o, n in zip(olds, news):
                diff_node(o, n, scope, parent_old, prev_old, next_old, out)
            continue
        for o in olds:
            out.append(Change(o, None, scope, parent_old, prev_old, next_old))
        for n in news:
            out.append(Change(None, n, scope, parent_old, prev_old, next_old))


def header_dump(node) -> str:
    fields = []
    for name, value in ast.iter_fields(node):
        if name in BODY_FIELDS:
            continue
        if isinstance(value, list):
            fields.append((name, [dump(v) if isinstance(v, ast.AST) else repr(v) for v in value]))
        elif isinstance(value, ast.AST):
            fields.append((name, dump(value)))
        else:
            fields.append((name, repr(value)))
    return repr(fields)


def diff_node(o, n, scope, parent_old, prev_old, next_old, out):
    if type(o) is not type(n) or header_dump(o) != header_dump(n):
        out.append(Change(o, n, scope, parent_old, prev_old, next_old))
        return
    if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)):
        sub = "func"
    elif isinstance(o, ast.ClassDef):
        sub = "func" if scope == "func" else f"class:{o.name}"
    else:
        sub = scope
    for field in BODY_FIELDS:
        ob = getattr(o, field, None)
        nb = getattr(n, field, None)
        if ob is None and nb is None:
            continue
        if isinstance(ob, list) and isinstance(nb, list):
            diff_bodies(ob, nb, sub, o, out)


# ---------------------------------------------------------------- selection


class Selector:
    def __init__(self):
        self.tests = json.loads((DATA / "tests.json").read_text())["tests"]
        self.by_id = {t["id"]: t["number"] for t in self.tests}
        self.ms = {t["number"]: round(t["seconds"] * 1000) for t in self.tests}
        with gzip.open(MAP, "rt") as fh:
            m = json.load(fh)
        # coverage test index -> tests.json numbers
        self.cov_num = []
        for name in m["tests"]:
            num = self.by_id.get(name)
            if num is None:
                # parametrized / subtest ids: match the base id
                base = name.split("[")[0]
                num = self.by_id.get(base)
            self.cov_num.append(num)
        self.sets = m["sets"]
        self.lines = m["lines"]
        self.opened = m["opened"]
        self._set_cache = {}
        self._name_index = None
        self.all_tests = set(self.by_id.values())
        self.tests_by_file = {}
        for t in self.tests:
            self.tests_by_file.setdefault(t["file"], set()).add(t["number"])

    # coverage lookups
    def set_nums(self, sid: int) -> set[int]:
        r = self._set_cache.get(sid)
        if r is None:
            r = {self.cov_num[i] for i in self.sets[sid] if self.cov_num[i] is not None}
            self._set_cache[sid] = r
        return r

    def covering(self, path: str, lo: int, hi: int) -> set[int]:
        per = self.lines.get(path)
        out: set[int] = set()
        if not per:
            return out
        for ln in range(lo, hi + 1):
            sid = per.get(str(ln))
            if sid is not None:
                out |= self.set_nums(sid)
        return out

    def covering_file(self, path: str) -> set[int]:
        out: set[int] = set()
        for sid in self.lines.get(path, {}).values():
            out |= self.set_nums(sid)
        return out

    def name_index(self):
        if self._name_index is None:
            idx: dict[str, list[tuple[str, int]]] = {}
            for p in SRC.joinpath("sqlglot").rglob("*.py"):
                rel = str(p.relative_to(SRC))
                for i, line in enumerate(p.read_text().split("\n"), 1):
                    for w in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)):
                        idx.setdefault(w, []).append((rel, i))
            self._name_index = idx
        return self._name_index

    def name_refs(self, names: set[str]) -> set[int]:
        idx = self.name_index()
        out: set[int] = set()
        for nm in names:
            for path, ln in idx.get(nm, ()):
                out |= self.covering(path, ln, ln)
        return out

    # per file kinds
    def py_source(self, path, info, why):
        old_text = (SRC / path).read_text() if (SRC / path).exists() else ""
        old_lines = old_text.split("\n")
        new_lines, removed, inserts = apply_hunks(old_lines, info["hunks"])
        new_text = "\n".join(new_lines)
        try:
            new_tree = ast.parse(new_text)
        except SyntaxError:
            why.append(f"{path}: syntax error -> all")
            return set(self.all_tests)
        try:
            old_tree = ast.parse(old_text)
        except SyntaxError:
            return set(self.all_tests)
        if dump(old_tree) == dump(new_tree):
            why.append(f"{path}: no AST change")
            return set()
        changes: list[Change] = []
        diff_bodies(old_tree.body, new_tree.body, "module", None, changes)
        sel: set[int] = set()
        for ch in changes:
            sel |= self.change_tests(path, ch, why)
        pickling = path.endswith("serde.py") or re.search(r"__(reduce|reduce_ex|getstate|setstate|getnewargs)__", new_text) and any(
            re.search(r"__(reduce|reduce_ex|getstate|setstate|getnewargs)__", (ast.unparse(c.old) if c.old else "") + (ast.unparse(c.new) if c.new else "")) for c in changes)
        if pickling:
            # spawned process pools pickle expressions; tracing can't see it
            sel |= self.mentions("check_file") | self.mentions("_mp_execute") | self.mentions("self.pool")
            why.append(f"{path}: pickling -> pool users")
        return sel

    def change_tests(self, path, ch: Change, why) -> set[int]:
        sel: set[int] = set()
        if ch.scope == "func":
            if ch.old is not None:
                sel |= self.covering(path, ch.old.lineno, end(ch.old))
            else:
                if ch.prev_old is not None:
                    sel |= self.covering(path, end(ch.prev_old), end(ch.prev_old))
                    sel |= self.covering(path, ch.prev_old.lineno, end(ch.prev_old))
                if ch.next_old is not None:
                    sel |= self.covering(path, ch.next_old.lineno, ch.next_old.lineno)
                if ch.parent_old is not None:
                    sel |= self.covering(path, ch.parent_old.lineno, ch.parent_old.lineno)
                    if not sel:
                        sel |= self.covering(path, ch.parent_old.lineno, end(ch.parent_old))
            why.append(f"{path}: func-level stmt -> {len(sel)}")
            return sel
        # module / class level statement
        node = ch.old if ch.old is not None else ch.new
        names = set()
        for n in (ch.old, ch.new):
            if n is None:
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            for sub in ast.walk(n) if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else []:
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.id)
                elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.attr)
                elif isinstance(sub, ast.alias):
                    names.add((sub.asname or sub.name).split(".")[0])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for sub in n.decorator_list:
                    pass
        if ch.old is not None:
            sel |= self.covering(path, ch.old.lineno, end(ch.old))
        lazy = self.mentions("test_lazy_load") | {self.by_id.get("tests/dialects/test_dialect.py::TestDialect::test_lazy_load")}
        sel |= {n for n in lazy if n is not None}
        dialect = self.dialect_of(path)
        if dialect:
            sel |= self.dialect_users(dialect)
            why.append(f"{path}: {ch.scope} dialect={dialect} -> {len(sel)}")
            return sel
        if ch.scope.startswith("class:"):
            cls = ch.scope.split(":", 1)[1]
            sel |= self.name_refs({cls} | self.subclasses(cls))
            if ch.parent_old is not None:
                sel |= self.covering(path, ch.parent_old.lineno, end(ch.parent_old))
        else:
            sel |= self.name_refs(names)
        if not sel:
            sel |= self.covering_file(path)
        why.append(f"{path}: {ch.scope} stmt names={sorted(names)[:6]} -> {len(sel)}")
        return sel

    DIALECT_DIRS = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/", "sqlglot/tokenizers/")

    def dialect_of(self, path: str):
        for d in self.DIALECT_DIRS:
            if path.startswith(d):
                stem = Path(path).stem
                if stem in ("__init__", "dialect", "base"):
                    return None
                return stem
        return None

    def dialect_children(self):
        if getattr(self, "_dchildren", None) is None:
            parent = {}
            for p in SRC.joinpath("sqlglot/dialects").glob("*.py"):
                try:
                    tree = ast.parse(p.read_text())
                except SyntaxError:
                    continue
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        for b in node.bases:
                            if isinstance(b, ast.Name):
                                parent[p.stem] = (node.name, b.id)
            name_to_stem = {cls: stem for stem, (cls, _) in parent.items()}
            children = {}
            for stem, (_, base) in parent.items():
                if base in name_to_stem:
                    children.setdefault(name_to_stem[base], set()).add(stem)
            self._dchildren = children
        return self._dchildren

    def dialect_users(self, stem: str) -> set[int]:
        stems = {stem}
        todo = [stem]
        ch = self.dialect_children()
        while todo:
            s = todo.pop()
            for c in ch.get(s, ()):
                if c not in stems:
                    stems.add(c)
                    todo.append(c)
        out: set[int] = set()
        for s in stems:
            out |= self.tests_by_file.get(f"tests/dialects/test_{s}.py", set())
            out |= self.mentions(s)
            out |= self.mentions(s.upper())
            for d in self.DIALECT_DIRS:
                p = f"{d}{s}.py"
                out |= self.covering_file(p)
            # data fixtures that mention the dialect
            for f, idxs in self.opened.items():
                fp = SRC / f
                if fp.suffix in (".sql", ".json", ".py", ".txt") and fp.exists():
                    try:
                        if re.search(rf"\b{re.escape(s)}\b", fp.read_text(errors="ignore"), re.I):
                            out |= {self.cov_num[i] for i in idxs if self.cov_num[i] is not None}
                    except OSError:
                        pass
        return out

    def subclasses(self, cls: str) -> set[int]:
        return set()

    def test_funcs(self):
        """[(test number, source text of the test function, class-level text)]"""
        if getattr(self, "_funcs", None) is None:
            self._funcs = []
            for f in self.tests_by_file:
                p = SRC / f
                if not p.exists():
                    continue
                text = p.read_text()
                src_lines = text.split("\n")
                tree = ast.parse(text)
                for node in tree.body:
                    if not isinstance(node, ast.ClassDef):
                        continue
                    other = []
                    for fn in node.body:
                        seg = "\n".join(src_lines[fn.lineno - 1 : end(fn)])
                        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test"):
                            num = self.by_id.get(f"{f}::{node.name}::{fn.name}")
                            if num is not None:
                                self._funcs.append((num, seg, node.name, f))
                        else:
                            other.append(seg)
                    self._funcs.append((None, "\n".join(other), node.name, f))
        return self._funcs

    def mentions(self, word: str) -> set[int]:
        """Tests whose function (or class setup code) mentions word."""
        if not word:
            return set()
        out = set()
        funcs = self.test_funcs()
        for num, seg, cls, f in funcs:
            if word in seg:
                if num is not None:
                    out.add(num)
                else:
                    out |= {n for n, _, c, ff in funcs if c == cls and ff == f and n is not None}
        return out

    def test_file(self, path, info, why) -> set[int]:
        nums = self.tests_by_file.get(path)
        if nums is None:
            # helper module: every test file importing it
            mod = Path(path).stem
            out = set()
            for f, ns in self.tests_by_file.items():
                p = SRC / f
                if p.exists() and re.search(rf"\b{re.escape(mod)}\b", p.read_text()):
                    out |= ns
            why.append(f"{path}: test helper -> {len(out)}")
            return out
        old_text = (SRC / path).read_text()
        new_lines, removed, inserts = apply_hunks(old_text.split("\n"), info["hunks"])
        try:
            new_tree = ast.parse("\n".join(new_lines))
            old_tree = ast.parse(old_text)
        except SyntaxError:
            return set(nums)
        if dump(old_tree) == dump(new_tree):
            return set()
        changes: list[Change] = []
        diff_bodies(old_tree.body, new_tree.body, "module", None, changes)
        # find enclosing test functions of changed nodes
        out = set()
        funcs = []
        for node in ast.walk(old_tree):
            if isinstance(node, ast.ClassDef):
                for f in node.body:
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        funcs.append((f"{path}::{node.name}::{f.name}", f.lineno, end(f)))
        for ch in changes:
            n = ch.old or ch.prev_old or ch.next_old or ch.parent_old
            hit = None
            if n is not None:
                for tid, lo, hi in funcs:
                    if lo <= n.lineno <= hi:
                        hit = tid
            if hit and hit in self.by_id:
                out.add(self.by_id[hit])
            else:
                return set(nums)
        return out

    def select(self, patch_text: str):
        why: list[str] = []
        files = parse_patch(patch_text)
        sel: set[int] = set()
        for path, info in files.items():
            if path.startswith("sqlglot/") and path.endswith(".py"):
                sel |= self.py_source(path, info, why)
            elif path.startswith("tests/") and path.endswith(".py"):
                sel |= self.test_file(path, info, why)
            elif path.startswith("tests/"):
                rel = path
                hits = set()
                for i in self.opened.get(rel, []):
                    n = self.cov_num[i]
                    if n is not None:
                        hits.add(n)
                stem = Path(path).name.split(".")[0]
                hits |= self.mentions(f'"{stem}"') | self.mentions(f"'{stem}'") | self.mentions(f"/{stem}")
                if not hits:
                    hits |= self.mentions(stem)
                if not hits:
                    hits |= self.mentions(Path(path).parent.name)
                why.append(f"{path}: data -> {len(hits)}")
                sel |= hits
            else:
                why.append(f"{path}: ignored")
        return sel, why

    def split(self, nums: set[int]) -> list[list[int]]:
        if not nums:
            return []
        items = sorted(nums, key=lambda n: -self.ms[n])
        best = None
        for k in range(1, 9):
            loads = [0] * k
            ws = [[] for _ in range(k)]
            for n in items:
                j = loads.index(min(loads))
                ws[j].append(n)
                loads[j] += self.ms[n]
            cost = max(loads) + 2000 * k
            if best is None or cost < best[0]:
                best = (cost, [w for w in ws if w])
        return best[1]


def main() -> None:
    s = Selector()
    for patch in sorted((DATA / "commits").glob("*.patch")):
        try:
            text = patch.read_text()
            sel, why = s.select(text)
            num_to_id = {t["number"]: t["id"] for t in s.tests}
            big = {num_to_id[n]: n for n in sel if num_to_id[n] in bigrun.BIG}
            if big:
                try:
                    v = bigrun.verdicts(text, big, src=str(SRC), log=log)
                    sel -= {big[t] for t, x in v.items() if x == "pass"}
                except Exception as e:
                    log(patch.stem, "bigrun error", repr(e))
        except Exception as e:  # never crash: fall back to everything
            log(patch.stem, "ERROR", repr(e))
            sel, why = set(s.all_tests), []
        workers = s.split(sel)
        log(patch.stem, len(sel), sum(s.ms[n] for n in sel) / 1000, "; ".join(why)[:300])
        print(json.dumps({"commit": patch.stem, "workers": workers}))


if __name__ == "__main__":
    main()
