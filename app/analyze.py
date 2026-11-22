"""Static analysis: which tests can a patch affect?

Every function returns a superset guess plus notes on how it was reached. The
map (see tools/build_map.py) says, for the unmodified checkout, which tests
executed each line, opened each file and read each class attribute.
"""

from __future__ import annotations

import ast
import collections.abc
import functools
import gzip
import os
import pickle
import re
import sys
import types
from dataclasses import dataclass, field

import patchfile

ALL = "ALL"  # marker: could affect any test


@dataclass
class Region:
    start: tuple[int, int]  # (line, utf8 col) in the old file, inclusive
    end: tuple[int, int]  # exclusive end; equal to start for a pure insertion


@dataclass
class FileVerdict:
    path: str
    kind: str  # "none", "lines", "import", "all", ...
    tests: set[int] = field(default_factory=set)
    everything: bool = False
    notes: list[str] = field(default_factory=list)
    import_time: bool = False
    import_bindings: list[tuple[str | None, str]] = field(default_factory=list)


class Pickled(collections.abc.Mapping):
    """A mapping whose values arrive pickled and are unpickled the first time they are read."""

    def __init__(self, blobs: dict):
        self._blobs = blobs
        self._values: dict = {}

    def __getitem__(self, key):
        try:
            return self._values[key]
        except KeyError:
            value = self._values[key] = pickle.loads(self._blobs[key])
            return value

    def __iter__(self):
        return iter(self._blobs)

    def __len__(self) -> int:
        return len(self._blobs)


class PackedSets:
    """The map's sets of test numbers, as slices of one buffer of 16-bit numbers."""

    def __init__(self, numbers: bytes, offsets: bytes):
        self._numbers = memoryview(numbers).cast("H")
        self._offsets = memoryview(offsets).cast("I")

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, number: int):
        if not 0 <= number < len(self._offsets) - 1:
            raise IndexError(number)
        return self._numbers[self._offsets[number] : self._offsets[number + 1]]


def load_packed(path) -> Pickled:
    """The map as app/packmap.py wrote it: the same data, each piece unpickled when first read."""
    with open(path, "rb") as handle:
        packed = pickle.load(handle)
    data = Pickled(packed["sections"])
    data._values["sets"] = PackedSets(*packed["sets"])
    data._blobs["sets"] = b""
    for name in ("lines", "keyreaders"):
        data._values[name] = Pickled(packed[name])
        data._blobs[name] = b""
    return data


def load_map(path, packed=None):
    """The map's data: from the packed file when there is one that loads, else from the gzipped pickle."""
    if packed is not None and os.path.exists(packed):
        try:
            return load_packed(packed)
        except Exception as error:
            print(f"selector: cannot load {packed}: {error!r}", file=sys.stderr, flush=True)
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


class Map:
    def __init__(self, data):
        self.data = data
        self.tests: list[str] = data["tests"]
        self.sets = data["sets"]  # number -> the test indices of that set (any iterable)
        self.lines: dict[str, dict[int, int]] = data["lines"]
        self.files: dict[str, int] = data["files"]
        self.test_file = [nodeid.split("::", 1)[0] for nodeid in self.tests]
        self._attrs_by_class: dict[str, list[str]] | None = None
        self._subclasses: dict[str, set[str]] | None = None

    # read on demand: with the packed map a section is only unpickled when something asks for it
    @functools.cached_property
    def attrs(self) -> dict[tuple[str, str], int]:
        return self.data["attrs"]

    @functools.cached_property
    def import_lines(self) -> dict[str, list[int]]:
        return self.data["import_lines"]

    @functools.cached_property
    def import_codes(self) -> set[tuple]:
        return {tuple(code) for code in self.data.get("import_codes", ())}

    @functools.cached_property
    def classes(self) -> dict[str, list[str]]:
        return self.data["classes"]

    def tests_on_lines(self, path: str, first: int, last: int) -> set[int]:
        found: set[int] = set()
        by_line = self.lines.get(path)
        if not by_line:
            return found
        for line in range(first, last + 1):
            number = by_line.get(line)
            if number is not None:
                found.update(self.sets[number])
        return found

    def any_coverage(self, path: str, first: int, last: int) -> bool:
        by_line = self.lines.get(path, {})
        imported = self.import_lines.get(path, ())
        return any(line in by_line for line in range(first, last + 1)) or any(
            first <= line <= last for line in imported
        )

    def file_readers(self, path: str) -> set[int]:
        number = self.files.get(path)
        return set(self.sets[number]) if number is not None else set()

    def attr_readers(self, owner: str, name: str) -> set[int]:
        number = self.attrs.get((owner, name))
        return set(self.sets[number]) if number is not None else set()

    def subclasses(self, owner: str) -> set[str]:
        if self._subclasses is None:
            self._subclasses = {}
            for qual, mro in self.classes.items():
                for base in mro:
                    self._subclasses.setdefault(base, set()).add(qual)
        return self._subclasses.get(owner, {owner})

    def class_users(self, owner: str) -> set[int]:
        """Tests that read any recorded attribute of the class or a subclass."""
        if self._attrs_by_class is None:
            self._attrs_by_class = {}
            for (qual, name) in self.attrs:
                self._attrs_by_class.setdefault(qual, []).append(name)
        found: set[int] = set()
        for qual in self.subclasses(owner) | set(self.classes.get(owner, [owner])):
            for name in self._attrs_by_class.get(qual, ()):
                found |= self.attr_readers(qual, name)
        return found

    def tests_in_file(self, path: str) -> set[int]:
        return {i for i, f in enumerate(self.test_file) if f == path}


# ---------------------------------------------------------------- regions


def _byte_col(line: str, char_col: int) -> int:
    return len(line[:char_col].encode("utf-8"))


def regions(old_text: str, patch: patchfile.FilePatch) -> list[Region]:
    """Changed spans of each hunk, in old-file coordinates, trimmed to the characters that differ."""
    old_lines = old_text.splitlines(keepends=True)
    found: list[Region] = []
    for hunk in patch.hunks:
        old_no = hunk.old_start if hunk.old_len > 0 else hunk.old_start + 1
        removed: list[int] = []
        added: list[str] = []

        def flush():
            if not removed and not added:
                return
            if removed:
                first = removed[0]
                old_block = "".join(old_lines[n - 1] for n in removed)
            else:
                first = old_no
                old_block = ""
            new_block = "".join(a + "\n" for a in added)
            prefix = 0
            limit = min(len(old_block), len(new_block))
            while prefix < limit and old_block[prefix] == new_block[prefix]:
                prefix += 1
            suffix = 0
            while (
                suffix < limit - prefix
                and old_block[len(old_block) - 1 - suffix] == new_block[len(new_block) - 1 - suffix]
            ):
                suffix += 1
            start = _offset_to_pos(old_lines, first, old_block, prefix)
            end = _offset_to_pos(old_lines, first, old_block, len(old_block) - suffix)
            if not removed:
                start = end = (first, 0)
            found.append(Region(start, end))
            removed.clear()
            added.clear()

        for line in hunk.lines:
            tag, body = line[:1], line[1:]
            if tag == " ":
                flush()
                old_no += 1
            elif tag == "-":
                if added and not removed:
                    flush()
                removed.append(old_no)
                old_no += 1
            elif tag == "+":
                added.append(body)
        flush()
    return found


def _offset_to_pos(old_lines, first_line, block, offset):
    line = first_line
    consumed = 0
    for text in block.splitlines(keepends=True):
        if consumed + len(text) > offset or (consumed + len(text) == offset and not text.endswith("\n")):
            source = old_lines[line - 1] if line - 1 < len(old_lines) else ""
            return (line, _byte_col(source, offset - consumed))
        consumed += len(text)
        line += 1
    return (line, 0)


# ---------------------------------------------------------------- python


def _pos(node) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if getattr(node, "lineno", None) is None or getattr(node, "end_lineno", None) is None:
        return None
    return (node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)


def _contains(span, region: Region) -> bool:
    (start, end) = span
    if region.start == region.end:
        return start <= region.start and region.end <= end and start != region.start
    return start <= region.start and region.end <= end


def chain_for(tree: ast.AST, region: Region) -> list[ast.AST]:
    chain = [tree]
    node = tree
    while True:
        nxt = None
        for child in ast.iter_child_nodes(node):
            span = _pos(child)
            if span is None:
                # arguments, comprehension, match_case... look one level deeper
                for grand in ast.iter_child_nodes(child):
                    gspan = _pos(grand)
                    if gspan is not None and _contains(gspan, region):
                        nxt = grand
                        break
                if nxt is not None:
                    break
                continue
            if _contains(span, region):
                nxt = child
                break
        if nxt is None:
            return chain
        chain.append(nxt)
        node = nxt


def _body_span(node):
    if isinstance(node, ast.Lambda):
        return _pos(node.body)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
        first = node.body[0]
        last = node.body[-1]
        # the gap right after the "def ...:" line belongs to the body too
        return ((node.lineno, node.col_offset + 1), (last.end_lineno, last.end_col_offset))
    return None


def _is_docstring(node, parent) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and isinstance(parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and parent.body
        and parent.body[0] is node
    )


def code_equal(a: types.CodeType, b: types.CodeType) -> bool:
    if (
        a.co_code != b.co_code
        or a.co_names != b.co_names
        or a.co_varnames != b.co_varnames
        or a.co_freevars != b.co_freevars
        or a.co_cellvars != b.co_cellvars
        or a.co_flags != b.co_flags
        or a.co_argcount != b.co_argcount
        or a.co_posonlyargcount != b.co_posonlyargcount
        or a.co_kwonlyargcount != b.co_kwonlyargcount
        or a.co_name != b.co_name
        or a.co_qualname != b.co_qualname
        or a.co_exceptiontable != b.co_exceptiontable
        or len(a.co_consts) != len(b.co_consts)
    ):
        return False
    for x, y in zip(a.co_consts, b.co_consts):
        if isinstance(x, types.CodeType) or isinstance(y, types.CodeType):
            if not (isinstance(x, types.CodeType) and isinstance(y, types.CodeType) and code_equal(x, y)):
                return False
        elif type(x) is not type(y) or x != y:
            if not (x != x and y != y):  # NaN constants
                return False
    return True


def same_bytecode(old_text: str, new_text: str, filename: str) -> bool | None:
    """True when both sources compile to the same code apart from line numbers."""
    try:
        old_code = compile(old_text, filename, "exec", dont_inherit=True)
    except SyntaxError:
        return None
    try:
        new_code = compile(new_text, filename, "exec", dont_inherit=True)
    except SyntaxError:
        return False
    return code_equal(old_code, new_code)


IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def bound_names(stmt: ast.stmt) -> set[str]:
    names: set[str] = set()
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(stmt.name)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for alias in stmt.names:
            names.add((alias.asname or alias.name).split(".")[0])
    else:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.attr)
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
                base = node.value
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value if isinstance(base, ast.Subscript) else base
                    if isinstance(base, ast.Attribute):
                        names.add(base.attr)
                        break
                if isinstance(base, ast.Name):
                    names.add(base.id)
    return names


def module_name(path: str) -> str:
    name = path[:-3].replace("/", ".")
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    return name


# ---------------------------------------------------------------- AST diff


@dataclass
class Change:
    """One difference between the old and new tree, located in the old tree."""

    chain: list  # [(node, field_name_leading_to_next)] from the module down to the changed node
    node: ast.AST  # old node that differs (or the parent whose list changed)
    inserted: list = field(default_factory=list)  # new statements inserted under `node`
    new: object = None  # the new node replacing `node`, when the types differ


def _dump(node) -> str:
    return ast.dump(node, include_attributes=False) if isinstance(node, ast.AST) else repr(node)


def _source_lines(text: str) -> list[str]:
    """The text line by line, with the line numbers the parser gives its nodes."""
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def ast_changes(old: ast.AST, new: ast.AST, old_text: str | None = None, new_text: str | None = None) -> list[Change]:
    """The differences between two trees. With the texts they were parsed from, statements
    written with the same characters are not walked: they have the same tree."""
    import difflib

    out: list[Change] = []
    old_lines = new_lines = None
    if old_text is not None and new_text is not None:
        old_lines, new_lines = _source_lines(old_text), _source_lines(new_text)

    def same_source(a, b) -> bool:
        try:
            if a.col_offset != b.col_offset or a.end_col_offset != b.end_col_offset:
                return False
            first_a, first_b = a.lineno, b.lineno
            for decorator in getattr(a, "decorator_list", ()):
                first_a = min(first_a, decorator.lineno)
            for decorator in getattr(b, "decorator_list", ()):
                first_b = min(first_b, decorator.lineno)
            return old_lines[first_a - 1 : a.end_lineno] == new_lines[first_b - 1 : b.end_lineno]
        except (AttributeError, TypeError):  # a node without a position
            return False

    def walk(a, b, chain):
        if type(a) is not type(b):
            out.append(Change(chain, a, new=b))
            return
        if old_lines is not None and isinstance(a, ast.stmt) and same_source(a, b):
            return
        for name in a._fields:
            x = getattr(a, name, None)
            y = getattr(b, name, None)
            here = chain + [(a, name)]
            if isinstance(x, list) and isinstance(y, list):
                if len(x) == len(y) and all(type(p) is type(q) for p, q in zip(x, y)):
                    for p, q in zip(x, y):
                        if isinstance(p, ast.AST):
                            walk(p, q, here)
                        elif differ(p, q):
                            out.append(Change(chain, a))
                    continue
                dx = [_dump(p) for p in x]
                dy = [_dump(q) for q in y]
                matcher = difflib.SequenceMatcher(a=dx, b=dy, autojunk=False)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "equal":
                        continue
                    if tag == "replace" and i2 - i1 == j2 - j1:
                        for p, q in zip(x[i1:i2], y[j1:j2]):
                            if isinstance(p, ast.AST) and isinstance(q, ast.AST):
                                walk(p, q, here)
                            else:
                                out.append(Change(chain, a))
                        continue
                    for p in x[i1:i2]:
                        if isinstance(p, ast.AST):
                            out.append(Change(here, p))
                        else:
                            out.append(Change(chain, a))
                    if j2 > j1:
                        out.append(Change(here, a, inserted=[q for q in y[j1:j2]]))
            elif isinstance(x, ast.AST) and isinstance(y, ast.AST):
                walk(x, y, here)
            elif isinstance(x, ast.AST) or isinstance(y, ast.AST):
                out.append(Change(chain, a))
            elif differ(x, y):
                out.append(Change(chain, a))

    walk(old, new, [])
    return out


def differ(x, y) -> bool:
    """Field values compared by type and spelling: True == 1, 0 == False, 1 == 1.0 and
    -0.0 == 0.0 under ==, but a program can tell each pair apart."""
    return type(x) is not type(y) or x != y or repr(x) != repr(y)


FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
if hasattr(ast, "TryStar"):
    COMPOUND = COMPOUND + (ast.TryStar,)


@dataclass
class Located:
    kind: str  # "runtime" or "import"
    lines: tuple[int, int] | None = None  # runtime: old-file line span whose coverage decides
    owner: list[str] = field(default_factory=list)  # import: enclosing class names, outermost first
    stmt: ast.stmt | None = None  # import: innermost changed statement at class or module level
    header_of: ast.AST | None = None  # import: def/class whose header (not body) changed
    inserted: list = field(default_factory=list)
    docstring: bool = False
    scope: ast.AST | None = None  # runtime: the innermost function or lambda holding the change
    new_node: object = None  # the new node when the change replaced a node by one of another type


def locate(change: Change) -> Located:
    full = change.chain if change.inserted else change.chain + [(change.node, None)]
    # innermost function whose body holds the change
    scope = None
    for index, (node, name) in enumerate(full):
        if isinstance(node, FUNCS) and name == "body":
            if index < len(full) - 1 or change.inserted:
                scope = index
    if scope is not None:
        fn = full[scope][0]
        stmt = None
        for node, _ in full[scope + 1 :]:
            if isinstance(node, ast.stmt):
                stmt = node
        if stmt is None:
            if isinstance(fn, ast.Lambda):
                return Located("runtime", (fn.body.lineno, fn.body.end_lineno), scope=fn)
            return Located("runtime", (fn.body[0].lineno, fn.end_lineno), scope=fn)
        parent = None
        for node, _ in full:
            if node is stmt:
                break
            parent = node
        if _is_docstring(stmt, parent) and change.new is None and not change.inserted and stmt is not change.node:
            # the docstring's text changed and it is still a docstring: nothing reads a function's
            # __doc__ in sqlglot or its tests
            return Located("runtime", (stmt.lineno, stmt.end_lineno), docstring=True, scope=fn)
        if _is_docstring(stmt, parent) or isinstance(stmt, (ast.Pass, ast.Global, ast.Nonlocal)) or (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
            # a docstring, bare constant, pass, global or nonlocal runs no line of its own, yet
            # it can change what the whole function does: whoever ran the function
            whole = (fn.body.lineno, fn.body.end_lineno) if isinstance(fn, ast.Lambda) else (fn.body[0].lineno, fn.end_lineno)
            return Located("runtime", whole, docstring=_is_docstring(stmt, parent), scope=fn)
        return Located("runtime", (stmt.lineno, stmt.end_lineno), scope=fn)
    owner: list[str] = []
    stmt = None
    header_of = None
    parents = [node for node, _ in full]
    for index, (node, name) in enumerate(full):
        if isinstance(node, ast.ClassDef):
            if name == "body":
                owner.append(node.name)
                continue
            header_of = node
            stmt = node
            break
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            header_of = node
            stmt = node
            break
        if isinstance(node, ast.stmt):
            stmt = node
            if isinstance(node, COMPOUND) and name in ("body", "orelse", "finalbody", "handlers", "cases"):
                continue  # a statement nested in a module level if/try: keep looking
            break
    docstring = False
    if stmt is not None and stmt in parents:
        idx = parents.index(stmt)
        docstring = idx > 0 and _is_docstring(stmt, parents[idx - 1])
    return Located(
        "import",
        owner=owner,
        stmt=stmt,
        header_of=header_of,
        inserted=change.inserted,
        docstring=docstring,
        new_node=change.new,
    )


def swapped_comparison(change: Change) -> tuple[ast.Compare, str, str] | None:
    """(compare node, old op, new op) when the change only swaps a single comparison operator."""
    if not isinstance(change.node, ast.cmpop) or not isinstance(change.new, ast.cmpop) or change.inserted:
        return None
    if not change.chain:
        return None
    parent, name = change.chain[-1]
    if not isinstance(parent, ast.Compare) or name != "ops" or len(parent.ops) != 1:
        return None
    return parent, type(change.node).__name__, type(change.new).__name__
