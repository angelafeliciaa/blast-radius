"""Turn a patch into the set of tests (map indices) it could make fail.

Whatever this module cannot reason about selects every test (Verdict.all, or a fallback
of None when the run-time diff is unavailable), never nothing. NOTES.md ("When the
selector runs every test") lists those inputs; keep it in step with the verdict.all calls.
"""

from __future__ import annotations

import ast
import collections
import gzip
import os
import pickle
import re
import types
from dataclasses import dataclass, field

import analyze
import patchfile

CONFIG_FILES = {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "setup.py"}

# Class attributes whose dict values are only read while classes are created, so a
# change of values (same keys, same order) reaches tests only through the attributes
# derived from it, which the fingerprint diff reports separately. In the pinned
# sqlglot, Expr.arg_types values are read only by Expr.__init_subclass__ (to derive
# required_args); everything else uses its keys, their order or its length.
VALUE_BLIND = {"arg_types"}
# a changed constant default value reaches only the callers that ran the body without passing
# the parameter (tools/build_defaults.py). Under the measured-risk policy: 203 labelled samples,
# 0 misses, 7.03 s saved per def-header commit; at ~0.73 such commits per 45 that is ~5.2 s
# against a rule-of-three miss bound of ~3.2 s. BR_DEFAULTS=0 turns it off.
DEFAULTS = os.environ.get("BR_DEFAULTS", "1") == "1"
# with BR_DEFAULTS=1: also leave out the callers that always pass the parameter themselves
# (app/defaults.pkl.gz, tools/build_defaults.py); BR_DEFAULT_USE=0 keeps every caller
DEFAULT_USE = os.environ.get("BR_DEFAULT_USE", "1") == "1"
DEFAULT_USE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults.pkl.gz")
# Lines that run while process-pool data is pickled or unpickled (tools/pickle_map.py)
PICKLE_LINES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle_lines.pkl.gz")
# Per comparison site, the tests that reached it and the tests where a sibling operator
# would have answered differently (tools/build_profile.py). BR_COMPARE=0 turns the rule off.
COMPARE_PATH = os.environ.get("BR_COMPARE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.pkl.gz"))
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "words.bin")


class Words:
    """Analyzer.word_index() as app/packmap.py stored it: the words are kept by first letter
    and each word's lines as (path number, line) pairs, both decoded when first asked for."""

    def __init__(self, paths: list[str], letters: dict[str, bytes]):
        self._paths = paths
        self._letters = letters
        self._words: dict[str, dict[str, bytes]] = {}
        self._lines: dict[str, list[tuple[str, int]]] = {}

    def get(self, word: str, default=()):
        found = self._lines.get(word)
        if found is None:
            letter = word[:1]
            words = self._words.get(letter)
            if words is None:
                blob = self._letters.get(letter)
                words = self._words[letter] = pickle.loads(blob) if blob is not None else {}
            pairs = words.get(word)
            if pairs is None:
                return default
            numbers = memoryview(pairs).cast("I")
            paths = self._paths
            found = self._lines[word] = [(paths[numbers[i]], numbers[i + 1]) for i in range(0, len(numbers), 2)]
        return found


def load_words(path: str, checkout: str):
    """The stored word index, or None when there is none that was made from this checkout."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        if data["checkout"] != os.path.abspath(checkout):
            return None
        return Words(data["paths"], data["letters"])
    except Exception as error:  # noqa: BLE001
        print(f"selector: cannot load {path}: {error!r}", file=__import__("sys").stderr, flush=True)
        return None


class Tracked(set):
    """A set of tests that remembers which line of this program added each one. Used in
    place of a plain set when BR_EXPLAIN=1, so a run can say which rule selected a test."""

    def __init__(self, *args):
        super().__init__(*args)
        self.why: dict[str, set[int]] = {}

    def _note(self, other) -> None:
        if isinstance(other, Tracked):
            for label, tests in other.why.items():
                self.why.setdefault(label, set()).update(tests)
            return
        import linecache
        import sys

        frame = sys._getframe(2)
        text = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
        self.why.setdefault(f"{frame.f_code.co_name}:{frame.f_lineno}: {text}", set()).update(other)

    def __ior__(self, other):
        self._note(other)
        return super().__ior__(other)

    def update(self, *others):
        for other in others:
            self._note(other)
        super().update(*others)


new_tests = Tracked if os.environ.get("BR_EXPLAIN") == "1" else set


@dataclass
class Verdict:
    everything: bool = False
    tests: set[int] = field(default_factory=new_tests)
    notes: list[str] = field(default_factory=list)
    import_time: list[tuple[str, list[str], set[str]]] = field(default_factory=list)
    new_texts: dict[str, str | None] = field(default_factory=dict)
    # Import-time changes inside sqlglot/: a run-time fingerprint diff decides them.
    # Without one, `fallback` (or everything, if fallback_all) is used instead.
    needs_diff: bool = False
    fallback: set[int] = field(default_factory=new_tests)
    fallback_all: bool = False
    patched_modules: set[str] = field(default_factory=set)
    # tests picked only because a data file they read changed, and whether every
    # changed line of such files was a comment or blank (then they may well pass)
    data_tests: set[int] = field(default_factory=set)
    # every file the patch touches: a check run on the patched copy is only meaningful when
    # each of them is in new_texts
    paths: list = field(default_factory=list)
    # set when the change moves state between tests (class fixtures): a check on the patched
    # copy runs the candidates in its own order and alone, so it cannot clear them
    no_verify: bool = False
    # tests the check must not clear: a broken shared pool fails them in the real run, while
    # the check's in-process stand-in fails only the call that raised
    unverified: set = field(default_factory=set)
    data_comment_only: bool = True
    # Keys whose required-ness an edit of `arg_types` dict literals flipped; None once
    # anything else at import time changed in sqlglot/ (then required_args can move for
    # other reasons and the narrow rule in diff_tests is off).
    arg_flips: set[str] | None = field(default_factory=set)
    # For checking fixture-driven tests on part of their fixtures (app/items.py): the changed
    # lines inside sqlglot functions, the changed data files, the tests picked for reading a
    # changed module's source, and whether the patch is nothing but function-body statements
    # and data files (an import-time effect, test code or any fallback makes it False).
    split_ok: bool = True
    split_lines: dict = field(default_factory=dict)
    data_paths: list = field(default_factory=list)
    source_readers: set = field(default_factory=set)
    # files of the patch that no test run reads (docs, CI, packaging)
    inert: list = field(default_factory=list)
    # new test modules (their own tests have no number): the import-first stage imports each
    new_test_modules: list = field(default_factory=list)

    def all(self, why: str) -> None:
        self.everything = True
        self.split_ok = False
        self.notes.append(f"ALL: {why}")

    lazy: list = field(default_factory=list)

    def defer_lazy(self, guess, why: str) -> None:
        """Like defer, with the fallback computed only if the diff is unavailable."""
        self.needs_diff = True
        self.split_ok = False
        self.lazy.append(guess)
        self.notes.append(f"import-time: {why}")

    def resolve_fallback(self) -> None:
        for guess in self.lazy:
            tests = guess()
            if tests is None:
                self.fallback_all = True
            else:
                self.fallback |= tests
        self.lazy.clear()

    def defer(self, tests: set[int] | None, why: str) -> None:
        """An import-time effect: the diff decides; `tests` is the static fallback (None: all)."""
        self.needs_diff = True
        self.arg_flips = None
        self.split_ok = False
        if tests is None:
            self.fallback_all = True
        else:
            self.fallback |= tests
        self.notes.append(f"import-time: {why}")


class Analyzer:
    def __init__(self, the_map: analyze.Map, checkout: str):
        self.map = the_map
        self.checkout = checkout
        self._word_index: dict[str, list[tuple[str, int]]] | None = None
        self._sources: dict[str, str] = {}
        self._module_users: dict[str, set[int]] = {}
        self._has_missing: bool | None = None
        self._private: dict[str, bool] = {}
        self._new_trees: dict[str, ast.AST] = {}

    # ------------------------------------------------------------ helpers

    def source(self, path: str) -> str | None:
        if path not in self._sources:
            full = os.path.join(self.checkout, path)
            try:
                with open(full, encoding="utf-8", newline="") as handle:  # exactly as on disk
                    self._sources[path] = handle.read()
            except (OSError, UnicodeDecodeError):
                return None
        return self._sources[path]

    def word_index(self):
        """identifier -> lines of sqlglot/ and tests/ Python files that mention it (.get(word, ())).
        Read from the file the image build made of it (app/packmap.py --words) when there is
        one for this checkout: making it takes a fifth of a second, looking a word up none."""
        if self._word_index is None:
            self._word_index = load_words(WORDS_PATH, self.checkout) or self.build_word_index()
        return self._word_index

    def build_word_index(self) -> dict[str, list[tuple[str, int]]]:
        index: dict[str, list[tuple[str, int]]] = {}
        for top in ("sqlglot", "tests"):
            for folder, _, files in os.walk(os.path.join(self.checkout, top)):
                for name in files:
                    if not name.endswith(".py"):
                        continue
                    rel = os.path.relpath(os.path.join(folder, name), self.checkout)
                    text = self.source(rel)
                    if text is None:
                        continue
                    for number, line in enumerate(text.split("\n"), start=1):
                        for word in set(WORD.findall(line)):
                            index.setdefault(word, []).append((rel, number))
        return index

    def tests_mentioning(self, name: str, skip: tuple[str, set[int]] = ("", set())) -> set[int]:
        found: set[int] = set()
        for path, line in self.word_index().get(name, ()):
            if path == skip[0] and line in skip[1]:
                continue
            found |= self.map.tests_on_lines(path, line, line)
        return found

    def method_callers(self, owner: str, name: str, related: set[str] | None = None) -> set[int]:
        """Tests that ran the body of a method called `name` defined on any of `related`
        (default: the class, its bases and its subclasses)."""
        found: set[int] = set()
        if related is None:
            related = set(self.map.subclasses(owner)) | set(self.map.classes.get(owner, [owner]))
        for qual in related:
            module, _, rest = qual.partition(".")
            # qualified names look like sqlglot.dialects.tsql.TSQL; find the file
            path = self.class_file(qual)
            if path is None:
                continue
            tree = self.tree(path)
            if tree is None:
                continue
            node = find_class(tree, qual_in_module(qual, path))
            if node is None:
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    found |= self.map.tests_on_lines(path, item.body[0].lineno, item.end_lineno)
        return found

    def class_file(self, qual: str) -> str | None:
        parts = qual.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            base = "/".join(parts[:cut])
            for candidate in (base + ".py", base + "/__init__.py"):
                if os.path.exists(os.path.join(self.checkout, candidate)):
                    return candidate
        return None

    _trees: dict[str, ast.AST] = {}

    def tree(self, path: str):
        if path not in self._trees:
            text = self.source(path)
            try:
                # parsed from bytes, as the interpreter reads the file: a BOM or a coding line counts
                self._trees[path] = ast.parse(text.encode("utf-8")) if text is not None else None
            except (SyntaxError, ValueError):
                self._trees[path] = None
        return self._trees[path]

    def class_body_names(self, qual: str) -> set[str] | None:
        path = self.class_file(qual)
        if path is None:
            return None
        tree = self.tree(path)
        node = find_class(tree, qual_in_module(qual, path)) if tree is not None else None
        if node is None:
            return None
        names: set[str] = set()
        for item in node.body:
            names |= analyze.bound_names(item)
        return names

    def class_change(self, owner: str, names: set[str]) -> set[int]:
        """Tests that could see a changed class level binding of `owner`.

        Reads of the attribute on the class or any subclass, plus reads of every
        attribute a metaclass or __init_subclass__ derived at import time (those
        can depend on the changed one), plus callers of same-named methods.
        """
        found: set[int] = set()
        class_attrs = self.map.data.get("class_attrs", {})
        for qual in self.map.subclasses(owner) | {owner}:
            attrs = set(class_attrs.get(qual, ()))
            body = self.class_body_names(qual)
            derived = attrs - body if body is not None else attrs
            for attr in names | derived:
                found |= self.map.attr_readers(qual, attr)
        for name in names:
            found |= self.method_callers(owner, name)
            found |= self.dispatch_readers(owner, name)
        return found

    def dispatch_readers(self, owner: str, name: str) -> set[int]:
        """Methods sqlglot finds by name rather than by reference: the generator's dispatch
        table collects every public `<key>_sql` of a class (dir(cls) in _build_dispatch, and
        simplify's copy of it), and Connector nodes call getattr(self, f"{key}_sql"). A new,
        removed or changed one reaches every test that generated that expression with the
        class or a subclass, whether or not it ran the method before."""
        if not name.endswith("_sql") or name.startswith("_"):
            return set()
        key = name[:-4]
        targets = {"t:" + qual for qual in self.map.classes
                   if qual.startswith("sqlglot.expressions.") and qual.rsplit(".", 1)[-1].lower() == key}
        keyreaders = self.map.data.get("keyreaders", {})
        class_attrs = self.map.data.get("class_attrs", {})
        found: set[int] = set()
        owners = set()
        for sub in self.map.subclasses(owner) | {owner}:
            for base in self.map.classes.get(sub, [sub]):  # the dispatch is recorded under TRANSFORMS' owner
                if "TRANSFORMS" in class_attrs.get(base, ()):
                    owners.add(base)
                    break
        for base in owners:
            keyed = keyreaders.get((base, "TRANSFORMS"), {})
            for entry in targets | {"*"}:
                number = keyed.get(entry)
                if number is not None:
                    found.update(self.map.sets[number])
        lines: dict[str, list[str]] = {}
        for path, line in self.word_index().get("getattr", ()):
            if path not in lines:
                lines[path] = (self.source(path) or "").split("\n")
            source_line = lines[path][line - 1] if line <= len(lines[path]) else ""
            if '_sql"' in source_line and 'f"' in source_line:
                found |= self.map.tests_on_lines(path, line, line)
        return found

    # ------------------------------------------------------------ main entry

    def analyze_patch(self, text: str | None) -> Verdict:
        verdict = Verdict()
        if text is None:
            verdict.all("patch is not UTF-8")
            return verdict
        try:
            patches = patchfile.parse(text)
        except patchfile.PatchError as error:
            verdict.all(f"unparsable patch: {error}")
            return verdict
        if not patches:
            if text.strip():
                verdict.all("patch text with no file in it")
            else:
                verdict.notes.append("empty patch")
            return verdict
        paths = [fp.path for fp in patches]
        if len(set(paths)) != len(paths):
            verdict.all("a file appears twice in the patch")
            return verdict
        problem = patch_shape_problem(text, patches)
        if problem:
            verdict.all(problem)
            return verdict
        verdict.paths = [fp.path for fp in patches] + [fp.new_path for fp in patches if fp.new_path]
        for fp in patches:
            try:
                self.analyze_file(fp, verdict)
            except Exception as error:  # any surprise: be safe for this file
                verdict.all(f"{fp.path}: analysis error {error!r}")
        if verdict.inert:
            verdict.paths = [path for path in verdict.paths if path not in verdict.inert]
        return verdict

    def outside_allowlist(self, fp: patchfile.FilePatch) -> str | None:
        """Why this file's edit is not one the selector can narrow (None: it is). Only text
        edits to files that exist in the checkout are analysed: Python modules under sqlglot/
        and tests/, and other files that tests were seen opening. Anything else (new, deleted
        or renamed files, binary or mode changes, files no test reads, configuration) selects
        every test."""
        path = fp.path
        if fp.binary:
            return "binary patch"
        if fp.old_path and fp.new_path and fp.old_path != fp.new_path:
            return f"renamed to {fp.new_path}"
        if fp.old_path is None:
            if fp.new_path is not None and self.new_file_open(fp.new_path):
                return None
            return "new file"
        if fp.new_path is None:
            return "deleted file"
        if fp.mode_only or not fp.hunks:
            return "mode change"
        if not os.path.isfile(os.path.join(self.checkout, path)):
            return "not a file in the checkout"
        if os.path.basename(path) == "conftest.py" or (os.path.basename(path) in PYTEST_CONFIGS
                                                       and path not in CHECKED_CONFIGS):
            return "test configuration"
        if path.endswith(".py") and (path.startswith("sqlglot/") or path.startswith("tests/")):
            if path.startswith("tests/") and os.path.basename(path) == "__init__.py":
                return "test package"
            return None
        if self.map.file_readers(path):
            return None  # a data file whose readers the map knows
        if INERT.match(path):
            return None  # docs, CI, packaging: analyze_file checks what pytest would read
        return "a file no test was seen reading"

    def analyze_file(self, fp: patchfile.FilePatch, verdict: Verdict) -> None:
        path = fp.path
        reason = self.outside_allowlist(fp)
        if reason is not None:
            verdict.all(f"{path}: {reason}")
            return
        if fp.binary:
            readers = self.map.file_readers(path)
            verdict.tests |= readers
            verdict.notes.append(f"{path}: binary, {len(readers)} readers")
            if path.startswith("sqlglot/"):
                verdict.all(f"{path}: binary change inside the package")
            elif path.endswith(".py"):
                # git writes a binary patch when the new file holds a NUL byte: such a module no
                # longer imports, and its new text is unknown
                if fp.new_path is None:
                    verdict.notes.append(f"{path}: binary delete")
                self.unimportable(path, verdict, "binary change to a Python file")
            return
        if fp.old_path != fp.new_path and fp.old_path and fp.new_path:
            self.analyze_rename(fp, verdict)
            return
        if fp.mode_only:
            verdict.notes.append(f"{path}: mode change only")
            return
        changed = [line for hunk in fp.hunks for line in hunk.lines if line[:1] in ("+", "-")]
        if changed and all(line[1:].startswith("Subproject commit ") for line in changed):
            # a submodule pointer: the run never fetches submodules, so nothing reads it
            verdict.notes.append(f"{path}: submodule pointer")
            return
        old_text = self.source(fp.old_path) if fp.old_path else None
        if fp.old_path and old_text is None:
            verdict.all(f"{path}: not in checkout")
            return
        try:
            new_text = patchfile.apply(old_text, fp)
        except patchfile.PatchError as error:
            verdict.all(f"{path}: does not apply ({error})")
            return
        verdict.new_texts[path] = new_text
        name = os.path.basename(path)
        if name == "conftest.py" or (name in CONFIG_FILES and "pytest" in (old_text or "") + (new_text or "")):
            verdict.all(f"{path}: test configuration")
            return
        if name in PYTEST_CONFIGS and os.path.dirname(path) in ("", "tests"):
            problem = config_problem(name, old_text, new_text)
            if problem:
                verdict.all(f"{path}: {problem}")
                return
        if path.endswith(".py") and "/" not in path:
            problem = shadowing(path[:-3])
            if problem:
                verdict.all(f"{path}: {problem}")
                return
        if fp.new_path is None:
            if path.endswith(".py") and (path.startswith("sqlglot/") or path.startswith("tests/")):
                self.deleted_module(path, verdict)
                return
            readers = self.map.file_readers(path) or self.unread_file(path, verdict)
            verdict.tests |= readers
            verdict.notes.append(f"{path}: deleted, {len(readers)} readers")
            return
        if REQUIREMENTS.match(os.path.basename(path)):
            verdict.all(f"{path}: requirements file")
            return
        if not path.endswith(".py") or not (path.startswith("sqlglot/") or path.startswith("tests/")):
            readers = self.map.file_readers(path)
            if not readers and INERT.match(path):
                # a file of a kind no test run reads (docs, CI, packaging), edited in place or new
                # (outside_allowlist), past the configuration checks above: as if not in the patch
                verdict.new_texts.pop(path, None)
                verdict.inert.append(path)
                verdict.notes.append(f"{path}: not read by tests (docs, CI or packaging)")
                return
            if not readers and (fp.old_path is not None or not (INERT.match(path) or path.startswith("tests/fixtures/"))):
                # an existing file no test opened, or a new one we cannot place
                readers = self.unread_file(path, verdict)
                if verdict.everything:
                    return
            verdict.tests |= readers
            verdict.data_tests |= readers
            verdict.data_paths.append(path)
            if not comment_like(fp):
                verdict.data_comment_only = False
            verdict.notes.append(f"{path}: data file, {len(readers)} readers")
            return
        if fp.old_path is None:
            # a new module nobody imports yet; the rest of the patch decides
            if os.path.basename(path) == "__init__.py":
                verdict.all(f"{path}: new package")
                return
            if path.startswith("tests/"):
                problem = self.new_test_module_problem(path, new_text)
                if problem:
                    verdict.all(f"{path}: new test module, {problem}")
                    return
                # its own tests have no number; pytest imports it while collecting, so the
                # import-first stage imports it on the patched copy (a failure: every test)
                verdict.new_test_modules.append(analyze.module_name(path))
                verdict.split_ok = False
                verdict.notes.append(f"{path}: new test module that writes nothing outside itself")
                return
            verdict.notes.append(f"{path}: new module")
            if path.startswith("sqlglot/"):
                # importing every module is part of the diff: it shows a new module that breaks others
                verdict.patched_modules.add(analyze.module_name(path))
                verdict.defer(set(), f"{path}: new module in the package")
            return
        self.analyze_python(path, old_text, new_text, verdict)

    _pool_tests: set[int] | None = None
    _pickle_lines: dict | None = None
    _shared_pool: set[int] | None = None

    def pickle_path(self, path: str, first: int, last: int) -> bool:
        """Whether a line in first..last runs while pool data is pickled or unpickled."""
        if self._pickle_lines is None:
            try:
                with gzip.open(PICKLE_LINES, "rb") as handle:
                    self._pickle_lines = {p: frozenset(v) for p, v in pickle.load(handle)["lines"].items()}
            except Exception as error:  # noqa: BLE001
                print(f"selector: cannot load {PICKLE_LINES}: {error!r}; every sqlglot line counts", flush=True,
                      file=__import__("sys").stderr)
                self._pickle_lines = {}
        if not self._pickle_lines:
            return path.startswith("sqlglot/")
        lines = self._pickle_lines.get(path, ())
        return any(first <= line <= last for line in lines)

    def shared_pool_tests(self) -> set[int]:
        """Tests of classes that keep one process pool for all their tests (cls.pool = ...
        ProcessPoolExecutor(...) in setUpClass or the class body): when an unpickle raises,
        concurrent.futures breaks the pool and every later test of the class fails."""
        if self._shared_pool is None:
            found: set[int] = set()
            for path in sorted(set(self.map.test_file)):
                text = self.source(path) or ""
                if "Executor" not in text and "Pool" not in text:
                    continue
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    found |= self.map.tests_in_file(path)
                    continue
                for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                    if any(_makes_pool(node, False) for node in ast.walk(cls)) or \
                            any(_makes_pool(node, True) for node in cls.body):
                        found |= {i for i in self.map.tests_in_file(path) if self.map.tests[i].split("::")[1:2] == [cls.name]}
            self._shared_pool = found
        return set(self._shared_pool)

    def pool_breaks(self, path: str, first: int, last: int, verdict: Verdict) -> None:
        if not self.pickle_path(path, first, last):
            return
        shared = self.shared_pool_tests()
        verdict.tests |= shared
        verdict.unverified |= shared
        verdict.split_ok = False
        verdict.notes.append(f"{path}:{first}-{last} runs while pool data is (un)pickled, {len(shared)} tests share a pool")
    _code_readers: set[int] | None = None

    def _tests_in_files_matching(self, pattern: re.Pattern, words: tuple[str, ...]) -> set[int]:
        """Tests of the test files the pattern is found in. `words`: plain strings one of which
        every match contains, so that a file with none of them is not searched at all."""
        found: set[int] = set()
        for path in sorted(set(self.map.test_file)):
            text = self.source(path) or ""
            if any(word in text for word in words) and pattern.search(text):
                found |= self.map.tests_in_file(path)
        return found

    def pool_tests(self) -> set[int]:
        """Tests in modules that send work to process pools (pickled expressions)."""
        if self._pool_tests is None:
            self._pool_tests = self._tests_in_files_matching(
                re.compile(r"ProcessPoolExecutor|multiprocessing|get_context"),
                ("ProcessPoolExecutor", "multiprocessing", "get_context"))
        return set(self._pool_tests)

    def code_reading_tests(self) -> set[int]:
        """Tests whose modules read function code objects (compared without being called)."""
        if self._code_readers is None:
            self._code_readers = self._tests_in_files_matching(
                re.compile(r"__code__|\bco_code\b|\bco_consts\b"), ("__code__", "co_code", "co_consts"))
        return set(self._code_readers)

    _codes: dict = {}

    def code_of(self, path: str):
        """The pinned file's compiled module code (None if it does not compile)."""
        if path not in self._codes:
            text = self.source(path)
            try:
                self._codes[path] = compile(text.encode("utf-8"), path, "exec", dont_inherit=True) if text is not None else None
            except (SyntaxError, ValueError):
                self._codes[path] = None
        return self._codes[path]

    def members_changed(self, path: str, scope: tuple) -> set[int] | None:
        """A def, class or binding added to or removed from a class or module body: sqlglot
        enumerates members (dir() for generator dispatch, inspect.getmembers, registries), so
        every test that uses the class hierarchy (or the module) may see it. None: no bound."""
        module = analyze.module_name(path)
        if not scope:
            return self.module_users(module, path)
        owner = module + "." + ".".join(scope)
        if owner not in self.map.classes:
            return None
        found = self.map.class_users(owner)
        for qual in self.map.subclasses(owner) | set(self.map.classes.get(owner, [owner])):
            qual_path = self.class_file(qual)
            tree = self.tree(qual_path) if qual_path else None
            node = find_class(tree, qual_in_module(qual, qual_path)) if tree is not None else None
            if node is not None:
                found |= self.map.tests_on_lines(qual_path, node.lineno, node.end_lineno)
        return found

    def test_methods_only(self, path: str, old_tree, new_tree, old_text: str, new_text: str) -> set[int] | None:
        """For an edit of a test module that no other test module imports or names, where only
        test methods changed, were added or were removed (test_method_edits) and each of them,
        before and after, writes nothing outside itself (self_contained): the tests that run a
        changed body (its own tests, inherited ones, callers in the map) and the tests that now run
        another class's method of the same name. None when that does not hold."""
        name = os.path.basename(path)
        if not name.startswith("test_"):
            return None
        own = self.map.tests_in_file(path)
        if self.module_users(analyze.module_name(path), path) - own:
            return None
        word = re.compile(rf"\b{re.escape(name[:-3])}\b")
        for other in sorted(set(self.map.test_file)) + ["tests/helpers.py", "tests/__init__.py"]:
            if other != path and word.search(self.source(other) or ""):
                return None
        if changed_fixture_classes(old_tree, new_tree):
            return None
        edits = test_method_edits(old_tree, new_tree, old_text, new_text)
        if not edits:
            return None
        added_imports = edits.pop(None, [])
        if added_imports and not self.imports_quiet(old_tree, added_imports):
            return None
        self._import_at_collection = bool(added_imports) or any(
            parts["added"] or parts["removed"] or any(_header(old) != _header(new) for old, new in parts["changed"])
            for parts in edits.values())
        old_scope, new_scope = self.scope(old_tree), self.scope(new_tree)
        found: set[int] = set()
        for cls, parts in edits.items():
            for old_def, new_def in parts["changed"]:
                if self_contained(old_def, old_scope) or self_contained(new_def, new_scope):
                    return None
                found |= self.method_tests(path, old_tree, cls, old_def.name)
                # the def line itself runs at import, for every test of the file: the body's lines
                found |= self.map.tests_on_lines(path, old_def.body[0].lineno, old_def.end_lineno)
            for new_def in parts["added"]:
                if self_contained(new_def, new_scope):
                    return None
                found |= self.method_tests(path, old_tree, cls, new_def.name)  # overrides an inherited one
            for old_def in parts["removed"]:
                if self_contained(old_def, old_scope):
                    return None
                # callers of it fail now; its own tests run an inherited method of that name, if any
                callers = self.map.tests_on_lines(path, old_def.body[0].lineno, old_def.end_lineno)
                mine = self.method_tests(path, old_tree, cls, old_def.name)
                inherited = old_scope.method(old_def.name)
                found |= (callers - mine) | (mine if len(inherited) > 1 else set())
        return found

    _test_imports: set | None = None
    _import_at_collection = False

    def imports_quiet(self, old_tree, statements: list) -> bool:
        """Import statements added at a test module's top: absolute, of sqlglot or of a module some
        test module of the checkout imports at its top already (loaded in the run anyway), binding
        names the module did not bind. Whether they import is the import-first stage's check."""
        if self._test_imports is None:
            found = set()
            for path in sorted(set(self.map.test_file)) + ["tests/helpers.py"]:
                tree = self.tree(path)
                for stmt in (tree.body if tree is not None else []):
                    if isinstance(stmt, ast.Import):
                        found |= {alias.name.split(".")[0] for alias in stmt.names}
                    elif isinstance(stmt, ast.ImportFrom) and not stmt.level:
                        found.add((stmt.module or "").split(".")[0])
            self._test_imports = found - {"tests"}
        bound = set()
        for stmt in old_tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                bound |= {(alias.asname or alias.name).split(".")[0] for alias in stmt.names}
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(stmt.name)
            else:
                bound |= {node.id for node in ast.walk(stmt) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        for stmt in statements:
            if isinstance(stmt, ast.ImportFrom):
                if stmt.level or any(alias.name == "*" for alias in stmt.names):
                    return False
                modules = [stmt.module or ""]
            else:
                modules = [alias.name for alias in stmt.names]
            if any(module.split(".")[0] not in self._test_imports | {"sqlglot"} for module in modules):
                return False
            names = {(alias.asname or alias.name).split(".")[0] for alias in stmt.names}
            if names & bound or any(name in CLOSED_NAMES for name in names):
                return False
            bound |= names
        return True

    def method_tests(self, path: str, tree, cls: str, method: str) -> set[int]:
        """Numbered tests path::C::method for cls and the classes of the module that derive from it."""
        classes = {cls}
        grown = True
        while grown:
            grown = False
            for stmt in tree.body:
                if isinstance(stmt, ast.ClassDef) and stmt.name not in classes and \
                        any(_dotted(base) in classes for base in stmt.bases):
                    classes.add(stmt.name)
                    grown = True
        found = set()
        for index in self.map.tests_in_file(path):
            parts = self.map.tests[index].split("::")
            if len(parts) == 3 and parts[1] in classes and parts[2] == method:
                found.add(index)
        return found

    def new_file_open(self, path: str) -> bool:
        """A new file the analysis can place: a test module in an existing test package (its
        content is checked in analyze_file), a fixture file, or docs, CI and packaging files no
        test run reads. Not a Python file anywhere else, nothing pytest may read or collect
        (configuration, conftest, test*.txt doctest files), and not a path that already exists."""
        name = os.path.basename(path)
        if os.path.lexists(os.path.join(self.checkout, path)):
            return False
        if NEW_TEST_MODULE.match(path):
            return True
        if path.endswith(".py") or name == "conftest.py" or name in PYTEST_CONFIGS or REQUIREMENTS.match(name) \
                or name.startswith("test") or name.startswith(".git") or path.startswith("sqlglot-integration-tests"):
            return False
        return bool(INERT.match(path)) or path.startswith("tests/fixtures/")

    def new_test_module_problem(self, path: str, text: str | None) -> str | None:
        """Why a new test module could change the run of the suite's numbered tests (None: it
        cannot). Its own tests have no number. pytest imports it while collecting (the import-first
        stage checks that it imports), and it runs its tests among the others: its module and
        class level may only import and bind names, its classes must be plain test classes, and
        every function in it must write nothing outside itself (self_contained)."""
        if text is None:
            return "no text"
        if not os.path.isfile(os.path.join(self.checkout, os.path.dirname(path), "__init__.py")):
            return "not in a test package"
        stem = os.path.basename(path)[:-3]
        word = re.compile(rf"\b{re.escape(stem)}\b")
        for other in sorted(set(self.map.test_file)) + ["tests/helpers.py", "tests/__init__.py"]:
            if word.search(self.source(other) or ""):
                return f"named by {other}"
        if text.startswith("\ufeff") or "\x00" in text or "\r" in text or coding_cookie_changed("", text):
            return "source encoding"
        try:
            tree = ast.parse(text.encode("utf-8"))
            compile(text.encode("utf-8"), path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as error:
            return f"does not compile ({error.__class__.__name__})"
        own_classes: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                module = stmt.module or ""
                if stmt.level or any(alias.name == "*" for alias in stmt.names):
                    return "relative or star import"
                if module.split(".")[0] == "tests" and not (module == "tests.helpers" or (
                        module == "tests.dialects.test_dialect" and all(a.name == "Validator" for a in stmt.names))):
                    return f"imports from {module}"
            elif isinstance(stmt, ast.Import):
                if any(alias.name.split(".")[0] == "tests" for alias in stmt.names):
                    return "imports a test module"
            elif isinstance(stmt, ast.ClassDef):
                if stmt.keywords or _decorators_open(stmt) or not all(
                        _dotted(base) in TEST_BASES or _dotted(base) in own_classes for base in stmt.bases):
                    return f"class {stmt.name} is not a plain test class"
                for item in stmt.body:
                    if not (_quiet(item) and not isinstance(item, ast.ClassDef)) and not (
                            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _decorators_open(item)
                            and all(_plain(d) for d in item.args.defaults + [d for d in item.args.kw_defaults if d])):
                        return f"class {stmt.name}: line {item.lineno} does more than bind names"
                own_classes.add(stmt.name)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _decorators_open(stmt) or not all(_plain(d) for d in stmt.args.defaults + [d for d in stmt.args.kw_defaults if d]):
                    return f"function {stmt.name}: decorators or defaults"
            elif isinstance(stmt, ast.If) and not stmt.orelse and isinstance(stmt.test, ast.Compare) \
                    and _dotted(stmt.test.left) == "__name__" and len(stmt.test.comparators) == 1 \
                    and isinstance(stmt.test.comparators[0], ast.Constant) and stmt.test.comparators[0].value == "__main__":
                continue  # never runs under pytest
            elif not _quiet(stmt):
                return f"line {stmt.lineno} does more than bind names"
            bound = [stmt.name] if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else \
                [t.id for t in (stmt.targets if isinstance(stmt, ast.Assign) else [getattr(stmt, "target", None)])
                 if isinstance(t, ast.Name)]
            if any(name.startswith("pytest") or (name.startswith("__") and name != "__all__") for name in bound):
                return f"line {stmt.lineno} binds a name pytest or Python reads ({bound[0]})"
        scope = self.scope(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                problem = self_contained(node, scope)
                if problem:
                    return problem
        return None

    _validator_scope = None

    def scope(self, tree) -> Scope:
        """The Scope of a test module, with Validator's module behind it when it imports Validator."""
        base = None
        imports_validator = any(isinstance(stmt, ast.ImportFrom) and stmt.module == "tests.dialects.test_dialect"
                                and any(alias.name == "Validator" for alias in stmt.names) for stmt in tree.body)
        if imports_validator:
            if self._validator_scope is None:
                dialect_tree = self.tree("tests/dialects/test_dialect.py")
                validator = find_class(dialect_tree, ["Validator"]) if dialect_tree is not None else None
                only = ast.Module(body=[validator] if validator is not None else [], type_ignores=[])
                self._validator_scope = Scope(only)
                if dialect_tree is not None:  # what Validator's methods see of their own module
                    whole = Scope(dialect_tree)
                    self._validator_scope.functions = whole.functions
                    self._validator_scope.classes = whole.classes - {"Validator"}
                    self._validator_scope.others = whole.others
                    self._validator_scope.constants = whole.constants
            base = self._validator_scope
        return Scope(tree, base)

    def unread_file(self, path: str, verdict: Verdict) -> set[int]:
        """An existing file no test was seen opening. Under tests/: the tests whose files name
        it. Elsewhere: nothing if it is one of the kinds no test run reads (docs, CI, build
        files), else every test: a file the map knows nothing about is not a file we can
        say is harmless."""
        if path.startswith("tests/"):
            name = os.path.basename(path)
            stem = name.split(".", 1)[0]
            found = set()
            for index, test_path in enumerate(self.map.test_file):
                text = self.source(test_path) or ""
                if name in text or re.search(rf"\b{re.escape(stem)}\b", text):
                    found.add(index)
            verdict.notes.append(f"{path}: never opened, named by {len(found)} tests")
            return found
        if INERT.match(path):
            verdict.notes.append(f"{path}: not read by tests (docs, CI or packaging)")
            return set()
        verdict.all(f"{path}: a file the map has no reader for")
        return set()

    def analyze_rename(self, fp: patchfile.FilePatch, verdict: Verdict) -> None:
        """A rename is the old path deleted and the new one added (with the hunks applied)."""
        old, new = fp.old_path, fp.new_path
        if any(os.path.basename(p) in ("__init__.py", "conftest.py") or os.path.basename(p) in PYTEST_CONFIGS
               for p in (old, new)) or any(p.endswith(".py") and "/" not in p for p in (old, new)):
            verdict.all(f"rename {old} -> {new}")
            return
        old_text = self.source(old)
        if old_text is None:
            verdict.all(f"rename {old}: not in checkout")
            return
        try:
            new_text = patchfile.apply(old_text, fp)
        except patchfile.PatchError as error:
            verdict.all(f"rename {old}: does not apply ({error})")
            return
        verdict.new_texts[old] = None
        verdict.new_texts[new] = new_text
        if old.endswith(".py") and (old.startswith("sqlglot/") or old.startswith("tests/")):
            self.deleted_module(old, verdict)
        else:
            readers = self.map.file_readers(old)
            verdict.tests |= readers
            verdict.notes.append(f"{old}: renamed away, {len(readers)} readers")
        if new.endswith(".py") and new.startswith("sqlglot/"):
            verdict.patched_modules.add(analyze.module_name(new))
            verdict.defer(set(), f"{new}: new module in the package")
        verdict.notes.append(f"rename {old} -> {new}")

    def unimportable(self, path: str, verdict: Verdict, why: str) -> None:
        """A module that may no longer import, or whose every line may mean something else
        (new source encoding): like a deleted one, everything that ran, read or names it,
        and in sqlglot/ the diff finds every module that imports it at import time."""
        name = os.path.basename(path)
        if name == "__init__.py" or (path.startswith("tests/") and not name.startswith("test_")):
            verdict.all(f"{path}: {why}")
            return
        verdict.notes.append(f"{path}: {why}")
        self.deleted_module(path, verdict)

    def deleted_module(self, path: str, verdict: Verdict) -> None:
        """A deleted module: everything that ran it, read it or names it, and (in sqlglot/) the
        modules the diff finds broken without it."""
        if os.path.basename(path) == "__init__.py":
            verdict.all(f"{path}: deleted package")
            return
        module = analyze.module_name(path)
        found = self.module_users(module, path) | self.map.file_readers(path)
        if path.startswith("tests/"):
            found |= self.map.tests_in_file(path)
        else:
            verdict.patched_modules.add(module)
            verdict.defer(None, f"{path}: deleted module")
        verdict.tests |= found
        verdict.notes.append(f"{path}: deleted module, {len(found)} users")

    def analyze_python(self, path: str, old_text: str, new_text: str, verdict: Verdict) -> None:
        readers = self.map.file_readers(path)  # tests that read this source as text
        verdict.tests |= readers
        verdict.source_readers |= readers
        old_tree = self.tree(path)
        if old_tree is None:
            verdict.all(f"{path}: original does not parse")
            return
        if coding_cookie_changed(old_text, new_text) or new_text.startswith("\ufeff") != old_text.startswith("\ufeff"):
            # a new coding line or BOM changes how every byte of the file decodes
            verdict.all(f"{path}: source encoding changed")
            return
        if new_text.count("\r") != old_text.count("\r") or "\x00" in new_text:
            verdict.all(f"{path}: line endings or NUL bytes changed")
            return
        try:
            new_tree = ast.parse(new_text.encode("utf-8"))  # bytes: decoded as the interpreter would
        except (SyntaxError, ValueError) as error:
            verdict.all(f"{path}: does not parse ({error.__class__.__name__})")
            return
        self._new_trees[path] = new_tree
        changes = analyze.ast_changes(old_tree, new_tree, old_text, new_text)
        if not changes:
            verdict.notes.append(f"{path}: same AST (comments/formatting), {len(readers)} readers")
            return
        try:
            new_code = compile(new_text.encode("utf-8"), path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as error:
            verdict.all(f"{path}: does not compile ({error.__class__.__name__})")
            return
        # import_effects reads nothing inside function or lambda bodies: edits that all sit in
        # one cannot change it
        in_bodies = all(analyze.locate(change).kind == "runtime" for change in changes)
        if not path.startswith("tests/") and not in_bodies and import_effects(old_tree) != import_effects(new_tree):
            # a call that sets, patches or registers now runs (or no longer runs) at import:
            # in a binding's value, a default argument, a decorator or a class header
            verdict.all(f"{path}: import-time call with process-wide effects changed")
            return
        if effects_changed(old_tree, new_tree, old_text, new_text):
            # code that sets or mutates state outside its own frame (module globals, other
            # modules, the process: logging, os.environ, sys) was added, removed or changed:
            # whatever runs later in the same test process can see it
            verdict.all(f"{path}: changes code with process-wide effects")
            return
        is_test = path.startswith("tests/")
        if is_test and alias_writers(old_tree) != alias_writers(new_tree):
            # test code that writes through a local bound to something it did not build
            # (x = exp.Literal; x.number = ...): the write lands on shared state, wherever the
            # effect rules above (which follow names, not aliases) cannot see it
            verdict.all(f"{path}: test code writing through an alias changed")
            return
        if is_test:
            narrowed = self.test_methods_only(path, old_tree, new_tree, old_text, new_text)
            if narrowed is not None and self._import_at_collection:
                # what runs while the class is made changed (imports, decorators, defaults,
                # annotations, a method another class-level name may refer to): the
                # import-first stage imports the patched module
                verdict.new_test_modules.append(analyze.module_name(path))
            if narrowed is not None:
                # only test methods changed, were added or removed, and every one of them (old and
                # new) writes nothing outside itself: the file's other tests run as before
                verdict.tests |= narrowed
                verdict.split_ok = False
                verdict.notes.append(f"{path}: test methods only, each self-contained, {len(narrowed)} tests")
                return
        if not is_test and verdict.arg_flips is not None:
            flips = arg_type_flips(old_tree, new_tree)
            verdict.arg_flips = None if flips is None else verdict.arg_flips | flips
        if is_test:
            for cls_name in changed_fixture_classes(old_tree, new_tree):
                # a test that sets or mutates its class's shared fixtures (self.__class__.x = ...,
                # self.pool.shutdown(), self.conn.execute(...)) hands that state to the class's
                # other tests
                same_class = {i for i in self.map.tests_in_file(path) if self.map.tests[i].split("::")[1:2] == [cls_name]}
                verdict.tests |= same_class
                verdict.no_verify = True
                verdict.notes.append(f"{path}: {cls_name} touches class-level fixtures, {len(same_class)} tests")
        else:
            if path == "sqlglot/serde.py" or (any(hook in old_text or hook in new_text for hook in PICKLING_HOOKS)
                                              and pickling_hooks_changed(old_tree, new_tree)):
                # expressions cross process pools through pickle: every test that hands them to a
                # pool depends on these, whether or not it ran the changed line in the map
                found = self.pool_tests()
                verdict.tests |= found
                verdict.split_ok = False
                verdict.notes.append(f"{path}: pickling path changed, {len(found)} pool-using tests")
            if path.startswith("sqlglot/typing/"):
                found = self.code_reading_tests()
                verdict.tests |= found
                verdict.notes.append(f"{path}: {len(found)} tests compare these functions' code objects")
        if is_test:
            verdict.split_ok = False
            module = analyze.module_name(path)
            importers = self.module_users(module, path)
            own_tests = self.map.tests_in_file(path)
            if not os.path.basename(path).startswith("test_") or importers - own_tests:
                # helpers and test modules others import (Validator): every importer can see it
                verdict.tests |= importers | own_tests
                verdict.notes.append(f"{path}: shared test module, {len(importers)} tests import it")
        else:
            for scope in member_changes(old_tree, new_tree, old_text, new_text):
                verdict.arg_flips = None  # added or removed members can move required_args too
                found = self.members_changed(path, scope)
                if found is None:
                    verdict.all(f"{path}: members of {'.'.join(scope) or 'the module'} added or removed")
                    return
                verdict.tests |= found
                verdict.split_ok = False
                verdict.notes.append(f"{path}: members of {'.'.join(scope) or 'the module'} changed, {len(found)} tests")
        profile = self.map.data.get("profile", {})
        # A function whose names change kind (a new local, free or global; an import alias,
        # except-as or def in a branch no test runs) or that becomes or stops being a generator
        # or coroutine behaves differently on every call: all its lines' tests, edited or not.
        old_code = self.code_of(path)
        if old_code is None:
            verdict.all(f"{path}: original does not compile")
            return
        for first, last in changed_function_kinds(old_code, new_code, old_tree):
            found = self.map.tests_on_lines(path, first, last)
            verdict.tests |= found
            # every call of it changes: a fixture-driven test is cut down to the items that
            # reach any of its lines, not only the edited ones (red team i1-i3: a check run on
            # no items passed and dropped test_execute_tpcds, which fails)
            verdict.split_lines.setdefault(path, set()).update(range(first, last + 1))
            if not is_test:
                self.pool_breaks(path, first, last, verdict)
            verdict.notes.append(f"{path}:{first}-{last} function kind or scope changed, {len(found)} tests")
        for change in changes:
            swap = analyze.swapped_comparison(change)
            same_answer: set[int] = set()
            if swap is not None:
                compare, old_op, new_op = swap
                site = f"{path}:{compare.lineno}:{compare.col_offset}"
                differed = profile.get(site)
                # the profile only evaluated the operators of old_op's family (mapgen_plugin
                # _FAMILY): any other swap is unknown
                if differed is not None and new_op in COMPARE_FAMILY.get(old_op, ()) and new_op not in differed:
                    verdict.notes.append(f"{site} {old_op}->{new_op} never differs in any test: equivalent")
                    continue
                if differed is not None and new_op in COMPARE_FAMILY.get(old_op, ()):
                    same_answer = self.swap_same_answer(site, new_op)
            loc = analyze.locate(change)
            if not is_test:
                # an edit to code that runs while expressions are unpickled can make that raise
                spans = [loc.lines] if loc.kind == "runtime" else \
                    [(node.lineno, node.end_lineno) for node, _ in change.chain if isinstance(node, ast.ClassDef)]
                for first, last in spans:
                    self.pool_breaks(path, first, last, verdict)
            if loc.kind == "runtime":
                # an insertion's change.node is the container it went into: scan what was inserted
                edited = list(change.inserted) if change.inserted else [change.node, change.new]
                touched = {name for node in (*edited, loc.stmt, loc.new_node, *loc.inserted)
                           if isinstance(node, ast.AST) for name in _names(node)}
                if touched & PROCESS_NAMES:
                    # code run by a test that reaches process-wide state (sys, os, gc, builtins,
                    # an import hook...) can change every later test, however it is called
                    verdict.all(f"{path}:{loc.lines[0]}: code touching {sorted(touched & PROCESS_NAMES)[:3]}")
                    return
            if loc.kind == "runtime" and is_test:
                # a test can reach its class's or module's shared state (class attributes,
                # setUpClass fixtures such as pools and connections, module globals) through
                # anything: self, an alias, getattr, a lambda, the class's own name. The file's
                # other tests see what it leaves, in an order a check run does not reproduce.
                same_file = self.map.tests_in_file(path)
                verdict.tests |= same_file
                verdict.no_verify = True
                verdict.split_ok = False
                verdict.notes.append(f"{path}:{loc.lines[0]}-{loc.lines[1]} test code, {len(same_file)} tests of the file")
            if loc.kind == "runtime":
                outermost = next((node for node, _ in change.chain
                                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), loc.scope)
                class_writes: list = []
                if outermost is not None and is_test and not writes_shared_state(outermost, class_level=class_writes) \
                        and class_writes:
                    # a test that sets class-level state (cls.x) hands it to the rest of its class
                    owner = next((node.name for node, _ in change.chain if isinstance(node, ast.ClassDef)), None)
                    if owner is not None:
                        same_class = {i for i in self.map.tests_in_file(path)
                                      if self.map.tests[i].split("::")[1:2] == [owner]}
                        verdict.tests |= same_class
                        verdict.no_verify = True
                        verdict.notes.append(f"{path}: {owner} sets class-level state, {len(same_class)} tests")
                if outermost is not None and writes_shared_state(outermost):
                    # the function fills a registry or other state that outlives the call
                    # (Dialect._try_load re-registers dialects after a test clears them): tests
                    # that never ran it read what it left behind
                    verdict.all(f"{path}:{loc.lines[0]}: change in a function that writes shared state")
                    return
                found = self.map.tests_on_lines(path, *loc.lines)
                if same_answer and not is_test and loc.scope is not None and not self.ran_at_import(path, loc.scope):
                    # the swapped operator gives these tests the answer the old one gave, every
                    # time their own run reaches the statement: they run exactly as before
                    spared = found & same_answer
                    if spared:
                        found -= spared
                        verdict.notes.append(f"{site} {old_op}->{new_op} gives {len(spared)} tests the same answers")
                verdict.tests |= found
                verdict.notes.append(f"{path}:{loc.lines[0]}-{loc.lines[1]} runtime, {len(found)} tests")
                verdict.split_lines.setdefault(path, set()).update(range(loc.lines[0], loc.lines[1] + 1))
                if loc.scope is None:
                    verdict.split_ok = False
                if loc.scope is not None and self.ran_at_import(path, loc.scope):
                    if is_test:
                        verdict.tests |= self.map.tests_in_file(path)
                    else:
                        verdict.patched_modules.add(analyze.module_name(path))
                        verdict.defer(None, f"{path}:{loc.lines[0]} is in a function that runs at import")
                continue
            if not is_test and DEFAULTS and self.default_value_only(path, change, loc, verdict):
                continue
            if not is_test:
                verdict.patched_modules.add(analyze.module_name(path))
            verdict.split_ok = False
            self.import_time(path, loc, verdict, is_test)

    def default_value_only(self, path: str, change: analyze.Change, loc: analyze.Located, verdict: Verdict) -> bool:
        """A constant default value of a def replaced by another constant: only a call that
        leaves that parameter out can tell, and such a call runs the function's body, so the
        tests that ran the body are all it reaches (nothing in the pinned sqlglot or its tests
        reads __defaults__ or a signature's default values). Not for a function that also runs
        while modules load. True when the change was handled here."""
        fn = loc.header_of
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.body or change.inserted:
            return False
        if not isinstance(change.node, ast.Constant) or (change.new is not None and not isinstance(change.new, ast.Constant)):
            return False
        if not change.chain or not isinstance(change.chain[-1][0], ast.arguments) \
                or change.chain[-1][1] not in ("defaults", "kw_defaults"):
            return False
        if len(change.chain) < 2 or change.chain[-2][0] is not fn or self.ran_at_import(path, fn):
            return False
        found = self.map.tests_on_lines(path, fn.body[0].lineno, fn.end_lineno)
        spared = self.default_never_used(path, fn, change) & found if DEFAULT_USE else set()
        found = found - spared
        verdict.tests |= found
        verdict.split_lines.setdefault(path, set()).update(range(fn.body[0].lineno, fn.end_lineno + 1))
        more = f"; {len(spared)} more always pass that parameter themselves" if spared else ""
        verdict.notes.append(f"{path}: default value in the header of {fn.name}, {len(found)} callers{more}")
        return True

    _default_use: dict | None | bool = False

    def default_use(self, path: str, fn, change: analyze.Change):
        """(the recorded default-reliance data, this def's key, the changed parameter's name),
        or None without that data (tools/build_defaults.py) or when it is for another test list."""
        if self._default_use is False:
            self._default_use = None
            try:
                with gzip.open(DEFAULT_USE_PATH, "rb") as handle:
                    data = pickle.load(handle)
                if data["tests"] == self.map.tests:
                    self._default_use = data
            except Exception:  # no data: every caller stays
                pass
        if not self._default_use:
            return None
        args = fn.args
        if change.chain[-1][1] == "defaults":
            positional = args.posonlyargs + args.args
            names = [arg.arg for arg in positional[len(positional) - len(args.defaults):]]
            where = [i for i, node in enumerate(args.defaults) if node is change.node]
        else:
            names = [arg.arg for arg in args.kwonlyargs]
            where = [i for i, node in enumerate(args.kw_defaults) if node is change.node]
        if len(where) != 1:
            return None
        return self._default_use, (path, min([fn.lineno] + [d.lineno for d in fn.decorator_list]), fn.name), names[where[0]]

    def default_never_used(self, path: str, fn, change: analyze.Change) -> set[int]:
        """Tests that called this def in their own run (one fork each) and whose calls, there
        and in the one-process run, all started with the changed parameter holding another
        value than its default constant: they passed it themselves every time, so the new
        default is a value none of their calls ever sees. Empty without the data."""
        known = self.default_use(path, fn, change)
        if known is None:
            return set()
        data, key, param = known
        mask = data["called"].get(key, 0) & ~data["used"].get(key + (param,), 0)
        found = set()
        while mask:
            low = mask & -mask
            found.add(low.bit_length() - 1)
            mask ^= low
        return found

    def import_time(self, path: str, loc: analyze.Located, verdict: Verdict, is_test: bool) -> None:
        module = analyze.module_name(path)
        names: set[str] = set()
        if loc.stmt is not None:
            names |= analyze.bound_names(loc.stmt)
        for node in loc.inserted:
            if isinstance(node, ast.stmt):
                names |= analyze.bound_names(node)
        if loc.docstring and docstring_to_docstring(loc):
            verdict.notes.append(f"{path}: docstring at import level")
            return
        touched = [node for node in [loc.stmt, loc.new_node, *loc.inserted] if isinstance(node, ast.stmt)]
        if not is_test and any(isinstance(sub, (ast.Import, ast.ImportFrom)) for node in touched for sub in ast.walk(node)):
            # whether an import cycle resolves depends on which module the run imports first,
            # and the test run's order is not the diff's (red team ic2)
            verdict.all(f"{path}: import statement changed")
            return
        if any((not _quiet(node)) if is_test else process_effect(node) for node in touched):
            # an import-level statement that acts beyond binding names (a call, setting an
            # attribute or item of something else): it can change the whole test process
            verdict.all(f"{path}: import-level statement with effects beyond its bindings")
            return
        verdict.import_time.append((path, list(loc.owner), set(names)))
        if is_test:
            self.import_time_test(path, loc, names, verdict)
            return
        if not (loc.owner and names == {"arg_types"} and loc.header_of is None):
            verdict.arg_flips = None
        if isinstance(loc.header_of, (ast.FunctionDef, ast.AsyncFunctionDef)) and loc.header_of.body:
            # a new signature or decorator reaches whoever called the function, whatever the diff says
            fn = loc.header_of
            found = self.map.tests_on_lines(path, fn.body[0].lineno, fn.end_lineno)
            if self.ran_at_import(path, fn):
                # it also runs while modules load (registration, lazy dialect loading): a new
                # binding there reaches every test that uses the module, not just its callers
                found |= self.module_users(module, path)
            verdict.tests |= found
            verdict.notes.append(f"{path}: header of {fn.name}, {len(found)} callers")
        verdict.defer_lazy(lambda: self._static_guess(path, loc, names), f"{path}: {sorted(names)[:4]}")

    def _static_guess(self, path, loc, names):
        scratch = Verdict()
        self.import_time_static(path, loc, names, scratch)
        return None if scratch.everything else scratch.tests

    def import_time_static(self, path, loc, names, verdict) -> None:
        """Conservative guess at the tests an import-time change in sqlglot/ reaches."""
        module = analyze.module_name(path)
        if isinstance(loc.header_of, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = loc.header_of
            found = self.map.tests_on_lines(path, fn.body[0].lineno, fn.end_lineno)
            verdict.tests |= found
            if loc.owner:
                owner = module + "." + ".".join(loc.owner)
                verdict.tests |= self.class_change(owner, {fn.name})
            else:
                verdict.tests |= self.tests_mentioning(fn.name)
            verdict.notes.append(f"{path}: header of {fn.name}, {len(found)} callers")
            return
        if isinstance(loc.header_of, ast.ClassDef):
            owner = module + "." + ".".join(loc.owner + [loc.header_of.name])
            class_attrs = self.map.data.get("class_attrs", {})
            found = self.class_change(owner, set(class_attrs.get(owner, ())))
            verdict.tests |= found
            verdict.notes.append(f"{path}: header of class {owner}, {len(found)} tests")
            return
        if loc.owner:
            owner = module + "." + ".".join(loc.owner)
            if owner not in self.map.classes:
                verdict.all(f"{path}: class {owner} unknown to the map")
                return
            found = self.class_change(owner, names)
            verdict.tests |= found
            verdict.notes.append(f"{path}: class {owner} {sorted(names)}, {len(found)} tests")
            return
        if not names or isinstance(loc.stmt, (ast.Import, ast.ImportFrom)):
            verdict.all(f"{path}: module level change binding {sorted(names) or 'nothing'}")
            return
        found = set()
        for name in names:
            found |= self.tests_mentioning(name)
            qual = module + "." + name
            if qual in self.map.classes:
                class_attrs = self.map.data.get("class_attrs", {})
                found |= self.class_change(qual, set(class_attrs.get(qual, ())))
        verdict.tests |= found
        verdict.notes.append(f"{path}: module names {sorted(names)}, {len(found)} tests")

    _compare: dict | None | bool = False

    def swap_same_answer(self, site: str, new_op: str) -> set[int]:
        """Tests whose own run (one fork each, tools/build_profile.py) reached the statement
        of this comparison and in which `new_op` never gave another answer than the operator
        that is there: evaluated with the same result every time, or not evaluated at all.
        Empty when there is no per-test profile or it was made for another test list."""
        if self._compare is False:
            self._compare = None
            try:
                with gzip.open(COMPARE_PATH, "rb") as handle:
                    data = pickle.load(handle)
                if data["tests"] == self.map.tests:
                    self._compare = data
            except Exception:  # no per-test profile: nothing is spared
                pass
        if not self._compare:
            return set()
        entry = self._compare["sites"].get(site)
        if entry is None:
            return set()
        reach, differs = entry
        mask = reach & ~differs.get(new_op, 0) & self._compare["passed"]
        found = set()
        while mask:
            low = mask & -mask
            found.add(low.bit_length() - 1)
            mask ^= low
        return found

    def ran_at_import(self, path: str, scope) -> bool:
        codes = self.map.import_codes
        if isinstance(scope, ast.Lambda):
            key = (path, scope.lineno, "<lambda>")
        else:
            first = min([scope.lineno] + [d.lineno for d in scope.decorator_list])
            key = (path, first, scope.name)
        return key in codes

    def diff_tests(self, changed: set[tuple], failed_modules: list[str],
                   arg_flips: set[str] | None = None) -> tuple[set[int], bool, list[str]]:
        """Tests reached by the bindings a fingerprint diff found changed. (tests, everything, notes)

        `arg_flips`: the keys whose required-ness the patch flipped in `arg_types` literals, when
        nothing else in sqlglot/ changed at import time (see Verdict.arg_flips)."""
        found: set[int] = new_tests()
        notes: list[str] = []
        if failed_modules:
            # which modules fail can depend on the order they are imported in (an import cycle
            # breaks whoever comes first), and the test run imports them in another order
            return set(), True, [f"patched sqlglot fails to import {', '.join(failed_modules)[:120]}"]
        if ("process", "*", "#effects") in changed:
            # importing the patched package changes process-wide state the pinned import leaves
            # alone (recursion limit, logging, warnings, environment, a stdlib attribute...)
            return set(), True, ["importing patched sqlglot changes process-wide state"]
        if ("import", "sqlglot", "#loaded") in changed:
            spawners = self.spawning_tests()
            found |= spawners
            notes.append(f"bare import loads other modules: {len(spawners)} tests start an interpreter")
        changed = {key for key in changed if key[0] != "import"}
        class_attrs = self.map.data.get("class_attrs", {})
        keyreaders = self.map.data.get("keyreaders", {})
        key_changes: dict[tuple[str, str], set[str]] = {}
        for key in changed:
            if key[0] == "kv":
                key_changes.setdefault((key[1], key[2]), set()).add(key[3])
        for key in changed:
            kind, owner, name = key[0], key[1], key[2]
            if kind in ("keys", "kv"):
                continue  # bookkeeping for the dict rules below
            if kind == "c" and name in VALUE_BLIND and ("keys", owner, name) not in changed:
                notes.append(f"{owner.rsplit('.', 1)[-1]}.{name}: values only")
                continue
            if kind == "c" and name == "required_args" and arg_flips and self.flips_reach(owner, changed):
                # required_args is read only by Expression.error_messages, where a key matters
                # only if the validated instance of this exact class lacks it (None or [])
                missing = keyreaders.get((owner, "#missing"), {})
                for key in arg_flips:
                    number = missing.get(key)
                    if number is not None:
                        found.update(self.map.sets[number])
                notes.append(f"{owner.rsplit('.', 1)[-1]}.required_args: {sorted(arg_flips)[:3]} missing in "
                             f"{sum(len(self.map.sets[missing[k]]) for k in arg_flips if k in missing)} tests")
                continue
            if kind == "c":
                here = self._class_key_tests(key, changed, key_changes, keyreaders, class_attrs, notes)
                if here is None:
                    return set(), True, [f"unknown class {owner}"]
                if (name.startswith("__") and name.endswith("__")) or (not here and not self.recorded_kind(owner, name)):
                    # nothing recorded reads it, or a member dataclasses, slots or the class
                    # machinery generate (no def to find callers of): whoever names it or the
                    # class, and whoever used the class
                    here |= self.tests_mentioning(name) | self.tests_mentioning(owner.rsplit(".", 1)[-1])
                    here |= self.members_changed(self.class_file(owner) or "", tuple(qual_in_module(owner, self.class_file(owner) or ""))) or set()
                    notes.append(f"{owner.rsplit('.', 1)[-1]}.{name}: no direct readers, users of the class")
                found |= here
            else:
                found |= self.tests_mentioning(name)
                qual = f"{owner}.{name}"
                if qual in self.map.classes:
                    found |= self.class_change(qual, set(class_attrs.get(qual, ())))
                path = self.module_path(owner)
                tree = self.tree(path) if path else None
                if tree is not None:
                    for item in tree.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                            found |= self.map.tests_on_lines(path, item.body[0].lineno, item.end_lineno)
        notes.append(f"diff: {len(changed)} bindings changed -> {len(found)} tests")
        return found, False, notes

    def private_test_module(self, path: str) -> bool:
        """A tests/**/test_*.py module that no other test file names, where every module or class
        level statement the patch adds, removes or changes only binds a name without other
        effects at import: then only the module's own tests can see the change."""
        if path in self._private:
            return self._private[path]
        self._private[path] = False
        name = os.path.basename(path)
        old_tree, new_tree = self.tree(path), self._new_trees.get(path)
        if not name.startswith("test_") or old_tree is None or new_tree is None:
            return False
        old, new = _scoped_statements(old_tree), _scoped_statements(new_tree)
        if not all(_quiet(stmt) for key, stmt in (old | new).items() if key not in old or key not in new):
            return False
        word = re.compile(rf"\b{re.escape(name[:-3])}\b")
        for other in sorted(set(self.map.test_file)) + ["tests/helpers.py", "tests/__init__.py"]:
            if other != path and word.search(self.source(other) or ""):
                return False
        self._private[path] = True
        return True

    def flips_reach(self, owner: str, changed: set[tuple]) -> bool:
        """True when owner's arg_types (its own or the one it inherits) is a changed literal whose
        keys stayed the same, and the map recorded missing arguments at validation."""
        if self._has_missing is None:
            self._has_missing = any(pair[1] == "#missing" for pair in self.map.data.get("keyreaders", {}))
        if not self._has_missing:
            return False
        class_attrs = self.map.data.get("class_attrs", {})
        for base in self.map.classes.get(owner, [owner]):
            if ("c", base, "arg_types") in changed:
                return ("keys", base, "arg_types") not in changed
            if "arg_types" in class_attrs.get(base, ()):
                return False  # inherits an arg_types the patch did not change
        return False

    _spawners: set[int] | None = None

    def spawning_tests(self) -> set[int]:
        """Tests whose body starts another process: the map cannot see what runs there, so
        they join any change to what a fresh interpreter does on `import sqlglot`."""
        if self._spawners is None:
            found: set[int] = set()
            index = {nodeid: i for i, nodeid in enumerate(self.map.tests)}
            for path in sorted(set(self.map.test_file)):
                text = self.source(path) or ""
                if not SPAWN.search(text):
                    continue
                tree = self.tree(path)
                if tree is None:
                    found |= self.map.tests_in_file(path)
                    continue
                for cls in tree.body:
                    if not isinstance(cls, ast.ClassDef):
                        continue
                    for item in cls.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                                SPAWN.search(ast.get_source_segment(text, item) or ""):
                            number = index.get(f"{path}::{cls.name}::{item.name}")
                            if number is not None:
                                found.add(number)
            self._spawners = found
        return self._spawners

    def recorded_kind(self, owner: str, name: str) -> bool:
        """Whether the map would have seen every use of owner.name: a def in the class body
        (its callers ran its lines) or a literal value (reads go through the recorded class
        attribute). Members a decorator, __slots__ or an assigned call produced are not."""
        path = self.class_file(owner)
        tree = self.tree(path) if path else None
        node = find_class(tree, qual_in_module(owner, path)) if tree is not None else None
        if node is None:
            return False
        literal = (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple, ast.JoinedStr, ast.UnaryOp, ast.BinOp,
                   ast.DictComp, ast.ListComp, ast.SetComp)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                return not item.decorator_list or all(
                    isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod") for d in item.decorator_list)
            if isinstance(item, (ast.Assign, ast.AnnAssign)) and name in analyze.bound_names(item):
                return isinstance(item.value, literal)
        return False

    def _class_key_tests(self, key, changed, key_changes, keyreaders, class_attrs, notes) -> set[int] | None:
        """Tests reached by one changed class-level binding ("c", class, attribute); None when
        the class cannot be placed at all."""
        owner, name = key[1], key[2]
        here: set[int] = set()
        if owner not in self.map.classes:
            # a class the map has never seen (new, or defined differently): its module's users
            module = owner.rsplit(".", 1)[0]
            while module and self.module_path(module) is None:
                module = module.rsplit(".", 1)[0] if "." in module else ""
            if not module:
                return None
            return self.module_users(module, self.module_path(module))
        if name == "__mro__":
            return self.class_change(owner, set(class_attrs.get(owner, ())))
        keyed = keyreaders.get((owner, name))
        entries = key_changes.get((owner, name))
        if keyed is not None and entries and "?" not in entries and name in class_attrs.get(owner, ()):
            # a dict whose lookups the map recorded key by key
            for entry in entries | {"*"}:
                number = keyed.get(entry)
                if number is not None:
                    here.update(self.map.sets[number])
            if ("keys", owner, name) in changed and keyed.get("#len") is not None:
                here.update(self.map.sets[keyed["#len"]])
            notes.append(f"{owner.rsplit('.', 1)[-1]}.{name}: keys {sorted(entries)[:3]}")
            return here
        if keyed is not None and not entries and ("keys", owner, name) in changed \
                and name in class_attrs.get(owner, ()):
            # same entries in a new order: only what iterates over the dict can tell
            number = keyed.get("*")
            if number is not None:
                here.update(self.map.sets[number])
            notes.append(f"{owner.rsplit('.', 1)[-1]}.{name}: key order")
            return here
        related = set(self.map.subclasses(owner)) | {owner}
        if name not in class_attrs.get(owner, ()):
            # new or removed attribute: it changes what lookups find on base classes
            related |= set(self.map.classes.get(owner, [owner]))
            here |= self.method_callers(owner, name, related)
        else:
            here |= self.method_callers(owner, name, {owner})
        for qual in related:
            here |= self.map.attr_readers(qual, name)
        here |= self.dispatch_readers(owner, name)
        return here

    def module_path(self, module: str) -> str | None:
        base = module.replace(".", "/")
        for candidate in (base + ".py", base + "/__init__.py"):
            if os.path.exists(os.path.join(self.checkout, candidate)):
                return candidate
        return None

    def module_users(self, module: str, path: str) -> set[int]:
        """Tests that would notice the module failing to import."""
        if module not in self._module_users:
            self._module_users[module] = self._module_users_uncached(module, path)
        return set(self._module_users[module])

    def _module_users_uncached(self, module: str, path: str) -> set[int]:
        found: set[int] = set()
        by_line = self.map.lines.get(path, {})
        for number in set(by_line.values()):
            found.update(self.map.sets[number])
        for qual in self.map.classes:
            if qual.startswith(module + "."):
                for attr in self.map.data.get("class_attrs", {}).get(qual, ()):
                    found |= self.map.attr_readers(qual, attr)
        last = module.rsplit(".", 1)[-1]
        word = re.compile(rf"\b{re.escape(last)}\b")
        names: dict[str, bool] = {}
        for index, test_path in enumerate(self.map.test_file):
            if test_path not in names:
                text = self.source(test_path) or ""
                # the word can only be found as a whole word where it is found at all
                names[test_path] = module in text or (last in text and word.search(text) is not None)
            if names[test_path]:
                found.add(index)
        return found

    def import_time_test(self, path, loc, names, verdict) -> None:
        """Import-level change in a test module: its tests, plus users of what it defines.
        A package __init__ is imported by every test module below it, and a module other test
        modules import (helpers, test_dialect's Validator) takes them all down if its import
        fails: both select every test that imports them."""
        if os.path.basename(path) == "__init__.py":
            verdict.all(f"{path}: import level change in a test package")
            return
        module = analyze.module_name(path)
        importers = self.module_users(module, path)
        own_tests = self.map.tests_in_file(path)
        if importers - own_tests or not os.path.basename(path).startswith("test_"):
            verdict.tests |= importers | own_tests
            verdict.notes.append(f"{path}: shared test module, {len(importers)} tests import it")
        found = self.map.tests_in_file(path)
        if self.private_test_module(path):
            # nothing outside the module sees its bindings, and binding them did nothing else
            verdict.tests |= found
            verdict.notes.append(f"{path}: test module level {sorted(names)} (quiet), {len(found)} tests")
            return
        own = set()
        if loc.stmt is not None:  # the changed binding itself is not a use of it
            own = set(range(loc.stmt.lineno, (loc.stmt.end_lineno or loc.stmt.lineno) + 1))
        for name in names:
            found |= self.tests_mentioning(name, skip=(path, own))
        if os.path.basename(path) in ("helpers.py", "__init__.py") or not os.path.basename(path).startswith("test_"):
            # imported by other test modules
            for name in names or {""}:
                if not name:
                    verdict.all(f"{path}: import level change in a shared test module")
                    return
        verdict.tests |= found
        verdict.notes.append(f"{path}: test module level {sorted(names)}, {len(found)} tests")


PYTEST_CONFIGS = {"pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"}
# configuration files of the checkout's root that pytest reads only for its own section:
# an edit in place goes on to config_problem, which compares what pytest would see
CHECKED_CONFIGS = {"pyproject.toml", "setup.cfg"}


def config_problem(name: str, old_text: str | None, new_text: str | None) -> str | None:
    """Why pytest could read this config file differently after the patch (None: it won't).
    pytest parses each of these while it looks for its configuration, so one that no
    longer parses stops the whole run."""
    if name in ("pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini"):
        return "pytest configuration file"
    if new_text is None:
        return None if old_text is None or "pytest" not in old_text else "pytest configuration removed"
    if name == "pyproject.toml":
        import tomllib

        try:
            new = tomllib.loads(new_text)
        except (tomllib.TOMLDecodeError, ValueError):
            return "does not parse as TOML"
        try:
            old = tomllib.loads(old_text) if old_text is not None else {}
        except (tomllib.TOMLDecodeError, ValueError):
            return "original does not parse"
        if old.get("tool", {}).get("pytest") != new.get("tool", {}).get("pytest"):
            return "pytest section changed"
        return None
    try:
        import iniconfig
    except ImportError:
        return "cannot check the ini syntax"
    try:
        iniconfig.IniConfig(name, data=new_text)
    except Exception:  # iniconfig.ParseError and anything else it raises: pytest would fail the same way
        return "does not parse as ini"
    return None


def shadowing(module: str) -> str | None:
    """A module at the checkout root is on sys.path first in every run: it can replace a
    library module or run at interpreter start."""
    if module in ("sitecustomize", "usercustomize", "sqlglot", "tests"):
        return "runs at start or shadows the package"
    import importlib.util
    import sys

    if module in sys.stdlib_module_names or module in sys.builtin_module_names:
        return "shadows a standard library module"
    try:
        if importlib.util.find_spec(module) is not None:
            return "shadows an installed module"
    except (ImportError, ValueError):
        return "shadows an installed module"
    return None


NEW_TEST_MODULE = re.compile(r"^tests/(?:[A-Za-z_]\w*/)*test_\w+\.py$")
REQUIREMENTS = re.compile(r"^(requirements.*\.(txt|in|lock)|constraints.*\.txt)$")
# Files outside sqlglot/ and tests/ that a test run never reads: documentation, CI and
# editor settings, packaging metadata (pytest's own config files are checked before this).
INERT = re.compile(
    r"^(?:[^/]+\.(?:md|rst|txt|png|svg|jpg|gif|yml|yaml|cfg|toml|in|json|lock)$"
    r"|(?:LICENSE|Makefile|setup\.py|CLAUDE\.md|AGENTS\.md|\.git\w*|\.pre-commit-config\.yaml|\.gitpod\.yml)$"
    r"|(?:posts|pdoc|benchmarks|docs|\.github|sqlglot-integration-tests)(?:/|$)"
    r"|sqlglot/py\.typed$)"
)
SPAWN = re.compile(r"\bsubprocess\b|\bPopen\b|sys\.executable|os\.system|os\.exec|os\.spawn|os\.posix_spawn")
_EQUALITY = ("Eq", "NotEq", "Is", "IsNot")
_ORDER = ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")
COMPARE_FAMILY = {op: tuple(x for x in _EQUALITY if x != op) for op in _EQUALITY}
COMPARE_FAMILY.update({op: tuple(x for x in _ORDER if x != op) for op in ("Lt", "LtE", "Gt", "GtE")})
COMPARE_FAMILY.update({"In": ("NotIn",), "NotIn": ("In",)})
CODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


def coding_cookie_changed(old_text: str, new_text: str) -> bool:
    old_head = [line for line in old_text.split("\n")[:2] if CODING.match(line)]
    new_head = [line for line in new_text.split("\n")[:2] if CODING.match(line)]
    return old_head != new_head


MUTATORS = re.compile(
    r"^(set|disable|enable|patch|register|unregister|update|clear|add|remove|discard|pop|append|extend|insert"
    r"|install|uninstall|configure|start|stop|reset|seed|delete|write|basicConfig|addHandler|removeHandler"
    r"|filterwarnings|simplefilter|chdir|putenv|unsetenv|freeze|collect|sort|reverse|move_to_end|__setitem__"
    r"|__delitem__|__setattr__|__delattr__)"
)
DYNAMIC_CALLS = {"setattr", "delattr", "exec", "eval", "globals", "__import__"}


def _root(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call, ast.Starred)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _calls_outside_functions(node) -> list:
    found, stack = [], [node]
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(current, ast.Call):
            found.append(current)
        stack.extend(ast.iter_child_nodes(current))
    return found


def process_effect(stmt) -> bool:
    """An import-level statement (or one nested in a module-level if/try/with) that does more
    than bind names: an expression statement other than a docstring, or an assignment or
    deletion whose target is an attribute or item of something."""
    stack = [stmt]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue  # a body runs when called, not at import: global_effects looks at those
        if isinstance(node, ast.Expr) and not (isinstance(node.value, ast.Constant)):
            return True
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and getattr(node, "value", None) is not None:
            # a binding whose value calls something that sets, patches or registers
            # (X = sys.setrecursionlimit(250), Y = logging.disable(...))
            for call in _calls_outside_functions(node.value):
                func = call.func
                name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
                if name in DYNAMIC_CALLS or MUTATORS.match(name):
                    return True
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, (ast.Attribute, ast.Subscript)) and isinstance(getattr(part, "ctx", None), (ast.Store, ast.Del)):
                        return True
        if isinstance(node, ast.stmt):  # class bodies and module-level if/try/with/for/while run at import
            for field in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(node, field, []) or []:
                    stack.extend(child.body if isinstance(child, ast.ExceptHandler) else [child])
    return False


def global_effects(tree) -> list[str]:
    """Every statement or call inside a function or lambda that reaches beyond the function's
    own frame: assigning or deleting through a name that is not local (module globals, imported
    modules, builtins), global/nonlocal declarations, and calls that set, patch, register or
    mutate through such a name. Compared as a whole between the old and the new file."""
    found: list[str] = []

    def scan(fn, path: str) -> None:
        local = set()
        args = fn.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
            if arg is not None:
                local.add(arg.arg)
        declared = set()
        body = fn.body if isinstance(fn.body, list) else [fn.body]
        for node in [n for b in body for n in ast.walk(b)]:
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                declared |= set(node.names)
                found.append(f"{path}:{type(node).__name__}:{','.join(node.names)}")
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local.add(node.id)
        local -= declared
        for node in [n for b in body for n in ast.walk(b)]:
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                for target in targets:
                    root = _root(target)
                    if isinstance(target, ast.Name) and target.id in declared or \
                            not isinstance(target, ast.Name) and root is not None and root not in local:
                        value = ast.dump(node.value) if getattr(node, "value", None) is not None else ""
                        found.append(f"{path}:set:{ast.dump(target)}={value}")
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in DYNAMIC_CALLS:
                    found.append(f"{path}:call:{ast.dump(node)}")
                elif isinstance(func, ast.Attribute) and MUTATORS.match(func.attr):
                    root = _root(func)
                    if root is None or root not in local:
                        found.append(f"{path}:call:{ast.dump(node)}")
                elif isinstance(func, ast.Name) and MUTATORS.match(func.id) and func.id not in local:
                    found.append(f"{path}:call:{ast.dump(node)}")

    def visit(node, path: str, in_function: bool = False) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef) and in_function and not all(
                    _dotted(base) in TEST_BASES for base in child.bases):
                # a class made when a function runs: its metaclass may register it process-wide
                found.append(f"{path}:class:{ast.dump(child)}")
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scan(child, f"{path}.{child.name}")
                visit(child, f"{path}.{child.name}", True)
            elif isinstance(child, ast.Lambda):
                scan(child, f"{path}.<lambda>")
                visit(child, path, in_function)
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{path}.{child.name}", in_function)
            else:
                visit(child, path, in_function)

    visit(tree, "")
    return sorted(found)


def effects_changed(old_tree, new_tree, old_text: str, new_text: str) -> bool:
    """global_effects(old_tree) != global_effects(new_tree), without walking what the two
    files share.

    global_effects is the sorted list of what each module-level statement contributes on its
    own, and a class contributes what its header and each statement of its body contribute
    under the class's name. Two statements written with the same characters at the same
    columns have the same tree, so the same contribution: such pairs are left out on both
    sides, which keeps the two lists equal or unequal as they were. Classes of one name that
    remain on both sides are compared the same way, body statement by body statement."""
    old_lines, new_lines = analyze._source_lines(old_text), analyze._source_lines(new_text)

    def source(stmt, lines) -> tuple:
        first = stmt.lineno
        for decorator in getattr(stmt, "decorator_list", ()):
            first = min(first, decorator.lineno)
        return (stmt.col_offset, stmt.end_col_offset, tuple(lines[first - 1 : stmt.end_lineno]))

    def unshared(body, keys, shared) -> list:
        left = dict(shared)
        rest = []
        for stmt, key in zip(body, keys):
            if left.get(key, 0) > 0:
                left[key] -= 1
            else:
                rest.append(stmt)
        return rest

    def reduce(old_body, new_body) -> tuple[list, list]:
        old_keys = [source(stmt, old_lines) for stmt in old_body]
        new_keys = [source(stmt, new_lines) for stmt in new_body]
        shared = collections.Counter(old_keys) & collections.Counter(new_keys)
        old_rest, new_rest = unshared(old_body, old_keys, shared), unshared(new_body, new_keys, shared)
        old_names = collections.Counter(s.name for s in old_rest if isinstance(s, ast.ClassDef))
        new_names = collections.Counter(s.name for s in new_rest if isinstance(s, ast.ClassDef))
        for name in [n for n in old_names if old_names[n] == 1 and new_names.get(n) == 1]:
            i = next(k for k, s in enumerate(old_rest) if isinstance(s, ast.ClassDef) and s.name == name)
            j = next(k for k, s in enumerate(new_rest) if isinstance(s, ast.ClassDef) and s.name == name)
            old_inner, new_inner = reduce(old_rest[i].body, new_rest[j].body)
            old_rest[i], new_rest[j] = _with_body(old_rest[i], old_inner), _with_body(new_rest[j], new_inner)
        return old_rest, new_rest

    try:
        old_rest, new_rest = reduce(old_tree.body, new_tree.body)
    except (AttributeError, TypeError):  # a statement without a position: compare the whole files
        return global_effects(old_tree) != global_effects(new_tree)
    return global_effects(ast.Module(body=old_rest, type_ignores=[])) != \
        global_effects(ast.Module(body=new_rest, type_ignores=[]))


def _with_body(cls: ast.ClassDef, body: list) -> ast.ClassDef:
    """The class with the same header and another body."""
    return ast.ClassDef(name=cls.name, bases=cls.bases, keywords=cls.keywords, body=body,
                        decorator_list=cls.decorator_list, type_params=getattr(cls, "type_params", []))


SHARED_ROOTS = {"cls", "mcs", "metacls", "klass"}


def writes_shared_state(fn, outer: frozenset = frozenset(), class_level: list | None = None) -> bool:
    """A function that assigns through its class (cls., type(self)., self.__class__.), through
    a global, or through a name that belongs to no enclosing function (a module, a class such
    as TestExecutor.cache): what it writes stays after it returns. Writes to a function's own
    locals, its parameters' objects and enclosing functions' locals (nonlocal) do not."""
    if isinstance(fn, ast.Lambda):
        return False
    args = fn.args
    local = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    local |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    declared_global, declared_nonlocal = set(), set()
    own = []  # this function's nodes, not those of nested functions
    stack = list(fn.body)
    nested = []
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested.append(node)
            local.add(node.name)
            continue
        if isinstance(node, (ast.Lambda, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                local.add(node.name)
            continue
        own.append(node)
        if isinstance(node, ast.Global):
            declared_global |= set(node.names)
        elif isinstance(node, ast.Nonlocal):
            declared_nonlocal |= set(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local.add(node.id)
        elif isinstance(node, ast.alias):
            local.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            local.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            local.add(node.name)
        stack.extend(ast.iter_child_nodes(node))
    local -= declared_global
    visible = (outer | local | declared_nonlocal) - declared_global
    fresh = set()  # cls.<attr> this function assigns outright: the class's own new object
    for node in own:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in SHARED_ROOTS:
                    fresh.add(target.attr)
    for node in own:
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in declared_global:
                    return True
                base = target
                through_class = False
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    if isinstance(base, ast.Attribute) and base.attr == "__class__" and base is not target:
                        through_class = True  # self.__class__.x = ...: the class, not the instance
                    base = base.value
                    if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "type":
                        through_class = True
                if through_class:
                    if class_level is not None:
                        class_level.append(target)
                    return True
                if base is not target:
                    if isinstance(base, ast.Name) and base.id in SHARED_ROOTS:
                        # cls.attr = ... sets the class's own attribute (class-level state); going
                        # further (cls.registry[key] = ...) mutates what the class may share,
                        # unless this function made that object (cls._COMMENTS = {...} first)
                        inner = target
                        while isinstance(inner, (ast.Attribute, ast.Subscript)) and inner.value is not base:
                            inner = inner.value
                        direct = inner is target
                        own_object = isinstance(inner, ast.Attribute) and inner.attr in fresh
                        if not direct and not own_object:
                            return True
                        if class_level is not None:
                            class_level.append(target)
                        continue
                    if isinstance(base, ast.Name) and base.id not in visible:
                        return True  # through a module, a global or a class name (TestExecutor.cache)
                    if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "type":
                        return True
                    if isinstance(base, ast.Attribute) and base.attr == "__class__":
                        return True
    return any(writes_shared_state(inner, frozenset(visible), class_level) for inner in nested)


_KIND_FLAGS = 0x20 | 0x80 | 0x100 | 0x200  # generator, coroutine, iterable coroutine, async generator


def _code_kinds(code) -> dict[str, list]:
    """qualified name -> [(locals, cells, frees, globals used, kind flags, first line)] for every
    function code object under a module's code, from the compiler's own symbol tables."""
    found: dict[str, list] = {}
    stack = [code]
    while stack:
        current = stack.pop()
        for const in current.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
        if current.co_name in ("<module>",) or current.co_flags & 0x02 == 0:  # module or class body
            continue
        found.setdefault(current.co_qualname, []).append((
            frozenset(current.co_varnames), frozenset(current.co_cellvars), frozenset(current.co_freevars),
            current.co_flags & _KIND_FLAGS, current.co_firstlineno))
    return found


def changed_function_kinds(old_code, new_code, old_tree) -> list[tuple[int, int]]:
    """Line spans (in the old file) of functions whose locals, cells, frees or generator/
    coroutine kind differ between the old and new module code."""
    old, new = _code_kinds(old_code), _code_kinds(new_code)
    spans = []
    defs = None  # first line -> the old file's defs there; built only if some function differs
    for qualname, entries in old.items():
        other = new.get(qualname)
        mine = sorted((e[:4] for e in entries), key=repr)
        theirs = sorted((e[:4] for e in other), key=repr) if other else None
        if mine == theirs:
            continue
        if defs is None:
            defs = {}
            for node in ast.walk(old_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    defs.setdefault(first, []).append(node)
        for entry in entries:
            for node in defs.get(entry[4], []):
                spans.append((node.lineno, node.end_lineno))
    return spans


UNHANDLED_HEADERS = ("copy from ", "copy to ", "old mode ", "new mode ", "similarity index ", "dissimilarity index ",
                     "rename from ", "rename to ", "GIT binary patch", "Binary files ")


def patch_shape_problem(text: str, patches) -> str | None:
    """A second, independent read of the patch's file headers: if it disagrees with the
    parser on how many files there are, or the patch uses a header the analysis does not
    fully handle, nothing narrower than every test is safe."""
    lines = text.split("\n")
    git_headers = sum(1 for line in lines if line.startswith("diff --git "))
    pairs = sum(1 for a, b in zip(lines, lines[1:]) if a.startswith("--- ") and b.startswith("+++ "))
    if git_headers and git_headers != len(patches):
        return f"patch has {git_headers} file headers, parsed {len(patches)} files"
    if not git_headers and pairs != len(patches):
        return f"patch has {pairs} ---/+++ pairs, parsed {len(patches)} files"
    if git_headers and pairs > git_headers:
        return "more ---/+++ pairs than file headers"
    for line in lines:
        if line.startswith(UNHANDLED_HEADERS):
            return f"patch header not analysed: {line[:40]}"
    return None


PICKLING_HOOKS = {"__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__getnewargs__",
                  "__getnewargs_ex__", "__copy__", "__deepcopy__"}


def _effect_call(node) -> bool:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    return name in DYNAMIC_CALLS or bool(MUTATORS.match(name))


def import_effects(tree) -> list[str]:
    """What runs at import and acts beyond binding names, as dumps: process_effect statements,
    and effect calls in binding values, default arguments, decorators and class headers."""
    found: list[str] = []

    def calls_in(nodes) -> list[str]:
        return [ast.dump(call) for node in nodes if node is not None
                for call in _calls_outside_functions(node) if _effect_call(call)]

    def visit(body, path: str) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = stmt.args
                header = list(stmt.decorator_list) + list(args.defaults) + [d for d in args.kw_defaults if d is not None]
                found.extend(f"{path}.{stmt.name}:{c}" for c in calls_in(header))
                continue
            if isinstance(stmt, ast.ClassDef):
                header = list(stmt.decorator_list) + list(stmt.bases) + [k.value for k in stmt.keywords]
                found.extend(f"{path}.{stmt.name}:{c}" for c in calls_in(header))
                visit(stmt.body, f"{path}.{stmt.name}")
                continue
            if isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                found.extend(f"{path}:{c}" for c in calls_in([getattr(stmt, "test", None), getattr(stmt, "iter", None)]
                                                             + [i.context_expr for i in getattr(stmt, "items", [])]))
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(stmt, field, []) or [], path)
                for handler in getattr(stmt, "handlers", []) or []:
                    visit(handler.body, path)
                continue
            if process_effect(stmt):
                found.append(f"{path}:stmt:{ast.dump(stmt)}")
            else:
                found.extend(f"{path}:{c}" for c in calls_in([stmt]))

    visit(tree.body, "")
    return sorted(found)


def changed_fixture_classes(old_tree, new_tree) -> list[str]:
    """Test classes whose methods' writes or mutating calls on class-level state differ: through
    cls, type(self) or self.__class__, or on an attribute the class sets at class level
    (cls.conn = ... in setUpClass, or in the class body) reached through self."""
    def effects(tree) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            shared = {t.attr for n in ast.walk(cls) if isinstance(n, (ast.Assign, ast.AnnAssign))
                      for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                      if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in SHARED_ROOTS}
            shared |= {name for item in cls.body if isinstance(item, (ast.Assign, ast.AnnAssign))
                       for name in analyze.bound_names(item)}

            def class_level(node) -> bool:
                chain = []
                while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
                    chain.append(node)
                    node = node.func if isinstance(node, ast.Call) else node.value
                if isinstance(node, ast.Name) and node.id in SHARED_ROOTS:
                    return True
                if any(isinstance(n, ast.Attribute) and n.attr == "__class__" for n in chain):
                    return True
                if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "type" for n in chain):
                    return True
                if isinstance(node, ast.Name) and node.id == "self" and chain:
                    first = chain[-1]  # the attribute right after self
                    return isinstance(first, ast.Attribute) and first.attr in shared
                return False

            items = []
            for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for node in ast.walk(fn):
                    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                        for target in targets:
                            if isinstance(target, (ast.Attribute, ast.Subscript)) and class_level(target):
                                items.append(f"{fn.name}:set:{ast.dump(node)}")
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and class_level(node.func.value):
                        items.append(f"{fn.name}:call:{ast.dump(node)}")
            result[cls.name] = sorted(items)
        return result

    old, new = effects(old_tree), effects(new_tree)
    return sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))


def pickling_hooks_changed(old_tree, new_tree) -> bool:
    def hooks(tree) -> list[str]:
        return sorted(ast.dump(n) for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in PICKLING_HOOKS)
    return hooks(old_tree) != hooks(new_tree)


POOL_CALLS = {"ProcessPoolExecutor", "Pool"}  # concurrent.futures, multiprocessing / get_context(...)


def _makes_pool(node, class_body: bool) -> bool:
    """cls.x = ProcessPoolExecutor(...) / ...Pool(...) anywhere in a class, or x = ... directly in
    its body: one process pool that all the class's tests share."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    shared = class_body or any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                               and t.value.id in SHARED_ROOTS for t in targets)
    if not shared:
        return False
    for call in ast.walk(node.value):
        if isinstance(call, ast.Call):
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name in POOL_CALLS:
                return True
    return False


PROCESS_NAMES = frozenset((
    "sys", "os", "builtins", "__builtins__", "gc", "signal", "atexit", "locale", "warnings", "logging",
    "importlib", "__import__", "faulthandler", "resource", "ctypes", "threading", "multiprocessing",
    "setrecursionlimit", "setswitchinterval", "set_int_max_str_digits", "settrace", "setprofile",
    "environ", "putenv", "unsetenv", "chdir", "umask", "getcontext", "setcontext", "basicConfig",
    "setLoggerClass", "setLogRecordFactory", "register_at_fork", "excepthook", "displayhook",
    "__call__", "__globals__", "__code__", "__defaults__", "__kwdefaults__", "__builtins__",
    "meta_path", "path_hooks", "exec", "eval",
))


def _names(node) -> set[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
    return found


def docstring_to_docstring(loc) -> bool:
    """A docstring edit that leaves a docstring (not one replaced by a statement or removed)."""
    new = loc.new_node
    return not loc.inserted and (new is None or isinstance(new, ast.Expr) and isinstance(new.value, ast.Constant)
                                 and isinstance(new.value.value, str))


def member_changes(old_tree, new_tree, old_text: str | None = None, new_text: str | None = None) -> list[tuple]:
    """Class paths (() for the module) whose set of bound names differs between the trees.
    With the texts, the names a statement binds are worked out once for all the statements
    written with the same characters at the same columns (they have the same tree): nearly
    every statement of the new file is one the old file has too."""
    known: dict[tuple, set[str]] = {}

    def bound(stmt, lines) -> set[str]:
        if lines is None or isinstance(stmt, _NAMED_STATEMENTS):
            return analyze.bound_names(stmt)  # a def, class or import: read off the node
        try:
            key = (stmt.col_offset, stmt.end_col_offset, tuple(lines[stmt.lineno - 1 : stmt.end_lineno]))
        except (AttributeError, TypeError):  # a statement without a position
            return analyze.bound_names(stmt)
        names = known.get(key)
        if names is None:
            names = known[key] = analyze.bound_names(stmt)
        return names

    def scopes(tree, text) -> dict[tuple, set[str]]:
        found: dict[tuple, set[str]] = {}
        lines = analyze._source_lines(text) if text is not None else None

        def visit(body, owner: tuple) -> None:
            names = found.setdefault(owner, set())
            for stmt in body:
                names |= bound(stmt, lines)
                if isinstance(stmt, ast.ClassDef):
                    visit(stmt.body, owner + (stmt.name,))
                elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    for field in ("body", "orelse", "finalbody", "handlers"):
                        for inner in getattr(stmt, field, []) or []:
                            if isinstance(inner, ast.ExceptHandler):
                                visit(inner.body, owner)
                            elif isinstance(inner, ast.stmt):
                                visit([inner], owner)
        visit(tree.body, ())
        return found

    old, new = scopes(old_tree, old_text), scopes(new_tree, new_text)
    return sorted(scope for scope in set(old) | set(new) if old.get(scope) != new.get(scope))


_NAMED_STATEMENTS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)


def comment_like(fp: patchfile.FilePatch) -> bool:
    """Every added or removed line is blank or an SQL `--` comment."""
    for hunk in fp.hunks:
        for line in hunk.lines:
            if line[:1] in ("+", "-"):
                body = line[1:].strip()
                if body and not body.startswith("--"):
                    return False
    return True


def _scoped_statements(tree) -> dict[tuple, ast.stmt]:
    """(class path, statement dump) -> statement, for module and class level statements
    (a class contributes its header; its body statements are listed on their own)."""
    found: dict[tuple, ast.stmt] = {}

    def visit(body, owner: tuple) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                header = ast.ClassDef(stmt.name, stmt.bases, stmt.keywords, [], stmt.decorator_list,
                                      getattr(stmt, "type_params", []))
                found[(owner, ast.dump(header))] = header
                visit(stmt.body, owner + (stmt.name,))
            else:
                found[(owner, ast.dump(stmt))] = stmt
    visit(tree.body, ())
    return found


def _plain(node) -> bool:
    """An expression whose evaluation calls nothing (names, constants, attribute loads, operators)."""
    return node is None or not any(
        isinstance(sub, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)) for sub in ast.walk(node)
    )


TEST_BASES = {"TestCase", "unittest.TestCase", "Validator", "object"}


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _quiet(stmt) -> bool:
    """A module or class level statement whose only effect at import is binding names."""
    if isinstance(stmt, (ast.Pass,)):
        return True
    if isinstance(stmt, ast.Expr):
        return isinstance(stmt.value, ast.Constant)
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        return all(isinstance(target, ast.Name) for target in targets) and _plain(stmt.value) and \
            (not isinstance(stmt, ast.AnnAssign) or _plain(stmt.annotation))
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = stmt.args
        return all(isinstance(d, (ast.Name, ast.Attribute)) for d in stmt.decorator_list) and \
            all(_plain(d) for d in args.defaults + [d for d in args.kw_defaults if d is not None])
    if isinstance(stmt, ast.ClassDef):
        # only plain test case classes: any other base can run a metaclass or __init_subclass__
        # that registers the class for the whole process (class DuckDB(Dialect) does)
        return not stmt.decorator_list and not stmt.keywords and all(
            _dotted(base) in TEST_BASES for base in stmt.bases)
    if isinstance(stmt, ast.Import):
        return all(alias.name.split(".")[0] not in ("sqlglot", "tests") for alias in stmt.names)
    if isinstance(stmt, ast.ImportFrom):
        return stmt.level == 0 and (stmt.module or "").split(".")[0] not in ("sqlglot", "tests") and \
            all(alias.name != "*" for alias in stmt.names)
    return False


def _arg_types_values(tree) -> dict[str, ast.expr]:
    """Class path -> the value of its `arg_types = ...` assignment, for every class in the module."""
    found: dict[str, ast.expr] = {}

    def visit(body, prefix: str) -> None:
        for node in body:
            if not isinstance(node, ast.ClassDef):
                continue
            name = prefix + node.name
            for item in node.body:
                targets = item.targets if isinstance(item, ast.Assign) else \
                    [item.target] if isinstance(item, ast.AnnAssign) and item.value is not None else []
                if any(isinstance(target, ast.Name) and target.id == "arg_types" for target in targets):
                    found[name] = item.value
            visit(node.body, name + ".")

    visit(tree.body, "")
    return found


def arg_type_flips(old_tree, new_tree) -> set[str] | None:
    """Keys whose truthiness differs between the old and new `arg_types` dict literals of the
    module's classes (what required_args is derived from). None when a changed one is not a
    plain dict literal."""
    old, new = _arg_types_values(old_tree), _arg_types_values(new_tree)
    flips: set[str] = set()
    for name in set(old) | set(new):
        a, b = old.get(name), new.get(name)
        if a is not None and b is not None and ast.dump(a) == ast.dump(b):
            continue
        try:
            x = ast.literal_eval(a) if a is not None else {}
            y = ast.literal_eval(b) if b is not None else {}
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
        if not isinstance(x, dict) or not isinstance(y, dict):
            return None
        flips |= {key for key in set(x) | set(y) if bool(x.get(key)) != bool(y.get(key))}
    return flips


def qual_in_module(qual: str, path: str) -> list[str]:
    module = analyze.module_name(path)
    rest = qual[len(module) + 1 :] if qual.startswith(module + ".") else qual
    return rest.split(".")


def find_class(tree, names: list[str]):
    body = tree.body
    node = None
    for name in names:
        node = None
        for item in body:
            if isinstance(item, ast.ClassDef) and item.name == name:
                node = item
                break
        if node is None:
            return None
        body = node.body
    return node


# ------------------------------------------------------------ test code that reaches only itself
#
# The suite runs its test modules in one process (or a few): a test that writes anything outside
# its own frame (a class or module attribute, a registry, a shared fixture, the process) can change
# what later tests see. `self_contained` accepts a function only when every write it makes stays in
# its own locals, its own instance (self.x = ...), its own class for a classmethod (cls.x = ...), or
# a container it built itself ([] {} or a comprehension); methods it calls on self and functions of
# the test modules it calls are held to the same rule. Library calls (sqlglot, the standard library)
# are taken as the pinned suite runs them: the other tests make the same calls.

TEST_API = re.compile(r"^(?:assert\w*|fail\w*|subTest|skipTest|addCleanup|addClassCleanup|enterContext|id"
                      r"|shortDescription|addTypeEqualityFunc|countTestCases|setUp|tearDown)$")
# names MUTATORS takes for mutators by their first letters that only read
READERS = {"startswith", "removeprefix", "removesuffix"}
TEST_DECORATORS = {"unittest.skip", "unittest.skipIf", "unittest.skipUnless", "unittest.expectedFailure",
                   "skip", "skipIf", "skipUnless", "expectedFailure", "classmethod", "staticmethod"}
# names whose use can reach the process or other objects' internals, or patch them
CLOSED_NAMES = (PROCESS_NAMES - {"os"}) | {
    "getattr", "setattr", "delattr", "hasattr", "vars", "globals", "locals", "exec", "eval", "compile",
    "__import__", "open", "breakpoint", "input", "exit", "quit", "random", "mock", "patch", "Mock", "MagicMock",
    "PropertyMock", "shutil", "subprocess", "pathlib", "Path", "tempfile", "socket", "asyncio", "concurrent",
    "ProcessPoolExecutor", "ThreadPoolExecutor", "weakref", "copyreg", "inspect", "functools", "sqlite3", "duckdb",
}
OPEN_DUNDERS = {"__name__", "__doc__"}


BORROWING_CALLS = {"getattr", "type", "vars", "super"}


def _base(node):
    """What an attribute/item chain hangs from (a.b[c].d -> a)."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node


def _bound_names(target) -> set[str]:
    """The names an assignment target binds (x, (a, b), *rest), not those of x.attr = or x[i] =."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _bound_names(item)}
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return set()


def alias_writers(tree) -> collections.Counter:
    """The functions of a module (as their trees, positions left out) that write to something they
    did not build: through a global or imported name, through a local bound to one or to what is
    reached from one (x = exp.Literal), through a loop variable over such things (for c in
    (exp.A, exp.B), for d in Dialect.classes.values()), through what getattr, type, vars or super
    give, or through a parameter other than self/cls. Writing: assigning or deleting an attribute
    or item, or naming a mutating method (MUTATORS), called or handed on as a callback."""
    found: collections.Counter = collections.Counter()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        params = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs + [a for a in (args.vararg, args.kwarg) if a]]
        stored = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        own = stored | set(params)
        # the instance or class a method gets, unless the function binds that name to something else
        first = params[0] if params and params[0] in ("self", "cls") and params[0] not in stored else None
        borrowed = set(params[1:] if first is not None else params) - ({params[0]} if params and params[0] in ("self", "cls") else set())
        called = {id(node.func) for node in ast.walk(fn) if isinstance(node, ast.Call)}

        def lent(node) -> bool:
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                return any(lent(item) for item in node.elts)
            base = _base(node)
            if isinstance(base, ast.Call):
                return _dotted(base.func) in BORROWING_CALLS
            if isinstance(base, ast.Name):
                return base.id != first and (base.id in borrowed or base.id not in own)
            return False

        def lent_items(node) -> bool:
            if lent(node):
                return True
            base = _base(node)
            if isinstance(base, ast.Call):  # .values(), sorted(...), zip(...) over something lent
                receiver = base.func.value if isinstance(base.func, ast.Attribute) else None
                return (receiver is not None and lent(receiver)) or any(lent(arg) for arg in base.args)
            return False

        for _ in range(3):  # aliases of aliases
            for node in ast.walk(fn):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and getattr(node, "value", None) is not None:
                    if lent(node.value):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        borrowed |= {name for t in targets for name in _bound_names(t)}
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    if lent_items(node.iter):
                        borrowed |= _bound_names(node.target)
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        if item.optional_vars is not None and lent(item.context_expr):
                            borrowed |= _bound_names(item.optional_vars)

        def writes(node) -> bool:
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                return lent(node.value)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and MUTATORS.match(node.attr) \
                    and node.attr not in READERS:
                if id(node) not in called:
                    return lent(node.value)  # handed on as a callback
                # called: through a local alias here (a call through a global name is the effect
                # rules' to judge, and many are module functions: exp.update builds an UPDATE)
                base = _base(node.value)
                return (isinstance(base, ast.Name) and base.id in borrowed and base.id in own) or \
                    (isinstance(base, ast.Call) and _dotted(base.func) in BORROWING_CALLS)
            return False

        if any(writes(node) for node in ast.walk(fn)):
            found[ast.dump(fn)] += 1
    return found


def _header(fn) -> str:
    """What of a def runs when its class is made: decorators, defaults, annotations."""
    return ast.dump(ast.FunctionDef(name=fn.name, args=fn.args, body=[], decorator_list=fn.decorator_list,
                                    returns=fn.returns, type_params=getattr(fn, "type_params", [])))


def _fresh_locals(fn) -> set[str]:
    """Names this function binds only to containers it builds itself ([...], {...}, comprehensions)."""
    built: dict[str, bool] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            fresh = isinstance(node.value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp))
            built[name] = built.get(name, True) and fresh
        else:
            for sub in (node.targets if isinstance(node, ast.Assign) else
                        [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor,
                                                           ast.NamedExpr, ast.comprehension)) else
                        [item.optional_vars for item in node.items if item.optional_vars is not None]
                        if isinstance(node, (ast.With, ast.AsyncWith)) else []):
                for name in ast.walk(sub):
                    if isinstance(name, ast.Name):
                        built[name.id] = False
            if isinstance(node, ast.ExceptHandler) and node.name:
                built[node.name] = False
            if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
                # a nested function's parameter of the same name is another variable
                inner = node.args
                for arg in inner.posonlyargs + inner.args + inner.kwonlyargs + [a for a in (inner.vararg, inner.kwarg) if a]:
                    built[arg.arg] = False
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
                built[node.name] = False
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    built[(alias.asname or alias.name).split(".")[0]] = False
    args = fn.args
    for arg in args.posonlyargs + args.args + args.kwonlyargs + [a for a in (args.vararg, args.kwarg) if a]:
        built[arg.arg] = False
    return {name for name, fresh in built.items() if fresh}


def _decorators_open(fn) -> str | None:
    for decorator in fn.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _dotted(target) not in TEST_DECORATORS:
            return f"decorator {_dotted(target) or '?'}"
        if isinstance(decorator, ast.Call) and not all(
                _plain(arg) for arg in decorator.args + [kw.value for kw in decorator.keywords]):
            return "decorator arguments call something"
    return None


class Scope:
    """What the names of one test module are bound to, for self_contained: its classes' methods
    and their class-body constants (with Validator's, when it imports Validator), its own module
    functions, classes, immutable constants and other bindings, and the names it imports."""

    def __init__(self, tree, base_scope=None):
        self.methods: dict[str, list] = {}
        self.class_constants: set[str] = set()
        self.class_variables: set[str] = set()
        self.functions: dict[str, list] = {}
        self.classes: set[str] = set()
        self.constants: set[str] = set()
        self.others: set[str] = set()
        self.base = base_scope  # Validator's module, for its methods
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                self.classes.add(stmt.name)
                self._class_body(stmt.body)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(stmt.name, []).append(stmt)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            (self.constants if _immutable(stmt.value) else self.others).add(name.id)
            elif not isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Expr, ast.Pass)):
                # compound statements (if, try, with, for): whatever they bind is not a plain constant
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        self.others.add(node.id)
                    elif isinstance(node, ast.ClassDef):
                        self.classes.add(node.name)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.others.add(node.name)
        self.constants -= self.others | self.classes | set(self.functions)

    def _class_body(self, body) -> None:
        for item in body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods.setdefault(item.name, []).append((item, self))
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        (self.class_constants if _immutable(item.value) else self.class_variables).add(target.id)

    def method(self, name: str) -> list:
        found = list(self.methods.get(name, []))
        if self.base is not None:
            found += self.base.methods.get(name, [])
        return found

    def constant_attribute(self, name: str) -> bool:
        variable = name in self.class_variables or (self.base is not None and name in self.base.class_variables)
        constant = name in self.class_constants or (self.base is not None and name in self.base.class_constants)
        return constant and not variable


def _immutable(node) -> bool:
    """A literal that no one can change in place: a constant, or a tuple or frozenset of them."""
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple):
        return all(_immutable(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return True
    return False


UNITTEST_ATTRIBUTES = {"maxDiff", "longMessage", "failureException"}


def self_contained(fn, scope: Scope, seen: set | None = None) -> str | None:
    """None when fn (a FunctionDef of the module `scope` describes) can write nothing outside its
    own frame and instance, else why it may. Shared objects are only reachable through imported
    (library) names here: self is used only for its checked methods, unittest's API, class-body
    constants and the instance attributes the function sets itself; never handed on as a value;
    the module's own classes and mutable globals are not used; what a module function or method
    it reaches does is checked the same way."""
    seen = set() if seen is None else seen
    if id(fn) in seen:
        return None
    seen.add(id(fn))
    problem = _decorators_open(fn)
    if problem:
        return f"{fn.name}: {problem}"
    args = fn.args.posonlyargs + fn.args.args
    # the instance (or class) a method gets; a plain function's first parameter is just a parameter
    first = args[0].arg if args and args[0].arg in ("self", "cls") else None
    is_class_method = any(_dotted(d) == "classmethod" for d in fn.decorator_list)
    fresh = _fresh_locals(fn)
    parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
    own_attributes = {node.attr for node in ast.walk(fn) if isinstance(node, ast.Attribute)
                      and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id == first}

    def callees(name: str) -> str | None:
        found = scope.method(name)
        if not found:
            return f"{fn.name}: self.{name}, not found"
        for callee, callee_scope in found:
            problem = self_contained(callee, callee_scope, seen)
            if problem:
                return problem
        return None

    for node in ast.walk(fn):
        if isinstance(node, ast.Global):
            return f"{fn.name}: global"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super":
            parent = parents.get(node)
            if node.args or not (isinstance(parent, ast.Attribute) and parent.value is node) \
                    or not (isinstance(parents.get(parent), ast.Call) and parents[parent].func is parent):
                return f"{fn.name}: super() not used for a method call"
            if not (TEST_API.match(parent.attr) and not scope.method(parent.attr)):
                problem = callees(parent.attr)
                if problem:
                    return problem
            continue
        if isinstance(node, ast.Name):
            if node.id in CLOSED_NAMES:
                return f"{fn.name}: uses {node.id}"
            if node.id == "os" and not (isinstance(parents.get(node), ast.Attribute) and parents[node].attr == "path"):
                return f"{fn.name}: uses os"
            if first is not None and node.id == first:
                parent = parents.get(node)
                if not (isinstance(parent, ast.Attribute) and parent.value is node):
                    return f"{fn.name}: hands {first} on"
                if isinstance(parent.ctx, (ast.Store, ast.Del)):
                    if first == "cls" and not is_class_method:
                        return f"{fn.name}: sets cls.{parent.attr}"
                    continue
                called = isinstance(parents.get(parent), ast.Call) and parents[parent].func is parent
                if TEST_API.match(parent.attr) and not scope.method(parent.attr):
                    continue
                if scope.method(parent.attr):
                    problem = callees(parent.attr)
                    if problem:
                        return problem
                    continue
                if called:
                    return f"{fn.name}: self.{parent.attr}, not found"
                if parent.attr in own_attributes or parent.attr in UNITTEST_ATTRIBUTES or \
                        scope.constant_attribute(parent.attr):
                    continue
                return f"{fn.name}: reads {first}.{parent.attr}"
            if isinstance(node.ctx, ast.Load):
                if node.id in scope.classes or node.id in scope.others:
                    return f"{fn.name}: uses the module's {node.id}"
                for callee in scope.functions.get(node.id, ()):
                    problem = self_contained(callee, scope, seen)
                    if problem:
                        return problem
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in OPEN_DUNDERS:
                return f"{fn.name}: {node.attr}"
            owner = node.value
            on_self = isinstance(owner, ast.Name) and first is not None and owner.id == first
            on_super = isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "super"
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                if not on_self and not (isinstance(owner, ast.Name) and owner.id in fresh):
                    return f"{fn.name}: sets {_dotted(node) or 'an attribute'}"
            elif MUTATORS.match(node.attr) and node.attr not in READERS and not on_self and not on_super \
                    and not (isinstance(owner, ast.Name) and owner.id in fresh):
                return f"{fn.name}: {node.attr} of {_dotted(owner) or 'something'}"
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if not (isinstance(node.value, ast.Name) and node.value.id in fresh):
                return f"{fn.name}: sets an item of {_dotted(node.value) or 'something'}"
        elif isinstance(node, ast.ClassDef):
            if node.decorator_list or node.keywords or not all(_dotted(b) in TEST_BASES for b in node.bases):
                return f"{fn.name}: makes class {node.name}"
    return None


def test_method_edits(old_tree, new_tree, old_text: str, new_text: str) -> dict | None:
    """When the only difference between two versions of a test module is in test methods (defs
    named test* directly in a class body: changed, added or removed) and module-level import
    statements added, {class: {"changed": [(old def, new def)], "added": [new def], "removed": [old
    def]}, None: [added import statements]}; None for any other difference."""
    old_lines, new_lines = analyze._source_lines(old_text), analyze._source_lines(new_text)

    def text(stmt, lines) -> tuple:
        first = min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list", ())])
        return (stmt.col_offset, tuple(lines[first - 1: stmt.end_lineno]))

    def header(cls, lines) -> tuple:
        first = min([cls.lineno] + [d.lineno for d in cls.decorator_list])
        body_start = cls.body[0].lineno if cls.body else cls.end_lineno + 1
        if cls.body and getattr(cls.body[0], "decorator_list", None):
            body_start = min([body_start] + [d.lineno for d in cls.body[0].decorator_list])
        return (cls.col_offset, tuple(lines[first - 1: body_start - 1]))

    def is_test_def(stmt) -> bool:
        return isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name.startswith("test")

    old_level = [s for s in old_tree.body if not isinstance(s, ast.ClassDef)]
    new_level = [s for s in new_tree.body if not isinstance(s, ast.ClassDef)]
    old_keys = [text(s, old_lines) for s in old_level]
    added_imports = []
    position = 0
    for stmt in new_level:  # the old statements, in order, with import statements put in between
        if position < len(old_keys) and text(stmt, new_lines) == old_keys[position]:
            position += 1
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            added_imports.append(stmt)
        else:
            return None
    if position != len(old_keys):
        return None
    old_classes = [s for s in old_tree.body if isinstance(s, ast.ClassDef)]
    new_classes = [s for s in new_tree.body if isinstance(s, ast.ClassDef)]
    if [c.name for c in old_classes] != [c.name for c in new_classes] or \
            len({c.name for c in old_classes}) != len(old_classes):
        return None
    edits: dict[str, dict] = {}
    for old_cls, new_cls in zip(old_classes, new_classes):
        if header(old_cls, old_lines) != header(new_cls, new_lines):
            return None
        if [text(s, old_lines) for s in old_cls.body if not is_test_def(s)] != \
                [text(s, new_lines) for s in new_cls.body if not is_test_def(s)]:
            return None
        old_defs = {s.name: s for s in old_cls.body if is_test_def(s)}
        new_defs = {s.name: s for s in new_cls.body if is_test_def(s)}
        if len(old_defs) != sum(map(is_test_def, old_cls.body)) or len(new_defs) != sum(map(is_test_def, new_cls.body)):
            return None  # a name defined twice: the later def wins, which a by-name match would miss
        found = {"changed": [], "added": [], "removed": []}
        for name in sorted(set(old_defs) | set(new_defs)):
            old_def, new_def = old_defs.get(name), new_defs.get(name)
            if old_def is None:
                found["added"].append(new_def)
            elif new_def is None:
                found["removed"].append(old_def)
            elif text(old_def, old_lines)[1] != text(new_def, new_lines)[1] or old_def.col_offset != new_def.col_offset:
                found["changed"].append((old_def, new_def))
        if any(found.values()):
            edits[old_cls.name] = found
    if added_imports:
        edits[None] = added_imports
    return edits
