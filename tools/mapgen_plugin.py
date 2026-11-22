"""pytest plugin that records, per test, what of the pinned sqlglot checkout it touches.

Loaded with `-p mapgen_plugin` while running sqlglot's own suite on the unmodified
checkout. It records:
  * lines of sqlglot/ and tests/ executed (sys.monitoring LINE events),
  * files under the checkout that were opened (audit hook),
  * reads of data-valued class attributes of sqlglot classes (descriptor wrap).
Every process pool is replaced by an in-process executor that still pickles
arguments and results, so work done in pool workers is attributed to the test.

Writes a pickle to $MAPGEN_OUT.
"""

import concurrent.futures
import json
import os
import pickle
import sys
import time
import resource

ROOT = os.path.realpath(os.environ.get("MAPGEN_ROOT", os.getcwd())) + os.sep
PACKAGE = ROOT + "sqlglot" + os.sep
OUT = os.environ["MAPGEN_OUT"]


class InProcessExecutor:
    """Runs each submitted call immediately, round-tripping data through pickle."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        self._shutdown = False
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            result = fn(*args, **kwargs)
            future.set_result(pickle.loads(pickle.dumps(result)))
        except BaseException as error:  # the worker's exception travels back pickled
            try:
                error = pickle.loads(pickle.dumps(error))
            except Exception:
                pass
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


concurrent.futures.ProcessPoolExecutor = InProcessExecutor
import concurrent.futures.process as _cfp  # noqa: E402

_cfp.ProcessPoolExecutor = InProcessExecutor

mon = sys.monitoring
TOOL = 4
mon.use_tool_id(TOOL, "mapgen")
DISABLE = mon.DISABLE

STATE = {"lines": set(), "files": set(), "attrs": set(), "keys": set(), "defaults": set(), "called": set()}
RESULTS = {}  # key -> {"lines":..., "files":..., "attrs":...}
OUTCOMES = {}
CPU = {}


IMPORT_CODES = set()  # code objects that ran while modules were imported or collected


LAZY = [False]  # MAPGEN_MODE=collect_lazy: leave out what runs while a sqlglot module is imported


def _under_sqlglot_import() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        code = frame.f_code
        if code.co_name == "<module>" and code.co_filename.startswith(PACKAGE):
            return True
        frame = frame.f_back
    return False


def _on_line(code, line):
    filename = code.co_filename
    if filename.startswith(ROOT):
        if LAZY[0] and _under_sqlglot_import():
            return None  # import-time work: keep listening, a later call may run it for the test
        STATE["lines"].add((filename[len(ROOT):], line))
        if STATE.get("key", ("",))[0] in ("import", "collect"):
            IMPORT_CODES.add((filename[len(ROOT):], code.co_firstlineno, code.co_name))
    return DISABLE


def _audit(event, args):
    if event == "open" and args:
        path = args[0]
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if isinstance(path, str):
            full = os.path.abspath(path)
            if full.startswith(ROOT) and not full.endswith(".pyc"):
                STATE["files"].add(full[len(ROOT):])


# ---- optional: the default values a test relied on (MAPGEN_DEFAULTS=1) ----
# For every def in sqlglot/ with a constant default value: the parameters that, at the start
# of some call in the test, held that constant (the call left the parameter out, or passed
# the same constant), and whether the def was called at all. tools/build_defaults.py turns
# the fork and seq passes into app/defaults.pkl.gz.
DEFAULTS_ON = os.environ.get("MAPGEN_DEFAULTS") == "1"
DEFAULT_CALLS_MAX = 5000  # calls of one def watched per test; past that every default counts as used
DEFAULT_TABLES = {}  # file -> {(first line, name): [(parameter, constant)]}
DEFAULT_SPECS = {}  # code object -> [(parameter, constant)] (empty: nothing to watch)
DEFAULT_LEFT = {}  # code object -> [{parameter: constant} not seen yet in this test, calls so far]


def _default_table(relpath):
    import ast

    table = {}
    try:
        with open(ROOT + relpath, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return table
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
            pairs += [(arg, default) for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is not None]
            found = [(arg.arg, default.value) for arg, default in pairs if isinstance(default, ast.Constant)]
            if found:
                table[(min([node.lineno] + [d.lineno for d in node.decorator_list]), node.name)] = found
    return table


def _defaults_at_start(code):
    """Called from _on_start. True while this def still has a default that no call of the
    test has used (so its calls must go on being watched)."""
    spec = DEFAULT_SPECS.get(code)
    if spec is None:
        relpath = code.co_filename[len(ROOT):]
        spec = ()
        if relpath.startswith("sqlglot/"):
            table = DEFAULT_TABLES.get(relpath)
            if table is None:
                table = DEFAULT_TABLES[relpath] = _default_table(relpath)
            spec = table.get((code.co_firstlineno, code.co_name), ())
        DEFAULT_SPECS[code] = spec
    if not spec:
        return False
    key = (code.co_filename[len(ROOT):], code.co_firstlineno, code.co_name)
    state = DEFAULT_LEFT.get(code)
    if state is None:
        state = DEFAULT_LEFT[code] = [dict(spec), 0]
        STATE["called"].add(key)
    left = state[0]
    if not left:
        return False
    values = sys._getframe(2).f_locals  # 0: here, 1: _on_start, 2: the def that starts
    for param in list(left):
        constant = left[param]
        try:
            value = values[param]
        except KeyError:
            continue
        if value is constant or (type(value) is type(constant) and value == constant):
            STATE["defaults"].add(key + (param,))
            del left[param]
    state[1] += 1
    if left and state[1] >= DEFAULT_CALLS_MAX:
        for param in left:
            STATE["defaults"].add(key + (param,))
        left.clear()
    return bool(left)


def _on_start(code, offset):
    # A one-line lambda (body on the lambda's own line) gets no LINE event in 3.13
    # unless PY_START is monitored, so count its first line from here as well.
    filename = code.co_filename
    if filename.startswith(ROOT):
        if LAZY[0] and _under_sqlglot_import():
            return None
        STATE["lines"].add((filename[len(ROOT):], code.co_firstlineno))
        if STATE.get("key", ("",))[0] in ("import", "collect"):
            IMPORT_CODES.add((filename[len(ROOT):], code.co_firstlineno, code.co_name))
        if DEFAULTS_ON and _defaults_at_start(code):
            return None  # a later call may leave another parameter out
    return DISABLE


mon.register_callback(TOOL, mon.events.LINE, _on_line)
mon.register_callback(TOOL, mon.events.PY_START, _on_start)
mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
sys.addaudithook(_audit)


def _begin(key):
    STATE["lines"] = set()
    STATE["files"] = set()
    STATE["attrs"] = set()
    STATE["keys"] = set()
    STATE["defaults"] = set()
    STATE["called"] = set()
    DEFAULT_LEFT.clear()
    STATE["key"] = key
    mon.restart_events()


def _end():
    key = STATE["key"]
    slot = RESULTS.setdefault(key, {"lines": set(), "files": set(), "attrs": set(), "keys": set()})
    slot.setdefault("defaults", set()).update(STATE["defaults"])
    slot.setdefault("called", set()).update(STATE["called"])
    slot["lines"] |= STATE["lines"]
    slot["files"] |= STATE["files"]
    slot["attrs"] |= STATE["attrs"]
    slot["keys"] |= STATE["keys"]


# ---- optional: comparison profile (MAPGEN_PROFILE=1) ----
# Every single-operator comparison in sqlglot/ is rewritten into a call that
# also evaluates the sibling operators (is/==/!=/is not, </<=/>/>=, in/not in)
# and remembers whether any of them ever gave a different answer. A site where
# `is` and `==` always agreed makes that swap an equivalent mutant.
PROFILE = {}
PROFILE_ON = os.environ.get("MAPGEN_PROFILE") == "1"

if PROFILE_ON:
    import ast as _ast
    import builtins
    import enum as _enum
    import importlib.abc
    import importlib.machinery
    import operator as _op

    _OPS = {
        "Eq": _op.eq, "NotEq": _op.ne, "Lt": _op.lt, "LtE": _op.le, "Gt": _op.gt, "GtE": _op.ge,
        "Is": _op.is_, "IsNot": _op.is_not,
        "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b,
    }
    _EQUALITY = ("Eq", "NotEq", "Is", "IsNot")
    _ORDER = ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")
    _FAMILY = {op: tuple(x for x in _EQUALITY if x != op) for op in _EQUALITY}
    _FAMILY.update({op: tuple(x for x in _ORDER if x != op) for op in ("Lt", "LtE", "Gt", "GtE")})
    _FAMILY["In"] = ("NotIn",)
    _FAMILY["NotIn"] = ("In",)
    _ATOMS = (type(None), bool, int, float, str, bytes)

    def _safe(value, depth=0):
        kind = type(value)
        if kind in _ATOMS:
            return True
        if isinstance(value, _enum.Enum):
            return kind.__eq__ in (object.__eq__, str.__eq__, int.__eq__, _enum.Enum.__eq__)
        if isinstance(value, type):
            return type(value) is type
        if kind is tuple and depth < 3 and len(value) <= 8:
            return all(_safe(item, depth + 1) for item in value)
        return (
            kind.__eq__ is object.__eq__
            and kind.__ne__ is object.__ne__
            and kind.__lt__ is object.__lt__
            and kind.__le__ is object.__le__
            and kind.__gt__ is object.__gt__
            and kind.__ge__ is object.__ge__
        )

    # MAPGEN_PROFILE_NODES=1 (the per-test profile of tools/build_profile.py): a sqlglot
    # expression compared with None, a bool, a number or a string is evaluated too. Expression.__eq__
    # answers False for an operand of another type before it looks at either one, so asking
    # it changes nothing (`node is None` against `node == None` is the usual case).
    NODES_ON = os.environ.get("MAPGEN_PROFILE_NODES") == "1"

    def _node(value):
        eq = type(value).__eq__
        return getattr(eq, "__qualname__", "") == "Expression.__eq__" and getattr(eq, "__module__", "") == "sqlglot.expressions.core"

    def _pair_safe(a, b):
        if _safe(a) and _safe(b):
            return True
        return NODES_ON and ((type(a) in _ATOMS and _node(b)) or (type(b) in _ATOMS and _node(a)))

    def _br_cmp_(site, op, a, b):
        result = _OPS[op](a, b)
        entry = PROFILE.get(site)
        if entry is None:
            entry = PROFILE[site] = set()
        family = _FAMILY[op]
        if len(entry) < len(family):
            if op in ("In", "NotIn") or not _pair_safe(a, b) or type(result) is not bool:
                entry.update(family)
            else:
                for alt in family:
                    if alt in entry:
                        continue
                    try:
                        other = _OPS[alt](a, b)
                        if type(other) is not bool or other != result:
                            entry.add(alt)
                    except BaseException:
                        entry.add(alt)
        return result

    builtins._br_cmp_ = _br_cmp_

    class _Rewrite(_ast.NodeTransformer):
        def __init__(self, relpath):
            self.relpath = relpath

        def visit_Compare(self, node):
            self.generic_visit(node)
            if len(node.ops) != 1:
                return node
            site = f"{self.relpath}:{node.lineno}:{node.col_offset}"
            call = _ast.Call(
                func=_ast.Name(id="_br_cmp_", ctx=_ast.Load()),
                args=[
                    _ast.Constant(site),
                    _ast.Constant(type(node.ops[0]).__name__),
                    node.left,
                    node.comparators[0],
                ],
                keywords=[],
            )
            return _ast.copy_location(call, node)

    class _Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = self.get_filename(fullname)
            source = self.get_data(path)
            tree = _ast.parse(source, filename=path)
            tree = _Rewrite(path[len(ROOT):]).visit(tree)
            _ast.fix_missing_locations(tree)
            return compile(tree, path, "exec", dont_inherit=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if not (fullname == "sqlglot" or fullname.startswith("sqlglot.")):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                return spec
            if not spec.origin or not spec.origin.startswith(ROOT):
                return spec
            spec.loader = _Loader(fullname, spec.origin)
            return spec

    sys.meta_path.insert(0, _Finder())


def _run_one_test(nodeid):
    """Run one unittest-style test the way pytest's unittest support does (class and module
    fixtures included); returns a short outcome."""
    import importlib
    import unittest

    path, _, rest = nodeid.partition("::")
    cls_name, _, method = rest.partition("::")
    module = importlib.import_module(path[:-3].replace("/", "."))
    case = getattr(module, cls_name)(method)
    result = unittest.TestResult()
    unittest.TestSuite([unittest.TestSuite([case])]).run(result)
    bad = result.errors + result.failures + [(t, "unexpected success") for t in result.unexpectedSuccesses]
    return "passed" if not bad else "failed: " + str(bad[0][1])[-200:]


def _collect_each_module(per_test=False):
    """MAPGEN_MODE=collect: import every test module in its own fork of this process, the
    way collection would, and record what each import touched. pytest collects all modules
    in one process, where whatever the first module caches (lazy attributes, imported test
    helpers) is free for the rest; a fork per module gives each module its own record.
    MAPGEN_MODE=collect_lazy does the same from a process that has imported only `sqlglot`,
    as pytest's has when collection starts; what runs while a sqlglot module is being
    imported is left to the import-time analysis.

    MAPGEN_MODE=fresh (per_test) goes one step further: every test gets its own fork of the
    process that has imported only `sqlglot`, imports its module there and runs. Nothing
    another module's import or an earlier test cached is there, so each test runs every lazy
    loader and cache fill it depends on, and a patch that breaks one reaches all of them."""
    import glob
    import importlib

    if per_test:
        with open(os.environ["MAPGEN_TESTS"]) as handle:
            paths = [ROOT + test["id"] for test in json.load(handle)["tests"]]
    else:
        paths = sorted(glob.glob(os.path.join(ROOT, "tests", "**", "test_*.py"), recursive=True))
    workers = int(os.environ.get("MAPGEN_JOBS", os.cpu_count() or 1))
    scratch = os.environ.get("MAPGEN_SCRATCH", "/tmp/mapgen")
    os.makedirs(scratch, exist_ok=True)
    pending = list(enumerate(paths))
    pending.reverse()
    running = {}
    sys.stdout.flush()
    sys.stderr.flush()
    while pending or running:
        while pending and len(running) < workers:
            index, path = pending.pop()
            out = os.path.join(scratch, f"collect{index}.pkl")
            rel = path[len(ROOT):]
            pid = os.fork()
            if pid == 0:
                code = 0
                try:
                    IMPORT_CODES.clear()
                    key = ("test", rel) if per_test else ("collect", rel)
                    _begin(key)
                    error = None
                    try:
                        if per_test:
                            outcome = _run_one_test(rel)
                            if outcome != "passed":
                                error = outcome
                        else:
                            importlib.import_module(rel[:-3].replace("/", "."))
                    except BaseException as caught:  # a module that fails to import fails all its tests anyway
                        error = repr(caught)
                    mon.set_events(TOOL, 0)
                    _end()
                    with open(out, "wb") as handle:
                        pickle.dump({"key": key, "result": RESULTS[key], "codes": IMPORT_CODES, "error": error},
                                    handle, protocol=pickle.HIGHEST_PROTOCOL)
                except BaseException as caught:
                    print(f"mapgen: collect child failed on {rel}: {caught!r}", file=sys.stderr)
                    code = 1
                finally:
                    os._exit(code)
            running[pid] = (rel, out)
        pid, status = os.wait()
        rel, out = running.pop(pid)
        if os.waitstatus_to_exitcode(status) != 0 or not os.path.exists(out):
            OUTCOMES[rel] = "collect_child_failed"
            continue
        with open(out, "rb") as handle:
            data = pickle.load(handle)
        os.remove(out)
        RESULTS[data["key"]] = data["result"]
        IMPORT_CODES.update(data["codes"])
        if data["error"]:
            OUTCOMES[rel] = "import_error " + data["error"][:200]
    with open(OUT, "wb") as handle:
        pickle.dump(
            {"root": ROOT, "results": RESULTS, "outcomes": OUTCOMES, "cpu": {}, "classes": globals().get("CLASS_INFO", {}),
             "class_attrs": globals().get("CLASS_ATTRS", {}), "import_codes": IMPORT_CODES, "profile": {}},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"mapgen: {len(paths)} {'tests' if per_test else 'test modules'} one fork each into {OUT}; "
          f"{len(OUTCOMES)} did not pass", file=sys.stderr, flush=True)
    os._exit(0)


# ---- import every sqlglot module under the "import" context ----
_begin(("import", "sqlglot"))
import importlib  # noqa: E402
import pkgutil  # noqa: E402

import sqlglot  # noqa: E402

# ---- optional: arguments missing at validation (MAPGEN_MISSING=1) ----
# required_args is read only by Expression.error_messages, and only an argument that is
# None or an empty list there can turn a change of required-ness into a different error
# list. Record, per test, (class, "#missing", key) for every such argument of every
# validated expression; build_map files them under keyreaders[(class, "#missing")].
if os.environ.get("MAPGEN_MISSING") == "1":
    from sqlglot.expressions.core import Expression as _Expression

    _original_error_messages = _Expression.error_messages

    def _error_messages(self, args=None):
        kind = type(self)
        qual = f"{kind.__module__}.{kind.__qualname__}"
        for name in kind.arg_types:
            value = self.args.get(name)
            if value is None or (isinstance(value, list) and not value):
                STATE["keys"].add((qual, "#missing", name))
        return _original_error_messages(self, args)

    _Expression.error_messages = _error_messages


if os.environ.get("MAPGEN_MODE") in ("collect_lazy", "fresh"):
    _end()
    LAZY[0] = True
    _collect_each_module(per_test=os.environ.get("MAPGEN_MODE") == "fresh")

# MAPGEN_MODE=fork_lazy runs the tests (one fork each) without importing every module first,
# as pytest does: the lazy loaders (Dialect._try_load, package __getattr__) then run inside
# the tests that need them instead of once, here. Its lines are what it adds; what runs
# while a sqlglot module is imported stays with the import-time analysis.
FORK_LAZY = os.environ.get("MAPGEN_MODE") == "fork_lazy"
if FORK_LAZY:
    LAZY[0] = True
MODULES = []
for info in [] if FORK_LAZY else pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
    if info.name.endswith("__main__"):
        continue
    try:
        MODULES.append(importlib.import_module(info.name))
    except Exception as error:  # report and keep going
        print(f"mapgen: cannot import {info.name}: {error}", file=sys.stderr)
_end()


# ---- wrap data-valued class attributes so reads are recorded ----
class Recorded:
    __slots__ = ("value", "key")

    def __init__(self, value, key):
        self.value = value
        self.key = key

    def __get__(self, obj, objtype=None):
        STATE["attrs"].add(self.key)
        return self.value


def _classes():
    import enum

    seen = set()
    stack = []
    for module in [sqlglot, *MODULES]:
        for value in list(vars(module).values()):
            if isinstance(value, type) and getattr(value, "__module__", "").startswith("sqlglot"):
                stack.append(value)
    while stack:
        klass = stack.pop()
        if id(klass) in seen:
            continue
        seen.add(id(klass))
        if issubclass(klass, enum.Enum):
            continue
        yield klass
        for value in list(vars(klass).values()):
            if isinstance(value, type) and getattr(value, "__module__", "").startswith("sqlglot"):
                stack.append(value)


WRAPPED = 0
CLASS_INFO = {}
CLASS_ATTRS = {}
ALL_CLASSES = list(_classes())  # collect first: wrapping hides nested classes from the walk
for klass in ALL_CLASSES:
    qual = f"{klass.__module__}.{klass.__qualname__}"
    CLASS_INFO[qual] = [f"{b.__module__}.{b.__qualname__}" for b in klass.__mro__]
    CLASS_ATTRS[qual] = sorted(name for name in vars(klass) if not (name.startswith("__") and name.endswith("__")))
    slots = set(getattr(klass, "__slots__", ()) or ())
    for name, value in list(vars(klass).items()):
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in slots or hasattr(type(value), "__get__"):
            continue
        type.__setattr__(klass, name, Recorded(value, (qual, name)))
        WRAPPED += 1
print(f"mapgen: wrapped {WRAPPED} class attributes", file=sys.stderr)

# ---- optional: key-level reads of class-level dicts (MAPGEN_KEYS=1) ----
# Every dict-valued class attribute becomes a dict subclass that records which keys
# a test looked up ("*" for anything that reads them all, "#len" for size checks).
# The generator's dispatch table is wrapped the same way, under the TRANSFORMS of the
# class it was built for, since it is a copy of TRANSFORMS made once per class.
KEYS_ON = os.environ.get("MAPGEN_KEYS") == "1"
SUSPEND = [False]
if KEYS_ON:
    sys.path.insert(0, os.environ.get("MAPGEN_APP", "/app"))
    from fingerprint import canon_key

    def _rec(owner, key):
        if not SUSPEND[0]:
            name = canon_key(key)
            STATE["keys"].add((owner[0], owner[1], name if name is not None else "*"))

    def _rec_all(owner, what="*"):
        if not SUSPEND[0]:
            STATE["keys"].add((owner[0], owner[1], what))

    class KeyDict(dict):
        __slots__ = ("_br_owner",)
        __hash__ = None

        def __getitem__(self, key):
            _rec(self._br_owner, key)
            return dict.__getitem__(self, key)

        def get(self, key, default=None):
            _rec(self._br_owner, key)
            return dict.get(self, key, default)

        def __contains__(self, key):
            _rec(self._br_owner, key)
            return dict.__contains__(self, key)

        def setdefault(self, key, default=None):
            _rec(self._br_owner, key)
            return dict.setdefault(self, key, default)

        def pop(self, key, *default):
            _rec(self._br_owner, key)
            return dict.pop(self, key, *default)

        def __iter__(self):
            _rec_all(self._br_owner)
            return dict.__iter__(self)

        def __reversed__(self):
            _rec_all(self._br_owner)
            return dict.__reversed__(self)

        def keys(self):
            _rec_all(self._br_owner)
            return dict.keys(self)

        def values(self):
            _rec_all(self._br_owner)
            return dict.values(self)

        def items(self):
            _rec_all(self._br_owner)
            return dict.items(self)

        def popitem(self):
            _rec_all(self._br_owner)
            return dict.popitem(self)

        def __len__(self):
            _rec_all(self._br_owner, "#len")
            return dict.__len__(self)

        def __eq__(self, other):
            _rec_all(self._br_owner)
            return dict.__eq__(self, other)

        def __ne__(self, other):
            _rec_all(self._br_owner)
            return dict.__ne__(self, other)

        def __or__(self, other):
            _rec_all(self._br_owner)
            return dict.__or__(dict(dict.items(self)), other)

        def __ror__(self, other):
            _rec_all(self._br_owner)
            return dict.__ror__(dict(dict.items(self)), other)

        def copy(self):
            _rec_all(self._br_owner)
            return dict(dict.items(self))

        def __repr__(self):
            _rec_all(self._br_owner)
            return dict.__repr__(self)

        def __reduce_ex__(self, protocol):
            _rec_all(self._br_owner)
            return (dict, (dict(dict.items(self)),))

    _KEYED = {}
    for klass in ALL_CLASSES:
        for name, value in list(vars(klass).items()):
            if isinstance(value, Recorded) and type(value.value) is dict:
                shared = _KEYED.get(id(value.value))
                if shared is None:
                    shared = KeyDict(value.value)
                    shared._br_owner = value.key
                    _KEYED[id(value.value)] = shared
                value.value = shared
    print(f"mapgen: {len(_KEYED)} class-level dicts record their keys", file=sys.stderr)

    import sqlglot.generator as _generator

    _original_build = _generator._build_dispatch

    def _build_dispatch(cls):
        SUSPEND[0] = True
        try:
            dispatch = _original_build(cls)
        finally:
            SUSPEND[0] = False
        owner = next((k for k in cls.__mro__ if "TRANSFORMS" in vars(k)), None)
        if owner is None:
            return dispatch
        keyed = KeyDict(dispatch)
        keyed._br_owner = (f"{owner.__module__}.{owner.__qualname__}", "TRANSFORMS")
        return keyed

    _generator._build_dispatch = _build_dispatch


if os.environ.get("MAPGEN_MODE") == "collect":
    _collect_each_module()

_begin(("collect", ""))


def pytest_collection_finish(session):
    _end()


def _cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


import pytest  # noqa: E402
from _pytest.runner import runtestprotocol  # noqa: E402

PHASE_CPU = {}


SEQUENTIAL = os.environ.get("MAPGEN_MODE", "fork") == "seq"


def _phase_begin(when, item):
    if SEQUENTIAL:
        _begin(("test", item.nodeid))


def _phase_end(when, item):
    if SEQUENTIAL:
        _end()
        CPU.setdefault(item.nodeid, {})[when] = PHASE_CPU.get(when, 0.0)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _phase_begin("setup", item)
    start = _cpu()
    yield
    PHASE_CPU["setup"] = _cpu() - start
    _phase_end("setup", item)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _phase_begin("call", item)
    start = _cpu()
    yield
    PHASE_CPU["call"] = _cpu() - start
    _phase_end("call", item)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    _phase_begin("teardown", item)
    yield
    _phase_end("teardown", item)


def pytest_runtest_logreport(report):
    if SEQUENTIAL and report.failed:
        OUTCOMES[report.nodeid] = OUTCOMES.get(report.nodeid, "") + report.when + ":failed;"


def _run_child(item, path):
    """Runs one test in this freshly forked process and saves what it touched."""
    _begin(("test", item.nodeid))
    PROFILE.clear()
    reports = runtestprotocol(item, log=False, nextitem=None)
    mon.set_events(TOOL, 0)
    data = {
        "lines": STATE["lines"],
        "files": STATE["files"],
        "attrs": STATE["attrs"],
        "keys": STATE["keys"],
        "defaults": STATE["defaults"],
        "called": STATE["called"],
        "outcomes": [(report.when, report.outcome) for report in reports],
        "cpu": dict(PHASE_CPU),
        "profile": PROFILE,
    }
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


CHILD_PROFILE = {}
PROFILE_TESTS = {}  # nodeid -> {site: operators that differed in that test}, every site the test evaluated


# ---- MAPGEN_MODE=items: what each fixture item of a fixture-driven test touches ----
# tests/helpers.py hands tests their fixtures through two generators, load_sql_fixture_pairs
# and load_sql_fixtures; a test then does the same work for every item they yield. Here each
# such test runs (one fork a run) with the loaders cut down: to nothing, and to one item at a
# time. The lines a one-item run adds to the no-item run are that item's own. A verification
# run can then hand a test only the items that reach a changed line, instead of all of them.
ITEM_LOADERS = ("load_sql_fixture_pairs", "load_sql_fixtures")
ITEM_STATE = {"restrict": None, "loads": {}, "depth": 0, "raw": set()}


def _item_wrapper(original, name):
    def wrapper(filename):
        key = (name, filename)
        restrict = ITEM_STATE["restrict"]
        keep = None if restrict is None else restrict.get(key)
        loads = ITEM_STATE["loads"]
        loads.setdefault(key, 0)
        ITEM_STATE["depth"] += 1
        try:
            for index, item in enumerate(original(filename)):
                if index + 1 > loads[key]:
                    loads[key] = index + 1
                if keep is None or index in keep:
                    ITEM_STATE["depth"] -= 1  # the consumer's code is not the loader's
                    try:
                        yield item
                    finally:
                        ITEM_STATE["depth"] += 1
        finally:
            ITEM_STATE["depth"] -= 1

    wrapper._br_original = original
    return wrapper


def _item_audit(event, args):
    if event == "open" and args and ITEM_STATE["depth"] == 0:
        path = args[0]
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if isinstance(path, str):
            full = os.path.abspath(path)
            if full.startswith(ROOT + "tests" + os.sep + "fixtures"):
                ITEM_STATE["raw"].add(full[len(ROOT):])


def _install_item_wrappers():
    import tests.helpers as helpers

    for name in ITEM_LOADERS:
        original = getattr(helpers, name)
        if hasattr(original, "_br_original"):
            continue
        wrapped = _item_wrapper(original, name)
        for module in list(sys.modules.values()):
            if getattr(module, name, None) is original:
                setattr(module, name, wrapped)


def _item_child(item, restrict, path):
    ITEM_STATE["restrict"] = restrict
    ITEM_STATE["loads"] = {}
    ITEM_STATE["raw"] = set()
    _begin(("test", item.nodeid))
    reports = runtestprotocol(item, log=False, nextitem=None)
    mon.set_events(TOOL, 0)
    data = {
        "lines": {entry for entry in STATE["lines"] if entry[0].startswith("sqlglot/")},
        "all_lines": len(STATE["lines"]),
        "test_lines": {entry for entry in STATE["lines"] if not entry[0].startswith("sqlglot/")},
        "loads": ITEM_STATE["loads"],
        "raw": ITEM_STATE["raw"],
        "ok": all(report.outcome == "passed" for report in reports),
        "cpu": dict(PHASE_CPU),
    }
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _item_jobs(jobs, workers, scratch):
    """jobs: [(key, item, restrict)] -> {key: data or None}, each in its own fork."""
    results = {}
    pending = list(enumerate(jobs))
    pending.reverse()
    running = {}
    sys.stdout.flush()
    sys.stderr.flush()
    while pending or running:
        while pending and len(running) < workers:
            index, (key, item, restrict) = pending.pop()
            path = os.path.join(scratch, f"item{index}.pkl")
            pid = os.fork()
            if pid == 0:
                code = 0
                try:
                    _item_child(item, restrict, path)
                except BaseException as error:
                    print(f"mapgen: item child failed on {key}: {error!r}", file=sys.stderr)
                    code = 1
                finally:
                    os._exit(code)
            running[pid] = (key, path)
        pid, status = os.wait()
        key, path = running.pop(pid)
        if os.waitstatus_to_exitcode(status) != 0 or not os.path.exists(path):
            results[key] = None
            continue
        with open(path, "rb") as handle:
            results[key] = pickle.load(handle)
        os.remove(path)
    return results


def _items_loop(session):
    workers = int(os.environ.get("MAPGEN_JOBS", os.cpu_count() or 1))
    scratch = os.environ.get("MAPGEN_SCRATCH", "/tmp/mapgen")
    os.makedirs(scratch, exist_ok=True)
    sys.addaudithook(_item_audit)
    _install_item_wrappers()
    by_id = {item.nodeid: item for item in session.items}
    # 1. every test of a module that can see a loader, as it is: which fixtures it loads
    sources = {}
    wanted = []
    for item in session.items:
        path = str(item.fspath)
        if path not in sources:
            with open(path, encoding="utf-8") as handle:
                sources[path] = handle.read()
        if "load_sql_fixture" in sources[path] and os.environ.get("MAPGEN_ITEMS_ONLY", "") in item.nodeid:
            wanted.append(item)
    full = _item_jobs([(item.nodeid, item, None) for item in wanted], workers, scratch)
    print(f"mapgen: items: {len(wanted)} tests probed", file=sys.stderr, flush=True)
    out = {}
    jobs = []
    for nodeid, data in full.items():
        if data is None or not data["ok"] or not data["loads"]:
            continue
        loads = data["loads"]
        out[nodeid] = {"loads": loads, "raw": data["raw"], "cpu": data["cpu"], "files": {}, "ok": True}
        nothing = {key: () for key in loads}
        jobs.append(((nodeid, "base"), by_id[nodeid], nothing))
        for key in loads:  # only this file emptied: does the test use its items at all?
            jobs.append(((nodeid, "without", key), by_id[nodeid], {key: ()}))
    done = _item_jobs(jobs, workers, scratch)
    print(f"mapgen: items: {len(jobs)} runs without items", file=sys.stderr, flush=True)
    jobs = []
    for nodeid, entry in out.items():
        base = done.get((nodeid, "base"))
        if base is None or not base["ok"]:
            entry["ok"] = False
            continue
        entry["base_lines"] = base["lines"]
        entry["base_cpu"] = base["cpu"]
        for key, count in entry["loads"].items():
            without = done.get((nodeid, "without", key))
            if without is None or not without["ok"]:
                entry["ok"] = False
                break
            same = without["lines"] == full[nodeid]["lines"] and without["test_lines"] == full[nodeid]["test_lines"]
            entry["files"][key] = {"n": count, "dependent": not same, "items": {}, "cpu": {}}
            if not same:
                nothing = {other: () for other in entry["loads"]}
                for index in range(count):
                    jobs.append(((nodeid, key, index), by_id[nodeid], {**nothing, key: (index,)}))
    print(f"mapgen: items: {len(jobs)} one-item runs", file=sys.stderr, flush=True)
    solo = _item_jobs(jobs, workers, scratch)
    for (nodeid, key, index), data in solo.items():
        entry = out[nodeid]
        if data is None or not data["ok"]:
            entry["ok"] = False
            continue
        entry["files"][key]["items"][index] = data["lines"] - entry["base_lines"]
        entry["files"][key]["cpu"][index] = round(
            max(data["cpu"].get("call", 0.0) - entry["base_cpu"].get("call", 0.0), 0.0), 4)
    for nodeid, entry in out.items():
        if not entry["ok"]:
            continue
        explained = set(entry["base_lines"])
        for info in entry["files"].values():
            for lines in info["items"].values():
                explained |= lines
        entry["unexplained"] = full[nodeid]["lines"] - explained
    with open(OUT, "wb") as handle:
        pickle.dump({"root": ROOT, "items": out}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    good = sum(1 for entry in out.values() if entry["ok"])
    print(f"mapgen: items: wrote {OUT}: {good} of {len(out)} fixture-driven tests split by item", file=sys.stderr, flush=True)
    return True


def pytest_runtestloop(session):
    """Runs every test in its own fork of this process, several at a time."""
    if SEQUENTIAL:
        return None
    if os.environ.get("MAPGEN_MODE") == "items":
        return _items_loop(session)
    workers = int(os.environ.get("MAPGEN_JOBS", os.cpu_count() or 1))
    scratch = os.environ.get("MAPGEN_SCRATCH", "/tmp/mapgen")
    os.makedirs(scratch, exist_ok=True)
    running = {}
    pending = list(enumerate(session.items))
    pending.reverse()
    done = 0
    sys.stdout.flush()
    sys.stderr.flush()
    while pending or running:
        while pending and len(running) < workers:
            index, item = pending.pop()
            path = os.path.join(scratch, f"{index}.pkl")
            pid = os.fork()
            if pid == 0:
                code = 0
                try:
                    _run_child(item, path)
                except BaseException as error:
                    print(f"mapgen: child failed on {item.nodeid}: {error!r}", file=sys.stderr)
                    code = 1
                finally:
                    os._exit(code)
            running[pid] = (item, path, time.time())
        pid, status = os.wait()
        item, path, started = running.pop(pid)
        if os.waitstatus_to_exitcode(status) != 0 or not os.path.exists(path):
            OUTCOMES[item.nodeid] = "child_failed"
            continue
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        os.remove(path)
        RESULTS[("test", item.nodeid)] = {
            "lines": data["lines"],
            "files": data["files"],
            "attrs": data["attrs"],
            "keys": data.get("keys", set()),
            "defaults": data.get("defaults", set()),
            "called": data.get("called", set()),
        }
        CPU[item.nodeid] = data["cpu"]
        for site, differed in data.get("profile", {}).items():
            CHILD_PROFILE.setdefault(site, set()).update(differed)
        if PROFILE_ON:
            PROFILE_TESTS[item.nodeid] = {site: tuple(sorted(differed)) for site, differed in data.get("profile", {}).items()}
        bad = [f"{when}:{outcome}" for when, outcome in data["outcomes"] if outcome == "failed"]
        if bad:
            OUTCOMES[item.nodeid] = ";".join(bad)
        done += 1
        if done % 100 == 0:
            print(f"mapgen: {done}/{len(session.items)} tests", file=sys.stderr, flush=True)
    return True


def _merged_profile():
    merged = {site: set(differed) for site, differed in PROFILE.items()}
    for site, differed in CHILD_PROFILE.items():
        merged.setdefault(site, set()).update(differed)
    return merged


def pytest_sessionfinish(session):
    mon.set_events(TOOL, 0)
    if os.environ.get("MAPGEN_MODE") == "items":
        return
    with open(OUT, "wb") as handle:
        pickle.dump(
            {
                "root": ROOT,
                "results": RESULTS,
                "outcomes": OUTCOMES,
                "cpu": CPU,
                "classes": CLASS_INFO,
                "class_attrs": CLASS_ATTRS,
                "import_codes": IMPORT_CODES,
                "profile": _merged_profile(),
                "profile_tests": PROFILE_TESTS,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"mapgen: wrote {OUT} with {len(RESULTS)} contexts", file=sys.stderr)
