"""Fingerprint the import-time state of sqlglot: every module level name and
every class attribute, as short digests.

Comparing the fingerprints of the pinned checkout with those of a patched one
tells which bindings an import-time change altered, including attributes that
metaclasses and __init_subclass__ derive. Function bodies are left out (line
coverage handles them); their defaults and closure contents are kept.

Usage (build time): python fingerprint.py <out.pickle>
"""

from __future__ import annotations

import enum
import functools
import hashlib
import importlib
import pickle
import pkgutil
import re
import os
import sys
import types

_ATOMS = (type(None), bool, int, float, complex, str, bytes)
_ONLY_STR = {str}
_LEAVES = (type, enum.Enum)  # fingerprinted by name
_ORDERED = (str, int, enum.Enum)  # keys hashed by value: a dict of them keeps its order from run to run


class Fingerprinter:
    def __init__(self):
        self.memo: dict[int, str] = {}
        self.keep: list = []  # keep objects alive so ids stay unique
        self.strings: dict[str, str] = {}  # plain strings met so far

    def digest(self, text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).hexdigest()

    def fp(self, value, depth: int = 0) -> str:
        if type(value) is str:  # most of what there is: dict keys, names
            text = self.strings.get(value)
            if text is None:
                text = f"str:{value!r}"
                if len(text) >= 64:
                    text = self.digest(text)
                self.strings[value] = text
            return text
        if value is None:
            return "NoneType:None"
        kind = type(value)
        if kind is bool:
            return "bool:True" if value else "bool:False"
        if kind is int:
            text = f"int:{value!r}"
            return text if len(text) < 64 else self.digest(text)
        key = id(value)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        if isinstance(value, _ATOMS):
            if isinstance(value, float) and value != value:
                return "float:nan"
            text = f"{type(value).__name__}:{value!r}"
            if len(text) >= 64:
                text = self.digest(text)
            if type(value) not in _ATOMS:
                # an enum member that is also a str or an int: its repr is Python code and it
                # turns up thousands of times, so it is remembered like any other object
                self.memo[key] = text
                self.keep.append(value)
            return text
        if depth > 60:
            return "deep"
        self.memo[key] = "cycle"
        self.keep.append(value)
        result = self._fp(value, depth + 1)
        self.memo[key] = result
        return result

    def _sequence(self, value, depth: int) -> str:
        fp = self.fp
        parts = [fp(x, depth) for x in value]
        if value and all(isinstance(x, type) for x in value):
            parts.sort()  # tuples of classes are often built from a set, so their order varies
        return self.digest(type(value).__name__ + "(" + ",".join(parts) + ")")

    def _mapping(self, value, depth: int) -> str:
        fp = self.fp
        if len(value) < 16:  # most dicts are small (an object's attributes, a trie node)
            pairs = [f"{fp(k, depth)}:{fp(v, depth)}" for k, v in value.items()]
            for k in value:
                if type(k) is not str and not isinstance(k, _ORDERED):
                    pairs.sort()
                    break
            return self.digest(f"{type(value).__qualname__}{{{','.join(pairs)}}}")
        memo = self.memo
        keys = list(value)
        kinds = set(map(type, keys))
        if kinds == _ONLY_STR or all(kind in _ATOMS or issubclass(kind, _LEAVES) for kind in kinds):
            # Keys that are atoms, classes or enum members: fingerprinting one touches nothing
            # else, so all the keys can go before all the values, and the lookups can be done
            # by the dozen instead of one call each.
            if kinds == _ONLY_STR:
                left = list(map(self.strings.get, keys))
            else:
                left = list(map(memo.get, map(id, keys)))
            if None in left:
                left = [fp(k, depth) for k in keys]
            values = list(value.values())
            right = list(map(memo.get, map(id, values)))
            if None in right:
                # in order: fingerprinting one value can store the answer for a later one
                right = [b if b is not None else fp(v, depth) for v, b in zip(values, right)]
            pairs = list(map(":".join, zip(left, right)))
        else:
            pairs = [f"{fp(k, depth)}:{fp(v, depth)}" for k, v in value.items()]
        if not all(issubclass(kind, _ORDERED) for kind in kinds):
            # keyed by classes or other objects: built from sets, they come out in id order
            pairs.sort()
        # otherwise the order counts: positional builders map arguments by arg_types order
        return self.digest(f"{type(value).__qualname__}{{{','.join(pairs)}}}")

    def _function(self, value, depth: int) -> str:
        fp = self.fp
        cells = []
        for cell in value.__closure__ or ():
            try:
                cells.append(fp(cell.cell_contents, depth))
            except ValueError:  # empty cell
                cells.append("empty")
        # A def'd function's body is line coverage's business, and its qualified name already
        # tells it apart; lambdas and functions a factory makes share names, so their code counts
        anonymous = value.__name__ == "<lambda>" or "<locals>" in value.__qualname__
        code = code_digest(value.__code__) if anonymous else ""
        return self.digest(
            f"func:{value.__module__}.{value.__qualname__}|{signature(value.__code__)}|{code}"
            f"|{fp(value.__defaults__, depth)}|{fp(value.__kwdefaults__, depth)}|{','.join(cells)}"
        )

    def _fp(self, value, depth: int) -> str:
        fp = self.fp
        kind = type(value)
        # the common kinds first; none of them is an instance of anything tested before
        # its own turn in the chain below
        if kind is dict:
            return self._mapping(value, depth)
        if kind is types.FunctionType:
            return self._function(value, depth)
        if kind is tuple or kind is list:
            return self._sequence(value, depth)
        if isinstance(value, type):
            return f"type:{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', '?')}"
        if isinstance(value, enum.Enum):
            return f"enum:{type(value).__module__}.{type(value).__qualname__}.{value.name}"
        if isinstance(value, types.ModuleType):
            return f"module:{value.__name__}"
        if isinstance(value, (tuple, list)):
            return self._sequence(value, depth)
        if isinstance(value, (set, frozenset)):
            items = list(value)
            parts = list(map(self.memo.get, map(id, items)))  # what is already known, in one go
            if None in parts:
                parts = [part if part is not None else fp(x, depth) for x, part in zip(items, parts)]
            parts.sort()
            return self.digest(type(value).__name__ + "{" + ",".join(parts) + "}")
        if isinstance(value, dict):
            return self._mapping(value, depth)
        if isinstance(value, types.FunctionType):
            return self._function(value, depth)
        if isinstance(value, (types.BuiltinFunctionType, types.MethodWrapperType, types.WrapperDescriptorType)):
            return f"builtin:{getattr(value, '__qualname__', repr(type(value)))}"
        if isinstance(value, types.MethodType):
            return self.digest(f"method:{fp(value.__func__, depth)}:{fp(value.__self__, depth)}")
        if isinstance(value, (staticmethod, classmethod)):
            return self.digest(f"{type(value).__name__}:{fp(value.__func__, depth)}")
        if isinstance(value, property):
            return self.digest(f"property:{fp(value.fget, depth)}:{fp(value.fset, depth)}:{fp(value.fdel, depth)}")
        if isinstance(value, functools.partial):
            return self.digest(f"partial:{fp(value.func, depth)}:{fp(value.args, depth)}:{fp(value.keywords, depth)}")
        if isinstance(value, re.Pattern):
            return self.digest(f"re:{value.pattern!r}:{value.flags}")
        kind = type(value)
        name = f"{kind.__module__}.{kind.__qualname__}"
        state = []
        if hasattr(value, "__dict__") and isinstance(getattr(value, "__dict__", None), dict):
            state.append(fp(value.__dict__, depth))
        for klass in kind.__mro__:
            for slot in getattr(klass, "__dict__", {}).get("__slots__", ()) or ():
                if isinstance(slot, str) and slot not in ("__dict__", "__weakref__"):
                    try:
                        state.append(f"{slot}={fp(getattr(value, slot), depth)}")
                    except AttributeError:
                        state.append(f"{slot}=unset")
        if not state:
            # a value type kept in C (datetime, Decimal, Fraction...): its repr is its state,
            # unless it is the default one that only shows an address
            text = repr(value)
            if " at 0x" not in text:
                state.append("repr=" + text)
        return self.digest(f"obj:{name}:{'|'.join(state)}")


_CALL_FLAGS = 0x04 | 0x08 | 0x20 | 0x80 | 0x100 | 0x200  # *args, **kwargs, generator, coroutine kinds


def signature(code: types.CodeType) -> str:
    """How a function is called: parameter names, kinds and generator-ness. Bodies are
    left to line coverage, but a renamed or reordered parameter breaks callers that
    never reach the body."""
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & 0x04) + bool(code.co_flags & 0x08)
    return (
        f"{code.co_argcount}/{code.co_posonlyargcount}/{code.co_kwonlyargcount}/"
        f"{code.co_flags & _CALL_FLAGS}/{','.join(code.co_varnames[:count])}"
    )


_CODE_DIGESTS: dict[int, tuple] = {}  # id(code) -> (code, digest); holding the code keeps the id its own


def code_digest(code: types.CodeType) -> str:
    """What a function does, apart from where it sits: bytecode, constants and names, without
    line numbers. Two lambdas with one signature in a dispatch table differ only here."""
    cached = _CODE_DIGESTS.get(id(code))
    if cached is not None:
        return cached[1]
    parts = [code.co_code.hex(), ",".join(code.co_names), ",".join(code.co_varnames),
             ",".join(code.co_freevars), ",".join(code.co_cellvars)]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            parts.append("code:" + code_digest(const))
        else:
            parts.append(f"{type(const).__name__}:{const!r}")
    digest = hashlib.blake2b("|".join(parts).encode("utf-8", "surrogatepass"), digest_size=8).hexdigest()
    _CODE_DIGESTS[id(code)] = (code, digest)
    return digest


def prime_code_digests(package: str = "sqlglot") -> None:
    """Work out code_digest for the functions of the package that exist now and that the
    collector has not been told to forget (gc.freeze): a process forked after this has them
    ready. A code object never changes, so this only moves work, it does not change answers."""
    import gc

    for value in gc.get_objects():
        if type(value) is types.FunctionType:
            module = value.__module__
            if isinstance(module, str) and (module == package or module.startswith(package + ".")):
                code_digest(value.__code__)


def _qual(klass) -> str:
    return f"{klass.__module__}.{klass.__qualname__}"


def canon_key(key, depth: int = 0) -> str | None:
    """A stable name for a dict key, or None when it has none. Shared with the map
    generator, which records which keys of class-level dicts each test looked up."""
    if key is None:
        return "n"
    kind = type(key)
    if kind is str:
        return "s:" + key
    if kind is bool:
        return "b:" + repr(key)
    if kind is int:
        return "i:" + repr(key)
    if isinstance(key, enum.Enum):
        return f"e:{_qual(type(key))}.{key.name}"
    if isinstance(key, type):
        return "t:" + _qual(key)
    if kind is tuple and depth < 3:
        parts = [canon_key(item, depth + 1) for item in key]
        if all(part is not None for part in parts):
            return "u:(" + "|".join(parts) + ")"
    return None


IMPORT_KEY = ("import", "sqlglot", "#loaded")


def dict_entries(finger: Fingerprinter, out: dict, qual: str, attr: str, value: dict) -> None:
    """The per-key digests of one class-level dict."""
    # the keys alone, so a change of values only can be told apart; their order
    # counts only where it is stable (keys hashed by value, not by id)
    keys = [finger.fp(key) for key in value]
    if not all(isinstance(key, (str, int, enum.Enum)) for key in value):
        keys.sort()
    out[("keys", qual, attr)] = finger.digest(",".join(keys))
    # and every entry on its own, so a changed entry maps to the tests that looked it up
    odd = []
    for key, item in value.items():
        name = canon_key(key)
        if name is None:
            odd.append(finger.fp(key) + ":" + finger.fp(item))
        else:
            out[("kv", qual, attr, name)] = finger.fp(item)
    if odd:
        out[("kv", qual, attr, "?")] = finger.digest(",".join(sorted(odd)))


PROCESS_KEY = ("process", "*", "#effects")
PROCESS_DELTA: list = []  # what the last load() changed in process-wide state (process_delta)
# loaded before the first snapshot, so that build and run see the same modules
STATE_MODULES = ("os", "builtins", "logging", "warnings", "decimal", "random", "locale", "signal",
                 "threading", "socket", "copyreg", "gc", "atexit", "faulthandler", "json", "re")


def process_state(full: bool = True) -> dict:
    """Process-wide state that code run at import can change for every later test, whatever
    the syntax that changes it (getattr(sys, ...)(), map(), a lambda default, an annotation).
    Values are plain data except under "attrs" and "builtins", which hold object ids."""
    import importlib

    for name in STATE_MODULES:
        importlib.import_module(name)
    import atexit, copyreg, decimal, faulthandler, gc, locale, logging, os, random, signal, socket, threading, warnings  # noqa: E401

    def hook_name(value) -> str:
        return type(value).__qualname__ if not callable(value) else getattr(value, "__qualname__", type(value).__qualname__)

    def logger_state(logger) -> tuple:
        return (logger.level, logger.disabled, logger.propagate, tuple(hook_name(h) for h in logger.handlers),
                len(logger.filters))

    loggers = {"": logger_state(logging.getLogger())}
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger):
            config = logger_state(logger)
            if config != (0, False, True, (), 0):  # creating a logger alone changes nothing
                loggers[name] = config
    signals = {}
    for name in ("SIGINT", "SIGTERM", "SIGCHLD", "SIGPIPE", "SIGHUP", "SIGUSR1", "SIGUSR2"):  # SIGALRM: the check's own timer
        if hasattr(signal, name):
            handler = signal.getsignal(getattr(signal, name))
            signals[name] = handler if isinstance(handler, int) else hook_name(handler)
    attrs = {}
    for name, module in list(sys.modules.items()) if full else ():
        if module is None or name.split(".", 1)[0] in ("sqlglot", "tests", "__main__"):
            continue
        try:
            attrs[name] = {key: id(value) for key, value in list(vars(module).items())}
        except TypeError:
            continue
    return {
        "sys": {"recursionlimit": sys.getrecursionlimit(), "switchinterval": sys.getswitchinterval(),
                "int_max_str_digits": sys.get_int_max_str_digits(), "trace": sys.gettrace() is None,
                "profile": sys.getprofile() is None, "dont_write_bytecode": sys.dont_write_bytecode},
        "streams": {name: id(getattr(sys, name)) for name in ("stdin", "stdout", "stderr", "excepthook",
                                                              "displayhook", "breakpointhook", "unraisablehook")},
        "path": tuple(sys.path), "path_hooks": len(sys.path_hooks),
        "meta_path": tuple(type(f).__qualname__ for f in sys.meta_path if type(f).__name__ != "_ForkAt"),
        "environ": dict(os.environ), "cwd": os.getcwd(),
        "gc": (gc.isenabled(), gc.get_threshold()), "atexit": atexit._ncallbacks(),
        "faulthandler": faulthandler.is_enabled(), "threading": (threading.stack_size(), len(threading.enumerate())),
        "socket": socket.getdefaulttimeout(), "locale": locale.setlocale(locale.LC_ALL),
        "decimal": (lambda c: (c.prec, c.rounding, c.Emin, c.Emax, c.capitals, c.clamp,  # not flags: any
                            tuple(sorted(k.__name__ for k, v in c.traps.items() if v))))(decimal.getcontext()),  # arithmetic sets them
        "logging": (logging.root.manager.disable, logging.getLoggerClass().__qualname__, logging.getLogRecordFactory() is logging.LogRecord),
        "loggers": loggers, "signals": signals,
        "warnings": frozenset(repr(f) for f in warnings.filters), "showwarning": hook_name(warnings.showwarning),
        "copyreg": frozenset(getattr(k, "__qualname__", repr(k)) for k in copyreg.dispatch_table),
        "builtins": {key: id(value) for key, value in vars(__import__("builtins")).items()},
        "attrs": attrs,
    }


def process_delta(before: dict, after: dict) -> list:
    """What changed between two process_state() calls, as plain data: the same for the same
    imports whatever the environment (paths, variables) the process started with."""
    out: list = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if key == "attrs":
            # modules present before only: attributes they had, rebound or removed (an added one,
            # such as a submodule or a lazily made alias, depends on what was imported earlier)
            for name in sorted(set(old) & set(new)):
                if old[name] != new[name]:
                    # one that was None and now holds something is a cache filled on first use
                    moved = tuple(sorted(a for a in old[name] if old[name][a] != new[name].get(a)
                                         and not (old[name][a] == NONE_ID and a in new[name])))
                    if moved:
                        out.append((key, name, moved))
            out.extend((key, name, "removed") for name in sorted(set(old) - set(new)))
        elif key in ("builtins", "streams"):
            out.append((key, tuple(sorted(a for a in set(old) | set(new) if old.get(a) != new.get(a)))))
        elif isinstance(old, dict) and isinstance(new, dict):
            for item in sorted(set(old) | set(new)):
                if old.get(item, ()) != new.get(item, ()):
                    out.append((key, item, repr(new.get(item))))
        elif isinstance(old, frozenset) and isinstance(new, frozenset):
            out.append((key, tuple(sorted(new - old)), tuple(sorted(old - new))))
        elif key == "path":
            out.append((key, tuple(p for p in new if p not in old), tuple(p for p in old if p not in new)))
        else:
            out.append((key, repr(new)))
    return out


NONE_ID = id(None)
_BEFORE: list = []  # the process_state() the running load() started from


def _after_fork() -> None:
    """A fork mid-import (runtime.diff_server) reseeds random and rebuilds threading's locks in
    the child: start the child's comparison from those, not the parent's."""
    if _BEFORE:
        import random, threading  # noqa: E401

        before = _BEFORE[0]
        before["threading"] = (threading.stack_size(), len(threading.enumerate()))
        before["attrs"]["threading"] = {key: id(value) for key, value in list(vars(threading).items())}


def load(package: str = "sqlglot") -> tuple[list | None, list[str], str]:
    """Imports every module of the package: (the modules, or None when the package itself
    fails; import errors; the names a bare import of the package loaded). Sets PROCESS_DELTA."""
    errors: list[str] = []
    # sqlglot checks at import whether it runs under a test runner (expressions/core.py:
    # UNITTEST = "unittest" in sys.modules or "pytest" in sys.modules) and then runs more code
    # at import; the real run imports it under pytest, so this import must see unittest too
    import unittest  # noqa: F401

    before = process_state()
    if not _BEFORE:
        os.register_at_fork(after_in_child=_after_fork)  # after random's and threading's own
    _BEFORE[:] = [before]
    PROCESS_DELTA[:] = []
    try:
        root = importlib.import_module(package)
    except BaseException as error:  # the package itself no longer imports
        return None, [f"{package}: {error.__class__.__name__}: {error}"], ""
    # What a bare `import sqlglot` loads, before anything else is imported: tests that start a
    # fresh interpreter (test_lazy_load counts the dialects it registers) see exactly this.
    loaded = sorted(name for name in sys.modules if name == package or name.startswith(package + "."))
    eager = ",".join(loaded)
    modules = [root]
    for info in pkgutil.walk_packages(root.__path__, package + "."):
        if info.name.endswith("__main__"):
            continue
        try:
            modules.append(importlib.import_module(info.name))
        except BaseException as error:  # an import failure is itself the finding
            errors.append(f"{info.name}: {error.__class__.__name__}: {error}")
    PROCESS_DELTA[:] = process_delta(before, process_state())
    if os.environ.get("BR_DEBUG_PROCESS") == "1":
        print(f"selector: process delta {PROCESS_DELTA!r}"[:2000], file=sys.stderr, flush=True)
    return modules, errors, eager


def snapshot(package: str = "sqlglot", finger: Fingerprinter | None = None, lazy: list | None = None,
             repeated: set | None = None, loaded: tuple | None = None) -> tuple[dict, list[str]]:
    """Imports every module of the package (or takes what load() returned) and fingerprints its state.

    With `lazy`, the per-key digests of class-level dicts ("keys" and "kv" entries) are left
    out: each such dict is appended to `lazy` as (class, attribute, dict) so the caller can
    add them with dict_entries() for the dicts it needs. Every digest those entries are made
    of is already memoized by then, so adding them later gives what adding them here would.
    `repeated` collects the names of classes fingerprinted more than once (two classes with
    one qualified name), whose entries overwrite each other in order.
    """
    modules, errors, eager = loaded if loaded is not None else load(package)
    if modules is None:
        return {}, errors
    if finger is None:
        finger = Fingerprinter()
    out: dict[tuple, str] = {IMPORT_KEY: finger.digest(eager), PROCESS_KEY: finger.digest(repr(PROCESS_DELTA))}
    seen_classes: set[int] = set()
    seen_quals: set[str] = set()

    def add_class(klass) -> None:
        if id(klass) in seen_classes:
            return
        seen_classes.add(id(klass))
        qual = _qual(klass)
        if qual in seen_quals and repeated is not None:
            repeated.add(qual)
        seen_quals.add(qual)
        out[("c", qual, "__mro__")] = ",".join(_qual(base) for base in klass.__mro__)
        for attr, value in list(vars(klass).items()):
            if attr in ("__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__firstlineno__",
                        "__static_attributes__", "__annotations__", "__annotate__", "__slots__"):
                continue
            if isinstance(value, type) and value.__module__ == klass.__module__ and \
                    value.__qualname__ == f"{klass.__qualname__}.{attr}":
                add_class(value)
                out[("c", qual, attr)] = "nested:" + _qual(value)
                continue
            out[("c", qual, attr)] = finger.fp(value)
            if type(value) is dict:
                if lazy is None:
                    dict_entries(finger, out, qual, attr, value)
                else:
                    lazy.append((qual, attr, value))

    for module in modules:
        for name, value in list(vars(module).items()):
            if name in ("__builtins__", "__cached__", "__file__", "__loader__", "__spec__", "__path__",
                        "__doc__", "__package__", "__name__"):
                continue
            if isinstance(value, type) and value.__module__ == module.__name__ and value.__qualname__ == name:
                add_class(value)
                out[("m", module.__name__, name)] = "class:" + _qual(value)
                continue
            out[("m", module.__name__, name)] = finger.fp(value)
    return out, errors


def diff(original: dict, unstable: set, patched_modules: set[str], entries: dict | None = None,
         loaded: tuple | None = None) -> dict:
    """Run in a process that has the patched package on sys.path: import it (unless load()
    already did: `loaded`), fingerprint it and report what differs from the original fingerprints.

    `original` holds the module and class bindings; `entries` holds, per (class, attribute)
    of a class-level dict, its "keys" and "kv" digests (see split_state). Those are worked
    out here only for the dicts whose own digest differs from the original's: nothing reads
    the entries of a dict that did not change.
    """
    if entries is None:
        original, entries = split_state(original)
    finger = Fingerprinter()
    lazy: list = []
    repeated: set = set()
    state, errors = snapshot(finger=finger, lazy=lazy, repeated=repeated, loaded=loaded)
    failed = [error.split(":", 1)[0] for error in errors]
    expanded: dict = {}
    same = set()
    for qual, attr, value in lazy:
        if qual in repeated or original.get(("c", qual, attr)) != state.get(("c", qual, attr)):
            dict_entries(finger, expanded, qual, attr, value)
        else:
            same.add((qual, attr))
    differing = [key for key, value in state.items() if original.get(key) != value]
    differing += [key for key in original if key not in state]
    for pair in entries:
        if pair not in same:
            differing += [key for key, value in _group(entries, pair).items() if expanded.get(key) != value]
    differing += [key for key in expanded if key not in _group(entries, (key[1], key[2]))]

    changed = set()
    for key in differing:
        owner = key[1]
        if any(owner == f or owner.startswith(f + ".") for f in failed):
            continue
        if key in unstable and not any(owner == m or owner.startswith(m + ".") for m in patched_modules):
            continue
        changed.add(key)
    return {"changed": changed, "errors": errors}


def _group(entries: dict, pair: tuple) -> dict:
    """The entries of one class-level dict. state.pkl keeps each group pickled, since a
    diff reads only the few whose dict changed."""
    group = entries.get(pair, ())
    if isinstance(group, bytes):
        group = entries[pair] = pickle.loads(group)
    return group


def split_state(state: dict) -> tuple[dict, dict]:
    """(module and class bindings, {(class, attribute): its "keys" and "kv" digests})."""
    main: dict = {}
    entries: dict = {}
    for key, value in state.items():
        if key[0] in ("keys", "kv"):
            entries.setdefault((key[1], key[2]), {})[key] = value
        else:
            main[key] = value
    return main, entries


def _child(out: str) -> None:
    state, errors = snapshot()
    with open(out, "wb") as handle:
        pickle.dump({"state": state, "errors": errors}, handle, protocol=4)


def main() -> None:
    """Build time: fingerprint the pinned checkout in a few separate processes and keep
    the keys whose digest varies between processes (iteration order of sets of classes)."""
    import os
    import subprocess
    import tempfile

    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _child(sys.argv[2])
        return
    if len(sys.argv) != 2:
        sys.exit("usage: fingerprint.py <out.pickle>")
    runs = []
    with tempfile.TemporaryDirectory() as scratch:
        for index in range(3):
            out = os.path.join(scratch, f"{index}.pkl")
            subprocess.run([sys.executable, __file__, "--child", out], check=True)
            with open(out, "rb") as handle:
                runs.append(pickle.load(handle))
    for run in runs:
        if run["errors"]:
            sys.exit("import errors in the pinned checkout: " + "; ".join(run["errors"]))
    base = runs[0]["state"]
    unstable = set()
    for run in runs[1:]:
        state = run["state"]
        unstable |= {key for key in set(base) | set(state) if base.get(key) != state.get(key)}
    main_part, entries = split_state(base)
    entries = {pair: pickle.dumps(group, protocol=4) for pair, group in entries.items()}
    with open(sys.argv[1], "wb") as handle:
        pickle.dump({"state": main_part, "entries": entries, "unstable": unstable}, handle, protocol=4)
    print(f"fingerprint: {len(base)} bindings, {len(unstable)} unstable", file=sys.stderr)


if __name__ == "__main__":
    main()
