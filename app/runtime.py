"""Run things against a patched copy of sqlglot inside /tmp.

The patched copy is a tree of symlinks into the read-only checkout with only the
changed files written out, so making one costs a few milliseconds and almost no
space. Work happens in forked children that report through a file; the parent
enforces a deadline and kills a child that overruns.
"""

from __future__ import annotations

import gc
import os
import pickle
import shutil
import signal
import sys
import time

CHECKOUT = os.environ.get("BR_CHECKOUT", "/opt/sqlglot")
SCRATCH = os.environ.get("BR_SCRATCH", os.path.join(os.environ.get("TMPDIR", "/tmp"), "br"))


# ------------------------------------------------------------------ overlay


def _materialize(root: str, rel_dir: str) -> None:
    """Turn root/rel_dir (a symlink to the checkout's directory) into a real directory of symlinks."""
    target = os.path.join(root, rel_dir) if rel_dir else root
    if os.path.isdir(target) and not os.path.islink(target):
        return
    source = os.path.join(CHECKOUT, rel_dir)
    if os.path.lexists(target):
        os.unlink(target)
    os.mkdir(target)
    if os.path.isdir(source):
        for entry in os.listdir(source):
            os.symlink(os.path.join(source, entry), os.path.join(target, entry))


def make_overlay(name: str, new_texts: dict[str, str | None]) -> str:
    """A copy of the checkout made of symlinks, with the patched files written out.

    Only directories on the way to a patched file become real directories; every
    other entry (including __pycache__ folders, so unchanged modules load their
    compiled bytecode) points into the read-only checkout.
    """
    root = os.path.join(SCRATCH, name)
    if os.path.lexists(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    _materialize(root, "")
    write_files(root, new_texts)
    return root


def write_files(root: str, new_texts: dict[str, str | None]) -> None:
    """Put a commit's files into the tree at root."""
    for path, text in sorted(new_texts.items()):
        parts = path.split("/")
        for depth in range(1, len(parts)):
            _materialize(root, "/".join(parts[:depth]))
        target = os.path.join(root, path)
        if os.path.lexists(target):
            os.unlink(target)
        if text is None:
            continue
        if isinstance(text, bytes):  # a file whose bytes are not UTF-8 text
            with open(target, "wb") as handle:
                handle.write(text)
        else:
            with open(target, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
        os.utime(target, (1, 1))  # never matches a cached .pyc of the original


def reset_tree(root: str) -> None:
    """Undo write_files: root is again nothing but symlinks to the checkout's entries."""
    wanted = set(os.listdir(CHECKOUT))
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if entry in wanted and os.path.islink(full) and os.readlink(full) == os.path.join(CHECKOUT, entry):
            continue
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        else:
            os.unlink(full)
    for entry in wanted:
        full = os.path.join(root, entry)
        if not os.path.lexists(full):
            os.symlink(os.path.join(CHECKOUT, entry), full)


def remove_overlay(root: str) -> None:
    shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ children

# Standard library modules that importing sqlglot loads. Loaded once in the parent, every
# forked child shares them. Not unittest or pytest: sqlglot looks for those in sys.modules
# when it is imported, and the fingerprints were taken without them.
SHARED_MODULES = (
    "logging", "importlib.metadata", "datetime", "decimal", "numbers", "math", "random", "bisect",
    "string", "textwrap", "traceback", "threading", "tempfile", "zipfile", "fractions", "statistics",
)


def preload() -> None:
    import importlib

    for name in SHARED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:
            print(f"selector: cannot preload {name}: {error!r}", file=sys.stderr, flush=True)



class Child:
    """A forked child running `function(*args, out_path)`; its result is pickled to out_path."""

    def __init__(self, label: str, function, args: tuple, deadline: float, out_path: str):
        self.label = label
        self.out_path = out_path
        self.deadline = deadline
        self.started = time.time()
        self.result = None
        self.status = "running"
        sys.stdout.flush()
        sys.stderr.flush()
        gc.freeze()  # the child's collector then neither walks nor copies what the parent built
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.setpgid(0, 0)
                # stdout carries the answers: nothing a child prints may reach it
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                function(*args, out_path)
            except BaseException as error:  # report and exit; never return into the parent's code
                try:
                    print(f"child {label}: {error!r}", file=sys.stderr, flush=True)
                except Exception:
                    pass
                code = 1
            finally:
                os._exit(code)
        self.pid = pid

    def poll(self) -> bool:
        """True once the child has finished (or was killed)."""
        if self.status != "running":
            return True
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            pid, status = self.pid, 1 << 8
        if pid == 0:
            if time.time() > self.deadline:
                self.kill()
                return True
            return False
        self._finish(os.waitstatus_to_exitcode(status))
        return True

    def kill(self) -> None:
        for target in (-self.pid, self.pid):
            try:
                os.kill(target, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass
        self.status = "timeout"
        self.result = self._load()

    def _finish(self, code: int) -> None:
        self.status = "ok" if code == 0 else f"exit {code}"
        self.result = self._load()

    def _load(self):
        return take(self.out_path)


def take(out_path: str):
    """What a child pickled to out_path (None if it did not get that far); the file is removed."""
    try:
        with open(out_path, "rb") as handle:
            return pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ------------------------------------------------------------------ child side: fingerprints


_STATE: dict = {}


def load_state(state_path: str) -> dict:
    """The pinned checkout's fingerprints. Loaded once: forked children share the parent's copy."""
    if state_path not in _STATE:
        with open(state_path, "rb") as handle:
            data = pickle.load(handle)
        data["unstable"] = set(data["unstable"])
        _STATE[state_path] = data
    return _STATE[state_path]


def child_import_first(root: str, modules: list[str], out_path: str) -> None:
    """Import each patched module right after `import sqlglot`, before anything else: the
    order a test that imports it first sees. The diff imports in another order, where an
    import cycle or a lazy load at import time can resolve when it would not in the run."""
    sys.path.insert(0, root)
    os.chdir(root)
    import importlib

    errors = []
    for module in modules:
        # each in its own fresh process: importing one patched module first can resolve a cycle
        # that importing another first does not (red team ic4)
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                importlib.import_module("sqlglot")
                importlib.import_module(module)
            except BaseException:  # the finding
                code = 3
            finally:
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        if os.waitstatus_to_exitcode(status) != 0:
            errors.append(f"{module}: fails when imported first")
    with open(out_path, "wb") as handle:
        pickle.dump({"errors": errors}, handle, protocol=4)


def child_diff(root: str, state_path: str, patched_modules: set[str], out_path: str) -> None:
    """Import the patched package, fingerprint it and pickle what differs from state_path."""
    if os.environ.get("BR_DIFF_GC") == "1":
        gc.enable()
    sys.path.insert(0, root)
    os.chdir(root)
    import fingerprint

    data = load_state(state_path)
    result = fingerprint.diff(data["state"], data["unstable"], patched_modules, data.get("entries"))
    with open(out_path, "wb") as handle:
        pickle.dump(result, handle, protocol=4)


# ------------------------------------------------------------------ one import shared by several commits

TREE = "tree"  # the shared tree: symlinks to the checkout, plus one commit's files while its child runs


def diff_triggers(new_texts: dict[str, str | None]) -> set[str] | None:
    """The modules whose first import is the first thing an import of the patched sqlglot does
    differently from the pinned one. None when the commit does more than rewrite existing
    modules of the package (an added or deleted file changes what gets imported and when),
    which leaves it to a child of its own."""
    triggers: set[str] = set()
    for path, text in new_texts.items():
        if path.startswith("tests/"):
            continue  # importing sqlglot reads nothing there
        if not (path.startswith("sqlglot/") and path.endswith(".py")) or text is None:
            return None
        if not os.path.isfile(os.path.join(CHECKOUT, path)):
            return None
        name = path[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        triggers.add(name)
    return triggers or None


def _wait(pid: int, deadline: float) -> str:
    """Wait for a forked child; kill it once the deadline has passed."""
    while True:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return "exit 1"
        if done:
            code = os.waitstatus_to_exitcode(status)
            return "ok" if code == 0 else f"exit {code}"
        if time.time() > deadline:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            return "timeout"
        time.sleep(0.005)


class _ForkAt:
    """A meta path finder that finds nothing. It sees every module looked up for import and,
    when one is a module a waiting commit rewrites, forks a child for that commit there."""

    def __init__(self, root: str, jobs: list[dict], budget_end: float, finish, prepare):
        self.root = root
        self.budget_end = budget_end
        self.finish = finish
        self.prepare = prepare
        self.job: dict | None = None  # in a forked child: the commit it now imports
        self.outcomes: dict[str, str] = {}
        self.waiting = len(jobs)
        self.by_module: dict[str, list[dict]] = {}
        for job in jobs:
            for module in job["triggers"]:
                self.by_module.setdefault(module, []).append(job)

    def find_spec(self, name, path=None, target=None):
        jobs = self.by_module.get(name)
        if jobs:
            for job in jobs:
                if job.get("started"):
                    continue
                job["started"] = True
                self.waiting -= 1
                if self._run(job):
                    return None  # the child: carry on importing, now from the patched tree
            if not self.waiting:
                self.finish()  # the server: nothing later in the import is of use to anyone
        return None

    def _run(self, job: dict) -> bool:
        """Fork a child for the commit at this point of the import. True in the child."""
        if time.time() >= self.budget_end:
            return False
        try:
            write_files(self.root, job["new_texts"])
            sys.stdout.flush()
            sys.stderr.flush()
            self.prepare()  # work every later child would repeat, done once here
            gc.freeze()
            pid = os.fork()
        except OSError as error:
            print(f"selector: no shared diff for {job['commit']}: {error!r}", file=sys.stderr, flush=True)
            reset_tree(self.root)
            return False
        if pid == 0:
            import importlib

            self.job = job
            self.by_module = {}
            importlib.invalidate_caches()  # directory listings cached before the files were written
            return True
        self.outcomes[job["commit"]] = _wait(pid, min(time.time() + 30, self.budget_end + 15))
        reset_tree(self.root)
        return False


def diff_server(jobs: list[dict], state_path: str, budget_end: float, out_path: str) -> None:
    """Import the pinned sqlglot once for several commits (each a dict with the commit, its
    new_texts, its triggers (see diff_triggers), its patched modules and its out path).

    The import runs on a tree of symlinks into the checkout. Just before a module that a
    commit rewrites is first looked up, the commit's files are written into the tree and a
    child is forked. Nothing has read a rewritten file by then, so what is imported so far is
    what the patched tree would have given as well; the child imports the rest, fingerprints
    and reports like child_diff. The parent waits for it, puts the tree back and goes on
    importing for the commits whose modules come later. out_path gets {commit: status} for
    the commits that had a child; the others are left to the caller.
    """
    import fingerprint

    data = load_state(state_path)
    root = make_overlay(TREE, {})
    sys.path.insert(0, root)
    os.chdir(root)

    def finish() -> None:
        with open(out_path, "wb") as handle:
            pickle.dump(hook.outcomes, handle, protocol=4)
        os._exit(0)

    hook = _ForkAt(root, jobs, budget_end, finish, fingerprint.prime_code_digests)
    sys.meta_path.insert(0, hook)
    loaded = fingerprint.load()
    job = hook.job
    if job is None:  # the server got through the whole import: some commit's module was never loaded
        finish()
    result = fingerprint.diff(data["state"], data["unstable"], job["modules"], data.get("entries"), loaded=loaded)
    with open(job["out"], "wb") as handle:
        pickle.dump(result, handle, protocol=4)


def keeps_bare_import(new_texts: dict[str, str | None], core: set[str]) -> bool:
    """True when `import sqlglot` does on the patched tree exactly what it does on the pinned
    one: no module it loads (`core`, as paths) is touched and nothing is added to or removed
    from the package (an added sqlglot/_version.py, for one, would be imported)."""
    for path, text in new_texts.items():
        if path.startswith("sqlglot/"):
            if path in core or text is None or not path.endswith(".py"):
                return False
            if not os.path.isfile(os.path.join(CHECKOUT, path)):
                return False
        elif path.startswith("tests/"):
            if os.path.basename(path) == "__init__.py":
                return False  # runs before the test modules import sqlglot
        else:
            return False
    return True


def first_server(jobs: list[dict], budget_end: float, out_path: str) -> None:
    """Do `import sqlglot` once for the import-first checks of several commits (each a dict
    with the commit, its new_texts and its patched modules).

    child_import_first imports sqlglot afresh for every patched module of every commit. For
    a commit that leaves that import as it is (keeps_bare_import), the import of the pinned
    tree gives the same process, so one is done here: the commit's files go into the tree
    and one child per module is forked to import it, as the first thing after `import
    sqlglot`. out_path gets {commit: (status, [errors])} with the errors spelled as
    child_import_first spells them; the other commits are left to the caller.
    """
    import importlib

    root = make_overlay(TREE, {})
    sys.path.insert(0, root)
    os.chdir(root)
    importlib.import_module("sqlglot")
    core: set[str] = set()
    for name, module in list(sys.modules.items()):
        if name == "sqlglot" or name.startswith("sqlglot."):
            file = getattr(module, "__file__", None) or ""
            if not file.startswith(root + os.sep):
                raise RuntimeError(f"{name} was not loaded from the tree")
            core.add(file[len(root) + 1 :])
    outcomes: dict[str, tuple] = {}
    for job in jobs:
        if not keeps_bare_import(job["new_texts"], core):
            continue
        try:
            write_files(root, job["new_texts"])
        except OSError as error:
            print(f"selector: no shared import-first check for {job['commit']}: {error!r}", file=sys.stderr, flush=True)
            reset_tree(root)
            continue
        deadline = min(time.time() + 20, budget_end + 20)
        status, errors = "ok", []
        for module in job["modules"]:
            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                code = 0
                try:
                    importlib.invalidate_caches()  # directory listings cached before the files were written
                    importlib.import_module("sqlglot")
                    importlib.import_module(module)
                except BaseException:  # the finding
                    code = 3
                finally:
                    os._exit(code)
            ended = _wait(pid, deadline)
            if ended == "timeout":
                status = "timeout"
                break
            if ended != "ok":
                errors.append(f"{module}: fails when imported first")
        reset_tree(root)
        outcomes[job["commit"]] = (status, errors)
    with open(out_path, "wb") as handle:
        pickle.dump(outcomes, handle, protocol=4)


def verify_server(jobs: list[dict], budget_end: float, out_path: str) -> None:
    """Do `import sqlglot` once for several verification runs (each a dict with the commit,
    its new_texts, nodeids, limits, CPU estimate and out path).

    The server prepares itself like a verification child and imports the pinned sqlglot from
    a tree of symlinks. For each commit that leaves that import as it is (keeps_bare_import),
    it writes the commit's files into the tree, forks a child that imports the test modules
    and runs the tests as child_unittest does, waits and puts the tree back. out_path gets
    {commit: (results, status, seconds)}; the other commits are left to the caller.
    """
    gc.enable()
    root = make_overlay(TREE, {})
    _prepare_test_process(root)
    import sqlglot  # noqa: F401

    core: set[str] = set()
    for name, module in list(sys.modules.items()):
        if name == "sqlglot" or name.startswith("sqlglot."):
            file = getattr(module, "__file__", None) or ""
            if not file.startswith(root + os.sep):
                raise RuntimeError(f"{name} was not loaded from the tree")
            core.add(file[len(root) + 1 :])
    outcomes: dict[str, tuple] = {}
    if DIALECT_STAGE:
        # the runs that start by importing tests.dialects.test_dialect go through a fork of this
        # process that has imported it once; whatever it cannot take comes back to the loop below
        leading = [job for job in jobs if keeps_bare_import(job["new_texts"], core) and _starts_with_base(job, root)]
        if len(leading) >= 2:
            out = out_path + ".dialects"
            stage = Child("dialect-stage", _dialect_stage, (root, leading, budget_end), budget_end + 15, out)
            while not stage.poll():
                time.sleep(0.005)
            if stage.status != "ok" or not isinstance(stage.result, dict):
                print(f"selector: dialect stage ended with {stage.status}", file=sys.stderr, flush=True)
            else:
                outcomes.update(stage.result)
            reset_tree(root)
    for job in jobs:
        if job["commit"] in outcomes:
            continue
        if not keeps_bare_import(job["new_texts"], core) or time.time() + job["estimate"] >= budget_end:
            continue
        outcome = _run_check(root, job, budget_end)
        if outcome is not None:
            outcomes[job["commit"]] = outcome
    with open(out_path, "wb") as handle:
        pickle.dump(outcomes, handle, protocol=4)


def _run_check(root: str, job: dict, budget_end: float) -> tuple | None:
    """Write the commit's files into the tree, fork a child that runs its tests, wait for it and
    put the tree back: (results, status, seconds), or None when the files cannot be written."""
    try:
        write_files(root, job["new_texts"])
    except OSError as error:
        print(f"selector: no shared run for {job['commit']}: {error!r}", file=sys.stderr, flush=True)
        reset_tree(root)
        return None
    deadline = min(time.time() + 3 * job["estimate"] + 5, budget_end + 10)
    child = Child(job["commit"], child_unittest, (root, job["nodeids"], job["limits"], job.get("options")),
                  deadline, job["out"])
    while not child.poll():
        time.sleep(0.005)
    outcome = (collect_results(child, job["out"]), child.status, time.time() - child.started)
    reset_tree(root)
    return outcome


DIALECT_STAGE = os.environ.get("BR_DIALECT_STAGE", "1") == "1"
DIALECT_BASE = "tests.dialects.test_dialect"  # the Validator class: what dialect test modules import first
_LEADS: dict[str, bool] = {}


def _starts_with_base(job: dict, root: str) -> bool:
    """Is importing DIALECT_BASE the first thing the job's first test module does that loads
    anything this process (which has only imported sqlglot) does not have yet? Then a process
    that has imported DIALECT_BASE already is where a fresh one would be at that point."""
    path = job["nodeids"][0].split("::", 1)[0]
    if path in job["new_texts"] or not path.startswith("tests/dialects/"):
        return False
    if path not in _LEADS:
        try:
            with open(os.path.join(root, path), encoding="utf-8") as handle:
                _LEADS[path] = path[:-3].replace("/", ".") == DIALECT_BASE or _leads_with(handle.read(), DIALECT_BASE)
        except (OSError, UnicodeDecodeError):
            _LEADS[path] = False
    return _LEADS[path]


def _leads_with(text: str, module: str) -> bool:
    """True when the source imports `module` at its top level and every statement before that
    is a docstring or an import that loads nothing but standard-library modules: of the
    standard library itself, or of modules this process has, for names they already hold (a
    name they do not hold could be a submodule, or one their __getattr__ loads)."""
    import ast

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.ImportFrom) and not stmt.level and stmt.module:
            if stmt.module == module:
                return True
            if stmt.module.split(".")[0] in sys.stdlib_module_names:
                continue
            loaded = sys.modules.get(stmt.module)
            if loaded is None or any(alias.name not in vars(loaded) for alias in stmt.names):
                return False
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == module:
                    return True
                if alias.name.split(".")[0] not in sys.stdlib_module_names and alias.name not in sys.modules:
                    return False
        else:
            return False
    return False


def _dialect_stage(root: str, jobs: list[dict], budget_end: float, out_path: str) -> None:
    """In a fork of verify_server: import DIALECT_BASE from the pinned tree and run from there
    the jobs for which a fresh process would be in this very state once its first test module
    had imported it. That holds when the commit touches none of the files this import read
    (modules, and anything opened meanwhile) and adds or removes none, and when no test
    module loaded so far holds a fixture loader of tests.helpers by name (items.install_filter
    replaces those in the child, before the job's own test modules bind them).
    out_path gets {commit: (results, status, seconds)} for the jobs run here."""
    import builtins
    import importlib
    import io

    import items

    opened: list = []
    real_open, real_os_open = builtins.open, os.open

    def open_and_note(file, *args, **kwargs):
        opened.append(file)
        return real_open(file, *args, **kwargs)

    def os_open_and_note(path, *args, **kwargs):
        opened.append(path)
        return real_os_open(path, *args, **kwargs)

    builtins.open = io.open = open_and_note
    os.open = os_open_and_note
    try:
        importlib.import_module(DIALECT_BASE)
    finally:
        builtins.open = io.open = real_open
        os.open = real_os_open
    read: set[str] = set()
    for name, module in list(sys.modules.items()):
        if name.split(".", 1)[0] in ("sqlglot", "tests"):
            file = getattr(module, "__file__", None) or ""
            if not file.startswith(root + os.sep):
                raise RuntimeError(f"{name} was not loaded from the tree")
            read.add(file[len(root) + 1 :])
    for file in opened:
        try:
            path = os.path.abspath(os.fspath(file))
        except TypeError:  # a file descriptor
            continue
        for base in (root, CHECKOUT):
            if path.startswith(base + os.sep):
                read.add(path[len(base) + 1 :])
    helpers = sys.modules.get("tests.helpers")
    loaders = [vars(helpers)[name] for name in items.LOADERS if name in vars(helpers)] if helpers is not None else []
    for name, module in list(sys.modules.items()):
        if name.startswith("tests.") and module is not helpers:
            if any(value is loader for value in vars(module).values() for loader in loaders):
                raise RuntimeError(f"{name} holds a fixture loader")
    outcomes: dict[str, tuple] = {}
    for job in jobs:
        if time.time() + job["estimate"] >= budget_end:
            continue
        if any(path in read or text is None or not os.path.isfile(os.path.join(CHECKOUT, path))
               for path, text in job["new_texts"].items()):
            continue
        outcome = _run_check(root, job, budget_end)
        if outcome is not None:
            outcomes[job["commit"]] = outcome
    with open(out_path, "wb") as handle:
        pickle.dump(outcomes, handle, protocol=4)


def collect_results(child: Child, out_path: str) -> dict[str, str]:
    """What a finished (or killed) verification child found, test by test."""
    if isinstance(child.result, dict):
        results = child.result.get("results", {})
    else:
        results = read_progress(out_path)
    try:
        os.unlink(out_path + ".progress")
    except OSError:
        pass
    return results


# ------------------------------------------------------------------ child side: running tests


class InProcessExecutor:
    """Stands in for ProcessPoolExecutor in verification runs: calls run at once in this
    process, but arguments and results still travel through pickle as they would
    between processes (sqlglot expressions pickle through serde)."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures

        future = concurrent.futures.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            result = fn(*args, **kwargs)
            future.set_result(pickle.loads(pickle.dumps(result)))
        except BaseException as error:
            try:
                error = pickle.loads(pickle.dumps(error))
            except Exception:
                pass
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _TestTimeout(Exception):
    pass


def _prepare_test_process(root: str) -> None:
    import importlib

    os.chdir(root)
    if sys.path[:1] != [root]:  # a child of verify_server is already set up
        sys.path.insert(0, root)
    importlib.invalidate_caches()
    os.environ["SKIP_INTEGRATION"] = "0"
    import concurrent.futures
    import concurrent.futures.process

    concurrent.futures.ProcessPoolExecutor = InProcessExecutor
    concurrent.futures.process.ProcessPoolExecutor = InProcessExecutor


PROCESS_CHECK = os.environ.get("BR_PROCESS_CHECK", "1") == "1"


def child_unittest(root: str, nodeids: list[str], limits: dict[str, float], options: dict | None,
                   out_path: str) -> None:
    """Run the given tests on the patched copy with unittest, in this (forked) process.

    sqlglot's tests are unittest.TestCase classes, so this gives pytest's verdicts
    (a failing subtest fails its test; an unexpected success counts as a failure)
    without pytest's start-up cost. The tests run in the order given. options:
      restrict  cuts fixture-driven tests down to the fixture items named for them (app/items.py);
      worth, cpu, ratio  each test's recorded milliseconds and estimated CPU seconds: the run
                ends early once the tests still to run could no longer pay for their CPU
                (the ones not run stay selected, like the ones that failed);
      scaled    tests run as their stand-in (app/scaled.py): {nodeid: replacements};
      tpcmemo   the executor tests' set-up takes csv rows and duckdb answers from the store
                made when the image was built (app/tpcmemo.py).
    """
    import importlib
    import unittest

    gc.enable()  # tests run with the collector on, as they do under pytest
    _prepare_test_process(root)
    options = options or {}
    running: list = [None]
    if options.get("restrict"):
        import items

        items.install_filter(options["restrict"], nodeids, lambda: running[0])
    progress = open(out_path + ".progress", "w")
    results: dict[str, str] = {}
    worth, cpu_left, ratio = options.get("worth"), options.get("cpu") or {}, options.get("ratio", 0.5)
    pending = set(nodeids)
    failed: list[str] = []
    halt = [False]
    stand_ins = options.get("scaled") or {}
    prepared: set[str] = set()

    def pays() -> bool:
        """Can the tests still to run, if they all pass, save more than running them costs?"""
        import packing

        stay = {n: worth.get(n, 0) for n in failed}
        bound = packing.pack({**stay, **{n: worth.get(n, 0) for n in pending}})[0] - packing.pack(stay)[0]
        return sum(cpu_left.get(n, 0.0) for n in pending) * 1000 <= ratio * bound

    def record(nodeid: str, status: str) -> None:
        if results.get(nodeid) in ("failed", "skipped"):
            return
        results[nodeid] = status
        progress.write(f"{nodeid}\t{status}\n")
        progress.flush()
        pending.discard(nodeid)
        if process_before and not nodeid.startswith("PROCESS:"):
            # a test that leaves process-wide state changed (recursion limit, logging, an
            # environment variable...) changes every later test of the real run, which this
            # run does not include: report it at once, so that a run killed later still has it
            delta = fingerprint.process_delta(process_before[0], fingerprint.process_state(full=False))
            if delta:
                record("PROCESS:" + repr(delta)[:300], "changed")
                process_before.clear()
        if status == "failed":
            failed.append(nodeid)
            # checked at the 1st, 2nd, 4th, 8th... failure: packing many tests is not free
            if worth is not None and len(failed) & (len(failed) - 1) == 0 and not pays():
                halt[0] = True

    def on_alarm(signum, frame):
        raise _TestTimeout("verification time limit")

    by_test: dict[int, str] = {}

    class Result(unittest.TestResult):
        def startTest(self, test):
            super().startTest(test)
            running[0] = by_test.get(id(test))
            signal.signal(signal.SIGALRM, on_alarm)
            signal.setitimer(signal.ITIMER_REAL, limits.get(by_test.get(id(test), ""), 5.0))

        def stopTest(self, test):
            signal.setitimer(signal.ITIMER_REAL, 0)
            running[0] = None
            self.shouldStop = halt[0]
            nodeid = by_test.get(id(test))
            if nodeid is not None:
                record(nodeid, "passed")
            super().stopTest(test)

        def _fail(self, test):
            nodeid = by_test.get(id(test))
            if nodeid is not None:
                record(nodeid, "failed")
                return
            # a class or module fixture failed: fail every test it could cover
            text = str(test)
            for value in by_test.values():
                path, _, rest = value.partition("::")
                cls = rest.split("::", 1)[0]
                if cls in text or path[:-3].replace("/", ".") in text:
                    record(value, "failed")

        def addError(self, test, err):
            super().addError(test, err)
            self._fail(test)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self._fail(test)

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err is not None:
                self._fail(test)

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test)
            self._fail(test)

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            nodeid = by_test.get(id(test))
            if nodeid is not None:
                record(nodeid, "skipped")

    suites: dict[str, unittest.TestSuite] = {}
    order: list[str] = []
    for nodeid in nodeids:
        path, _, rest = nodeid.partition("::")
        cls_name, _, method = rest.partition("::")
        module_name = path[:-3].replace("/", ".")
        try:
            module = importlib.import_module(module_name)
            if options.get("tpcmemo") and module_name not in prepared:
                prepared.add(module_name)
                import tpcmemo

                if module_name == tpcmemo.MODULE:
                    tpcmemo.install(module)
            if nodeid in stand_ins:
                import scaled

                if not scaled.install(getattr(module, cls_name), method, stand_ins[nodeid]):
                    pending.discard(nodeid)  # no stand-in: not run, so it stays selected
                    continue
            test = getattr(module, cls_name)(method)
        except BaseException:  # the test's file cannot be imported, or the test is gone
            record(nodeid, "failed")
            continue
        if halt[0]:
            break
        by_test[id(test)] = nodeid
        key = f"{module_name}.{cls_name}"
        if key not in suites:
            suites[key] = unittest.TestSuite()
            order.append(key)
        suites[key].addTest(test)
    process_before: list = []
    if PROCESS_CHECK:
        import fingerprint

        process_before.append(fingerprint.process_state(full=False))
        full_before = fingerprint.process_state()
    result = Result()
    # a test with a failed subtest has failed: unittest then ends it at its next subtest instead
    # of running the rest of its fixture items (stopTest lets the other tests go on)
    result.failfast = True
    result.shouldStop = halt[0]
    unittest.TestSuite([suites[key] for key in order]).run(result)
    if PROCESS_CHECK and process_before:
        delta = fingerprint.process_delta(full_before, fingerprint.process_state())
        if delta:
            record("PROCESS:" + repr(delta)[:300], "changed")
    with open(out_path, "wb") as handle:
        pickle.dump({"results": results}, handle, protocol=4)


def read_progress(out_path: str) -> dict[str, str]:
    """What a killed verification child managed to report. Lines are
    "nodeid<TAB>status", written once a test has finished; a failure is sticky."""
    found: dict[str, str] = {}
    try:
        with open(out_path + ".progress") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                nodeid, status = parts
                if found.get(nodeid) in ("failed", "skipped"):
                    continue
                found[nodeid] = status
    except OSError:
        pass
    return found
