"""What every check run of the executor tests recomputes from the pinned TPC fixtures alone.

tests/test_executor.py::TestExecutor.setUpClass parses the 32 TPC csv files in Python (0.45 s
on the box) and makes a duckdb view of each (0.38 s), and its three TPC tests ask duckdb for
the result of the same SQL on the same files (0.19 s a query). None of it depends on the
commit as long as it leaves the test module, tests/helpers.py and the csv files alone:

- the rows of a csv file are what the parsing loop of setUpClass makes of the file with csv
  and ast only. They are taken here from a file made of that very loop's output when the image
  was built; Table(...) and everything after it run as written;
- duckdb's answer to a SQL text over those files. The class's connections are wrapped: a text
  seen when the image was built (and answered twice the same) gets a copy of that answer, any
  other text goes to duckdb as before. What the patched sqlglot makes of a query (transpile)
  still runs and decides the text;
- the views are only needed by a text the store does not hold: their CREATE VIEW statements
  are kept and run, in their order, before the first such text reaches duckdb.

Nothing is assumed about the patched code: no selection changes, only the CPU of these runs.
`python tpcmemo.py <out dir>` (cwd and PYTHONPATH on the pinned checkout) writes the store.
"""

from __future__ import annotations

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("BR_TPCMEMO_STORE", os.path.join(HERE, "tpcmemo"))
ON = os.environ.get("BR_TPCMEMO", "1") == "1"
MODULE = "tests.test_executor"
FIXTURES = "tests/fixtures/optimizer/"
SUITES = ("tpc-h", "tpc-ds")
LEFT_ALONE = ("tests/test_executor.py", "tests/helpers.py", "tests/__init__.py")

# the loop of setUpClass that turns one csv file into rows, as it stands in the pinned test
# module (dedented by the class's indentation), and what takes its place
PARSE = "\n".join([
    '        reader = csv.reader(open_file(file_name), delimiter="|")',
    "        rows = []",
    "        ctypes = []",
    "        tables[table] = rows",
    "",
    "        next(reader)",
    "",
    "        for row in reader:",
    "            if not ctypes:",
    "                for i, v in enumerate(row):",
    "                    try:",
    "                        ctypes.append(type(ast.literal_eval(v)))",
    "                    except (ValueError, SyntaxError):",
    "                        ctypes.append(str)",
    "",
    "            rows.append(",
    '                tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row))',
    "            )",
    "",
])
TAKEN = "        rows = __br_tpc_rows__(file_name)\n"
# the statement of setUpClass that makes the view of one csv file: its text is kept for later
VIEW = "        conn.execute(\n            f\"\"\"\n            CREATE VIEW {table} AS\n"
LATER = "        __br_tpc_view__(conn,\n            f\"\"\"\n            CREATE VIEW {table} AS\n"

_loaded: dict = {}


def usable(new_texts: dict) -> bool:
    """The store holds what the pinned files give: not when the commit touches the test module,
    the helpers or a TPC fixture other than its queries."""
    if not ON or not os.path.isfile(os.path.join(STORE, "rows.pkl")):
        return False
    for path in new_texts:
        if path in LEFT_ALONE:
            return False
        if path.startswith(FIXTURES) and path[len(FIXTURES):].split("/", 1)[0] in SUITES and not path.endswith(".sql"):
            return False
    return True


def _store() -> dict:
    if not _loaded:
        with open(os.path.join(STORE, "rows.pkl"), "rb") as handle:
            rows = pickle.load(handle)
        with open(os.path.join(STORE, "frames.pkl"), "rb") as handle:
            frames = pickle.load(handle)
        _loaded.update(rows=rows, frames=frames)
    return _loaded


def _rows(file_name: str) -> list:
    """The rows setUpClass's loop gives for a pinned csv file; an error for any other file."""
    import runtime

    parts = os.path.realpath(file_name).replace(os.sep, "/").split("/")
    key = "/".join(parts[-2:])
    pinned = os.path.realpath(os.path.join(runtime.CHECKOUT, FIXTURES, key))
    if os.path.realpath(file_name) != pinned:
        raise LookupError(f"{file_name} is not a pinned fixture")
    return list(_store()["rows"][key])


class _Answer:
    def __init__(self, frame):
        self._frame = frame

    def fetchdf(self):
        return self._frame.copy(deep=True)


class _Memo:
    """A duckdb connection that answers the SQL texts of the store from the store, and gets
    its views when something else is asked of it."""

    def __init__(self, connection, frames: dict, views: list):
        self._connection = connection
        self._frames = frames
        self._views = views

    def _ready(self):
        while self._views:
            self._connection.execute(self._views.pop(0))
        return self._connection

    def execute(self, query, *args, **kwargs):
        frame = self._frames.get(query) if isinstance(query, str) and not args and not kwargs else None
        if frame is None:
            return self._ready().execute(query, *args, **kwargs)
        return _Answer(frame)

    def close(self):
        return self._connection.close()

    def __getattr__(self, name):
        return getattr(self._ready(), name)


def install(module) -> bool:
    """Make module.TestExecutor.setUpClass take csv rows from the store and wrap the class's
    duckdb connections. False, with nothing changed, unless the set-up code is the pinned one."""
    import inspect
    import textwrap

    try:
        store = _store()
        cls = module.TestExecutor
        function = getattr(cls.__dict__.get("setUpClass"), "__func__", None)
        if not inspect.isfunction(function):
            return False
        source = textwrap.dedent(inspect.getsource(function))
        if source.count(PARSE) != 1 or source.count(VIEW) != 1:
            return False
        source = source.replace(PARSE, TAKEN).replace(VIEW, LATER)
        namespace: dict = {}
        exec(compile(source, "<setUpClass on the stored rows>", "exec"), function.__globals__, namespace)
        made = namespace.get("setUpClass")
        inner = getattr(made, "__func__", made)
        if not inspect.isfunction(inner):
            return False
        frames = store["frames"]
        views: dict = {}

        def view(connection, statement):
            views.setdefault(id(connection), []).append(statement)

        def setUpClass(cls):
            views.clear()
            inner(cls)
            cls.tpch_conn = _Memo(cls.tpch_conn, frames["tpc-h"], views.get(id(cls.tpch_conn), []))
            cls.tpcds_conn = _Memo(cls.tpcds_conn, frames["tpc-ds"], views.get(id(cls.tpcds_conn), []))

        module.__br_tpc_rows__ = _rows
        module.__br_tpc_view__ = view
        cls.setUpClass = classmethod(setUpClass)
        return True
    except Exception as error:
        print(f"selector: executor set-up left as it is: {error!r}", file=sys.stderr, flush=True)
        return False


def build(out_dir: str) -> None:
    """Run the pinned set-up once and keep the rows of every csv file and duckdb's answers to
    the SQL the three TPC tests send it."""
    import importlib

    os.environ["SKIP_INTEGRATION"] = "0"
    module = importlib.import_module(MODULE)
    from sqlglot import transpile

    cls = module.TestExecutor
    cls.setUpClass()
    rows = {}
    for suite, tables in (("tpc-h", cls.tpch_tables), ("tpc-ds", cls.tpcds_tables)):
        for name, table in tables.items():
            rows[f"{suite}/{name}.csv.gz"] = table.rows
    frames: dict = {suite: {} for suite in SUITES}

    def remember(suite: str, connection, text: str) -> None:
        if text in frames[suite]:
            return
        first, second = connection.execute(text).fetchdf(), connection.execute(text).fetchdf()
        if first.equals(second) and list(first.dtypes) == list(second.dtypes) and list(first.columns) == list(second.columns):
            frames[suite][text] = first

    for meta, sql, optimized in cls.tpch_sqls:
        remember("tpc-h", cls.tpch_conn, transpile(sql, write="duckdb")[0])
        remember("tpc-h", cls.tpch_conn, transpile(optimized, write="duckdb")[0])
    for meta, sql, _ in cls.tpcds_sqls:
        if meta.get("execute"):
            remember("tpc-ds", cls.tpcds_conn, transpile(sql, write="duckdb")[0])
    cls.tearDownClass()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rows.pkl"), "wb") as handle:
        pickle.dump(rows, handle, protocol=4)  # equal cells are stored once: it loads in 0.06 s
    with open(os.path.join(out_dir, "frames.pkl"), "wb") as handle:
        pickle.dump(frames, handle, protocol=4)
    print(f"tpcmemo: {len(rows)} csv files, {sum(len(r) for r in rows.values())} rows, "
          f"{len(frames['tpc-h'])} + {len(frames['tpc-ds'])} duckdb answers", file=sys.stderr)


if __name__ == "__main__":
    build(sys.argv[1])
