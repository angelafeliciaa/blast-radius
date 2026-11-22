"""Check a fixture-driven test on only the fixture items a change can reach.

tests/helpers.py hands tests their fixtures item by item (load_sql_fixture_pairs and
load_sql_fixtures); a test then does the same work for every item. items.pkl.gz
(tools/build_items.py, from the pinned checkout) holds, per such test and fixture file, the
sqlglot lines each item's work reaches when the test runs on that item alone (beyond what
the test reaches with no items at all). A check run can then give the test only the items
that reach a changed line, or whose text changed, instead of all of them: test_execute_tpcds
on 3 of its 99 queries, say.

A test is only split when everything the patch changes is either a statement inside a sqlglot
function that does not run at import (the lines decide the items) or a fixture file the test
reads through the loaders alone (the items whose text differs decide). Anything else leaves
the test whole, and so does a changed line that the test reaches without any item (its own
assertions: an item may reach it too) or that no cut-down run accounts for.
"""

from __future__ import annotations

import gzip
import io
import os
import pickle
from pathlib import Path

ITEMS_PATH = Path(os.environ.get("BR_ITEMS", str(Path(__file__).resolve().parent / "items.pkl.gz")))
FIXTURES = "tests/fixtures/"
LOADERS = ("load_sql_fixture_pairs", "load_sql_fixtures")


class Items:
    def __init__(self, data: dict, checkout: str):
        self.tests: dict[str, dict] = data["tests"]
        self.checkout = checkout
        self._helpers: dict | None = None
        self._changed: dict[tuple, set[int] | None] = {}

    @classmethod
    def load(cls, checkout: str) -> "Items | None":
        try:
            with gzip.open(ITEMS_PATH, "rb") as handle:
                return cls(pickle.load(handle), checkout)
        except Exception:  # no per-item map: every test is checked whole
            return None

    # ------------------------------------------------------------ fixture text -> items

    def _loader(self, name: str):
        """tests/helpers.<name> of the pinned checkout, reading whatever text it is handed."""
        if self._helpers is None:
            path = os.path.join(self.checkout, "tests", "helpers.py")
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            namespace: dict = {"__file__": path, "__name__": "br_fixture_helpers"}
            exec(compile(source, path, "exec"), namespace)
            self._helpers = namespace
        return self._helpers[name]

    def loaded(self, loader: str, filename: str, text: str) -> list | None:
        """What the loader yields for a fixture file holding `text` (None: it raises)."""
        function = self._loader(loader)
        # the loaders open their file in text mode: universal newlines
        content = text.replace("\r\n", "\n").replace("\r", "\n")
        self._helpers["open"] = lambda *args, **kwargs: io.StringIO(content)
        try:
            return list(function(filename))
        except Exception:
            return None
        finally:
            self._helpers.pop("open", None)

    def changed_items(self, loader: str, filename: str, old: str, new: str) -> set[int] | None:
        """Indexes of the items whose content differs; None when the items cannot be paired
        (one side fails to load, or their number changes)."""
        key = (loader, filename, hash(old), hash(new))
        if key not in self._changed:
            before, after = self.loaded(loader, filename, old), self.loaded(loader, filename, new)
            if before is None or after is None or len(before) != len(after):
                self._changed[key] = None
            else:
                self._changed[key] = {i for i, (a, b) in enumerate(zip(before, after)) if a != b}
        return self._changed[key]

    # ------------------------------------------------------------ which items to run

    def split(self, nodeid: str, lines: dict[str, set[int]], data: dict[str, tuple[str, str]]):
        """{fixture file: sorted item indexes} that a run of the test must keep to still decide
        whether it fails, or None when it has to run whole.

        lines: changed sqlglot lines by path. data: (old text, new text) of every changed
        non-Python file this test reads."""
        entry = self.tests.get(nodeid)
        if entry is None:
            return None
        # a test that runs the same with and without the items of every file it loads (they are
        # loaded by its class for other tests) needs none of them, whatever changed
        uses_items = any(info["dependent"] for info in entry["files"].values())
        for path, changed in lines.items():
            # a line the test's own assertions or set-up reach: the per-item runs cannot say
            # which items reach it as well. A line no cut-down run reached: not accounted for.
            if uses_items and (changed & entry["base_lines"].get(path, frozenset())
                               or changed & entry["unexplained"].get(path, frozenset())):
                return None
        loaded_as = {FIXTURES + filename: filename for filename in entry["files"]}
        for path in data:
            if path not in loaded_as or path in entry["raw"]:
                return None  # read some other way than through the loaders
        keep: dict[str, list[int]] = {}
        for filename, info in entry["files"].items():
            mask = 0
            for path, changed in lines.items():
                by_line = info["lines"].get(path)
                if by_line:
                    for line in changed:
                        mask |= by_line.get(line, 0)
            indexes = {i for i in range(info["n"]) if mask >> i & 1}
            path = FIXTURES + filename
            if path in data:
                differing = self.changed_items(info["loader"], filename, *data[path])
                if differing is None:
                    return None
                if info["dependent"]:
                    indexes |= differing
            keep[filename] = sorted(indexes)
        return keep

    def cpu(self, nodeid: str, keep: dict[str, list[int]]) -> tuple[float, float]:
        """(setup, call) CPU seconds, on the map machine, of the test cut down to `keep`."""
        entry = self.tests[nodeid]
        call = entry["base"]
        for filename, indexes in keep.items():
            costs = entry["files"][filename]["cpu"]
            call += sum(costs[i] for i in indexes if i < len(costs))
        return entry["setup"], call


# ---------------------------------------------------------------- inside a check run


def install_filter(restrict: dict[str, dict[str, list[int]]], nodeids: list[str], current) -> None:
    """Make tests.helpers' loaders yield, for the tests in `restrict`, only the listed items.

    `current()` names the running test (None during class or module set-up). A test that is
    not in `restrict` gets every item. A setUpClass gets the union of what its class's tests
    in this run need, and every item as soon as one of them is not restricted; any other
    set-up code gets every item. Call before any test module is imported (they bind the
    loaders by name)."""
    import sys

    import tests.helpers as helpers

    everything = None
    by_class: dict[str, dict[str, set[int]] | None] = {}
    for nodeid in nodeids:
        owner = nodeid.rsplit("::", 1)[0]
        keep = restrict.get(nodeid)
        if keep is None:
            by_class[owner] = None
        elif by_class.setdefault(owner, {}) is not None:
            for filename, indexes in keep.items():
                by_class[owner].setdefault(filename, set()).update(indexes)

    def class_being_set_up() -> str | None:
        frame = sys._getframe(3)
        while frame is not None:
            if frame.f_code.co_name == "setUpClass":
                cls = frame.f_locals.get("cls")
                if isinstance(cls, type):
                    return f"{cls.__module__.replace('.', '/')}.py::{cls.__name__}"
            frame = frame.f_back
        return None

    def allowed(filename: str):
        nodeid = current()
        if nodeid is not None:
            keep = restrict.get(nodeid)
            return everything if keep is None or filename not in keep else set(keep[filename])
        owner = class_being_set_up()
        union = by_class.get(owner) if owner is not None else None
        return everything if union is None or filename not in union else union[filename]

    def wrap(original):
        def loader(filename):
            wanted = allowed(filename)
            for index, item in enumerate(original(filename)):
                if wanted is everything or index in wanted:
                    yield item

        return loader

    for name in LOADERS:
        setattr(helpers, name, wrap(getattr(helpers, name)))
