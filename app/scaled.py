"""Stand-ins for tests whose cost is the size of their input, not what they check.

test_max_nodes parses one statement of 10,000 identical columns twice (28.8 recorded seconds)
to see the node limit raise. The stand-in is the same test function with the width and the
two limits divided by 25: the same statements run in the same order, the first limit is still
passed in the last 1% of the columns and the second in the first 5%. A check run uses it in
place of the test (selector.verification_plan); the test is dropped only if the stand-in
passes on the patched copy. It is not proof: an edit whose effect needs more than 400 columns
(or more than the stand-in's time) to show would pass here and fail in the real run. At a
quarter of this width two of 360 mutants did just that (the parse slowed down with the size).
"""

from __future__ import annotations

import os

ON = os.environ.get("BR_SCALED", "1") == "1"

# nodeid -> (file of the test, replacements made once each in the function's source, CPU seconds
# to count for its run, seconds after which the stand-in is stopped and the test stays selected)
SCALED: dict[str, tuple[str, list[tuple[str, str]], float, float]] = {
    "tests/test_parser.py::TestParser::test_max_nodes": (
        "tests/test_parser.py",
        [
            ("width = 10000\n", "width = 400\n"),
            ("parse_one(sql, max_nodes=1_000_000)\n", "parse_one(sql, max_nodes=40_000)\n"),
            ("parse_one(sql, max_nodes=50_000)\n", "parse_one(sql, max_nodes=2_000)\n"),
            ('self.assertIn("(50000)", str(ctx.exception))', 'self.assertIn("(2000)", str(ctx.exception))'),
        ],
        2.0,  # with the imports of its run: 0.8 s on the box the other estimates come from
        # 0.5 s on the box. 5 of 360 mutants of lines the test runs made the parse so slow that
        # the real test timed out: the stand-in ran past a minute on each of them
        6.0,
    ),
}


def usable(nodeid: str, new_texts: dict) -> bool:
    """A stand-in is made from the pinned test: not when the commit touches the test's file."""
    return ON and nodeid in SCALED and SCALED[nodeid][0] not in new_texts


def install(cls, method: str, edits: list[tuple[str, str]]) -> bool:
    """Replace cls.method by the function compiled from its own source with `edits` made.
    False (and nothing replaced) unless every edit matches exactly once."""
    import inspect
    import textwrap

    function = cls.__dict__.get(method)
    if not inspect.isfunction(function):
        return False
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return False
    for old, new in edits:
        if source.count(old) != 1:
            return False
        source = source.replace(old, new)
    namespace: dict = {}
    exec(compile(source, f"<scaled {method}>", "exec"), function.__globals__, namespace)
    if not inspect.isfunction(namespace.get(method)):
        return False
    setattr(cls, method, namespace[method])
    return True
