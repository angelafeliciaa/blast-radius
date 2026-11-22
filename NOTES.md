# Blast Radius notes

## Score log (sample, 30 commits; `make sample` simulated + CPU from tools/measure.sh)

| # | Approach | Simulated | Misses | CPU | Total |
|---|----------|-----------|--------|-----|-------|
| 0 | Starter: every test, 3 workers | 1210.470 | 0 | ~0.1 | ~1210.6 |
| 1 | Static map: statement-level coverage, fixture readers, class-attribute readers, AST-equal = no-op | 173.785 | 0 | 0.98 | 174.76 |
| 2 | + comparison profile: swaps that never change a result select nothing | 121.008 | 0 | 0.88 | 121.88 |
| 3 | + run-time fingerprint diff of patched sqlglot's import-time state | 112.461 | 0 | 3.4 | 115.9 |
| 4 | + verify cheap candidate sets by running them (unittest, forked) | 108.379 | 0 | (noisy, see below) | |
| 5 | + key-level reads of class-level dicts; arg_types values-only rule | 99.301 | 0 | | |
| 6 | + map counts PY_START too (one-line lambdas had no LINE event: a miss on a mutation set) | 96.914 | 0 | | |
| 7 | d55d51b: signatures in fingerprints, config/rename/delete/syntax-error rules, collection map | 96.914 | 0 | 3.2 (box) | official practice 172.1 (164.4 + 7.7 CPU) |
| 8 | 415b3ff: map pass with every test in a clean process (lazy loaders), unread-file rule | 96.914 | 0 | 3.2-3.6 (box, loaded) | |
| 9 | 0f9690e: allowlist, red-team fixes (40+ patches), cpu-cut lane (zygote diffs, map.bin), waste arg_types rule | 96.914 | 0 | 1.9-2.2 (box) | official 163.7 (159.258 + 4.5 CPU), 0 missed |
| 10 | 779bc20: red-team round 2 (11 patches): import-time effects in values/defaults/decorators, test class fixtures, pickling, code-object readers | 98.914 | 0 | 1.5-2.0 (box) | official 164.0 (159.258 + 4.7 CPU), 0 missed; proxy 319: 1896.3 |
| 11 | 6ead3e1: proxy lane merge (fixture-item split checks, per-test comparison profile), pool-break rule (proxy MISS yjq2) | 98.903 | 0 | 2.9-3.7 (box, loaded) | proxy 319: 1663.3 (1624.9 + 38.5 CPU), 0 misses |
| 12 | 0bf784f: red-team round 3: process-wide state diff at import; test-code edits select their whole file | 98.903 | 0 | 2.4-2.9 (box) | official 146.3 (131.984 + 14.3 CPU), 0 missed; final pick |
| 13 | 99f235f: g599 fix (unittest loaded for the import diff), runtime process-effect checks, cpu-cut trims, def-default rule | 98.903 | 0 | 2.0-2.8 (box) | official 144.9 (131.984 + 12.95 CPU), 0 missed |
| 14 | 8569e7a: strategist: longest-candidate check runs, test_max_nodes stand-in, split fix for kind/scope changes | 98.903 | 0 | 2.8-3.0 (box) | official 145.1 (131.984 + 13.15 CPU), 0 missed; proxy 319: 1592.3 vs 1668.6 |
| 15 | e4764b5: strategist strat2: standard-library bytecode, TPC fixture/duckdb store for executor check runs (selections byte-identical) | 98.903 | 0 | same as 8569e7a (box, interleaved) | official 141.3 (131.984 + 9.27 CPU), 0 missed |
| 16 | 4a64c76: guard: supervisor/watchdog (12 s per commit, deadline 0.7 x limit), linear patch parser, novel kinds (docs/CI, new test files, test-method edits) | 98.903 | 0 | same as e4764b5 (box, interleaved) | official 141.8 (131.984 + 9.79 CPU), 0 missed; final pick (safer build) |

CPU numbers up to row 6 come from Docker on an Apple Silicon Mac running amd64 under Rosetta
(inflated, noisy); later ones from tools/measure.sh on a native x86 box (2 CPUs for the container).

Regression set (.work/regress, not shipped): 44 synthetic edit kinds + 60 random mutations, truth
from full pytest runs, rechecked file by file. e8f77b9: 3 misses; d55d51b: 1; 415b3ff: 0.

Lower bound if the failing set were known exactly: about 94 s.

## How the map is made

`tools/gen_raw_maps.sh` (dev image from `tools/Dockerfile.dev`, native amd64) runs
sqlglot's suite under `tools/mapgen_plugin.py` in several passes: every test in its own
fork (no cross-test caches); all tests in one process (order effects); forked with the
comparison profile; forked with key-level reads of class-level dicts; each test module
imported in its own fork (collection-time work, with every module preloaded and with
only `import sqlglot` done); the suite without preloading (lazy loaders run in the tests);
and every test alone in a fork of a process that has only imported sqlglot (nothing
cached by collection or earlier tests). `tools/build_map.py` unions them into
`app/map.pkl.gz`: per line the tests that executed it, per file the tests that opened
it, per class attribute (and dict key) the tests that read it.

## When the selector runs every test (or a wide superset)

Allowlist (since 8d4c4fa): the selector narrows only text edits to files that exist in the
checkout, namely Python modules under sqlglot/ and tests/, and other files tests were seen
reading. Everything else selects every test:
- a patch that is not UTF-8, does not parse, does not apply, lists a file twice, or whose
  file headers a second read counts differently; copy/mode/similarity/binary headers;
- new, deleted or renamed files anywhere (except the new files below); binary or mode
  changes; files no test reads outside the INERT kinds; conftest.py,
  pytest.toml/.pytest.toml/pytest.ini, tox.ini, a pyproject.toml/setup.cfg change that pytest
  would read differently (config_problem) or that mentions pytest, requirements files;
  tests/__init__.py;
- a changed coding line, BOM, line endings or NUL byte; a source that no longer parses
  or compiles;
- an import statement changed in sqlglot/; any module the patched sqlglot fails to import,
  in the diff's order or when the patched modules are imported first;
- importing the patched package changes process-wide state the pinned import leaves alone
  (fingerprint.process_state: sys settings, streams and hooks, os.environ, cwd, sys.path,
  logging, warnings, decimal, locale, signals, gc, atexit, copyreg, builtins, any rebound
  attribute of an already loaded module), however the call is spelled (red team round 3);
- a runtime edit in a test file: every test of that file, with no check (a test reaches
  shared class or module state through self, aliases, getattr, lambdas, its class's name);
- an import-level statement that does more than bind names (a call, setting an attribute
  or item of something else); in a test module, anything but plain bindings and plain
  TestCase classes; in sqlglot/, a set/patch/register/mutate call added to or removed from
  anything that runs at import (a binding's value, a default argument, a decorator, a class
  header);
- a changed process-wide effect inside a function (assigning through a non-local name,
  global/nonlocal, calls that set/patch/register/mutate through one, classes made at run
  time); a change in a function that writes class or global state (Dialect._try_load);
- an analysis error; a check (diff, import-first) that could not run.
Wide supersets:
- a def, class or binding added to or removed from a sqlglot class or module: every test
  that uses the class hierarchy or module;
- a function whose compiled symbols change (locals, cells, frees, generator/coroutine):
  every test of any of its lines, and a fixture-driven test among them is checked on the
  items that reach any of its lines (red team i1-i3 were dropped on no items once a check
  run ran);
- a changed def header: its callers (and the module's users if it runs at import);
- a public <key>_sql method added or removed: every test that dispatched that expression;
- a changed member no recorded read sees (dunders, dataclass/slots members, assigned
  calls): whoever names it or the class and whoever uses the class;
- shared test modules (helpers, Validator): every test that imports them;
- a test method that sets or calls into class-level fixtures (self.__class__.x, cls.x,
  type(self).x, self.<attr the class sets in setUpClass or its body>): every test of the
  class, with no check on the patched copy;
- sqlglot/serde.py or a pickling hook (__reduce__, __getstate__, ...) changed: every test in
  modules that use process pools (expressions are pickled across a spawn pool);
- sqlglot/typing/ changed: tests that compare function code objects without calling them;
- an edit to a line that runs while process-pool data is pickled or unpickled
  (tools/pickle_map.py -> app/pickle_lines.pkl.gz: serde, Expression construction and
  linking, DateTrunc.__init__, ...), or to a class body around one: every test of a class
  that keeps one process pool across its tests (TestOptimizer's cls.pool), with no check
  (a raise while unpickling breaks the real pool for the rest of the class; the check's
  in-process stand-in fails only the one call) (proxy MISS yjq2);
- an import-time change: the run-time diff of every binding, class attribute and dict
  entry (with function code for lambdas, key order, C value types), mapped to readers.
The check on the patched copy runs only when every patched file is a module under sqlglot/
or tests/ (or a fixture under tests/fixtures/, read through tests/helpers) and the copy holds
all of them. Fixture-driven tests (optimizer fixture files, the TPC-DS queries) run on only
the fixture items the change reaches (proxy lane: app/items.py, items map pass).

Check runs of the longest candidates (strat lane, 0fb5478). When a check run of every
candidate is not expected to pay, or pays by less, only the longest candidates are run under
the same gate (the first of 5 s or more, then those of 1 s or more, at most 12): they set the
busiest worker. BR_PARTIAL=0 turns it off. Exact in the sense of every check run: a test is
dropped only when it passes on the patched copy.

Stand-in for test_max_nodes (app/scaled.py, BR_SCALED=0 turns it off). The test parses one
statement of 10,000 identical columns twice (28.8 s) and cannot pay for its own check. A check
run uses the same test function with the width and both node limits divided by 25 (0.5 s on
the box) and drops the test only when that passes on the patched copy. Not exact: an edit
whose effect needs more than 400 columns, or that only slows the parse past the scorer's time
limit, passes the stand-in and fails the test. Measured: on the 2,864 labelled proxy commits
it decided 129 (dropped 126, all pass; kept 3, all fail); on 345 further sample-style mutants
of lines the test runs it agreed with the real test every time (233 pass, 112 fail; width 200
gives the same 345 answers). Sample-drawn proxy set (1,048 commits in batches of 30):
-7.2 s simulated per 30 for +0.4 s box CPU per 30 (about +2 s official); 0.94 drops per 45
commits, so 0 misses in 359 drops bounds the penalty at 2.4 s per 45.

Executor check runs on stored fixtures (strat lane, app/tpcmemo.py; BR_TPCMEMO=0 turns it
off). Exact: no selection changes. A check run that holds a TestExecutor test spent 0.45 s in
setUpClass parsing the 32 TPC csv files, 0.38 s making their duckdb views and 0.19 s per query
in duckdb, all on files no narrowed commit can change. The image now holds the rows that
parsing loop gives and duckdb's answers to the SQL of the pinned queries. setUpClass takes
the rows from there (Table(...) and the rest run as written); the class's connections answer
a stored SQL text from the store and hand any other text to duckdb, after making the views.
Not used when the commit touches tests/test_executor.py, tests/helpers.py or a TPC fixture
other than its .sql file. The Dockerfile also compiles the standard library (the base image
ships it without bytecode: unittest.mock cost 0.15 s per process that imported it).

Rejected: test_execute_tpcds on scaled-down fixtures (same test method, every table cut to N
rows). On the 748 labelled commits where it is selected, N=100 passes every commit whose real
test passes (487) but also 7 of the 261 whose real test fails: edits that de-optimize a plan
(a join no longer eliminated, a predicate no longer pushed down), so that the real test runs
out of time or memory on the 10,000-row tables while small tables hide it; two of them pass
even at 2,000 rows. The real test's results are empty for 49 of its 57 executed queries, so
it mostly checks that the pipeline runs and how long it takes. A run still costs 4.6 box-s
(optimize and duckdb do not shrink with the data): 56 s at 4.5x per 30 commits against 18 s
of excess. Also measured and not worth a run: every candidate of a 1,500-test reader set
(10 box-s for the tests under 0.3 s alone).

Narrowings under the measured-risk policy (>=200 samples of the family, 0 misses, >=10 s
saved per commit of the family, red team rounds 1+2 at 0): none enabled. The def-default
rule (BR_DEFAULTS=1: a changed constant default selects the tests that ran the body) is off:
3 proxy commits, 7.6 s saved in all, no 200-sample count.

## Guard (app/guard.py, branch br-final-guard)

The container's process is a supervisor: it forks a worker that runs the whole selector and
streams its answer lines through a pipe, checks every line against SCORING.md rules 1-8 and is
the only writer of stdout. The worker reports which commit it works on in its own process
(static analysis, diff to tests, check-run planning, packing). A commit that holds it longer
than 12 s, or the worker dying while it holds one (exception, signal, OOM kill: workers get
oom_score_adj 500), kills the worker's session and forks a new worker (from the state before
any analysis) that gives that commit every test and does the others again; restarts go on while
24 s remain before the hard deadline. Static analysis that starts after 0.6 of the budget gives
every test; the budget is capped at 0.45 of the limit. The hard deadline is 0.7 of the scorer's
limit (6 s a commit, clamped to 180..270 s: 126 s for 30 commits, 189 s for 45): the worker is
killed and every commit without a good line gets every test. BR_GUARD=0 runs the old way.
Proof (bench/guard/inject.py, sample, score.py valid each time, the other lines byte-identical):
raise, Python hang, C-level regex hang, SIGKILL, a real OOM in the 4 GB cgroup, a hang while a
diff becomes tests, five faults at once, a hang outside any commit (hard deadline, all every
test), python as PID 1. patchfile.parse was quadratic in a hunk's length: main gave no answer
in 290 s on a batch with three 5 MB patches; it now parses one in 0.04 s.

## Novel kinds (branch br-final-guard)

Labelled with bench/proxy/truth_host.py: bench/novel/gen.py (65 hand-made: docs/CI/config in
place and new, config pytest reads, sqlglotc, test-method edits, new test files, mixes),
gen_testedits.py (90 random edits of test methods) and gen_newtests.py (24 new test modules
copied from the suite). Narrowings, each exact by construction:
- Docs, CI and packaging (INERT: root *.md/.txt/.yml/.toml/.cfg/.in/..., LICENSE, Makefile,
  setup.py, .git*, posts/, pdoc/, benchmarks/, docs/, .github/; sqlglotc/ taken out, its
  setup.py is run by test_sqlglotc), edited in place or new, contribute nothing: the commit is
  analysed as if they were not in it. A new file must not be Python, test*, conftest, a pytest
  config, a requirements file or .git*. pyproject.toml and setup.cfg in place go through
  config_problem first (TOML/ini parse, tool.pytest unchanged) and anything mentioning pytest
  stays every test. A new file under tests/fixtures/ has the readers the map knows (none).
- A new tests/**/test_*.py in an existing test package that no test names: its own tests have
  no number. Module and class level may only import (from tests: only Validator and
  tests.helpers) and bind plain values; classes are plain test classes; no pytest_* or dunder
  bindings; every function is self_contained. The import-first stage imports it after
  `import sqlglot` on the patched copy: a failure selects every test.
- An existing test module no other test module imports or names, where only test methods
  changed, were added or were removed and each of them, before and after, is self_contained:
  the changed methods' own tests (and same-named tests of derived classes) plus every test the
  map saw running their bodies; a check run may clear them. Import statements may also be
  added at its top (absolute, of sqlglot or of a module some pinned test module imports, binding
  new names). When anything that runs while the class is made changed (an added import, an added
  or removed method, a def header: decorators, defaults, annotations), the import-first stage
  imports the patched module (a failure: every test).
Test code writing through an alias (fixes a miss that main has too, found by bench/novel/gen.py
r01: `x = exp.Literal; x.number = ...` in a test method failed 4 tests outside its file while the
whole file was selected): when a function of a test module that writes to something it did not
build through a local alias, a loop variable over such things, getattr/type/vars/super, a
parameter, or a callback (map(exp.ALL_FUNCTIONS.append, ...)) changes, every test (alias_writers).
self_contained(fn): shared objects reachable only through imported (library) names. self is
used only to call checked methods or unittest's API, to read class-body constants or the
attributes the function sets itself, never handed on; the module's own classes and mutable
globals are not used; no global, no dunder access, no getattr/setattr/exec/open/mock/os (but
os.path)/sys/logging/...; stores only to self.x or to containers the function built ([] {} or a
comprehension); mutating methods (MUTATORS) only on those; nested classes only plain test
classes; module functions and self methods it reaches are checked the same way. Red team round
3 (alias, lambda parameter and class-name fixture abuse) is refused.

## Expected value per mechanism, scaled to the final (~45 commits)

No statistical narrowing is enabled (def-default rule off; the tpcds-drop rule is not merged).
The selection is narrowed only by run-based mechanisms. Expected saving per 45 commits =
(total with the mechanism off - total with it on) / 319 x 45 on the proxy set (box, d10026d),
total = simulated + 4.5 x box CPU (the scorer charged 4.5x box CPU on 29fdd21). Miss rate:
rule-of-three upper bound over the labelled commits each mechanism narrowed, all at 0 misses
on current main (proxy 2028 on 68a9411 + g599 fix, proxy set 319, red-team hunt 600, red team
rounds 1-3, regression 105, set1 81, mutations 48).

| Mechanism | Narrows (share of commits) | Narrowed with truth, misses | Upper miss rate | Narrowed per 45 | Penalty bound / 45 | Saving / 45 | Net |
|---|---|---|---|---|---|---|---|
| Check runs on the patched copy | 9.7% | ~275, 0 | 1.1% | 4.4 | 14.5 s | 21.3 s | +6.8 s |
| Fixture-item split checks | 8.2% | ~213, 0 | 1.4% | 3.7 | 15.5 s | 20.8 s | +5.3 s |
| Import diff + import-first | 9.7% | ~231, 0 | 1.3% | 4.4 | 17.0 s | 23.8 s | +6.8 s |

All three stay positive under the bound (which is loose: each mechanism is exact given its
map or run, and their misses so far came from blind spots that were then closed). The final
pick is chosen by expected final score: same mechanisms, so the lower official total wins,
and a lower check-run ratio (fewer narrowed commits) breaks ties.

## Residual risks (not turned into a miss by the red team)

- The check on the patched copy runs under PYTHONHASHSEED=0, in sorted order, with its own
  per-test time limit; a test whose result depends on the hash seed or on tests outside the
  candidate set could pass there and fail in the scorer's run. Measured on the proxy set
  (319 patches with truth): the check saves ~37 s over 319 commits (1914.9 -> 1869.2 s
  simulated for +8 s CPU), so it stays.
- Import-time state that depends on import order without failing is fingerprinted in one
  order only (the import-first check only catches failures).
- A runtime edit that makes a pool result contain an expression class whose no-argument
  constructor already raises at the pinned commit (none crosses a pool in the suite) would
  break TestOptimizer's pool without touching a recorded pickle line.
