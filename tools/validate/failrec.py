"""pytest plugin: write every failing nodeid (any phase, any subtest) to $FAILREC_OUT."""

import json
import os

FAILED = set()
COLLECT = set()


def pytest_runtest_logreport(report):
    if report.failed:
        FAILED.add(report.nodeid)


def pytest_collectreport(report):
    if report.failed:
        COLLECT.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["FAILREC_OUT"], "w") as handle:
        json.dump({"failed": sorted(FAILED), "collect": sorted(COLLECT), "exit": int(exitstatus)}, handle)
