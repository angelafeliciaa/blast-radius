"""The answer is printed whatever happens to the analysis.

The process the container starts does no analysis. It forks a worker that does all of it and
sends its answer lines back through a pipe; this process checks every line against the
scorer's rules and is the only one that writes stdout.

- The worker says which commit it is working on in its own process (static analysis, turning a
  diff into tests, planning a check run, packing). One commit held longer than PATCH_LIMIT s, or
  the worker dying while it holds one: the worker and everything it started are killed and a new
  worker is forked from this process (the state from before any analysis) that gives that commit
  every test and does the others again, the same way.
- At the hard deadline, or when the worker dies at any other point (an exception, a signal, the
  OOM killer) or sends a line that breaks a rule: every commit without a good line gets every
  test.
The worker's own deadlines (selector.py: the analysis cut-off and the stages' budget) end its
work long before the hard one.
"""

from __future__ import annotations

import json
import os
import select
import signal
import sys
import time

PATCH_LIMIT = float(os.environ.get("BR_PATCH_LIMIT", "12"))
HARD_SHARE = 0.7  # of the scorer's limit
RESTARTS = 20  # new workers at most; each one also needs 2 x PATCH_LIMIT s left before the hard deadline


def log(message: str) -> None:
    try:
        print(f"guard: {message}", file=sys.stderr, flush=True)
    except Exception:  # stderr gone: nothing left to tell
        pass


def time_limit(commits: int) -> float:
    """The scorer's limit for the whole batch: 3 minutes for 30 commits, 4.5 in the final, which
    has about 45. Never assumed longer than 4.5 minutes."""
    return min(max(180.0, 6.0 * commits), 270.0)


def send(fd: int, kind: str, text: str = "") -> None:
    view = memoryview((kind + text + "\n").encode("utf-8"))
    while view:
        view = view[os.write(fd, view):]


class Worker:
    """A forked worker running work(write_fd, skip); its messages come back line by line."""

    def __init__(self, work, skip: frozenset):
        read_fd, write_fd = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            code = 70
            try:
                os.close(read_fd)
                try:
                    os.setsid()  # everything the worker starts shares its session: killed together
                except OSError:  # then kill() still takes the worker itself; PID 1's exit, the rest
                    pass
                try:  # the OOM killer takes the worker's processes, never this one
                    with open("/proc/self/oom_score_adj", "w") as handle:
                        handle.write("500")
                except OSError:
                    pass
                os.dup2(2, 1)  # only the supervisor writes stdout
                code = work(write_fd, skip)
            except BaseException as error:
                log(f"worker failed ({error!r})")
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:  # exiting anyway
                    pass
                os._exit(code if isinstance(code, int) else 70)
        os.close(write_fd)
        self.pid = pid
        self.fd = read_fd
        self.pending = b""
        self.lines: dict[str, bytes] = {}
        self.holding: str | None = None  # commit the worker is working on in its own process
        self.since = 0.0
        self.ended = False
        self.exited: int | None = None

    def feed(self, data: bytes) -> None:
        self.pending += data
        *complete, self.pending = self.pending.split(b"\n")
        for message in complete:
            kind, body = message[:1], message[1:]
            if kind == b"P":
                try:
                    self.holding = str(json.loads(body))
                except Exception:  # a name the worker could not have sent: count it as held
                    self.holding = body.decode("utf-8", "replace")
                self.since = time.time()
            elif kind == b"Q":
                if self.holding is not None and time.time() - self.since > 2.0:
                    log(f"{self.holding} held the worker {time.time() - self.since:.1f}s")
                self.holding = None
            elif kind == b"L":
                try:
                    commit = json.loads(body).get("commit")
                except Exception:  # checked again, and replaced, when the answer is printed
                    commit = None
                if isinstance(commit, str):
                    self.lines.setdefault(commit, body)
            elif kind == b"E":
                self.ended = True

    def reap(self) -> None:
        """Collect the worker's exit, and any orphan of it that this process inherited (it is
        the container's first process)."""
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                if self.exited is None:
                    self.exited = -1
                return
            if pid == 0:
                return
            if pid == self.pid:
                self.exited = os.waitstatus_to_exitcode(status)

    def drain(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([self.fd], [], [], 0)
            except (OSError, ValueError):
                return
            if not ready:
                return
            data = os.read(self.fd, 1 << 16)
            if not data:
                return
            self.feed(data)

    def kill(self) -> None:
        """SIGKILL the worker and every process of its session (forked children set process
        groups of their own; the session stays) until none is left, then keep what it sent."""
        for _ in range(100):
            members = session_members(self.pid)
            if self.exited is None and self.pid not in members:
                members.append(self.pid)  # its own session, whatever setsid did
            for pid in members:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            self.reap()
            if not members and self.exited is not None:
                break
            time.sleep(0.01)
        self.drain()
        try:
            os.close(self.fd)
        except OSError:
            pass


def session_members(session: int) -> list[int]:
    members = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return [session]
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as handle:
                fields = handle.read().rsplit(b")", 1)[1].split()
        except (OSError, IndexError):
            continue
        # after the name: state, ppid, pgrp, session
        if len(fields) > 3 and fields[0] != b"Z" and int(fields[3]) == session:
            members.append(int(name))
    return members


def watch(worker: Worker, hard_end: float) -> str | None:
    """Wait for the worker to end. Returns the commit it was killed or died on when that was
    while it held one, else None."""
    while True:
        now = time.time()
        if now >= hard_end:
            log(f"hard deadline: worker killed{' on ' + worker.holding if worker.holding else ''}")
            worker.kill()
            return None
        if worker.holding is not None and now >= worker.since + PATCH_LIMIT:
            held = worker.holding
            log(f"{held} held the worker more than {PATCH_LIMIT:.0f}s: killed")
            worker.kill()
            return held
        wait = hard_end - now
        if worker.holding is not None:
            wait = min(wait, worker.since + PATCH_LIMIT - now)
        try:
            ready, _, _ = select.select([worker.fd], [], [], max(0.0, min(wait, 0.25)))
        except InterruptedError:
            ready = []
        data = os.read(worker.fd, 1 << 16) if ready else None
        if data:
            worker.feed(data)
            if worker.ended:
                worker.reap()
                if worker.exited is None:
                    # it exits right after its last message; anything still in its session is done
                    deadline = time.time() + 2.0
                    while worker.exited is None and time.time() < deadline:
                        time.sleep(0.005)
                        worker.reap()
                worker.kill()
                return None
            continue
        worker.reap()
        if data == b"" or worker.exited is not None:  # pipe closed or worker gone without "E"
            if worker.exited is None:
                worker.exited = -1
            worker.drain()
            held = worker.holding
            log(f"worker ended (status {worker.exited}) without finishing{' on ' + held if held else ''}")
            worker.kill()
            return held


def check(line: bytes, commits: set[str], count: int) -> str | None:
    """The commit of a line that keeps every rule of SCORING.md, else None."""
    try:
        text = line.decode("utf-8")
        if text.startswith("﻿") or "\n" in text or "\r" in text:
            return None

        def refuse(name):
            raise ValueError(name)

        item = json.loads(text, parse_constant=refuse)
    except Exception:  # not JSON
        return None
    if not isinstance(item, dict) or set(item) != {"commit", "workers"}:
        return None
    commit, workers = item["commit"], item["workers"]
    if not isinstance(commit, str) or commit not in commits or not isinstance(workers, list):
        return None
    seen: set[int] = set()
    busy = 0
    for worker in workers:
        if not isinstance(worker, list):
            return None
        for number in worker:
            if type(number) is not int or not 0 <= number < count or number in seen:
                return None
            seen.add(number)
        busy += 1 if worker else 0
    if busy > 8:
        return None
    return commit


def supervise(work, commits: list[str], durations: dict[int, int], started: float) -> int | None:
    """Run work(write_fd, skip) in a worker and print one good line for every commit. None when
    no worker could be forked (the caller then does the work itself)."""
    hard_end = started + HARD_SHARE * time_limit(len(commits))
    skip: set[str] = set()
    worker = None
    lines: dict[str, bytes] = {}
    try:
        while True:
            try:
                worker = Worker(work, frozenset(skip))
            except OSError as error:
                log(f"cannot fork a worker ({error!r})")
                if worker is None:
                    return None
                break
            held = watch(worker, hard_end)
            lines = worker.lines
            if held is None or held in skip or len(skip) >= RESTARTS or time.time() > hard_end - 2 * PATCH_LIMIT:
                break
            skip.add(held)
            log(f"new worker; every test for {', '.join(sorted(skip))}")
    except BaseException as error:  # print what there is, whatever went wrong here
        log(f"failed ({error!r})")
        if worker is not None:
            try:
                worker.kill()
                lines = worker.lines
            except BaseException:
                pass
    known = set(commits)
    every = None
    out = []
    for commit in commits:
        line = lines.get(commit)
        if line is None or check(line, known, len(durations)) != commit:
            log(f"{commit}: {'no' if line is None else 'a malformed'} answer from the worker: every test")
            if every is None:
                import packing

                every = packing.pack(durations)[1]
            line = json.dumps({"commit": commit, "workers": every}, separators=(",", ":")).encode()
        out.append(line + b"\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(b"".join(out))
    sys.stdout.buffer.flush()
    return 0
