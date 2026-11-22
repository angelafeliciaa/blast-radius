"""Parse a unified diff (git style) and apply it to files of the pinned checkout."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)  # each starts with ' ', '-', '+' (or '\\')


@dataclass
class FilePatch:
    old_path: str | None  # None for a new file
    new_path: str | None  # None for a deleted file
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False
    mode_only: bool = False

    @property
    def path(self) -> str:
        return self.old_path or self.new_path or ""


def _strip_prefix(path: str) -> str | None:
    path = path.strip()
    if "\t" in path:
        path = path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def parse(text: str) -> list[FilePatch]:
    patches: list[FilePatch] = []
    seen_minus: set[int] = set()
    current: FilePatch | None = None
    hunk: Hunk | None = None
    old_seen = new_seen = 0  # the hunk's lines so far that count toward old_len / new_len
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            match = re.match(r"^diff --git (\S+) (\S+)$", line)
            old = new = None
            if match:
                old, new = _strip_prefix(match.group(1)), _strip_prefix(match.group(2))
            current = FilePatch(old_path=old, new_path=new)
            patches.append(current)
            hunk = None
            i += 1
            continue
        if hunk is not None and current is not None:
            if hunk.old_len - old_seen > 0 or hunk.new_len - new_seen > 0:
                if line == "":
                    line = " "  # some tools drop the space of an empty context line
                tag = line[:1]
                if tag in (" ", "-", "+", "\\"):  # "\ No newline" can sit between - and +
                    hunk.lines.append(line)
                    old_seen += tag in (" ", "-")
                    new_seen += tag in (" ", "+")
                    i += 1
                    continue
                raise PatchError(f"bad hunk line: {line[:40]!r}")
            if line.startswith("\\"):
                hunk.lines.append(line)
                i += 1
                continue
            hunk = None
        if line.startswith("--- "):
            if current is None or current.hunks or id(current) in seen_minus:
                # a plain unified diff has no "diff --git" line: each "---" starts a file
                current = FilePatch(old_path=None, new_path=None)
                patches.append(current)
            seen_minus.add(id(current))
            current.old_path = _strip_prefix(line[4:])
        elif line.startswith("+++ "):
            if current is None:
                raise PatchError("+++ without ---")
            current.new_path = _strip_prefix(line[4:])
        elif line.startswith("new file mode"):
            if current is not None:
                current.old_path = None
        elif line.startswith("deleted file mode"):
            if current is not None:
                current.new_path = None
        elif line.startswith("rename from "):
            if current is not None:
                current.old_path = line[len("rename from "):]
        elif line.startswith("rename to "):
            if current is not None:
                current.new_path = line[len("rename to "):]
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            if current is not None:
                current.binary = True
        elif line.startswith("@@"):
            match = HUNK.match(line)
            if not match or current is None:
                raise PatchError(f"bad hunk header: {line[:60]!r}")
            hunk = Hunk(
                int(match.group(1)),
                int(match.group(2)) if match.group(2) is not None else 1,
                int(match.group(3)),
                int(match.group(4)) if match.group(4) is not None else 1,
            )
            current.hunks.append(hunk)
            old_seen = new_seen = 0
        i += 1
    for patch in patches:
        if not patch.hunks and not patch.binary:
            patch.mode_only = patch.old_path is not None and patch.new_path is not None
    return patches


def apply(original: str | None, patch: FilePatch) -> str | None:
    """Returns the new text of the file, or None when the patch deletes it."""
    if patch.new_path is None:
        return None
    if patch.binary:
        raise PatchError("binary patch")
    old_lines = (original or "").splitlines(keepends=True)
    out: list[str] = []
    cursor = 0  # index into old_lines
    for hunk in patch.hunks:
        start = hunk.old_start - 1 if hunk.old_len > 0 else hunk.old_start
        if start < cursor or start > len(old_lines):
            raise PatchError("hunk out of order")
        out.extend(old_lines[cursor:start])
        cursor = start
        new_block: list[str] = []
        previous = ""
        for line in hunk.lines:
            tag, body = line[:1], line[1:]
            if tag == "\\":
                # "\ No newline at end of file" applies to the previous line
                if previous == "-":
                    pass
                elif new_block and new_block[-1].endswith("\n"):
                    new_block[-1] = new_block[-1][:-1]
                continue
            if tag in (" ", "-"):
                if cursor >= len(old_lines):
                    raise PatchError("context past end of file")
                if old_lines[cursor].rstrip("\r\n") != body.rstrip("\r\n"):
                    raise PatchError("context mismatch")
                if tag == " ":
                    new_block.append(old_lines[cursor])
                cursor += 1
            elif tag == "+":
                new_block.append(body + "\n")
            previous = tag
        out.extend(new_block)
    out.extend(old_lines[cursor:])
    return "".join(out)
