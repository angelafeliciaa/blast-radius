"""Download sqlglot at the pinned commit from its own GitHub repository and unpack it.

Usage: python fetch_sqlglot.py <destination>
"""

import io
import sys
import tarfile
import urllib.request
from pathlib import Path

COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
URL = f"https://github.com/tobymao/sqlglot/archive/{COMMIT}.tar.gz"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: fetch_sqlglot.py <destination>")
    destination = Path(sys.argv[1])
    if destination.exists():
        sys.exit(f"{destination} already exists")
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    prefix = f"sqlglot-{COMMIT}/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = []
        for member in archive.getmembers():
            if not member.name.startswith(prefix) or not (member.isfile() or member.isdir()):
                continue
            member.name = member.name[len(prefix):]
            if member.name:
                members.append(member)
        archive.extractall(destination, members=members, filter="data")
    if not (destination / "sqlglot" / "__init__.py").is_file():
        sys.exit("download did not contain sqlglot/__init__.py")
    print(f"unpacked {len(members)} entries into {destination}", file=sys.stderr)


if __name__ == "__main__":
    main()
