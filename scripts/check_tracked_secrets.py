#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "literal bearer": re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9_-]{24,}"),
}


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True
    ).stdout.split(b"\0")
    findings: list[str] = []
    for raw in files:
        if not raw:
            continue
        path = Path(raw.decode())
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path}: possible {label}")
    if findings:
        print("\n".join(findings))
        return 1
    print(f"checked {len(files) - 1} tracked files: no known secret patterns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
