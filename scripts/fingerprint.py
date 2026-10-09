#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import stat
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print key-file fingerprints without values")
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.files:
        value = path.read_bytes().strip()
        mode = stat.S_IMODE(path.stat().st_mode)
        print(f"{path} mode={mode:04o} sha256={hashlib.sha256(value).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
