#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an Embedserve API key file")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    old_umask = os.umask(0o077)
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        value = secrets.token_urlsafe(48).encode("ascii")
        with os.fdopen(descriptor, "wb") as target:
            target.write(value + b"\n")
    finally:
        os.umask(old_umask)

    fingerprint = hashlib.sha256(value).hexdigest()
    print(f"created {args.output} mode=0600 sha256={fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
