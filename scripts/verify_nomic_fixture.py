#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Any


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Nomic output against synthetic fixtures")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).parents[1] / "tests/fixtures/nomic-baseline.json",
    )
    args = parser.parse_args()

    fixture: dict[str, Any] = json.loads(args.fixture.read_text())
    body = json.dumps(
        {"model": fixture["model"], "input": fixture["inputs"], "truncate": True},
        separators=(",", ":"),
    ).encode()
    key = args.key_file.read_text(encoding="ascii").strip()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/api/embed",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        actual = json.load(response)["embeddings"]

    expected: list[list[float]] = fixture["vectors"]
    if len(actual) != len(expected) or any(len(row) != 768 for row in actual):
        raise SystemExit("FAIL vector count or dimension mismatch")
    max_error = 0.0
    min_cosine = 1.0
    for actual_row, expected_row in zip(actual, expected, strict=True):
        if not all(math.isfinite(value) for value in actual_row):
            raise SystemExit("FAIL non-finite output")
        max_error = max(
            max_error,
            max(abs(a - b) for a, b in zip(actual_row, expected_row, strict=True)),
        )
        min_cosine = min(min_cosine, cosine(actual_row, expected_row))
        if any(
            abs(a - b) > 1e-6 + 1e-5 * abs(b)
            for a, b in zip(actual_row, expected_row, strict=True)
        ):
            raise SystemExit("FAIL component tolerance exceeded")
    if min_cosine < 0.999999:
        raise SystemExit("FAIL cosine tolerance exceeded")

    canonical = json.dumps(actual, separators=(",", ":"), allow_nan=False).encode()
    print(
        "PASS "
        f"vectors={len(actual)} dimension=768 max_abs_error={max_error:.9g} "
        f"min_cosine={min_cosine:.9f} sha256={hashlib.sha256(canonical).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
