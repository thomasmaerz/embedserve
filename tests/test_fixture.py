from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest


def test_nomic_baseline_fixture_is_well_formed() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/nomic-baseline.json").read_text())
    assert fixture["model"] == "nomic-embed-text:v1.5"
    assert len(fixture["inputs"]) == len(fixture["vectors"]) == 3
    assert all(len(row) == 768 for row in fixture["vectors"])
    assert all(math.isfinite(value) for row in fixture["vectors"] for value in row)
    canonical = json.dumps(
        fixture["vectors"], separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == fixture["canonical_sha256"]


def test_e5_baseline_fixture_is_well_formed() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/e5-baseline.json").read_text())
    assert fixture["model"] == "intfloat/multilingual-e5-base"
    assert fixture["revision"] == "d128750597153bb5987e10b1c3493a34e5a4502a"
    assert len(fixture["inputs"]) == len(fixture["vectors"]) == 2
    assert all(text.startswith("passage: ") for text in fixture["inputs"])
    assert all(len(row) == 768 for row in fixture["vectors"])
    assert all(math.isfinite(value) for row in fixture["vectors"] for value in row)
    assert all(
        math.sqrt(sum(value * value for value in row)) == pytest.approx(1.0, abs=1e-5)
        for row in fixture["vectors"]
    )
    canonical = json.dumps(
        fixture["vectors"], separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == fixture["canonical_sha256"]
