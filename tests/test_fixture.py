from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


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
