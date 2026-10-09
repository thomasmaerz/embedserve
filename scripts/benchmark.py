#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

FIXTURES = [
    "search_query: diagnose a synthetic database timeout in a test service",
    "search_document: synthetic incident report: a cache miss caused elevated latency",
    "search_document: redacted test fixture about safe API key rotation on a private LAN",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Embedserve with synthetic inputs")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    key = args.key_file.read_text(encoding="ascii").strip()

    def post(texts: list[str]) -> tuple[float, dict[str, Any]]:
        body = json.dumps(
            {"model": "nomic-embed-text:v1.5", "input": texts, "truncate": True},
            separators=(",", ":"),
        ).encode()
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/api/embed",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        return time.perf_counter() - started, payload

    post(FIXTURES[:1])
    post(FIXTURES[:1])
    results: dict[str, Any] = {}
    for size in (1, 8, 16, 32):
        texts = [FIXTURES[index % len(FIXTURES)] + f" sample-{index}" for index in range(size)]
        samples = []
        for _ in range(args.samples):
            elapsed, payload = post(texts)
            vectors = payload["embeddings"]
            if len(vectors) != size or any(len(row) != 768 for row in vectors):
                raise SystemExit("invalid benchmark response")
            samples.append(elapsed)
        ordered = sorted(samples)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        results[str(size)] = {
            "samples_seconds": samples,
            "p50_seconds": statistics.median(samples),
            "p95_seconds": p95,
            "vectors_per_second": size / statistics.mean(samples),
        }

    _, fixture_payload = post(FIXTURES)
    vectors = fixture_payload["embeddings"]
    canonical = json.dumps(vectors, separators=(",", ":"), allow_nan=False).encode()
    report = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "synthetic_text_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in FIXTURES],
        "fixture_vectors_sha256": hashlib.sha256(canonical).hexdigest(),
        "batch_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")
    for size, result in results.items():
        print(
            f"batch={size} p50={result['p50_seconds']:.6f}s "
            f"p95={result['p95_seconds']:.6f}s "
            f"vectors_per_second={result['vectors_per_second']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
