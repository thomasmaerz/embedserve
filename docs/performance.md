# Performance report

## Phase 0 baseline

Measured 2026-10-08 against the existing unauthenticated PyTorch prototype on a Quadro
P620 with 2 GiB VRAM, PyTorch `2.4.1+cu121`, sentence-transformers `3.0.1`, and
transformers `4.44.2`. Requests used deterministic synthetic operational text. Ten warm
samples were collected per batch.

| Batch | p50 | p95 | Throughput |
|---:|---:|---:|---:|
| 1 | 359 ms | 379 ms | 2.82 vectors/s |
| 8 | 604 ms | 661 ms | 13.21 vectors/s |
| 16 | 896 ms | 982 ms | 17.65 vectors/s |
| 32 | 1,460 ms | 1,819 ms | 21.05 vectors/s |

Observed resident resources after the run:

- GPU memory: 847 MiB used of 2,048 MiB
- service RSS/cgroup memory: approximately 606 MB
- LXC memory available: approximately 1,462 MB of 2,048 MB
- cached model load: approximately 3 seconds from load log to dimension validation
- first successful load during initial setup: approximately 11 seconds

The three synthetic baseline vectors are 768-dimensional and unnormalized. Their
combined canonical JSON SHA-256 is
`9b9522dfd1c314fec1c3f19089d5747f310829eeffaf4675f1916244e164d833`.

## Acceptance

The authenticated standalone deployment must show no material Nomic regression. A
p95 increase over 15% or throughput decrease over 15% requires explanation before
promotion. Authentication overhead should be negligible relative to GPU inference.
Raw benchmark artifacts are gitignored because they are environment-specific.
