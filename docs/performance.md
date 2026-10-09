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

## Phase 2 authenticated deployment

Measured after promotion with the same fixtures, sample count, model cache, GPU, and
batch sizes. Direct runtime dependencies were pinned to the repository versions.

| Batch | p50 | p95 | Throughput | p50 change | p95 change | Throughput change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 392 ms | 417 ms | 2.58 vectors/s | +9.1% | +10.0% | -8.3% |
| 8 | 627 ms | 660 ms | 12.74 vectors/s | +3.9% | -0.2% | -3.6% |
| 16 | 909 ms | 1,135 ms | 17.33 vectors/s | +1.4% | +15.6% | -1.8% |
| 32 | 1,453 ms | 1,543 ms | 21.97 vectors/s | -0.5% | -15.2% | +4.4% |

Batch 16 p95 exceeded the 15% review line by 0.6 percentage points while its p50 and
throughput remained within 2% of baseline. Other batch p95 values improved or remained
within 10%. This isolated tail sample is accepted as measurement variance rather than a
material regression. Synthetic vectors matched the baseline exactly.

## Phase 4 E5 feasibility

Measured 2026-10-09 on the target P620 before service integration, using pinned
`intfloat/multilingual-e5-base` revision
`d128750597153bb5987e10b1c3493a34e5a4502a`, normalized output, synthetic text, and an
internal encoder batch size of one:

| Input count | Warm latency | Throughput |
|---:|---:|---:|
| 1 | 275 ms | 3.64 vectors/s |
| 8 | 166 ms | 48.24 vectors/s |
| 16 | 326 ms | 49.12 vectors/s |
| 32 | 648 ms | 49.39 vectors/s |

- cold E5 load: 2.73 seconds from cached files;
- model CUDA allocation: approximately 1,112 MB;
- peak CUDA allocation through batch 32: approximately 1,122 MB;
- CUDA allocation after unload/cache clear: 8,519,680 bytes;
- maximum model objects resident: one;
- two-vector synthetic fixture SHA-256:
  `aa8f7d8c3acff505084e9a013a5d3dbec976b447756545a4da97fcfe91e04ac2`.

The model fits alone with measured headroom. The service conservatively keeps E5's
internal batch size at one until integrated HTTP and switch benchmarks justify a higher
value.

## Integrated E5 HTTP

Ten warm synthetic samples per input count through authenticated `/embed`:

| Input count | p50 | p95 | Throughput |
|---:|---:|---:|---:|
| 1 | 369 ms | 377 ms | 2.77 vectors/s |
| 8 | 667 ms | 718 ms | 11.87 vectors/s |
| 16 | 1,003 ms | 1,194 ms | 15.18 vectors/s |
| 32 | 1,721 ms | 1,909 ms | 18.34 vectors/s |

Three alternating synthetic switch cycles measured median 9.79 seconds Nomic-to-E5 and
9.05 seconds E5-to-Nomic, including reservation retries and the five-second anti-thrash
window. The range was 6.03-11.27 seconds after excluding an already-loaded E5 request.

## FreeHire stages

- canary: 5 jobs, 12 chunks, 8 seconds, zero failures;
- stage: 1,000 jobs in 292 seconds, zero failures;
- stage: 10,000 jobs in 2,728 seconds, zero failures;
- stable observed rate: approximately 3.4-3.7 jobs/s;
- projected 321,236-job initial corpus: approximately 24-26 hours;
- current measured corpus: 11,611 jobs and 29,480 chunks, all 768-dimensional,
  unit-normalized, current-model, and content-hash current;
- projected chunk count at current ratio: approximately 808,000.

An attempted 50,000-job stage accidentally used an older artifact and failed fast. Its
50,000 live outbox rows were reset to attempt zero after diagnosis; no failed vector rows
or mixed-model corpus were written. This is retained as rollback evidence, not counted as
successful throughput.

Long E5 drains later exposed a process-local CUDA runtime failure after sustained request
volume. The production mitigation recycles the one service process after 200 successful
E5 requests and immediately after a CUDA runtime/OOM failure. This converts a persistent
poisoned context into bounded `503` retry time while preserving single residency.

Interactive SlackQuery remained usable during a stable FreeHire stage: the first query
paid a 4.80-second model handoff, followed by four 142-154 ms warm queries. The reservation
and retry design therefore bounds switch delay while preserving warm interactive latency.

The LXC was raised from 2 GiB to 4 GiB only after measured cgroup memory reached about
2.96 GB during switching/backfill. No CPU limit was added. GPU peak during the canary was
1,203 MiB, with no model overlap.
