# Model lifecycle

## Startup

The process validates an exact allowlisted model ID and alias, then loads pinned weights
and pinned reviewed remote code before opening the HTTP listener. It sets
`max_seq_length=2048`, validates a native dimension of 768, and retains Nomic until the
first switch. One Uvicorn worker and one inference lock prevent accidental duplicate
loads and concurrent calls into the small GPU.

The API's compatibility digest is not a loader revision. It remains stable so existing
SlackQuery vector generations remain identifiable. Weight and remote-code revisions are
separate settings and health fields.

## Change control

A model, revision, remote-code, sequence-limit, prefix, truncation, normalization, or
dimension change creates a new embedding recipe. Before promotion:

1. review upstream weights and remote code;
2. pin immutable revisions;
3. run synthetic contract fixtures on the target GPU;
4. measure cold load, warm latency, RAM, and VRAM;
5. compare retrieval behavior against the current generation;
6. prepare a re-embedding and rollback plan;
7. never mix recipes inside one corpus.

## Multi-model lifecycle

An alternate request creates bounded intent and reserves the next switch. Existing work
drains, old-model arrivals receive `MODEL_BUSY`, and the first alternate retry claims the
reservation. The runtime drops all old-model references, synchronizes CUDA, runs garbage
collection, clears the CUDA cache, and checks residual allocator bytes before loading the
alternate model. A five-second minimum residency window limits model thrashing.

Intent and reservations expire so a dead client cannot wedge the service. The internal
switch task is shielded from claimant cancellation. If target loading fails after unload,
the coordinator attempts to restore the old model; otherwise it enters `DEGRADED` while
health remains available. See `docs/adr/0001-model-busy-and-fair-handoff.md`.

The Pascal/CUDA stack can enter a non-recoverable process-local CUDA error state after
long E5 drains. Embedserve therefore returns a retryable sanitized failure and schedules
a systemd-supervised process recycle on CUDA runtime/OOM failures. It also recycles after
200 successful E5 HTTP requests, before the observed long-run failure window. Clients
retain request state, honor `Retry-After`, and retry after the process reloads; no second
model process is started concurrently.
