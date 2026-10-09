# Model lifecycle

## Phase 1: static Nomic

The process validates an exact allowlisted model ID and alias, then loads pinned weights
and pinned reviewed remote code before opening the HTTP listener. It sets
`max_seq_length=2048`, validates a native dimension of 768, and retains one model until
process exit. One Uvicorn worker and one inference lock prevent accidental duplicate
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

## Planned multi-model lifecycle

E5 support will unload Nomic before loading E5 and will verify VRAM release before the
next load. It requires a synchronized state machine and starvation-safe reservation;
it is intentionally absent from the Phase 1 release. Stateless retries alone are not
accepted as a scheduler.
