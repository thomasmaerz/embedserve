# FreeHire integration

FreeHire uses Embedserve's authenticated `/embed` route for
`intfloat/multilingual-e5-base` and stores chunks with corpus identity
`intfloat/multilingual-e5-base-chunked-v1`.

## Worker contract

- request: `{"inputs":["passage: ..."]}`;
- response: bare normalized 768-dimensional vector arrays;
- prefixes: FreeHire owns `passage: ` and `query: `;
- configuration: `EMBED_URL`, `EMBED_API_KEY`, `EMBED_CONCURRENCY`;
- retries: bounded full jitter, `Retry-After` minimum, cancellation propagation;
- provenance: `semantic_embedded_model`, `semantic_embedded_hash`, and job
  `content_hash` must agree before retrieval.

The deployment patch remains out-of-tree from the pinned FreeHire submodule. It adds
only the client retry policy, bounded `EMBED_MAX_ITEMS`, release artifacts, and isolated
semantic query service.

## Semantic candidate retrieval

The ETL-loopback `semantic-agent` accepts one to six bounded query capsules and returns
canonical public open jobs, matched excerpts capped at 800 runes, semantic distance,
model, and content-hash provenance. It applies authoritative open/canonical/private,
date, geography, model, and content-hash filters before returning candidates.

Semantic retrieval is only a candidate-generation channel. Jobsearch fuses title,
lexical, canonical-skill, semantic, and external ranks with RRF (`k=60`), then reapplies
deterministic geography, date, language, authorization, liveness, deduplication, and
mandatory-requirement gates. Semantic similarity never overrides a blocker. If the
service is unavailable, the lexical and external workflow continues unchanged.

All validation uses synthetic lane capsules. Raw candidate profiles and resumes are not
sent to the semantic endpoint or stored in test fixtures.
