# Provenance

Embedserve was extracted on 2026-10-08 from
[`thomasmaerz/slackquery`](https://github.com/thomasmaerz/slackquery).

## Source record

- Authoritative SlackQuery branch at extraction: `main`
- Authoritative branch commit: `aa416b58699a3efa992a368187054b23a7306ab3`
- Server-introduction commit after repository history normalization:
  `51971cc9d7318f4ed441821bb85c4a1df71337c4`
- Equivalent pre-normalization local commit:
  `f2847a5dc633e9ab40ee21dadb54b8f1e77d1b56`
- Original implementation: `src/slackquery/embedding_server.py`
- Original tests: `tests/test_embedding_server.py`
- Original service unit: `deploy/slackquery-embedding.service`
- Original guide: `docs/pytorch-embedding-server.md`
- Original implementation SHA-256:
  `f2f467bf7265d3fd44798d00e1948b73f8640472adfccb5ea220a54afcdce3a3`

The original files span package, test, deployment, and documentation paths in one
feature commit, so a path-only subtree extraction would either omit required material
or import unrelated SlackQuery code. This repository uses a documented source import
instead. SlackQuery's embedded implementation remains in place as a frozen legacy copy.

## Deployed prototype

The pre-extraction deployment used a separate, unauthenticated FastAPI prototype at
`/opt/embedserve/server.py`, SHA-256
`a5222f9f8dc6ae79a1ee9e35a65e5534955bc7fac7a3ad3bdd5411cad43202d7`.
It established the production alias, compatibility digest, 768-dimensional raw output,
2048-token encoder limit, and unnormalized inference behavior. It was not copied as the
new authority because the latest SlackQuery implementation added authentication,
bounded input handling, pinned revisions, and contract tests.

## Authority

After extraction, this repository is authoritative for the deployed embedding service.
The SlackQuery copy documents history and remains byte-for-byte unchanged. SlackQuery
continues to own client-side prefixes, 512-dimensional Matryoshka truncation, and L2
normalization.
