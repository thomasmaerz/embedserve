# Contributing

Contributions are welcome through focused issues and pull requests.

## Development

Install the locked development environment and run the required checks:

```bash
uv sync --frozen --extra dev
uv run --frozen ruff check .
uv run --frozen mypy src/embedserve
uv run --frozen pytest
```

The unit suite uses a fake encoder and does not download models or require a GPU.
Hardware compatibility tests require a private deployment and synthetic fixtures.

## Compatibility rules

- Keep model IDs, aliases, revisions, dimensions, prefix ownership, truncation, and
  normalization behavior explicit.
- Treat an embedding recipe change as a vector-space migration, not a patch release.
- Reject arbitrary model names; every model and remote-code revision must be reviewed
  and allowlisted.
- Add contract tests for every endpoint change.
- Never commit keys, private endpoints, model caches, benchmark raw output, logs,
  candidate data, or machine-specific deployment state.

Explain compatibility, deployment, security, and rollback impact in each pull request.
Contributions are licensed under the MIT License.
