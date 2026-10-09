<div align="center">

# Embedserve

### A small, authenticated PyTorch embedding service for compatibility-sensitive local workloads

`Ollama API subset` · `OpenAI API subset` · `CUDA` · `Pinned Nomic` · `LAN deployment`

**Keep it private. Every route requires a bearer key. Do not expose port `11435` to the internet.**

</div>

---

Embedserve extracts SlackQuery's proven PyTorch path into an independent service. It
serves raw 768-dimensional `nomic-ai/nomic-embed-text-v1.5` vectors while preserving
SlackQuery's exact alias and digest contract. SlackQuery remains responsible for
adding `search_document: ` or `search_query: `, taking the first 512 Matryoshka
dimensions, and L2-normalizing the stored vector.

The current release is intentionally Nomic-only. Multi-model Nomic/E5 scheduling and
the FreeHire TEI route are post-checkpoint work and are not claimed by this release.

## API surface

| Route | Shape | Authentication | Purpose |
|---|---|---|---|
| `GET /health` | Embedserve | Bearer | Device, pinned revision, and dimensions |
| `GET /api/tags` | Ollama subset | Bearer | Alias and compatibility digest |
| `POST /api/embed` | Ollama subset | Bearer | Raw 768-dimensional vectors |
| `GET /v1/models` | OpenAI subset | Bearer | Model discovery |
| `POST /v1/embeddings` | OpenAI subset | Bearer | OpenAI-shaped vectors |

Only `nomic-embed-text:v1.5` is accepted. The server does not add task prefixes and
does not normalize or truncate vector dimensions. The encoder truncates token input at
Nomic's configured 2048-token sequence limit; `truncate=false` is rejected.

## Quick start

Python 3.11 or 3.12 is required. The Quadro P620 deployment uses the CUDA 12.1 wheel
for PyTorch 2.4.1 because it includes `sm_61` kernels.

```bash
uv sync --frozen --extra cuda-pascal
install -d -m 0700 .secrets
uv run python scripts/generate_api_key.py .secrets/api-key
cp .env.example .env
```

Set `EMBEDSERVE_API_KEY_FILE` in `.env`, export the remaining settings, then run one
process:

```bash
set -a
. ./.env
set +a
uv run --frozen embedserve
```

Probe with a key loaded from a private file without printing it or placing it in process
arguments:

```python
import json
import urllib.request
from pathlib import Path

key = Path(".secrets/api-key").read_text(encoding="ascii").strip()
request = urllib.request.Request(
    "http://127.0.0.1:11435/health",
    headers={"Authorization": f"Bearer {key}"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    print(json.load(response)["status"])
```

Run the snippet from a protected script or standard input. For managed deployment, use
the key-file and systemd workflow in `docs/deployment.md`.

## Documentation

- `docs/architecture.md` - trust boundary and component design
- `docs/api.md` - exact requests, responses, limits, and errors
- `docs/compatibility.md` - SlackQuery/Nomic compatibility matrix
- `docs/deployment.md` - CUDA and systemd deployment
- `docs/operations.md` - monitoring, rotation, and incident operations
- `docs/model-lifecycle.md` - pinning, loading, and vector-space invariants
- `docs/performance.md` - Quadro P620 baseline measurements
- `docs/security.md` - proportional LAN security review
- `docs/troubleshooting.md` - common failures and diagnosis
- `docs/rollback.md` - release, unit, key, and client rollback
- `PROVENANCE.md` - extraction history and authority boundary

## Development

```bash
uv sync --frozen --extra dev
uv run --frozen ruff check .
uv run --frozen mypy src/embedserve
uv run --frozen pytest
```

The project is licensed under the MIT License.
