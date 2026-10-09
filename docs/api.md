# API reference

All routes require `Authorization: Bearer <key>`. Missing, malformed, and incorrect
credentials return `401` with `WWW-Authenticate: Bearer` and an authentication error.
Keys and request text are never included in application error messages.

## Ollama-compatible embedding

`POST /api/embed`

```json
{
  "model": "nomic-embed-text:v1.5",
  "input": ["search_query: synthetic database timeout"],
  "truncate": true,
  "keep_alive": "5m"
}
```

```json
{
  "model": "nomic-embed-text:v1.5",
  "embeddings": [[0.01, -0.02]]
}
```

The example vector is abbreviated. Actual rows contain exactly 768 floats. `input` may
be one string or a non-empty string array. `keep_alive` is accepted and ignored because
the model remains resident for the process lifetime.

## OpenAI-compatible embedding

`POST /v1/embeddings` accepts the same `model` and `input`. It returns `object`, indexed
`data`, `model`, and approximate `usage` fields. Only `encoding_format: "float"` and
`dimensions: 768` are accepted.

## Discovery

- `GET /health` reports status, source model, alias, pinned model and remote-code
  revisions, compatibility digest, device, GPU name, and native dimension.
- `GET /api/tags` returns one Ollama-shaped model entry. Its `digest` is the preserved
  SlackQuery generation digest, not the Hugging Face commit.
- `GET /v1/models` returns one OpenAI-shaped model entry.

## Limits and errors

| Condition | Status | Retry |
|---|---:|---|
| Missing or incorrect key | `401` | No |
| Invalid JSON or option | `400` | No |
| Unsupported model alias | `404` | No |
| Body over 2,000,000 bytes | `413` | No |
| Batch over 256 items | `400` | No |
| Aggregate text over 1,000,000 characters | `400` | No |
| Invalid encoder output | `500` | Operator investigation |

Limits are configurable downward or upward within validated bounds. Phase 1 has no
model-busy response because it serves only one statically loaded model.
