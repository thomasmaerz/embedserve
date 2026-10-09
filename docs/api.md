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
the lifecycle coordinator controls residency.

## OpenAI-compatible embedding

`POST /v1/embeddings` accepts the same `model` and `input`. It returns `object`, indexed
`data`, `model`, and approximate `usage` fields. Only `encoding_format: "float"` and
`dimensions: 768` are accepted.

## TEI-compatible E5 embedding

`POST /embed` accepts exactly one `inputs` string array and returns a bare array of
vectors:

```json
{"inputs":["passage: Synthetic platform role."]}
```

```json
[[0.01, -0.02]]
```

The vector is abbreviated; each row contains 768 normalized floats. Prefixes are
client-owned. The endpoint allows at most 32 inputs and uses the pinned
`intfloat/multilingual-e5-base` revision.

## Model-busy response

An alternate-model request first reserves a bounded handoff and receives `503` with an
integer-seconds `Retry-After` header:

```json
{
  "error": {
    "code": "MODEL_BUSY",
    "message": "Requested model is waiting for the active model to finish.",
    "retryable": true,
    "loaded_model": "nomic",
    "requested_model": "e5",
    "state": "SWITCH_RESERVED",
    "retry_after_ms": 1500
  }
}
```

Clients retry only this documented error, honor `Retry-After`, and bound attempts and
total elapsed time. `429` remains reserved for rate limiting.

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
| Nomic batch over 256 or E5 batch over 32 | `400` | No |
| Aggregate text over 1,000,000 characters | `400` | No |
| Invalid encoder output | `500` | Operator investigation |
| Alternate model owns GPU | `503` | Yes, `MODEL_BUSY` only |
| Model load failed | `503` | Yes when `retryable=true` |

Limits are configurable within validated bounds.
