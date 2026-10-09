# Operations runbook

## Routine checks

1. Confirm `systemctl is-active embedserve.service` returns `active`.
2. Call authenticated `/health`; verify `status=ok`, `device=cuda`, dimension `768`,
   `single_residency_verified=true`, and the expected loaded model.
3. Call authenticated `/api/tags`; compare alias and digest with the SlackQuery client.
4. Inspect `nvidia-smi`; one process should own model VRAM.
5. Check recent unit logs for restarts, `401`, `413`, `500`, CUDA errors, or keys. Logs
   must never contain authorization values or request text.

## Key rotation

Generate a replacement into a new mode-`0600` file. Distribute by file copy or
stdin-to-file automation without terminal output. Compare SHA-256 fingerprints on all
hosts, atomically replace server and client files, restart the server and consumers,
then verify one authenticated request. Keep the old key only for the bounded rollback
window and securely remove it afterward.

Because one key is accepted at a time, rotation requires a short coordinated outage.
Do not weaken authentication to avoid that outage.

## Incident triage

- `401`: compare fingerprints and file modes; never print values.
- `404 unknown model`: reject the caller; do not add arbitrary model loading.
- `413`: validate caller batching before raising a limit.
- `503 MODEL_BUSY`: inspect switch frequency and client retry budgets; it is expected
  during bounded Nomic/E5 handoff, not a rate-limit signal.
- `503 MODEL_LOAD_FAILED`: stop new work, inspect the sanitized failure category and
  whether the previous model was restored.
- `500 INFERENCE_FAILED`: stop the current batch and inspect
  `last_inference_failure`; never enable body logging.
- scheduled E5 recycle: expected after 200 successful E5 requests; systemd restarts the
  single process and clients continue from their bounded retry loop.
- CUDA OOM: stop new work, capture VRAM/process metadata, restart once, then lower the
  batch size if the same synthetic request reproduces it.
- repeated restart: use `docs/rollback.md` before changing model or dependency pins.

## Monitoring scope

For this LAN deployment, systemd state, structured HTTP status counts, request latency,
process RSS, loaded-model state, switch time, retries, inference-failure count, and
`nvidia-smi` are sufficient. Do not add an enterprise telemetry stack unless an observed
failure mode justifies it.
