# Troubleshooting

| Symptom | Diagnosis | Action |
|---|---|---|
| `401` on every route | Missing/mismatched key | Compare SHA-256 fingerprints and mode; restart client |
| key-file startup rejection | Group/world permission bits | Set owner correctly and mode `0600` |
| CUDA required error | Driver/device/wheel mismatch | Verify `torch.cuda.is_available()` and `sm_61` wheel |
| model revision mismatch | Cache or config drift | Restore pinned commit; do not bypass validation |
| digest mismatch in SlackQuery | Compatibility config changed | Restore the documented 64-character digest |
| dimension mismatch | Wrong model or upstream change | Stop deployment; never coerce a different model |
| slow first request | cold model/cache/CUDA startup | Use health readiness and compare load baseline |
| `413` | caller exceeded body limit | reduce batch/body before considering a measured limit change |
| `503 MODEL_BUSY` | alternate model owns or reserved the GPU | honor `Retry-After`; inspect repeated switching only if retry budget exhausts |
| `503 MODEL_LOAD_FAILED` | target load failed | check restored model in response and sanitized health failure state |
| `500 INFERENCE_FAILED` | encoder/CUDA failure | stop batch, inspect sanitized category, restart or lower measured batch |
| GPU OOM | batch or duplicate process | verify one PID, reduce batch, restart, remeasure |
| remote timeout | bind/firewall/route | verify private address and source allowlist |

Use `journalctl -u embedserve.service` for service errors. Do not enable HTTP header or
body logging while diagnosing authentication or model requests.
