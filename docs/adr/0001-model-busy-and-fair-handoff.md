# ADR 0001: Model-busy responses and fair GPU handoff

- Status: accepted
- Date: 2026-10-09

## Context

Nomic and E5 cannot coexist in the Quadro P620's 2 GiB VRAM. Requests remain queued on
clients, but stateless retries are insufficient: continuous requests for the loaded model
can prevent the alternate model from ever observing an idle GPU.

## HTTP decision

A valid embedding request that cannot run because another model owns the GPU receives
`503 Service Unavailable`, an integer-seconds `Retry-After` header, and a bounded JSON
error. RFC 9110 defines 503 for temporary server inability and permits `Retry-After` to
communicate the expected delay. `Retry-After` accepts either an HTTP date or decimal
delay-seconds; Embedserve uses delay-seconds.

`429 Too Many Requests` is not used because RFC 6585 defines it for rate limiting. `409
Conflict` is not used because RFC 9110 defines it as a client-resolvable conflict with
the target resource's state. GPU model availability is temporary server capacity, not a
malformed request, rate limit, or target-resource edit conflict.

The server rejects the request before embedding begins. Although embedding uses POST,
its operation is read-only and the rejected request was not applied, so a documented
client retry is safe. Clients retry only `MODEL_BUSY` responses marked retryable.

Sources:

- RFC 9110, 503: https://www.rfc-editor.org/rfc/rfc9110.html#section-15.6.4
- RFC 9110, Retry-After: https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3
- RFC 9110, 409: https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.10
- RFC 9110, idempotency and retry: https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2
- RFC 6585, 429: https://www.rfc-editor.org/rfc/rfc6585.html#section-4

## Scheduling decision

One in-process coordinator owns loaded-model identity, active-request accounting,
alternate intent, switch reservation, switching task, last activity, last successful
switch, and load-failure state. It uses one `asyncio.Lock` only for short state changes;
model loading and inference do not hold the coordinator lock.

When an alternate request arrives, the coordinator records bounded intent and reserves
the next handoff immediately. Existing requests drain, while new requests for the old
model receive `MODEL_BUSY`. This admission barrier guarantees progress even under
continuous old-model traffic. The first alternate retry after the active count reaches
zero claims the reservation and performs one unload/load transition. Abandoned intent
and reservations expire, allowing old-model traffic to resume.

The reservation is process-local and protects one local GPU, so distributed consensus,
renewal, and fencing tokens would add complexity without reducing a realistic risk.
Cancellation cleanup uses `try/finally`; Python documents that cancellation should use
`try/finally`, and that `asyncio.shield` allows an internal operation to complete when
its caller is cancelled. The switching task is shielded so a disconnected claimant
cannot leave the GPU half-switched.

Clients use capped exponential backoff with full jitter:

```text
delay = random(0, min(cap, base * 2^attempt))
```

`Retry-After` is a minimum. Attempts and total elapsed time are bounded, cancellation is
propagated, and only the documented retryable error is retried. AWS demonstrates that
full jitter reduces synchronized retry bursts and total client work.

Sources:

- Python asyncio synchronization: https://docs.python.org/3/library/asyncio-sync.html
- Python cancellation and shield: https://docs.python.org/3/library/asyncio-task.html
- AWS full jitter: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- AWS retry guidance: https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/

## Phase 3 acceptance criteria

Pass requires automated evidence that:

- same-model requests do not switch models;
- an alternate request receives bounded `503 MODEL_BUSY` metadata;
- alternate intent blocks new old-model admissions and cannot starve;
- one alternate retry performs exactly one unload/load and reverse handoff also works;
- abandoned intent and reservations expire;
- load failure restores the known-good model where possible and remains retryable;
- claimant cancellation and request cancellation leak no active count or reservation;
- only one model is resident throughout every test;
- client retries honor `Retry-After`, use full jitter, preserve cancellation, and stop at
  bounded attempts or deadline;
- authentication, validation, unsupported-model, and non-busy errors are not retried.

Any failed invariant blocks E5 implementation.
