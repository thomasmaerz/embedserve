# Phase 4 acceptance: E5 and TEI support

Phase 4 passes only if all of the following hold before deployment:

- the exact FreeHire TEI request/response and prefix ownership are verified in its source;
  no live TEI service exists on the inspected deployment, so normalization, sequence
  limit, model identity, and synthetic vectors are established from the pinned model;
- Nomic and E5 weight and remote-code revisions are immutable and allowlisted;
- `/api/embed` remains Nomic/Ollama-compatible and `/embed` implements only the E5 TEI
  `{"inputs": [...]}` to bare vector-array contract;
- both routes require the existing bearer key and arbitrary model selection is rejected;
- Nomic output matches the Phase 0 fixture tolerance and E5 output matches a synthetic
  baseline from the existing FreeHire service within a declared tolerance;
- every successful vector has 768 finite dimensions and preserves input ordering;
- scheduler tests cover both switch directions, busy responses, reservation expiry,
  starvation, cancellation, timeout, load failure, and active-counter cleanup;
- model unload removes application references, runs garbage collection, clears the CUDA
  allocator, and verifies measured VRAM before the alternate load starts;
- instrumentation and tests prove at most one model object and one model's GPU allocation
  exist at every point;
- batch 1, 8, 16, and 32 are measured where the GPU can safely support them; an unsafe
  batch is rejected or reduced rather than allowed to OOM;
- service health remains available after a model-load failure and the coordinator can
  retry or restore the previous model.

Phase 4 fails and blocks FreeHire integration if E5 cannot fit by itself on the 2 GiB
GPU with measured headroom, if any overlap occurs during switching, if the existing TEI
contract cannot be reproduced, if Nomic compatibility regresses, or if a cancellation or
failure can wedge the coordinator.
