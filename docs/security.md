# Security review

Scope: trusted-LAN, low-consequence embedding service. Findings are classified as
fixed, accepted, or deferred.

| Area | Classification | Decision |
|---|---|---|
| Endpoint authentication | Fixed | Constant-time bearer check on every route |
| Key storage | Fixed | Mode-`0600` file outside Git; fingerprint-only verification |
| Arbitrary model loading | Fixed | Exact model ID and alias allowlist |
| Model provenance | Fixed | Weight and remote-code commits pinned separately |
| Request abuse | Fixed | Body, item, character, model, dimension, and format limits |
| Duplicate GPU loads | Fixed | One systemd process, one Uvicorn worker, one encoder |
| Service privilege | Fixed | Dedicated user with only required cache and GPU access |
| Logs | Fixed | No application logging of headers, key, or input text |
| Plain HTTP | Accepted | Trusted isolated LAN; add TLS/VPN before boundary changes |
| Host firewall | Deployment fix | Permit only approved consumer hosts on TCP `11435` |
| `trust_remote_code` | Accepted with control | Required by Nomic; exact reviewed code commit pinned |
| Dependency drift | Fixed | `uv.lock` plus exact direct dependency pins |
| Distributed rate limiting | Deferred | No realistic current need; bounded requests and one lock |

Unauthenticated public access, committed credentials, secret-bearing logs, arbitrary
model selection, and unpinned remote code are not acceptable.
