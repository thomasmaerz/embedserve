# Security policy

Report suspected vulnerabilities through GitHub private vulnerability reporting. Do
not open a public issue containing credentials, private endpoints, logs, or request
content.

## Deployment boundary

Embedserve is designed for a low-consequence trusted LAN, not the public internet.
Operators must:

- require bearer authentication on every route, including health checks;
- bind only where approved clients can route to the service;
- restrict ingress to those client hosts with a host or network firewall;
- keep key files mode `0600` and outside Git;
- run one service process per GPU;
- pin model weights, reviewed remote code, and Python dependencies;
- retain a tested rollback artifact before each deployment;
- inspect logs and Git history for secrets before publication.

Plain HTTP on a trusted isolated LAN is an explicitly accepted risk. Add TLS or a VPN
before crossing an untrusted network. Bearer authentication does not prevent passive
observers from reading a key sent over plaintext HTTP.

See `docs/security.md` for the current review and threat model.
