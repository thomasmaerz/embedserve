# Deployment

## Requirements

- Python 3.11 or 3.12
- one NVIDIA GPU visible to the service account
- enough RAM and VRAM for the measured peak
- a private route from approved clients
- one Uvicorn worker

For Pascal `sm_61`, install PyTorch 2.4.1 from the CUDA 12.1 index before syncing the
remaining locked packages:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install \
  torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv sync --frozen --extra cuda-pascal --no-install-package torch
```

Verify CUDA before deployment:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## Filesystem

Use immutable release directories and a `current` symlink:

```text
/opt/embedserve/releases/<git-commit>/
/opt/embedserve/current -> releases/<git-commit>
/opt/embedserve/venvs/<git-commit>/
/opt/embedserve/venv -> venvs/<git-commit>
/etc/embedserve/embedserve.env
/etc/embedserve/api-key
/var/cache/embedserve/
```

The selected virtual environment contains the release's locked runtime dependencies;
the release source is selected through `PYTHONPATH`. On constrained hosts it may reuse
an unchanged, verified CUDA/model package layer while overriding every changed package
with the lockfile version. The environment file contains only non-secret settings. The key file is owned by the
service user and mode `0600`. Generate it with `scripts/generate_api_key.py`; do not
print, paste, or pass the key as an argument.

## systemd

Install `deploy/embedserve.service`, create an unprivileged `embedserve` user with GPU
device access, then enable the unit:

```bash
systemctl daemon-reload
systemctl enable --now embedserve.service
systemctl status embedserve.service
```

Do not add arbitrary memory or CPU limits. Measure cold load and peak batches first,
then set limits above observed peaks if the host needs containment.

The tested P620 LXC was increased from 2 GiB to 4 GiB after integrated switching and
FreeHire work measured an approximately 2.96 GB cgroup peak. This is measured headroom,
not a portable default. The GPU remains 2 GiB and permits only one loaded model.

## Network

Bind `0.0.0.0` only when remote LAN clients require it. Restrict TCP `11435` to the
approved SlackQuery host during Phase 1. Do not expose this service through a public
NAT rule. TLS is optional only while traffic remains on the accepted trusted LAN.
