#!/usr/bin/env bash
# Hugging Face version of sync_server.sh: pick a dataset root (dataset_filtered, ...), then folders in it,
# and push them to / pull them from the private dataset repo named by hf_dataset_repo in global_config.yaml.
#   bash hf_sync.sh                          # interactive (push / pull, then pick folders)
#   bash hf_sync.sh pull dataset_filtered/libero_90_full_full   # non-interactive; --yes skips the prompt
#   bash hf_sync.sh watch --keep 3 --protect 050000,100000     # (tmux, RunPod) upload new checkpoints (public)
# Log in once per server: hf auth login   (or export HF_TOKEN=...)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/lerobot/examples/libero/configs/src/hf_sync.py" "$@"
