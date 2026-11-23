#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../remote.sh"
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../libero/configs/src" && pwd)/submit_args.sh"
