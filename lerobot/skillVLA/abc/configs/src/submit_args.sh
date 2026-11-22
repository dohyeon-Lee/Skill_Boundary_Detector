#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../remote.sh"
fi
# Share launch argument handling with LIBERO, including --JOB_ID.
source "$(dirname "${BASH_SOURCE[0]}")/../../../libero/configs/src/submit_args.sh"
