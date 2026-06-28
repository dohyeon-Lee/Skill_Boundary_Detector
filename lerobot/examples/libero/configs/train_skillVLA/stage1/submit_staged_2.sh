#!/usr/bin/env bash
# Stage-1 experiment: STAGED phase 1-2 → {run}/1-2/ (run AFTER staged_1; warm-starts from a chosen
# 1-1 checkpoint via yaml staged_phase2_warmstart). (yaml untouched)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_train.sh" staged_2 "$@"
