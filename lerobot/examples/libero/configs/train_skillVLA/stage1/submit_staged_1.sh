#!/usr/bin/env bash
# Stage-1 experiment: STAGED phase 1-1 → {run}/1-1/ (run FIRST). (yaml untouched)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_train.sh" staged_1 "$@"
