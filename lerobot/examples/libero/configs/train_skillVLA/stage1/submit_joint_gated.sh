#!/usr/bin/env bash
# Stage-1 experiment: JOINT + WEIGHTED_GATED. (yaml untouched)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_train.sh" joint_gated "$@"
