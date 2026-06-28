#!/usr/bin/env bash
# Stage-1 experiment: JOINT + PLAIN. (yaml untouched; experiment fixed by this script)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_train.sh" joint_plain "$@"
