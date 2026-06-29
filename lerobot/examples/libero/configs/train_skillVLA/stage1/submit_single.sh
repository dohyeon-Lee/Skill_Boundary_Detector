#!/usr/bin/env bash
# Stage-1 experiment: SINGLE-STAGE (CFG-style) → {base}/single/<oracle> (one run). (yaml untouched)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_train.sh" single "$@"
