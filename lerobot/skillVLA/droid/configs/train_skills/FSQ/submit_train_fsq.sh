#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"; fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../src/submit_workflow.sh"
droid_submit FSQ "${HERE}/fsq_config.yaml" "$@"
