#!/usr/bin/env bash
# ABC-130k episode visualizer — edit config.yaml, then run this.
#
#   ./run.sh                 # render the HTML gallery per config.yaml
#   ./run.sh --dry-run       # list episodes that would be rendered
#   (cap the count with `max_episodes:` in config.yaml)
#
# Extra flags pass straight through to src/visualize_abc.py.
# Needs: PyAV + mcap + mcap-protobuf-support (uv pip install into the project venv).
# Override the interpreter with:  PYTHON=/path/to/python ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"     # abcdl_RLLAB/ — on PYTHONPATH (in case you reuse abcdl helpers)

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$HERE"
echo "[run.sh] python=$PY"
exec "$PY" src/visualize_abc.py --config config.yaml "$@"
