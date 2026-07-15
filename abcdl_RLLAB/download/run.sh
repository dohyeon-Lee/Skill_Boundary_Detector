#!/usr/bin/env bash
# ABC-130k selective downloader — edit config.yaml, then run this.
#
#   ./run.sh                 # download everything listed in config.yaml
#   ./run.sh --dry-run       # resolve + print the plan, download nothing
#   ./run.sh --list-tasks    # print all task-folder names (fill your groups)
#   ./run.sh --list-tasks --counts   # + episode count per task (slow)
#
# Any extra flags pass straight through to src/download_abc.py.
# Override the interpreter with:  PYTHON=/path/to/python ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"     # abcdl_RLLAB/ — put on PYTHONPATH so `import abcdl` works (convert step)

# Pick an interpreter: $PYTHON > project venv > python3/python
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
exec "$PY" src/download_abc.py --config config.yaml "$@"
