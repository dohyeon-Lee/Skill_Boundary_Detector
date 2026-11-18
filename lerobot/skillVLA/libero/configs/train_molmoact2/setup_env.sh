#!/usr/bin/env bash
# Standalone MolmoAct2 setup; the root setup_env.sh invokes this by default.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${MODULE_DIR}/../../../../.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/third_party/lerobot_molmoact2"
SOURCE_REV="146d8474932c8b566717eb2738ba459337ab71c4"
VENV_DIR="${PROJECT_ROOT}/.venv_molmoact2_lerobot"
PYTHON="${VENV_DIR}/bin/python"
LOCK="${MODULE_DIR}/requirements.lock.txt"
CHECKER="${MODULE_DIR}/PT/src/check_environment.py"
CHECK_ONLY=false
CLONED_SOURCE=false

case "${1:-}" in
    "") ;;
    --check) CHECK_ONLY=true ;;
    -h|--help)
        echo "Usage: bash setup_env.sh [--check]"
        echo "Prepare pinned HF LeRobot + .venv_molmoact2_lerobot; no model/data downloads."
        echo "--check: read-only source, dependency and LeRobot trainer import checks."
        exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    echo "ERROR: Expected at most one argument." >&2
    exit 2
fi

fail() { echo "ERROR: $*" >&2; exit 1; }
source "${PROJECT_ROOT}/tools/setup/system_deps.sh"
if [ "$CHECK_ONLY" = false ]; then
    sbd_ensure_system_tools
fi
command -v git >/dev/null || fail "git is required."
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || \
    fail "This CUDA 12.8 lock targets Linux x86_64 / Python 3.12."

# A caller may have activated the project's OTHER LeRobot. Never inherit its imports.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export GIT_LFS_SKIP_SMUDGE=1
if [ -z "${SSL_CERT_FILE:-}" ]; then
    for cert in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt /etc/ssl/cert.pem; do
        if [ -f "$cert" ]; then export SSL_CERT_FILE="$cert"; break; fi
    done
fi

if [ "$CHECK_ONLY" = false ]; then
    UV="${HOME}/.local/bin/uv"
    if [ ! -x "$UV" ]; then
        if command -v uv >/dev/null; then
            UV="$(command -v uv)"
        else
            command -v curl >/dev/null || fail "Install uv or curl first."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            [ -x "$UV" ] || fail "uv installer did not create $UV."
        fi
    fi

    if [ ! -e "$SOURCE_DIR" ]; then
        echo "MolmoAct2: cloning official source (no recursive submodules)..."
        mkdir -p "${PROJECT_ROOT}/third_party"
        git -c submodule.recurse=false clone --no-recurse-submodules --no-checkout \
            https://github.com/huggingface/lerobot.git "$SOURCE_DIR"
        CLONED_SOURCE=true
    fi
fi

[ -d "$SOURCE_DIR" ] || fail "MolmoAct2 source missing. Run bash setup_env.sh --molmoact2-only."
# Do not mistake the outer SkillVLA Git checkout for the source repository.
SOURCE_TOP="$(git -C "$SOURCE_DIR" rev-parse --show-toplevel 2>/dev/null)" || \
    fail "$SOURCE_DIR is not a Git checkout; nothing was overwritten."
[ "$(cd "$SOURCE_TOP" && pwd -P)" = "$(cd "$SOURCE_DIR" && pwd -P)" ] || \
    fail "$SOURCE_DIR must be its own Git checkout; nothing was overwritten."
CURRENT_REV="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [ "$CURRENT_REV" != "$SOURCE_REV" ] || [ "$CLONED_SOURCE" = true ]; then
    [ "$CHECK_ONLY" = false ] || fail "MolmoAct2 revision $CURRENT_REV differs from $SOURCE_REV."
    if [ "$CLONED_SOURCE" = false ]; then
        [ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ] || \
            fail "MolmoAct2 has local changes at $CURRENT_REV; save them before changing revision."
    fi
    if ! git -C "$SOURCE_DIR" cat-file -e "${SOURCE_REV}^{commit}" 2>/dev/null; then
        git -C "$SOURCE_DIR" -c submodule.recurse=false fetch origin "$SOURCE_REV"
    fi
    git -C "$SOURCE_DIR" -c submodule.recurse=false checkout --detach "$SOURCE_REV"
fi
for required in pyproject.toml src/lerobot/policies/molmoact2/modeling_molmoact2.py src/lerobot/scripts/lerobot_train.py; do
    [ -f "$SOURCE_DIR/$required" ] || fail "Missing LeRobot training source: $SOURCE_DIR/$required"
done
echo "MolmoAct2 source: $SOURCE_REV"
if [ -n "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: preserving local MolmoAct2 source modifications."
fi

if [ "$CHECK_ONLY" = false ]; then
    if [ ! -e "$VENV_DIR" ]; then
        "$UV" venv "$VENV_DIR" --python 3.12
    fi
    [ -x "$PYTHON" ] && [ -f "$VENV_DIR/pyvenv.cfg" ] || \
        fail "$VENV_DIR exists but is not a usable venv; nothing was removed."
    "$PYTHON" -c 'import sys; from pathlib import Path
assert sys.version_info[:2] == (3, 12), "MolmoAct2 requires Python 3.12"
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), "Refusing to install outside the Molmo venv"' "$VENV_DIR"
    sbd_ensure_python_headers "$PYTHON"

    # egl-probe 1.0.2 still declares CMake 2.8.12. Use our validated CMake 3,
    # including inside uv's isolated builds, rather than the main venv's CMake 4.
    export PATH="${VENV_DIR}/bin:${PATH}"
    CMAKE_PIN="$(sed -n '/^cmake==/p' "$LOCK")"
    [ -n "$CMAKE_PIN" ] || fail "No CMake pin in $LOCK."
    "$UV" pip install --python "$PYTHON" "$CMAKE_PIN"
    "$UV" pip install --python "$PYTHON" --torch-backend cu128 \
        -r "$LOCK" \
        -e "${SOURCE_DIR}[molmoact2,training,libero]"
    "$UV" pip check --python "$PYTHON"
fi

[ -x "$PYTHON" ] || fail "MolmoAct2 venv missing. Run bash setup_env.sh --molmoact2-only."
"$PYTHON" "$CHECKER"
echo "MolmoAct2 준비 완료. 모델 가중치와 데이터는 별도로 다운로드하세요."
