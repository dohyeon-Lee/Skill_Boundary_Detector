#!/usr/bin/env bash
# Isolated Coffee simulator. Training checkpoints and datasets are separate.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${MODULE_DIR}/../../../../../.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/third_party/mimicgen"
SOURCE_REV="$(<"${MODULE_DIR}/source_revision.txt")"
VENV_DIR="${PROJECT_ROOT}/.venv_mimicgen"
PYTHON="${VENV_DIR}/bin/python"
CHECK_ONLY=false
RENDER=false

fail() { echo "ERROR: $*" >&2; exit 1; }
for argument in "$@"; do
    case "$argument" in
        --check) CHECK_ONLY=true ;;
        --render) RENDER=true ;;
        -h|--help)
            echo "Usage: bash setup_env.sh --mimicgen-only [--check] [--render]"
            echo "Install pinned MimicGen + robosuite 1.4.1 into .venv_mimicgen."
            echo "Checks Coffee D0 reset/step, Panda observations and camera calibration on CPU."
            echo "--check: validate an existing installation; no clone/install/download."
            echo "--render: additionally render both 256x256 cameras; defaults to MUJOCO_GL=egl."
            echo "No episode datasets, policy weights, or VLA training packages are downloaded."
            exit 0 ;;
        *) echo "ERROR: Unknown argument: $argument" >&2; exit 2 ;;
    esac
done
[[ "$SOURCE_REV" =~ ^[0-9a-f]{40}$ ]] || fail "Invalid MimicGen source revision."
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || \
    fail "The simulator lock targets Linux x86_64 / Python 3.12."
command -v git >/dev/null || fail "git is required."

# Do not inherit imports from the main/ABC/GR00T virtual environments.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
if [ -z "${SSL_CERT_FILE:-}" ]; then
    for cert in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt /etc/ssl/cert.pem; do
        if [ -f "$cert" ]; then export SSL_CERT_FILE="$cert"; break; fi
    done
fi

if [ "$CHECK_ONLY" = false ]; then
    source "${PROJECT_ROOT}/tools/setup/system_deps.sh"
    sbd_ensure_system_tools
    UV="${HOME}/.local/bin/uv"
    if [ ! -x "$UV" ]; then
        if command -v uv >/dev/null; then UV="$(command -v uv)"
        else
            curl -LsSf https://astral.sh/uv/install.sh | sh
            [ -x "$UV" ] || fail "uv installer did not create $UV."
        fi
    fi
    if [ ! -e "$SOURCE_DIR" ]; then
        mkdir -p "${PROJECT_ROOT}/third_party"
        git init -q "$SOURCE_DIR"
        git -C "$SOURCE_DIR" remote add origin https://github.com/NVlabs/mimicgen.git
    fi
fi

[ -d "$SOURCE_DIR" ] && [ ! -L "$SOURCE_DIR" ] || \
    fail "MimicGen source missing or symlinked. Run bash setup_env.sh --mimicgen-only."
SOURCE_TOP="$(git -C "$SOURCE_DIR" rev-parse --show-toplevel 2>/dev/null)" || \
    fail "MimicGen source must be its own Git checkout; nothing was overwritten."
[ "$(cd "$SOURCE_TOP" && pwd -P)" = "$(cd "$SOURCE_DIR" && pwd -P)" ] || \
    fail "MimicGen source resolves to the outer project checkout; nothing was overwritten."
[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ] || \
    fail "MimicGen has local source changes; nothing was overwritten."
CURRENT_REV="$(git -C "$SOURCE_DIR" rev-parse --verify HEAD 2>/dev/null || true)"
if [ "$CURRENT_REV" != "$SOURCE_REV" ]; then
    [ "$CHECK_ONLY" = false ] || fail "MimicGen revision differs from $SOURCE_REV."
    if ! git -C "$SOURCE_DIR" cat-file -e "${SOURCE_REV}^{commit}" 2>/dev/null; then
        git -C "$SOURCE_DIR" -c submodule.recurse=false fetch --depth 1 origin "$SOURCE_REV"
    fi
    git -C "$SOURCE_DIR" -c submodule.recurse=false checkout --detach "$SOURCE_REV"
fi
echo "MimicGen source: $SOURCE_REV"

[ ! -L "$VENV_DIR" ] || fail "Refusing to install through a venv symlink: $VENV_DIR"
if [ "$CHECK_ONLY" = false ]; then
    if [ ! -e "$VENV_DIR" ]; then "$UV" venv "$VENV_DIR" --python 3.12; fi
fi
[ -x "$PYTHON" ] && [ -f "$VENV_DIR/pyvenv.cfg" ] || \
    fail "MimicGen venv missing or invalid. Run bash setup_env.sh --mimicgen-only."
"$PYTHON" -c 'import sys; from pathlib import Path
assert sys.version_info[:2] == (3, 12), "MimicGen requires Python 3.12"
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), "Wrong venv prefix"
assert sys.prefix != sys.base_prefix, "Refusing a system Python installation"' "$VENV_DIR"

if [ "$CHECK_ONLY" = false ]; then
    sbd_ensure_python_headers "$PYTHON"
    "$UV" pip install --python "$PYTHON" --link-mode copy -r "${MODULE_DIR}/requirements.lock.txt"
    # Its runtime imports work without robomimic; policy learning/data generation
    # are separate workflows. Avoid installing a second Torch stack for simulation.
    "$UV" pip install --python "$PYTHON" --link-mode copy --no-deps -e "$SOURCE_DIR"
    "$UV" pip check --python "$PYTHON"
fi

CHECK_ARGS=()
if [ "$RENDER" = true ]; then CHECK_ARGS+=(--render); fi
"$PYTHON" "${MODULE_DIR}/check_environment.py" "${CHECK_ARGS[@]}"
echo "MimicGen Coffee 준비 완료: .venv_mimicgen (episode 데이터·체크포인트는 별도)."
