#!/usr/bin/env bash
# Standalone official ABC Simulator setup with all evaluation assets.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${MODULE_DIR}/../../../../../.." && pwd)"
SOURCE_DIR="${PROJECT_ROOT}/third_party/abc"
SOURCE_REV="d0832d12651d1b260a652861a14648dc5f3660c7"
VENV_DIR="${PROJECT_ROOT}/.venv_abc_sim"
PYTHON="${VENV_DIR}/bin/python"
LOCK="${MODULE_DIR}/requirements.lock.txt"
CHECKER="${MODULE_DIR}/check_environment.py"
SOURCE_INSTALL_MARKER="${VENV_DIR}/.abc_source_revision"
CHECK_ONLY=false
CLONED_SOURCE=false

case "${1:-}" in
    "") ;;
    --check) CHECK_ONLY=true ;;
    -h|--help)
        echo "Usage: bash setup_env.sh [--check]"
        echo "Prepare pinned amazon-far/abc + .venv_abc_sim + all sim/RoboCasa assets."
        echo "Episode data and policy checkpoints remain separate."
        echo "--check: read-only source, dependency, asset, and simulator import checks."
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
    command -v ffmpeg >/dev/null 2>&1 || sbd_install_system_packages ffmpeg
fi
command -v git >/dev/null || fail "git is required."
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || \
    fail "The pinned ABC CUDA 12.8 environment targets Linux x86_64 / Python 3.12."

# Never inherit the main, GR00T, or Molmo LeRobot environment.
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
        echo "ABC Sim: cloning official source (no recursive submodules)..."
        mkdir -p "${PROJECT_ROOT}/third_party"
        git -c submodule.recurse=false clone --no-recurse-submodules --no-checkout \
            https://github.com/amazon-far/abc.git "$SOURCE_DIR"
        CLONED_SOURCE=true
    fi
fi

[ -d "$SOURCE_DIR" ] || fail "ABC Sim source missing. Run bash setup_env.sh --abc-sim-only."
SOURCE_TOP="$(git -C "$SOURCE_DIR" rev-parse --show-toplevel 2>/dev/null)" || \
    fail "$SOURCE_DIR is not a Git checkout; nothing was overwritten."
[ "$(cd "$SOURCE_TOP" && pwd -P)" = "$(cd "$SOURCE_DIR" && pwd -P)" ] || \
    fail "$SOURCE_DIR must be its own Git checkout; nothing was overwritten."
CURRENT_REV="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [ "$CURRENT_REV" != "$SOURCE_REV" ] || [ "$CLONED_SOURCE" = true ]; then
    [ "$CHECK_ONLY" = false ] || fail "ABC Sim revision $CURRENT_REV differs from $SOURCE_REV."
    if [ "$CLONED_SOURCE" = false ]; then
        [ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ] || \
            fail "ABC Sim has local changes at $CURRENT_REV; save them before changing revision."
    fi
    if ! git -C "$SOURCE_DIR" cat-file -e "${SOURCE_REV}^{commit}" 2>/dev/null; then
        git -C "$SOURCE_DIR" -c submodule.recurse=false fetch origin "$SOURCE_REV"
    fi
    git -C "$SOURCE_DIR" -c submodule.recurse=false checkout --detach "$SOURCE_REV"
fi
for required in pyproject.toml abc_sim/env.py abc_minimal/config.py prepare.py; do
    [ -f "$SOURCE_DIR/$required" ] || fail "Missing official ABC source: $SOURCE_DIR/$required"
done
echo "ABC Sim source: $SOURCE_REV"
if [ -n "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: preserving local ABC Sim source modifications."
fi

if [ "$CHECK_ONLY" = false ]; then
    if [ ! -e "$VENV_DIR" ]; then
        "$UV" venv "$VENV_DIR" --python 3.12
    fi
    [ -x "$PYTHON" ] && [ -f "$VENV_DIR/pyvenv.cfg" ] || \
        fail "$VENV_DIR exists but is not a usable venv; nothing was removed."
    "$PYTHON" -c 'import sys; from pathlib import Path
assert sys.version_info[:2] == (3, 12), "ABC Sim requires Python 3.12"
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), "Refusing to install outside the ABC Sim venv"' "$VENV_DIR"
    sbd_ensure_python_headers "$PYTHON"

    export PATH="${VENV_DIR}/bin:${PATH}"
    # The official project deliberately sources CUDA Torch and CPU TorchCodec
    # from two different PyTorch indexes. The exact pins keep all selections
    # deterministic even though both indexes are visible to this install.
    MUJOCO_REPAIR_ARGS=()
    if "$PYTHON" -c 'import importlib.metadata as m; m.version("mujoco-core")' 2>/dev/null; then
        echo "Removing incompatible mujoco-core and repairing the pinned MuJoCo 3.8 package..."
        "$UV" pip uninstall --python "$PYTHON" mujoco-core
        MUJOCO_REPAIR_ARGS=(--reinstall-package mujoco)
    fi
    "$UV" pip install --python "$PYTHON" --link-mode copy "${MUJOCO_REPAIR_ARGS[@]}" \
        --index https://download.pytorch.org/whl/cu128 \
        --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match -r "$LOCK"
    if [ -f "$SOURCE_INSTALL_MARKER" ] \
        && [ "$(<"$SOURCE_INSTALL_MARKER")" = "$SOURCE_REV" ] \
        && "$PYTHON" -c 'import importlib.util, pathlib, sys
spec = importlib.util.find_spec("abc_sim")
raise SystemExit(0 if spec and spec.origin and pathlib.Path(spec.origin).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()) else 1)' "$SOURCE_DIR"; then
        echo "ABC Sim editable source already installed; keeping it."
    else
        "$UV" pip install --python "$PYTHON" --link-mode copy --no-deps -e "$SOURCE_DIR"
        printf '%s\n' "$SOURCE_REV" > "$SOURCE_INSTALL_MARKER"
    fi
    "$UV" pip check --python "$PYTHON"
    echo "ABC Sim: installing all standard and RoboCasa evaluation assets..."
    "$PYTHON" "$SOURCE_DIR/prepare.py" --sim --sim-robocasa
fi

[ -x "$PYTHON" ] || fail "ABC Sim venv missing. Run bash setup_env.sh --abc-sim-only."
"$PYTHON" "$CHECKER"
echo "ABC Simulator 준비 완료. 전체 sim/RoboCasa assets 포함; episode data와 policy checkpoints는 별도."
