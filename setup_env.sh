#!/bin/bash
# SBD 환경 재구성 스크립트 (uv 기준, Python 3.12)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${HOME}/.local/bin/uv"

# ── 1. uv 설치 확인 ────────────────────────────────────────────────
if ! command -v uv &>/dev/null && [ ! -f "$UV" ]; then
    echo "[1/4] uv 설치 중..."
    curl -Ls https://astral.sh/uv/install.sh | sh
    UV="${HOME}/.local/bin/uv"
else
    [ -f "$UV" ] || UV="$(which uv)"
    echo "[1/4] uv 확인: $($UV --version)"
fi

# ── 2. venv 생성 ────────────────────────────────────────────────────
echo "[2/4] .venv 생성 중 (python 3.12)..."
$UV venv "$SCRIPT_DIR/.venv" --python 3.12

# ── 3. 패키지 설치 ──────────────────────────────────────────────────
echo "[3/4] requirements.txt 설치 중..."
# hf-egl-probe: EGL headless GPU rendering용 C 확장 (PyPI 버전)
# /tmp/egl_probe 는 동일 패키지의 로컬 빌드본 - hf-egl-probe로 대체됨
$UV pip install --python "$SCRIPT_DIR/.venv/bin/python" \
    -r "$SCRIPT_DIR/requirements.txt"

# ── 4. lerobot editable 설치 ────────────────────────────────────────
echo "[4/4] lerobot editable 설치 중..."
$UV pip install --python "$SCRIPT_DIR/.venv/bin/python" \
    -e "$SCRIPT_DIR/lerobot"

echo ""
echo "완료! 아래 명령어로 환경 활성화:"
echo "  source $SCRIPT_DIR/.venv/bin/activate"
echo ""
echo "또는 PYTHONPATH 없이 직접 실행:"
echo "  $SCRIPT_DIR/.venv/bin/python examples/libero/replay_demo.py ..."
