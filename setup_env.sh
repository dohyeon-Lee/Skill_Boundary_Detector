#!/bin/bash
# SBD 환경 재구성 스크립트 (uv 기준, Python 3.12)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${HOME}/.local/bin/uv"

# ── 1. uv 설치 확인 ────────────────────────────────────────────────
if ! command -v uv &>/dev/null && [ ! -f "$UV" ]; then
    echo "[1/5] uv 설치 중..."
    curl -Ls https://astral.sh/uv/install.sh | sh
    UV="${HOME}/.local/bin/uv"
else
    [ -f "$UV" ] || UV="$(which uv)"
    echo "[1/5] uv 확인: $($UV --version)"
fi

# ── 2. venv 생성 ────────────────────────────────────────────────────
echo "[2/5] .venv 생성 중 (python 3.12)..."
$UV venv "$SCRIPT_DIR/.venv" --python 3.12

PYTHON="$SCRIPT_DIR/.venv/bin/python"

# ── 3. hf-egl-probe 먼저 설치 (CMakeLists.txt + setup.py 패치) ───────
# robomimic이 egl-probe를 의존성으로 당겨오기 전에 미리 설치해야 함
echo "[3/5] hf-egl-probe 패치 후 설치 중..."
TMP_EGL=$(mktemp -d)
$UV pip download --python "$PYTHON" "hf-egl-probe==1.0.2" --no-deps -d "$TMP_EGL"
EGL_SDIST=$(ls "$TMP_EGL"/*.tar.gz 2>/dev/null | head -1)
tar xzf "$EGL_SDIST" -C "$TMP_EGL"
EGL_SRC=$(find "$TMP_EGL" -maxdepth 1 -mindepth 1 -type d | head -1)
# CMakeLists.txt: cmake_minimum_required 버전 3.5로 올림
sed -i 's/cmake_minimum_required(VERSION [0-9.]*)/cmake_minimum_required(VERSION 3.5)/' "$EGL_SRC/CMakeLists.txt"
# setup.py: cmake 명령어에 정책 플래그 추가
sed -i 's/cmake \.\./cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ../' "$EGL_SRC/setup.py"
$UV pip install --python "$PYTHON" --no-build-isolation "$EGL_SRC"
rm -rf "$TMP_EGL"

# ── 4. 나머지 패키지 설치 ────────────────────────────────────────────
echo "[4/5] requirements.txt 설치 중..."
$UV pip install --python "$PYTHON" -r "$SCRIPT_DIR/requirements.txt"

# ── 5. lerobot editable 설치 ────────────────────────────────────────
echo "[5/5] lerobot editable 설치 중..."
$UV pip install --python "$PYTHON" -e "$SCRIPT_DIR/lerobot"

echo ""
echo "완료! 아래 명령어로 환경 활성화:"
echo "  source $SCRIPT_DIR/.venv/bin/activate"
echo ""
echo "또는 직접 실행:"
echo "  $PYTHON examples/libero/replay_demo.py ..."
