#!/bin/bash
# SBD 환경 재구성 스크립트 (uv 기준, Python 3.12)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${HOME}/.local/bin/uv"
VENV_DIR="${SBD_VENV_PATH:-${SCRIPT_DIR}/.venv}"

# Keep destructive cleanup scoped to a named environment inside this project.
case "${VENV_DIR}" in
    "${SCRIPT_DIR}/.venv"|"${SCRIPT_DIR}/.venv."*) ;;
    *)
        echo "ERROR: SBD_VENV_PATH must be ${SCRIPT_DIR}/.venv or ${SCRIPT_DIR}/.venv.*"
        exit 1
        ;;
esac

# ── 1. uv 확인 ──────────────────────────────────────────────────────
if ! command -v uv &>/dev/null && [ ! -f "$UV" ]; then
    echo "[1/5] uv 설치 중..."
    curl -Ls https://astral.sh/uv/install.sh | sh
    UV="${HOME}/.local/bin/uv"
else
    [ -f "$UV" ] || UV="$(which uv)"
    echo "[1/5] uv 확인: $($UV --version)"
fi

# ── 2. venv 생성 ────────────────────────────────────────────────────
echo "[2/5] 환경 생성 중: ${VENV_DIR} (python 3.12)..."
[ -e "${VENV_DIR}" ] && rm -rf -- "${VENV_DIR}"
$UV venv "${VENV_DIR}" --python 3.12
PYTHON="${VENV_DIR}/bin/python"

# Use an environment-local cmake so login nodes do not need a system package.
$UV pip install --python "$PYTHON" setuptools wheel cmake==4.1.3
export PATH="${VENV_DIR}/bin:${PATH}"
echo "      cmake 확인: $(cmake --version | head -1)"

# ── 3. hf-egl-probe: 패치 → wheel 빌드 → 로컬 wheel로 설치 ──────────
# robomimic이 requirements.txt 설치 시 egl-probe를 재다운로드하지 않도록
# 미리 패치된 wheel을 빌드해두고 --find-links로 그걸 쓰게 함
echo "[3/5] hf-egl-probe wheel 빌드 중 (cmake 패치)..."

TMP_EGL=$(mktemp -d)
WHEELS_DIR=$(mktemp -d)
# 스크립트 종료 시 임시 디렉토리 정리
cleanup() { rm -rf "$TMP_EGL" "$WHEELS_DIR"; }
trap cleanup EXIT

# PyPI에서 sdist 다운로드
# robomimic은 'egl-probe'에 의존하므로 egl-probe sdist를 받아야 함
# (hf-egl-probe는 다른 패키지명이라 uv가 별개로 취급)
$PYTHON - "$TMP_EGL/egl.tar.gz" <<'PYEOF'
import urllib.request, json, sys
for pkg in ['egl-probe', 'hf-egl-probe']:
    try:
        resp = urllib.request.urlopen(f'https://pypi.org/pypi/{pkg}/1.0.2/json')
        data = json.loads(resp.read())
        sdist = next((r for r in data['urls'] if r['packagetype'] == 'sdist'), None)
        if sdist:
            urllib.request.urlretrieve(sdist['url'], sys.argv[1])
            print(f"      downloaded ({pkg}): {sdist['url']}")
            break
    except Exception:
        continue
else:
    raise RuntimeError("egl-probe sdist를 찾을 수 없습니다")
PYEOF

tar xzf "$TMP_EGL/egl.tar.gz" -C "$TMP_EGL"

EGL_SETUP=$(find "$TMP_EGL" -name "setup.py" | head -1)
EGL_SRC="$(dirname "$EGL_SETUP")"
CMAKE_FILE=$(find "$TMP_EGL" -name "CMakeLists.txt" | head -1)

# CMakeLists.txt 패치
sed -i 's/cmake_minimum_required(VERSION [0-9.]*)/cmake_minimum_required(VERSION 3.5)/' "$CMAKE_FILE"
# setup.py 패치: cmake 명령에 정책 플래그 추가
sed -i 's/cmake \.\./cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ../' "$EGL_SRC/setup.py"

# 빌드에 필요한 패키지를 설치한 뒤 wheel 빌드
cd "$EGL_SRC"
$PYTHON setup.py bdist_wheel --dist-dir "$WHEELS_DIR" 2>/dev/null
cd "$SCRIPT_DIR"

# 빌드된 wheel 설치
WHEEL_FILE=$(ls "$WHEELS_DIR"/*.whl | head -1)
echo "      built wheel: $(basename "$WHEEL_FILE")"
$UV pip install --python "$PYTHON" "$WHEEL_FILE"

# ── 4. 나머지 패키지 설치 ────────────────────────────────────────────
echo "[4/5] requirements.txt + robomimic 설치 중..."
$UV pip install --python "$PYTHON" \
    --find-links "$WHEELS_DIR" \
    -r "$SCRIPT_DIR/requirements.txt"
# robomimic은 --no-deps로 설치 (egl-probe 재빌드 방지)
$UV pip install --python "$PYTHON" --no-deps robomimic==0.2.0

trap - EXIT
cleanup

# ── 5. lerobot editable 설치 ────────────────────────────────────────
echo "[5/5] lerobot editable 설치 중..."
$UV pip install --python "$PYTHON" -e "$SCRIPT_DIR/lerobot"

echo ""
echo "완료! 환경 활성화:"
echo "  source ${VENV_DIR}/bin/activate"
