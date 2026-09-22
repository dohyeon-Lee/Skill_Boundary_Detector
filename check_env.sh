#!/bin/bash
# SBD 환경 검증: 설치된 venv가 requirements.txt(전부 == 고정)와 정확히 같은지 확인한다.
# setup_env.sh 마지막 단계에서 자동 실행되며, 언제든 단독 실행 가능:
#   bash check_env.sh                       # ./.venv 검사
#   SBD_VENV_PATH=/path/to/venv bash check_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SBD_VENV_PATH:-${SCRIPT_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"
EXPECTED_PYTHON="${SBD_PYTHON_VERSION:-3.12}"   # 3.12 = 3.12.x 전부, 3.12.13 = 정확히 그 버전

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: ${PYTHON} 가 없습니다. 먼저 bash setup_env.sh 를 실행하세요." >&2
    exit 1
fi

# 패키지를 import하지 않고 메타데이터만 읽는다 (torch import 없이 수 초 안에 끝남).
"${PYTHON}" - "${SCRIPT_DIR}" "${EXPECTED_PYTHON}" <<'PY'
import importlib.metadata as metadata
import importlib.util
import platform
import re
import sys
from pathlib import Path

root, expected_python = Path(sys.argv[1]), sys.argv[2]


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def same_version(actual: str, expected: str) -> bool:
    """PEP 440 equality (e.g. 0.1.1-2209072238 == 0.1.1.post2209072238)."""
    try:
        from packaging.version import Version

        return Version(actual) == Version(expected)
    except Exception:  # noqa: BLE001 - packaging missing or a non-PEP 440 version
        return actual == expected


installed = {canonical(dist.metadata["Name"]): dist.version for dist in metadata.distributions()}
problems = []

actual_python = platform.python_version()
if actual_python != expected_python and not actual_python.startswith(expected_python + "."):
    problems.append(f"python {platform.python_version()} (기대: {expected_python})")

pins = {}
for line in (root / "requirements.txt").read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    match = re.fullmatch(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;]+)", line)
    if match is None:
        problems.append(f"requirements.txt 에 버전 고정(==)이 아닌 줄: {line!r}")
        continue
    pins[canonical(match.group(1))] = match.group(2)
for name, version in sorted(pins.items()):
    actual = installed.get(name)
    if actual is None:
        problems.append(f"미설치: {name}=={version}")
    elif not same_version(actual, version):
        problems.append(f"버전 불일치: {name} {actual} (기대: {version})")

# requirements.txt 밖에서 setup_env.sh 가 따로 설치하는 것들.
for name, version in (("egl-probe", "1.0.2"), ("robomimic", "0.2.0")):
    if installed.get(name) is None or not same_version(installed[name], version):
        problems.append(f"{name}=={version} 필요 (현재: {installed.get(name)})")
spec = importlib.util.find_spec("lerobot")
expected_src = (root / "lerobot" / "src").resolve()
if spec is None or spec.origin is None:
    problems.append("lerobot 미설치 (uv pip install -e lerobot)")
elif expected_src not in Path(spec.origin).resolve().parents:
    problems.append(f"lerobot 이 이 저장소가 아닌 곳을 가리킴: {spec.origin}")

extra = sorted(set(installed) - set(pins) - {"egl-probe", "robomimic", "lerobot"})
if problems:
    print("환경 검증 실패:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print(f"환경 검증 통과: python {platform.python_version()}, 고정 패키지 {len(pins)}개 일치, lerobot -> {spec.origin}")
if extra:
    print(f"  (참고) requirements.txt 밖 추가 패키지 {len(extra)}개: {', '.join(extra)}")
PY
