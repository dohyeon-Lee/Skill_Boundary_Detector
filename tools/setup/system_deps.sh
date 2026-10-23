#!/usr/bin/env bash
# Shared by the main, MolmoAct2 and GR00T installers. Sourcing does not install anything.
_SBD_APT_UPDATED=0

sbd_install_system_packages() {
    local -a packages=("$@")
    local -a privilege=()
    [ "${#packages[@]}" -gt 0 ] || return 0

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: 시스템 패키지 필요: ${packages[*]}" >&2
        echo "자동 설치는 Ubuntu/Debian의 apt-get을 지원합니다. 이 OS에 맞는 패키지를 설치한 뒤 다시 실행하세요." >&2
        return 1
    fi
    if [ "$(id -u)" != 0 ]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "ERROR: 시스템 패키지 설치에 root 또는 sudo 권한이 필요합니다: ${packages[*]}" >&2
            echo "관리자에게 요청할 명령: apt-get update && apt-get install -y ${packages[*]}" >&2
            return 1
        fi
        privilege=(sudo)
        echo "시스템 패키지 설치에 sudo를 사용합니다. 요청 시 현재 계정 비밀번호를 입력하세요."
    fi

    echo "누락된 시스템 패키지 설치: ${packages[*]}"
    if [ "$_SBD_APT_UPDATED" != 1 ]; then
        if ! "${privilege[@]}" apt-get update; then
            echo "ERROR: apt-get update 실패. 위 권한/네트워크 오류를 해결한 뒤 다시 실행하세요." >&2
            return 1
        fi
        _SBD_APT_UPDATED=1
    fi
    if ! "${privilege[@]}" apt-get install -y "${packages[@]}"; then
        echo "ERROR: 시스템 패키지 설치 실패: ${packages[*]}. 위 apt 오류를 확인하세요." >&2
        return 1
    fi
}

sbd_ensure_system_tools() {
    local entry executable package
    local -a missing=() packages=()
    local -a requirements=(
        gcc:build-essential g++:build-essential make:build-essential
        curl:curl git:git rsync:rsync tmux:tmux tar:tar
    )
    for entry in "${requirements[@]}"; do
        executable="${entry%%:*}"
        package="${entry#*:}"
        if ! command -v "$executable" >/dev/null 2>&1; then
            missing+=("$executable")
            if [[ " ${packages[*]} " != *" $package "* ]]; then
                packages+=("$package")
            fi
        fi
    done
    [ "${#packages[@]}" -gt 0 ] || return 0
    echo "누락된 시스템 도구: ${missing[*]}"
    sbd_install_system_packages "${packages[@]}" || return 1
    for executable in "${missing[@]}"; do
        if ! command -v "$executable" >/dev/null 2>&1; then
            echo "ERROR: 패키지 설치 후에도 $executable 실행 파일을 찾지 못했습니다. PATH를 확인하세요." >&2
            return 1
        fi
    done
}

_sbd_python_has_headers() {
    "$1" -c 'import sys, sysconfig
from pathlib import Path
header = Path(sysconfig.get_path("include")) / "Python.h"
config = Path(sysconfig.get_config_h_filename())
sys.exit(0 if header.is_file() and config.is_file() else 1)'
}

sbd_ensure_python_headers() {
    local python="$1" minor base_prefix include_dir
    if _sbd_python_has_headers "$python"; then
        return 0
    fi
    minor="$("$python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" || return 1
    base_prefix="$("$python" -c 'import sys; print(sys.base_prefix)')" || return 1
    include_dir="$("$python" -c 'import sysconfig; print(sysconfig.get_path("include"))')" || return 1
    if [ "$base_prefix" != /usr ] || [[ ! "$minor" =~ ^3\.[0-9]+$ ]]; then
        echo "ERROR: Python 개발 헤더(Python.h/pyconfig.h)가 없습니다: $include_dir" >&2
        echo "Python: $python (base: $base_prefix). 이 Python 설치의 개발 헤더를 복구한 뒤 다시 실행하세요." >&2
        echo "uv 관리 Python이면: uv python install --reinstall $minor" >&2
        return 1
    fi

    echo "Python $minor 개발 헤더 누락: $include_dir"
    sbd_install_system_packages "python${minor}-dev" || return 1
    if ! _sbd_python_has_headers "$python"; then
        echo "ERROR: python${minor}-dev 설치 후에도 개발 헤더가 없습니다: $include_dir" >&2
        return 1
    fi
}
