#!/usr/bin/env python3
"""Update a local SSH alias from RunPod's API (Python 3.9+, stdlib only).

Run on the computer whose SSH config is used by VS Code / ssh, not in a Pod.
This tool only lists Pods; it never creates, stops, or modifies a Pod.
"""
from __future__ import annotations

import argparse
import getpass
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import socket
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
import urllib.parse


API_URL = "https://api.runpod.io/v2/pods"
KEY_FILE = Path.home() / ".config" / "skill-boundary-detector" / "runpod_api_key"
CA_BUNDLE_PATHS = (
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/ssl/cert.pem",
)


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)  # mkstemp creates a private (0600) file on Unix.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def register_key(path: Path) -> str:
    if not os.isatty(0):
        raise ValueError("API 키가 없습니다. PC의 대화형 터미널에서 bash runpod_ssh.sh 를 실행하세요.")
    key = getpass.getpass("RunPod API key (입력은 표시되지 않음): ").strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError("비어 있거나 공백이 포함된 API 키는 저장하지 않습니다.")
    atomic_write(path, (key + "\n").encode("utf-8"))
    print(f"API 키 저장 완료: {path}")
    return key


def load_key(path: Path, register_missing: bool = False) -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key and path.is_file():
        key = path.read_text(encoding="utf-8").strip()
    if not key:
        if register_missing:
            print("처음 사용: RunPod API 키를 한 번 등록하면 다음부터 자동으로 사용합니다.")
            return register_key(path)
        raise ValueError("API 키가 없습니다. 먼저 bash runpod_ssh.sh 를 실행해 등록하세요.")
    if any(char.isspace() for char in key):
        raise ValueError("API 키에 공백이 있습니다. --save-api-key 로 다시 등록하세요.")
    return key


def api_opener(use_proxy: bool = False):
    # uv-managed Python may have no default CA file on RHEL/CentOS. Add the
    # OS trust bundles to the default store while keeping TLS verification on.
    context = ssl.create_default_context()
    for bundle in CA_BUNDLE_PATHS:
        if Path(bundle).is_file():
            context.load_verify_locations(cafile=bundle)
    handlers = [urllib.request.HTTPSHandler(context=context)]
    # Other tools in this repo likewise bypass inherited Codex/shell proxies.
    # Opt in for networks that require their configured proxy.
    if not use_proxy:
        handlers.insert(0, urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def fetch_pods(key: str, use_proxy: bool = False) -> list[dict]:
    pods = []
    cursor = None
    seen_cursors = set()
    try:
        opener = api_opener(use_proxy)
        while True:
            url = API_URL if cursor is None else API_URL + "?" + urllib.parse.urlencode({"cursor": cursor})
            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
            with opener.open(request, timeout=30) as response:
                page = json.load(response)
            if not isinstance(page, dict) or not isinstance(page.get("pods"), list):
                raise ValueError("RunPod API의 Pod 목록 형식이 예상과 다릅니다.")
            if not all(isinstance(pod, dict) for pod in page["pods"]):
                raise ValueError("RunPod API의 Pod 목록 형식이 예상과 다릅니다.")
            pods.extend(page["pods"])
            pagination = page.get("pagination")
            if not isinstance(pagination, dict) or not isinstance(pagination.get("hasNextPage"), bool):
                raise ValueError("RunPod API의 페이지 정보가 예상과 다릅니다.")
            if not pagination["hasNextPage"]:
                return pods
            cursor = pagination.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                raise ValueError("RunPod API의 다음 페이지 정보가 유효하지 않습니다.")
            seen_cursors.add(cursor)
    except urllib.error.HTTPError as error:
        # Do not print response bodies (Pod env fields can contain secrets).
        raise ValueError(f"RunPod API HTTP {error.code}: API 키와 Pod 조회 권한을 확인하세요.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = error.reason if isinstance(error, urllib.error.URLError) else error
        if isinstance(reason, ssl.SSLError):
            message = "RunPod API TLS 인증서 검증 실패: 시스템 CA 인증서 또는 SSL_CERT_FILE 설정을 확인하세요."
        elif isinstance(reason, socket.gaierror):
            message = "RunPod API 주소를 찾지 못했습니다. DNS·인터넷 연결을 확인하세요."
        elif isinstance(reason, TimeoutError):
            message = "RunPod API 연결 시간 초과: 네트워크·방화벽을 확인하세요."
        else:
            message = "RunPod API 연결 실패: 네트워크·프록시를 확인하세요."
        raise ValueError(message) from None
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("RunPod API 응답을 해석할 수 없습니다.") from None


def endpoint(pod: dict) -> tuple[str, int]:
    if pod.get("status") != "RUNNING":
        raise ValueError("Pod가 RUNNING 상태가 아닙니다.")
    ssh = pod.get("ssh") or {}
    direct = ssh.get("direct") if isinstance(ssh, dict) else None
    address = direct.get("host") if isinstance(direct, dict) else None
    port = direct.get("port") if isinstance(direct, dict) else None
    try:
        address = str(ipaddress.ip_address(address))
        if isinstance(port, bool) or not str(port).isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("SSH 주소가 준비되지 않았습니다. public IP와 22/tcp 설정을 확인하고 다시 실행하세요.") from None
    return address, int(port)


def select_pod(pods: list[dict], selector: str | None) -> dict:
    running = [pod for pod in pods if pod.get("status") == "RUNNING"]
    if selector:
        matches = [pod for pod in pods if pod.get("id") == selector]
        if not matches:
            matches = [pod for pod in pods if pod.get("name") == selector]
    else:
        matches = running
    if not selector and len(matches) > 1 and os.isatty(0):
        print("실행 중인 Pod가 여러 개입니다. 연결할 Pod를 선택하세요.")
        for index, pod in enumerate(matches, 1):
            print(f"  {index}. {pod.get('name', '')} ({pod.get('id', '')})")
        try:
            choice = input("Pod 번호 (취소: q): ").strip()
        except EOFError:
            raise ValueError("Pod 선택을 입력받지 못했습니다. --pod ID로 지정하세요.") from None
        if choice.lower() == "q":
            raise ValueError("Pod 선택을 취소했습니다. SSH 설정은 변경하지 않았습니다.")
        if not choice.isdecimal() or not 1 <= int(choice) <= len(matches):
            raise ValueError("목록에 있는 Pod 번호를 입력하세요.")
        matches = [matches[int(choice) - 1]]
    if not matches and not selector:
        raise ValueError("실행 중인 Pod가 없습니다. RunPod에서 Pod를 시작한 뒤 다시 실행하세요.")
    if len(matches) != 1:
        raise ValueError("대상 Pod를 하나로 정할 수 없습니다. --list 후 --pod ID 또는 정확한 이름으로 지정하세요.")
    endpoint(matches[0])  # Reject stopped / unready Pods before touching the config.
    return matches[0]


def directive(line: str) -> tuple[str, list[str]]:
    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9]*)(?:\s*=\s*|\s+)(.*)$", line.rstrip("\r\n"))
    if not match:
        return "", []
    try:
        return match[1].lower(), shlex.split(match[2], comments=True, posix=True)
    except ValueError:
        raise ValueError("SSH config에 닫히지 않은 따옴표가 있습니다. 파일을 확인하세요.") from None


def render_config(original: str, alias: str, address: str, port: int, identity: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", alias):
        raise ValueError("SSH 별칭은 영문·숫자·점·밑줄·하이픈만 사용할 수 있습니다.")
    if not identity or any(char in identity for char in '\r\n\0"'):
        raise ValueError("SSH 키 경로에 사용할 수 없는 문자가 있습니다.")
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines(keepends=True)
    parsed = [directive(line) for line in lines]
    boundaries = [i for i, (key, _) in enumerate(parsed) if key in ("host", "match")]
    targets = []
    for index, start in enumerate(boundaries):
        key, names = parsed[start]
        if key == "host" and alias.lower() in [name.lower() for name in names]:
            if len(names) != 1:
                raise ValueError("대상 Host가 여러 별칭과 묶여 있습니다. 독립된 Host 블록으로 분리하세요.")
            end = boundaries[index + 1] if index + 1 < len(boundaries) else len(lines)
            targets.append((start, end))
    if len(targets) > 1:
        raise ValueError("대상 Host 블록이 중복되어 있습니다. 중복을 정리하세요.")
    settings = [
        f"  HostName {address}{newline}",
        f"  Port {port}{newline}",
        f"  User root{newline}",
        f'  IdentityFile "{identity}"{newline}',
    ]
    if targets:
        start, end = targets[0]
        managed = {"hostname", "port", "user", "identityfile"}
        preserved = [line for i, line in enumerate(lines[start + 1:end], start + 1)
                     if parsed[i][0] not in managed]
        header = lines[start].rstrip("\r\n") + newline
        block = "".join([header] + settings + preserved)
        if not block.endswith(("\n", "\r")):
            block += newline
        # OpenSSH uses the first value it encounters. Keep the specific alias
        # ahead of an existing Host * so its default User/Port cannot win.
        position = boundaries[0]
        return "".join(lines[:position]) + block + "".join(lines[position:start] + lines[end:])
    block = f"Host {alias}{newline}" + "".join(settings) + newline
    # Insert before the first Host/Match so a later Host * cannot override it.
    # Keep global settings before the first Host in their original scope.
    position = boundaries[0] if boundaries else len(lines)
    prefix = "".join(lines[:position])
    if prefix and not prefix.endswith(("\n", "\r")):
        prefix += newline
    return prefix + block + "".join(lines[position:])


def update_config(path: Path, alias: str, address: str, port: int, identity: str, dry_run: bool) -> None:
    path = path.expanduser().resolve()  # Respect a config symlink without replacing the link itself.
    original = path.read_bytes() if path.exists() else b""
    rendered = render_config(original.decode("utf-8"), alias, address, port, identity).encode("utf-8")
    if dry_run:
        print(f"미리보기: {path} / Host {alias} → root@{address}:{port} / IdentityFile {identity}")
        return
    if rendered == original:
        print(f"변경 없음: {path} / Host {alias}")
        return
    backup = None
    if path.exists():
        fd, name = tempfile.mkstemp(prefix=f"{path.name}.before-runpod-", dir=path.parent)
        backup = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(original)
    # Fail instead of overwriting an edit made while this command was running.
    if (path.read_bytes() if path.exists() else b"") != original:
        raise ValueError("SSH config가 실행 중 변경되었습니다. 다시 실행하세요.")
    atomic_write(path, rendered)
    print(f"갱신 완료: {path} / Host {alias} → root@{address}:{port}")
    if backup:
        print(f"원본 백업: {backup}")
    print(f"이제 ssh {alias} 또는 VS Code Remote-SSH로 연결하세요.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-api-key", action="store_true", help="API 키를 숨김 입력으로 이 PC에 저장하고 종료")
    parser.add_argument("--api-key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--list", action="store_true", help="Pod 목록만 출력")
    parser.add_argument("--pod", help="대상 Pod ID 또는 정확한 이름; 생략 시 자동 선택 또는 번호 입력")
    parser.add_argument("--host", default="runpod", help="갱신할 SSH 별칭 (기본: runpod)")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh" / "config")
    parser.add_argument("--identity", default="~/.ssh/id_ed25519", help="이 PC의 SSH 개인키 경로")
    parser.add_argument("--dry-run", action="store_true", help="SSH config를 쓰지 않고 대상만 확인")
    parser.add_argument("--use-proxy", action="store_true", help="환경변수에 설정된 프록시 사용 (기본: 직접 연결)")
    args = parser.parse_args(argv)
    try:
        key_path = args.api_key_file.expanduser()
        if args.save_api_key:
            if args.list or args.pod or args.dry_run:
                raise ValueError("--save-api-key는 단독으로 사용하세요.")
            register_key(key_path)
            return 0
        # Dry runs stay read-only, including the credential file.
        key = load_key(key_path, register_missing=not args.dry_run)
        pods = fetch_pods(key, use_proxy=True) if args.use_proxy else fetch_pods(key)
        if args.list:
            if not pods:
                print("API 연결 성공. 등록된 Pod가 없습니다.")
            for pod in pods:
                # Show selected metadata only, never env / the raw API response.
                print(f"{pod.get('id', '')}\t{pod.get('status', '')}\t{pod.get('name', '')}")
            return 0
        pod = select_pod(pods, args.pod)
        address, port = endpoint(pod)
        update_config(args.ssh_config, args.host, address, port, args.identity, args.dry_run)
        return 0
    except KeyboardInterrupt:
        print("\n취소했습니다.", file=sys.stderr)
        return 130
    except (ValueError, OSError, EOFError) as error:
        # ValueError messages here are curated; OS errors can contain arbitrary paths.
        if isinstance(error, EOFError):
            message = "입력을 받지 못했습니다. PC의 대화형 터미널에서 다시 실행하세요."
        else:
            message = str(error) if isinstance(error, ValueError) else "파일을 읽거나 저장하지 못했습니다. 경로·권한을 확인하세요."
        print(f"ERROR: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
