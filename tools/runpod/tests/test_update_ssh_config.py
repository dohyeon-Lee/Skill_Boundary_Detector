"""Use fake API responses and temporary configs; never change real SSH files."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import shutil
import sys
from types import SimpleNamespace
import urllib.error

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "update_ssh_config.py"
spec = importlib.util.spec_from_file_location("runpod_ssh", SCRIPT)
ssh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssh)


def pod(name="train", identifier="pod-1", status="RUNNING", port=30086):
    return {"id": identifier, "name": name, "status": status,
            "ssh": {"direct": {"host": "81.27.69.178", "port": port, "username": "root"}},
            "env": {"HF_TOKEN": "never-print-this-secret"}}


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-api-key")
    monkeypatch.setattr(ssh, "fetch_pods", lambda key: [pod()])
    config = tmp_path / "ssh" / "config"
    config.parent.mkdir()
    config.write_bytes(
        b"# lab servers\r\nHost yonsei3_local\r\n  HostName 165.132.142.207\r\n  User mdorazi\r\n\r\n"
        b"Host runpod\r\n  HostName 10.0.0.1\r\n  Port 2222\r\n  User old\r\n"
        b"  IdentityFile ~/.ssh/old\r\n  LocalForward 8765 localhost:8765\r\n\r\n"
        b"Host RLLAB_node100\r\n  HostName 165.132.144.103\r\n  User dohyeon\r\n"
    )
    return config


def test_update_preserves_other_hosts_forwards_and_private_backup(local, capsys):
    before = local.read_bytes()
    assert ssh.main(["--ssh-config", str(local)]) == 0
    after = local.read_bytes()
    lab = before.split(b"Host runpod")[0].split(b"Host yonsei3_local", 1)[1]
    assert b"Host yonsei3_local" + lab in after
    assert after.split(b"Host RLLAB_node100")[1] == before.split(b"Host RLLAB_node100")[1]
    assert b"  HostName 81.27.69.178\r\n  Port 30086\r\n  User root\r\n" in after
    assert b"  LocalForward 8765 localhost:8765\r\n" in after
    backups = list(local.parent.glob("config.before-runpod-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    if ssh.os.name != "nt":
        assert stat.S_IMODE(local.stat().st_mode) == 0o600
        assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert "fake-api-key" not in capsys.readouterr().out
    assert ssh.main(["--ssh-config", str(local)]) == 0
    assert len(list(local.parent.glob("config.before-runpod-*"))) == 1


def test_dry_run_and_unready_pod_never_change_files(local, monkeypatch):
    before = local.read_bytes()
    assert ssh.main(["--ssh-config", str(local), "--dry-run"]) == 0
    monkeypatch.setattr(ssh, "fetch_pods", lambda key: [pod(port=None)])
    assert ssh.main(["--ssh-config", str(local)]) == 1
    assert local.read_bytes() == before
    assert not list(local.parent.glob("config.before-runpod-*"))


def test_ambiguous_pods_require_explicit_selection(local, monkeypatch, capsys):
    before = local.read_bytes()
    monkeypatch.setattr(ssh.os, "isatty", lambda fd: False)
    monkeypatch.setattr(ssh, "fetch_pods", lambda key: [pod(), pod(identifier="pod-2")])
    assert ssh.main(["--ssh-config", str(local)]) == 1
    assert local.read_bytes() == before
    assert ssh.main(["--ssh-config", str(local), "--pod", "pod-2"]) == 0
    assert ssh.main(["--list"]) == 0
    output = capsys.readouterr()
    assert "never-print-this-secret" not in output.out + output.err


@pytest.mark.parametrize("original", [
    "Host runpod another-host\n  User me\n",
    "Host runpod\n  User me\nHost RUNPOD\n  User other\n",
])
def test_shared_or_duplicate_alias_is_rejected(original):
    with pytest.raises(ValueError):
        ssh.render_config(original, "runpod", "81.27.69.178", 30086, "~/.ssh/id_ed25519")


@pytest.mark.parametrize("value", [None, 0, 65536, "22\nProxyCommand evil", True])
def test_invalid_api_port_is_rejected(value):
    with pytest.raises(ValueError):
        ssh.endpoint(pod(port=value))


def test_new_config_precedes_wildcard_and_is_valid_for_openssh(tmp_path):
    if not shutil.which("ssh"):
        pytest.skip("OpenSSH not installed")
    path = tmp_path / "config"
    original = "# defaults\nHost *\n  User nobody\n  Port 99\n"
    path.write_text(ssh.render_config(original, "runpod", "81.27.69.178", 30086, "~/.ssh/id_ed25519"))
    result = subprocess.run(["ssh", "-G", "-F", str(path), "runpod"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "hostname 81.27.69.178\n" in result.stdout
    assert "port 30086\n" in result.stdout
    assert "user root\n" in result.stdout


def test_existing_alias_moves_ahead_of_wildcard_without_changing_other_hosts(tmp_path):
    if not shutil.which("ssh"):
        pytest.skip("OpenSSH not installed")
    other_hosts = "Host *\n  Port 99\n  User nobody\n\nHost lab\n  HostName lab.example\n\n"
    original = other_hosts + "Host runpod\n  HostName old\n  Port 1234\n  User root"
    rendered = ssh.render_config(original, "runpod", "81.27.69.178", 30086, "~/.ssh/id_ed25519")
    assert rendered.endswith(other_hosts)
    path = tmp_path / "config"
    path.write_text(rendered)
    result = subprocess.run(["ssh", "-G", "-F", str(path), "runpod"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "port 30086\n" in result.stdout
    assert "user root\n" in result.stdout


def test_host_equals_syntax_and_match_scope_preserved():
    original = 'Host=runpod # GPU\n  HostName=old\nMatch host lab\n  User lab-user\n'
    rendered = ssh.render_config(original, "runpod", "81.27.69.178", 30086, "~/.ssh/id_ed25519")
    assert rendered.startswith('Host=runpod # GPU\n  HostName 81.27.69.178\n')
    assert rendered.endswith("Match host lab\n  User lab-user\n")


def test_api_uses_authorization_header_and_does_not_leak_error_body(monkeypatch):
    def reject(request, timeout):
        assert request.full_url == ssh.API_URL
        assert request.get_header("Authorization") == "Bearer fake-secret"
        assert timeout == 30
        raise urllib.error.HTTPError(ssh.API_URL, 401, "fake-secret", {}, None)

    monkeypatch.setattr(ssh, "api_opener", lambda *a: SimpleNamespace(open=reject))
    with pytest.raises(ValueError, match="HTTP 401") as error:
        ssh.fetch_pods("fake-secret")
    assert "fake-secret" not in str(error.value)


def test_api_decodes_documented_response(monkeypatch):
    import io

    page = {"pods": [pod()], "pagination": {"hasNextPage": False, "nextCursor": None}}
    monkeypatch.setattr(ssh, "api_opener", lambda *a: SimpleNamespace(
        open=lambda *a, **k: io.BytesIO(json.dumps(page).encode())))
    assert ssh.endpoint(ssh.fetch_pods("fake-key")[0]) == ("81.27.69.178", 30086)


def test_api_follows_v2_pagination_and_encodes_cursor(monkeypatch):
    import io

    pages = iter([
        {"pods": [pod()], "pagination": {"hasNextPage": True, "nextCursor": "a+/="}},
        {"pods": [pod(identifier="pod-2")], "pagination": {"hasNextPage": False, "nextCursor": None}},
    ])
    urls = []

    def open_page(request, timeout):
        urls.append(request.full_url)
        return io.BytesIO(json.dumps(next(pages)).encode())

    monkeypatch.setattr(ssh, "api_opener", lambda *a: SimpleNamespace(open=open_page))
    assert len(ssh.fetch_pods("fake-key")) == 2
    assert urls == [ssh.API_URL, ssh.API_URL + "?cursor=a%2B%2F%3D"]


@pytest.mark.parametrize("reason,expected", [
    (ssh.ssl.SSLCertVerificationError("private diagnostic"), "TLS"),
    (ssh.socket.gaierror("private diagnostic"), "DNS"),
    (TimeoutError("private diagnostic"), "시간 초과"),
    (OSError("private diagnostic"), "네트워크"),
])
def test_transport_errors_are_distinguished_without_leaking_details(monkeypatch, reason, expected):
    def fail(*args, **kwargs):
        raise urllib.error.URLError(reason)

    monkeypatch.setattr(ssh, "api_opener", lambda *a: SimpleNamespace(open=fail))
    with pytest.raises(ValueError, match=expected) as error:
        ssh.fetch_pods("private-api-key")
    assert "private" not in str(error.value)


@pytest.mark.parametrize("use_proxy", [False, True])
def test_opener_bypasses_inherited_proxy_by_default_and_keeps_tls_verification(monkeypatch, use_proxy):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:17891")
    monkeypatch.setattr(ssh.urllib.request, "build_opener", lambda *handlers: handlers)
    handlers = ssh.api_opener(use_proxy=use_proxy)
    proxies = [handler for handler in handlers if isinstance(handler, ssh.urllib.request.ProxyHandler)]
    assert len(proxies) == (0 if use_proxy else 1)
    if proxies:
        assert proxies[0].proxies == {}
    https = next(handler for handler in handlers if isinstance(handler, ssh.urllib.request.HTTPSHandler))
    assert https._context.check_hostname is True
    assert https._context.verify_mode == ssh.ssl.CERT_REQUIRED
    assert https._context.cert_store_stats()["x509_ca"] > 0


def test_api_empty_pods_are_reported_without_modifying_config(local, monkeypatch, capsys):
    monkeypatch.setattr(ssh, "fetch_pods", lambda key: [])
    before = local.read_bytes()
    assert ssh.main(["--list", "--ssh-config", str(local)]) == 0
    assert "API 연결 성공. 등록된 Pod가 없습니다." in capsys.readouterr().out
    assert ssh.main(["--ssh-config", str(local)]) == 1
    assert "실행 중인 Pod가 없습니다" in capsys.readouterr().err
    assert local.read_bytes() == before


def test_proxy_only_connection_is_not_used_for_direct_ssh():
    value = pod()
    value["ssh"] = {"proxy": {"host": "ssh.runpod.io", "port": 22}, "direct": None}
    with pytest.raises(ValueError, match="SSH 주소가 준비되지"):
        ssh.endpoint(value)


def test_api_key_environment_precedence_and_private_write(monkeypatch, tmp_path):
    path = tmp_path / "credentials" / "key"
    ssh.atomic_write(path, b"saved-key\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "environment-key")
    assert ssh.load_key(path) == "environment-key"
    monkeypatch.delenv("RUNPOD_API_KEY")
    assert ssh.load_key(path) == "saved-key"
    if ssh.os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_key_registration_uses_hidden_prompt_and_never_calls_api(monkeypatch, tmp_path, capsys):
    path = tmp_path / "credentials" / "key"
    monkeypatch.setattr(ssh.os, "isatty", lambda fd: True)
    monkeypatch.setattr(ssh.getpass, "getpass", lambda prompt: "registered-secret")
    monkeypatch.setattr(ssh, "fetch_pods", lambda key: pytest.fail("registration must not contact API"))
    assert ssh.main(["--save-api-key", "--api-key-file", str(path)]) == 0
    assert path.read_text() == "registered-secret\n"
    output = capsys.readouterr()
    assert "registered-secret" not in output.out + output.err


def test_first_run_registers_key_updates_config_and_next_run_reuses_it(local, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY")
    monkeypatch.setattr(ssh.os, "isatty", lambda fd: True)
    entered = []

    def enter_key(prompt):
        entered.append(prompt)
        return "first-run-secret"

    def fetch(key):
        assert key == "first-run-secret"
        return [pod()]

    monkeypatch.setattr(ssh.getpass, "getpass", enter_key)
    monkeypatch.setattr(ssh, "fetch_pods", fetch)
    key_path = tmp_path / "credentials" / "key"
    args = ["--ssh-config", str(local), "--api-key-file", str(key_path)]
    assert ssh.main(args) == 0
    assert key_path.read_text() == "first-run-secret\n"
    assert "HostName 81.27.69.178" in local.read_text()
    assert ssh.main(args) == 0
    assert len(entered) == 1
    output = capsys.readouterr()
    assert "first-run-secret" not in output.out + output.err


@pytest.mark.parametrize("dry_run,interactive", [(True, True), (False, False)])
def test_missing_key_does_not_prompt_or_write_in_dry_run_or_noninteractive_mode(
    local, monkeypatch, tmp_path, dry_run, interactive,
):
    monkeypatch.delenv("RUNPOD_API_KEY")
    monkeypatch.setattr(ssh.os, "isatty", lambda fd: interactive)
    monkeypatch.setattr(ssh.getpass, "getpass", lambda prompt: pytest.fail("must not prompt"))
    key_path = tmp_path / "credentials" / "key"
    before = local.read_bytes()
    args = ["--ssh-config", str(local), "--api-key-file", str(key_path)]
    if dry_run:
        args.append("--dry-run")
    assert ssh.main(args) == 1
    assert not key_path.exists()
    assert local.read_bytes() == before


def test_interactive_pod_choice_updates_selected_endpoint(local, monkeypatch):
    monkeypatch.setattr(ssh.os, "isatty", lambda fd: True)
    monkeypatch.setattr(ssh, "fetch_pods", lambda key: [pod(port=1111), pod(identifier="pod-2", port=2222)])
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    assert ssh.main(["--ssh-config", str(local)]) == 0
    assert "Port 2222" in local.read_text()


def test_cancelled_pod_choice_does_not_modify_config(local, monkeypatch):
    monkeypatch.setattr(ssh.os, "isatty", lambda fd: True)
    monkeypatch.setattr(ssh, "fetch_pods", lambda key: [pod(), pod(identifier="pod-2")])
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    before = local.read_bytes()
    assert ssh.main(["--ssh-config", str(local)]) == 1
    assert local.read_bytes() == before


def test_launcher_finds_compatible_python_and_preserves_paths_and_arguments(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    launcher = root / "runpod_ssh.sh"
    launcher.write_bytes((SCRIPT.parents[2] / "runpod_ssh.sh").read_bytes())
    binaries = tmp_path / "bin"
    binaries.mkdir()
    old_python = binaries / "python3"
    old_python.write_text("#!/bin/bash\nexit 1\n")
    old_python.chmod(0o755)
    compatible_python = binaries / "python"
    compatible_python.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "if sys.argv[1:2] == ['-c']: sys.exit(0)\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    compatible_python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher), "--pod", "pod with spaces"], cwd=tmp_path,
        env=dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"]),
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(root / "tools/runpod/update_ssh_config.py"), "--pod", "pod with spaces"]
