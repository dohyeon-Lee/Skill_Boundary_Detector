"""No real credentials, network requests, or environment installations."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("login_services", ROOT / "tools/setup/login_services.py")
auth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth)


@pytest.fixture
def sdk(monkeypatch):
    calls = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def hf(**kwargs):
        calls.append(("hf", kwargs))
        print(kwargs["token"])

    def wandb(**kwargs):
        calls.append(("wandb", kwargs))
        print(kwargs["key"], file=sys.stderr)
        return True

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(login=hf))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=wandb))
    return calls


def test_no_keys_skip_without_prompt_or_sdk_calls(sdk, capsys):
    assert auth.login_services() == 0
    assert sdk == []
    assert "자동 로그인 생략" in capsys.readouterr().out


def test_supplied_keys_are_verified_without_output(sdk, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf-test-secret")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-test-secret")
    assert auth.login_services() == 0
    assert [name for name, _ in sdk] == ["hf", "wandb"]
    assert sdk[0][1]["skip_if_logged_in"] is False
    assert sdk[1][1]["verify"] is True
    output = capsys.readouterr()
    assert "secret" not in output.out + output.err


def test_failure_does_not_print_secret_and_other_service_still_runs(sdk, monkeypatch, capsys):
    def reject(**kwargs):
        raise ValueError("invalid hf-test-secret")

    monkeypatch.setenv("HF_TOKEN", "hf-test-secret")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-test-secret")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(login=reject))
    assert auth.login_services() == 1
    assert sdk[0][0] == "wandb"
    output = capsys.readouterr()
    assert "실패" in output.out
    assert "secret" not in output.out + output.err


def test_unverified_wandb_is_failure(sdk, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "wandb-test-secret")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=lambda **kwargs: False))
    assert auth.login_services() == 1


def test_shell_login_only_requires_explicit_runpod(tmp_path):
    # Copy the launcher only, so no actual venv or credentials can be touched.
    script = tmp_path / "setup_env.sh"
    script.write_bytes((ROOT / "setup_env.sh").read_bytes())
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$SBD_SERVER" "$1"\n')
    python.chmod(0o755)
    env = dict(os.environ)
    env.pop("SBD_VENV_PATH", None)
    result = subprocess.run(["bash", str(script), "--runpod", "--login-only"], env=env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["runpod", str(tmp_path / "tools/setup/login_services.py")]
    result = subprocess.run(["bash", str(script), "--login-only"], env=env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
