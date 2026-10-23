"""Run the actual shell checks with fake apt/sudo; never modify the host OS."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tools/setup/system_deps.sh"
TOOLS = ("gcc", "g++", "make", "curl", "git", "rsync", "tmux", "tar")


@pytest.fixture
def machine(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"
    env = dict(os.environ, PATH=str(bin_dir), COMMAND_LOG=str(log),
               MACHINE_BIN=str(bin_dir), FAKE_UID="1000", FAKE_BASE="/usr",
               HEADER_MARKER=str(tmp_path / "headers"))

    def executable(name, body="exit 0\n"):
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
        return path

    for name in TOOLS:
        executable(name)
    executable("id", 'echo "$FAKE_UID"\n')
    executable("sudo", '''printf 'sudo %s\n' "$*" >> "$COMMAND_LOG"
if [ "${FAIL_SUDO:-0}" = 1 ]; then
    echo 'sudo fixture: permission denied' >&2
    exit 1
fi
exec "$@"
''')
    executable("apt-get", '''printf 'apt-get %s\n' "$*" >> "$COMMAND_LOG"
if [ "$1" = "${FAIL_APT_ACTION:-never}" ]; then
    echo 'apt fixture: installation failed' >&2
    exit 1
fi
if [ "$1" = install ]; then
    for package in "$@"; do
        case "$package" in
            build-essential) names=(gcc g++ make) ;;
            python3.12-dev)
                [ "${KEEP_HEADERS_MISSING:-0}" = 1 ] || : > "$HEADER_MARKER"
                continue ;;
            curl|git|rsync|tmux|tar) names=("$package") ;;
            *) continue ;;
        esac
        for name in "${names[@]}"; do
            printf '#!/bin/bash\nexit 0\n' > "$MACHINE_BIN/$name"
            /bin/chmod +x "$MACHINE_BIN/$name"
        done
    done
fi
''')
    python = executable("fixture-python", '''case "$2" in
    *get_config_h_filename*) [ -f "$HEADER_MARKER" ] ;;
    *version_info*) echo 3.12 ;;
    *base_prefix*) echo "$FAKE_BASE" ;;
    *get_path*) echo "$FAKE_BASE/include/python3.12" ;;
    *) echo 'unexpected Python query' >&2; exit 2 ;;
esac
''')
    return bin_dir, log, env, python


def run(machine, command):
    return subprocess.run(
        ["/bin/bash", "-eu", "-c", f"source {shlex.quote(str(HELPER))}; {command}"],
        env=machine[2], text=True, capture_output=True, timeout=10,
    )


def calls(machine):
    return machine[1].read_text().splitlines() if machine[1].exists() else []


def test_ready_tools_need_neither_apt_nor_sudo(machine):
    for name in ("apt-get", "sudo"):
        (machine[0] / name).unlink()
    result = run(machine, "sbd_ensure_system_tools")
    assert result.returncode == 0, result.stderr
    assert calls(machine) == []


@pytest.mark.parametrize("uid", ["0", "1000"])
def test_missing_tools_are_installed_once_with_appropriate_privilege(machine, uid):
    machine[2]["FAKE_UID"] = uid
    for name in ("gcc", "g++", "make", "rsync"):
        (machine[0] / name).unlink()
    result = run(machine, "sbd_ensure_system_tools; sbd_ensure_system_tools")
    assert result.returncode == 0, result.stderr
    log = calls(machine)
    assert [line for line in log if line.startswith("apt-get")] == [
        "apt-get update", "apt-get install -y build-essential rsync",
    ]
    assert any(line.startswith("sudo") for line in log) == (uid != "0")


@pytest.mark.parametrize("unavailable", ["sudo", "apt-get"])
def test_missing_privilege_or_package_manager_gives_actionable_error(machine, unavailable):
    (machine[0] / "gcc").unlink()
    (machine[0] / unavailable).unlink()
    result = run(machine, "sbd_ensure_system_tools")
    assert result.returncode != 0
    assert "build-essential" in result.stderr
    assert calls(machine) == []


@pytest.mark.parametrize("failure", ["sudo", "update", "install"])
def test_install_errors_remain_visible_and_stop_setup(machine, failure):
    (machine[0] / "gcc").unlink()
    if failure == "sudo":
        machine[2]["FAIL_SUDO"] = "1"
    else:
        machine[2]["FAIL_APT_ACTION"] = failure
    result = run(machine, "sbd_ensure_system_tools")
    assert result.returncode != 0
    assert "fixture:" in result.stderr
    if failure != "install":
        assert not any(line.startswith("apt-get install") for line in calls(machine))


def test_system_python_gets_matching_dev_package_and_rechecks_headers(machine):
    result = run(machine, f'sbd_ensure_python_headers "{machine[3]}"')
    assert result.returncode == 0, result.stderr
    assert "apt-get install -y python3.12-dev" in calls(machine)
    assert Path(machine[2]["HEADER_MARKER"]).exists()


def test_existing_real_python_headers_do_not_trigger_apt(machine):
    if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
        pytest.skip("The test runner itself has no Python development headers")
    result = run(machine, f"sbd_ensure_python_headers {shlex.quote(sys.executable)}")
    assert result.returncode == 0, result.stderr
    assert calls(machine) == []


def test_custom_python_with_missing_headers_does_not_install_unrelated_system_headers(machine):
    machine[2]["FAKE_BASE"] = "/opt/custom-python"
    result = run(machine, f'sbd_ensure_python_headers "{machine[3]}"')
    assert result.returncode != 0
    assert "/opt/custom-python/include/python3.12" in result.stderr
    assert calls(machine) == []


def test_apt_success_without_headers_still_fails(machine):
    machine[2]["KEEP_HEADERS_MISSING"] = "1"
    result = run(machine, f'sbd_ensure_python_headers "{machine[3]}"')
    assert result.returncode != 0
    assert "python3.12-dev" in result.stderr


def test_apt_index_is_shared_between_tool_and_header_installs(machine):
    (machine[0] / "gcc").unlink()
    result = run(machine, f'sbd_ensure_system_tools; sbd_ensure_python_headers "{machine[3]}"')
    assert result.returncode == 0, result.stderr
    assert calls(machine).count("apt-get update") == 1


def test_root_setup_preserves_existing_venv_when_system_tools_cannot_be_installed(machine, tmp_path):
    project = tmp_path / "project"
    (project / "tools/setup").mkdir(parents=True)
    shutil.copy(HELPER, project / "tools/setup/system_deps.sh")
    shutil.copy(ROOT / "setup_env.sh", project / "setup_env.sh")
    marker = project / ".venv/keep"
    marker.parent.mkdir()
    marker.write_text("existing environment")
    (machine[0] / "gcc").unlink()
    (machine[0] / "sudo").unlink()
    (machine[0] / "dirname").symlink_to(shutil.which("dirname"))
    result = subprocess.run(["/bin/bash", str(project / "setup_env.sh")], env=machine[2],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert "build-essential" in result.stderr
    assert marker.read_text() == "existing environment"


def test_root_setup_preserves_existing_venv_when_python_headers_cannot_be_installed(machine, tmp_path):
    project = tmp_path / "project"
    (project / "tools/setup").mkdir(parents=True)
    shutil.copy(HELPER, project / "tools/setup/system_deps.sh")
    shutil.copy(ROOT / "setup_env.sh", project / "setup_env.sh")
    marker = project / ".venv/keep"
    marker.parent.mkdir()
    marker.write_text("existing environment")
    (machine[0] / "sudo").unlink()
    (machine[0] / "dirname").symlink_to(shutil.which("dirname"))
    uv = machine[0] / "uv"
    uv.write_text('''#!/bin/bash
if [ "$1" = --version ]; then
    echo 'uv fixture'
elif [ "$1" = python ] && [ "$2" = find ] && [ "$3" = --system ]; then
    echo "$FAKE_PYTHON"
else
    echo 'unexpected uv call' >&2
    exit 1
fi
''')
    uv.chmod(0o755)
    machine[2].update(HOME=str(tmp_path / "home"), FAKE_PYTHON=str(machine[3]))
    result = subprocess.run(["/bin/bash", str(project / "setup_env.sh")], env=machine[2],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert "python3.12-dev" in result.stderr
    assert marker.read_text() == "existing environment"
