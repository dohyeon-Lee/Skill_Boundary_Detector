import importlib.util
from pathlib import Path
from unittest.mock import Mock, MagicMock


PATH = Path(__file__).resolve().parents[3] / "lerobot/examples/libero/configs/src/batch_proxy_env.py"
SPEC = importlib.util.spec_from_file_location("batch_proxy_env_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dead_inherited_local_proxies_removed_only_inside_slurm():
    connect = Mock(side_effect=ConnectionRefusedError())
    env = {"SLURM_JOB_ID": "68381", "HTTP_PROXY": "http://127.0.0.1:17891",
           "HTTPS_PROXY": "http://127.0.0.1:17891", "https_proxy": "http://127.0.0.1:17891"}
    assert MODULE.stale_loopback_proxies(env, connect) == ["HTTP_PROXY", "HTTPS_PROXY", "https_proxy"]
    assert connect.call_count == 1
    connect.reset_mock()
    env.pop("SLURM_JOB_ID")
    assert MODULE.stale_loopback_proxies(env, connect) == []
    connect.assert_not_called()


def test_live_job_local_and_nonlocal_proxies_are_preserved():
    connect = MagicMock()
    env = {"SLURM_JOB_ID": "1", "HTTPS_PROXY": "http://job:private@localhost:1234",
           "HTTP_PROXY": "http://corp.internal:8080", "ALL_PROXY": "socks5://10.0.0.1:1080"}
    assert MODULE.stale_loopback_proxies(env, connect) == []
    connect.assert_called_once_with(("localhost", 1234), timeout=0.4)


def test_ipv6_default_ports_and_malformed_values_do_not_expose_credentials():
    connect = Mock(side_effect=TimeoutError())
    env = {"SLURM_JOB_ID": "1", "HTTPS_PROXY": "https://user:private@[::1]",
           "HTTP_PROXY": "invalid", "all_proxy": "http://127.0.0.1:bad"}
    assert MODULE.stale_loopback_proxies(env, connect) == ["HTTPS_PROXY"]
    connect.assert_called_once_with(("::1", 443), timeout=0.4)
