"""CPU-only proxy protocol and job activation tests, with no external services."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import http.client
import importlib.util
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import threading

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'lerobot/examples/libero/configs/src'
spec = importlib.util.spec_from_file_location('wandb_ssh_proxy', SRC / 'wandb_ssh_proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


class ReplyAfterEOF(socketserver.BaseRequestHandler):
    def handle(self):
        data = bytearray()
        while chunk := self.request.recv(65536):
            data.extend(chunk)
        self.request.sendall(b'reply:' + data)


class FakeRelay:
    def __init__(self, address):
        self.address, self.targets = address, []

    def connect(self, target):
        self.targets.append(target)
        return socket.create_connection(self.address, timeout=3), None

    def release(self, process):
        pass


@contextlib.contextmanager
def running(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_connect_auth_remote_hostname_parallel_streams_and_half_close():
    with running(socketserver.ThreadingTCPServer(('127.0.0.1', 0), ReplyAfterEOF)) as upstream:
        relay = FakeRelay(upstream.server_address)
        stopped = threading.Event()
        with running(proxy.ConnectServer(relay, 'private-token', stopped)) as server:
            def exchange(index):
                client = http.client.HTTPConnection(*server.server_address, timeout=3)
                client.set_tunnel('remote-dns-only.invalid', 443,
                                  headers={'Proxy-Authorization': server.authorization})
                client.connect()
                payload = (b'large-payload-' * 10000) + str(index).encode()
                client.sock.sendall(payload)
                client.sock.shutdown(socket.SHUT_WR)
                received = bytearray()
                while chunk := client.sock.recv(65536):
                    received.extend(chunk)
                client.close()
                assert received == b'reply:' + payload
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(exchange, range(6)))
            assert relay.targets == ['remote-dns-only.invalid:443'] * 6


def test_proxy_rejects_other_users_without_job_credentials():
    relay = FakeRelay(('127.0.0.1', 1))
    with running(proxy.ConnectServer(relay, 'secret', threading.Event())) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=3)
        client.request('CONNECT', 'example.invalid:443')
        assert client.getresponse().status == 407
        client.close()
        assert not relay.targets


def test_owner_pid_identity():
    assert proxy.owner_start(os.getpid()) is not None
    assert proxy.owner_start(99999999) is None


@pytest.mark.parametrize('extra', [
    {'SLURMD_NODENAME': 'node01'},
    {'SLURMD_NODENAME': 'runpod', 'SLURM_JOB_ID': ''},
    {'SLURM_JOB_ID': ''},
    {'SBD_WANDB_TUNNEL': 'off'},
    {'WANDB_MODE': 'offline'},
    {'WANDB_MODE': 'disabled'},
    {'WANDB_ENABLE': 'false'},
    {'WANDB_DISABLED': 'true'},
])
def test_inactive_modes_keep_network_and_traps_unchanged(extra):
    env = dict(os.environ, SLURMD_NODENAME='node100', SLURM_JOB_ID='123',
               SBD_WANDB_TUNNEL='auto', WANDB_MODE='online', WANDB_ENABLE='true',
               WANDB_DISABLED='false', WANDB_HTTPS_PROXY='existing')
    env.update(extra)
    result = subprocess.run(['bash', '-c', '''set -euo pipefail
trap ':' EXIT
before=$(trap -p EXIT)
source "$1"
test "$(trap -p EXIT)" = "$before"
test "$WANDB_HTTPS_PROXY" = existing
test -z "${SBD_WANDB_TUNNEL_PID:-}"
''', '_', str(SRC / 'wandb_tunnel.sh')], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
