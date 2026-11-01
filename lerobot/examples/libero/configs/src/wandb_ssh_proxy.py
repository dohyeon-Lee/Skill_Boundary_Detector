"""Job-local HTTP CONNECT proxy; SSH resolves destinations on the relay node."""
from __future__ import annotations

import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import signal
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def owner_start(pid):
    """Distinguish a living owner from a subsequently reused PID."""
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class SSHRelay:
    def __init__(self, host, directory):
        self.host = host
        self.control = str(directory / 'ssh-control')
        self.options = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                        '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15',
                        '-o', 'ServerAliveCountMax=3', '-S', self.control]
        self.master = None
        self.channels = set()
        self.lock = threading.Lock()

    def start(self):
        Path(self.control).unlink(missing_ok=True)
        self.master = subprocess.Popen(
            ['ssh', *self.options, '-M', '-N', self.host], stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.master.poll() is not None:
                raise RuntimeError('SSH relay exited; check SSH access and known_hosts.')
            if Path(self.control).exists():
                return
            time.sleep(.1)
        stop_process(self.master)
        raise RuntimeError('SSH relay startup timed out.')

    def connect(self, target):
        local, remote = socket.socketpair()
        try:
            process = subprocess.Popen(
                ['ssh', *self.options, '-o', 'ControlMaster=no', '-W', target, self.host],
                stdin=remote, stdout=remote)
        except Exception:
            local.close()
            raise
        finally:
            remote.close()
        with self.lock:
            self.channels.add(process)
        return local, process

    def release(self, process):
        stop_process(process)
        with self.lock:
            self.channels.discard(process)

    def close(self):
        with self.lock:
            children = list(self.channels)
        for process in children:
            stop_process(process)
        if self.master is not None:
            stop_process(self.master)


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def copy_streams(client, upstream, stopped):
    """Pass encrypted TLS bytes unchanged, including TCP half-close semantics."""
    readers = [client, upstream]
    while readers and not stopped.is_set():
        ready, _, _ = select.select(readers, [], [], 1)
        for source in ready:
            destination = upstream if source is client else client
            data = source.recv(65536)
            if not data:
                readers.remove(source)
                try:
                    destination.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
            else:
                destination.sendall(data)


class ConnectHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        # Never write proxy credentials or signed artifact URLs to job logs.
        pass

    def do_CONNECT(self):
        if not hmac.compare_digest(self.headers.get('Proxy-Authorization', ''), self.server.authorization):
            self.send_error(407, 'Proxy authentication required')
            return
        try:
            target = urllib.parse.urlsplit('//' + self.path)
            if (not target.hostname or not target.port or target.username or target.password
                    or target.path or target.query or target.fragment):
                raise ValueError('Invalid CONNECT destination')
            host = target.hostname
            if any(c.isspace() for c in host) or host.startswith('-'):
                raise ValueError('Invalid CONNECT host')
            address = f'[{host}]:{target.port}' if ':' in host else f'{host}:{target.port}'
            upstream, process = self.server.relay.connect(address)
        except (ValueError, OSError):
            self.send_error(502, 'Cannot open SSH channel')
            return
        try:
            self.connection.settimeout(30)
            upstream.settimeout(30)
            self.send_response(200, 'Connection established')
            self.end_headers()
            self.wfile.flush()
            copy_streams(self.connection, upstream, self.server.stopped)
        except (OSError, ValueError):
            pass
        finally:
            upstream.close()
            self.server.relay.release(process)


class ConnectServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, relay, token, stopped):
        super().__init__(('127.0.0.1', 0), ConnectHandler)
        self.relay, self.stopped = relay, stopped
        self.authorization = 'Basic ' + base64.b64encode(('job:' + token).encode()).decode()


def probe(proxy, url):
    context = ssl.create_default_context()
    # uv's standalone Python can expect /etc/ssl/cert.pem on RHEL, where the
    # trusted system bundle lives elsewhere. Retain certificate verification.
    for bundle in ('/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt'):
        if Path(bundle).is_file():
            context.load_verify_locations(cafile=bundle)
            break
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': proxy}),
                                        urllib.request.HTTPSHandler(context=context))
    try:
        with opener.open(url, timeout=12):
            pass
    except urllib.error.HTTPError as error:
        # api.wandb.ai's root returns 404; TLS and the CONNECT route succeeded.
        if error.code >= 500:
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='node01')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--owner-pid', type=int, required=True)
    args = parser.parse_args()
    directory = args.directory
    directory.chmod(0o700)
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    owner = owner_start(args.owner_pid)
    if owner is None:
        raise RuntimeError('Job owner is not alive.')
    relay = SSHRelay(args.host, directory)
    server = None
    try:
        relay.start()
        token = secrets.token_urlsafe(24)
        server = ConnectServer(relay, token, stopped)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        proxy = f'http://job:{token}@127.0.0.1:{server.server_port}'
        probe(proxy, 'https://api.wandb.ai')
        ready = directory / 'ready.json'
        pending = directory / 'ready.tmp'
        pending.write_text(json.dumps({'proxy': proxy, 'pid': os.getpid()}))
        pending.chmod(0o600)
        pending.replace(ready)
        print(f'W&B relay ready via {args.host}', flush=True)
        while not stopped.wait(1) and owner_start(args.owner_pid) == owner:
            if relay.master.poll() is not None:
                print('W&B SSH connection dropped; reconnecting.', flush=True)
                try:
                    relay.start()
                except RuntimeError:
                    stopped.wait(3)
    finally:
        stopped.set()
        relay.close()
        if server is not None:
            server.shutdown()
            server.server_close()
        # Keep startup failures long enough for the parent shell to show the
        # reason; that shell removes the private directory after reading it.
        if (directory / 'ready.json').exists():
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == '__main__':
    main()
