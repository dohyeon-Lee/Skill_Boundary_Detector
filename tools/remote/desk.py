"""Persistent Yonsei monitoring and verified automatic result collection."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.remote.yonsei import SSH_OPTIONS


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def load(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def ensure_daemon(root=ROOT):
    folder = root / '.cache/yonsei_desk'
    folder.mkdir(parents=True, exist_ok=True)
    # The child owns an exclusive process-lifetime lock; duplicate starts exit.
    with (folder / 'daemon.log').open('ab', buffering=0) as log:
        subprocess.Popen([sys.executable, str(ROOT / 'tools/remote/desk.py'),
                          '--root', str(root), 'daemon'], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True, close_fds=True)


def query(host, base, ids):
    from tools.remote.yonsei import endpoint
    endpoint({'remote_host': host, 'remote_project_root': base}, {})
    code = (ROOT / 'tools/remote/probe.py').read_text()
    result = subprocess.run(['ssh', *SSH_OPTIONS, host, 'python3 -c ' + shlex.quote(code)],
                            input=json.dumps({'base': base, 'ids': ids}),
                            capture_output=True, text=True, check=True, timeout=100)
    return json.loads(result.stdout)


def destination(root, receipt, manifest):
    relative = Path(manifest['evaluation']['output'])
    if relative.is_absolute() or '..' in relative.parts or relative.name in {'outputs', ''}:
        raise ValueError('Invalid result destination')
    parent = root / relative.parent
    parent.resolve().relative_to(root.resolve())
    name = relative.name if relative.name.endswith('_yonsei') else relative.name + '_yonsei'
    target = parent / name
    # Keep existing local results, including other Yonsei attempts of the same run.
    owner = load(target / 'yonsei_origin.json', {})
    if target.exists() and owner.get('launch_id') != receipt.name:
        name = relative.name.removesuffix('_yonsei') + '_' + receipt.name + '_yonsei'
        target = parent / name
    return target


def repair_paths(folder, old, new):
    """Move embedded absolute report references without touching video bytes."""
    for base, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
        for name in files:
            p = Path(base) / name
            if p.suffix not in {'.html', '.json', '.js', '.css', '.m3u8'} or p.is_symlink():
                continue
            text = p.read_text(errors='strict')
            if old in text:
                p.write_text(text.replace(old, str(new)))


def collect(root, receipt, manifest):
    """Resume an interrupted transfer; publish only after an rsync checksum check."""
    relative = Path(manifest['evaluation']['output'])
    target = destination(root, receipt, manifest)
    if load(target / 'yonsei_origin.json', {}).get('launch_id') == receipt.name:
        return str(target)  # Recover a daemon restart immediately after publication.
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / ('.yonsei-' + receipt.name + '.partial')
    stage.mkdir(exist_ok=True)
    remote = str(Path(manifest['project']) / relative)
    if not remote.startswith(manifest['project'].rstrip('/') + '/'):
        raise ValueError('Invalid remote result path')
    transport = 'ssh ' + shlex.join(SSH_OPTIONS)
    common = ['rsync', '-rlt', '--safe-links', '--protect-args', '--timeout=120',
              '-e', transport]
    source = manifest['host'] + ':' + remote.rstrip('/') + '/'
    with (receipt / 'collection.log').open('ab', buffering=0) as log:
        subprocess.run([*common, '--partial-dir=.yonsei-partial', source, str(stage) + '/'],
                       stdout=log, stderr=log, check=True, timeout=24 * 3600)
        verified = subprocess.run([*common, '--checksum', '--dry-run', '--itemize-changes',
                                   source, str(stage) + '/'],
                                  capture_output=True, text=True, check=True, timeout=24 * 3600)
    if verified.stdout.strip():
        raise RuntimeError('Remote results changed during transfer; retrying next poll')
    repair_paths(stage, remote, target)
    atomic(stage / 'yonsei_origin.json', {'launch_id': receipt.name, 'host': manifest['host'],
           'remote_output': remote, 'collected_at': time.time(), 'verified': 'rsync-checksum'})
    # No replacement of an existing result folder. A completed receipt is immutable.
    if target.exists():
        raise FileExistsError('Result destination already exists: ' + str(target))
    stage.rename(target)
    return str(target)


class Worker:
    def __init__(self, root=ROOT, probe=query, collector=collect):
        self.root, self.probe, self.collector = root, probe, collector
        self.folder = root / '.cache/yonsei_desk'
        self.receipts = root / '.cache/yonsei_launches'
        self.inflight = {}
        self.lock = threading.Lock()

    def transfer(self, receipt, manifest):
        try:
            atomic(receipt / 'collection.json', {'state': 'TRANSFERRING', 'started_at': time.time()})
            target = self.collector(self.root, receipt, manifest)
            atomic(receipt / 'collection.json', {'state': 'COLLECTED', 'path': target,
                                                'finished_at': time.time()})
        except Exception as error:
            atomic(receipt / 'collection.json', {'state': 'RETRY', 'error': str(error)[:1000],
                                                'retry_after': time.time() + 120})
        finally:
            with self.lock:
                self.inflight.pop(receipt.name, None)

    def poll(self):
        manifests, groups, rows = {}, {}, {}
        for path in sorted(self.receipts.glob('*/launch.json'), reverse=True)[:200]:
            manifest = load(path)
            if not manifest or not (path.parent / 'remote.json').is_file():
                continue
            identifier = path.parent.name
            manifests[identifier] = manifest
            collection = load(path.parent / 'collection.json', {})
            cached = load(path.parent / 'desk_status.json', {})
            rows[identifier] = {'id': identifier, 'script': manifest.get('script', ''),
                'created_at': manifest.get('created_at') or path.stat().st_mtime,
                'host': manifest['host'], 'evaluation': manifest.get('evaluation'),
                'collection': collection, **cached}
            if collection.get('state') == 'COLLECTED' or cached.get('state') == 'FAILED':
                continue
            if cached.get('state') == 'COMPLETED' and not manifest.get('evaluation'):
                continue
            if manifest.get('dry_run'):
                rows[identifier]['state'] = 'DRY_RUN'
                continue
            key = (manifest['host'], manifest['profile']['project'])
            groups.setdefault(key, []).append(identifier)
        errors = []
        for (host, base), ids in groups.items():
            try:
                snapshots = self.probe(host, base, ids)
                for row in snapshots:
                    identifier = row['id']
                    receipt = self.receipts / identifier
                    atomic(receipt / 'desk_status.json', row)
                    rows[identifier].update(row)
                    rows[identifier].pop('connection_error', None)
                    collection = load(receipt / 'collection.json', {})
                    if row['state'] == 'COMPLETED' and manifests[identifier].get('evaluation'):
                        if (collection.get('state') != 'COLLECTED' and
                                collection.get('retry_after', 0) <= time.time()):
                            with self.lock:
                                if identifier not in self.inflight and len(self.inflight) < 2:
                                    thread = threading.Thread(target=self.transfer,
                                        args=(receipt, manifests[identifier]), daemon=True)
                                    self.inflight[identifier] = thread
                                    thread.start()
            except Exception as error:
                detail = str(error)[:500]
                errors.append({'host': host, 'error': detail})
                for identifier in ids:
                    rows[identifier]['connection_error'] = detail
        for identifier, row in rows.items():
            row['collection'] = load(self.receipts / identifier / 'collection.json', row['collection'])
        value = {'updated_at': time.time(), 'jobs': list(rows.values()), 'errors': errors,
                 'interval': 30, 'automatic_collection': True}
        atomic(self.folder / 'state.json', value)
        return value


class Refresh:
    def __init__(self, folder):
        self.folder = folder

    def set(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / 'refresh').touch()


class Monitor:
    """Desk adapter. The collector keeps running when this UI closes."""
    def __init__(self, root=ROOT):
        self.root = root
        self.folder = root / '.cache/yonsei_desk'
        self.refresh = Refresh(self.folder)

    def start(self):
        ensure_daemon(self.root)

    def close(self):
        pass

    def snapshot(self):
        state = load(self.folder / 'state.json', {'jobs': [], 'errors': [], 'updated_at': None})
        return {**state, 'stale': not state.get('updated_at') or time.time() - state['updated_at'] > 150}


def daemon(root):
    folder = root / '.cache/yonsei_desk'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'daemon.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        atomic(folder / 'daemon.json', {'pid': os.getpid(), 'started_at': time.time()})
        worker = Worker(root)
        while True:
            try:
                (folder / 'refresh').unlink(missing_ok=True)
                worker.poll()
            except Exception as error:
                print('Yonsei monitor: ' + str(error), flush=True)
            for _ in range(30):
                if (folder / 'refresh').exists():
                    break
                time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('action', choices=['daemon', 'once'])
    options = parser.parse_args()
    if options.action == 'daemon':
        daemon(options.root)
    else:
        print(json.dumps(Worker(options.root).poll(), ensure_ascii=False))
