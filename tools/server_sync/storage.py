"""Read local filesystem capacity without walking project contents."""
import os
import shutil
import socket
from pathlib import Path


def storage_snapshot(locations):
    volumes, devices = [], set()
    for name, path in locations:
        path = Path(path)
        try:
            if not path.exists():
                continue
            device = os.stat(path).st_dev
            if device in devices:
                continue
            usage = shutil.disk_usage(path)
            devices.add(device)
            available_total = usage.used + usage.free
            volumes.append({'name': name, 'path': str(path), 'total': usage.total,
                            'used': usage.used, 'free': usage.free,
                            'used_percent': round(100 * usage.used / available_total, 1) if available_total else 0})
        except OSError:
            volumes.append({'name': name, 'path': str(path), 'error': '용량을 조회할 수 없습니다.'})
    return {'hostname': socket.gethostname(), 'volumes': volumes}
