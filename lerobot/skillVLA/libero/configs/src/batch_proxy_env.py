"""Find unreachable login-host loopback proxies inherited by a Slurm worker."""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

PROXY_KEYS = (
    "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy",
    "WANDB_HTTP_PROXY", "WANDB_HTTPS_PROXY",
)


def stale_loopback_proxies(environment, connect=socket.create_connection):
    if not environment.get("SLURM_JOB_ID"):
        return []
    checked, stale = {}, []
    for key in PROXY_KEYS:
        value = environment.get(key)
        if not value:
            continue
        try:
            url = urlsplit(value)
            host = url.hostname
            if not host:
                continue
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host.lower() == "localhost"
            if not loopback:
                continue
            port = url.port or (443 if url.scheme == "https" else 80)
        except ValueError:
            continue
        address = host, port
        if address not in checked:
            try:
                with connect(address, timeout=0.4):
                    pass
            except OSError:
                checked[address] = False
            else:
                checked[address] = True
        if not checked[address]:
            stale.append(key)
    return stale


if __name__ == "__main__":
    # Only variable names cross back into Bash; never expose proxy credentials.
    print(" ".join(stale_loopback_proxies(os.environ)))
