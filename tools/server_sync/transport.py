"""Shared SSH/rsync endpoints for the manual UI and checkpoint archiver."""
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]


@dataclass(frozen=True)
class Endpoint:
    id: str
    name: str
    base: str
    host: str = ""

    def __post_init__(self):
        if not os.path.isabs(self.base) or self.base == "/":
            raise ValueError("Endpoint base must be an absolute workspace path")
        if self.host and (self.host.startswith("-") or not re.fullmatch(r"[\w.@:\[\]-]+", self.host)):
            raise ValueError("Invalid SSH host; use user@host or an SSH config alias")


def configured_endpoints(repo=REPO, env=None, hostname=None):
    env = os.environ if env is None else env
    hostname = socket.gethostname() if hostname is None else hostname
    mode = env.get("SYNC_MODE") or ("yonsei" if hostname.startswith("yonsei") else "rllab")
    if mode not in ("yonsei", "rllab"):
        raise ValueError("SYNC_MODE must be yonsei or rllab")
    host = env.get("YONSEI_HOST", "mdorazi@165.132.142.208") if mode == "rllab" else ""
    yonsei = [
        Endpoint("scratch", "yonsei /scratch", env.get("YONSEI_SCRATCH_BASE", "/scratch/mdorazi/Skill_Boundary_Detector"), host),
        Endpoint("scratch2", "yonsei /scratch2", env.get("YONSEI_SCRATCH2_BASE", "/scratch2/mdorazi/Skill_Boundary_Detector"), host),
    ]
    local = [Endpoint("data1", "rllab /data1", env.get("RLLAB_BASE", str(repo))),
             Endpoint("data2", "rllab /data2", env.get("RLLAB2_BASE", "/data2/dohyeon/SBD"))]
    return mode, yonsei if mode == "yonsei" else local + yonsei


def rsync_options():
    return ["rsync", "-a", "--protect-args", "--partial-dir=.sbd-sync-partial",
            "--info=progress2", "--human-readable", "--timeout=120"]
