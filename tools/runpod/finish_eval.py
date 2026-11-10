#!/usr/bin/env python3
"""Persistently archive one completed RunPod eval through HF Desk, then stop it.

Run on the lab server, detached from SSH/this chat. A failed eval, incomplete
copy, unexpected repository visibility, or hash mismatch never permits a Pod stop.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.runpod import update_ssh_config as api
from tools.runpod.desk import RunPodAPI
from tools.runpod.hf_desk_client import HFDesk


PROBE = r'''
import base64, hashlib, json, sys
from pathlib import Path
args=json.loads(base64.b64decode(sys.argv[1]))
root=Path(args['root']).resolve()
base=root/'.cache/local_jobs'/args['job']
job=json.loads(base.with_suffix('.json').read_text())
exit_path=base.with_suffix('.exit')
result={'exit': exit_path.read_text().strip() if exit_path.exists() else None}
out=Path(job['eval_output']).resolve()
out.relative_to(root)
if '/eval/outputs/' not in str(out): raise ValueError('Unexpected eval output path')
result['output']=str(out)
active=[]
for record in (root/'.cache/local_jobs').glob('*.pid'):
    b=record.with_suffix('')
    if b.with_suffix('.exit').exists(): continue
    try:
        pid=int(record.read_text())
        stat=Path('/proc')/str(pid)/'stat'
        fields=stat.read_text().rsplit(')',1)[1].split()
        marker=b.with_suffix('.pidstart')
        if fields[0]!='Z' and (not marker.exists() or marker.read_text().strip()==fields[19]):
            active.append(b.name)
    except (OSError, ValueError, IndexError): pass
result['active_jobs']=active
if args.get('hashes') and result['exit']=='0':
    if not (out/'metrics/eval_info_merged.json').is_file():
        raise ValueError('Final merged metrics missing')
    files={}
    for p in sorted(out.rglob('*')):
        if p.is_symlink(): raise ValueError('Result symlink is not supported')
        if not p.is_file(): continue
        before=p.stat()
        h=hashlib.sha256(); g=hashlib.sha1()
        g.update(b'blob '+str(before.st_size).encode()+b'\0')
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024), b''):
                h.update(chunk);g.update(chunk)
        after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Result changed while hashing')
        files[p.relative_to(out).as_posix()]={'size':before.st_size,'sha256':h.hexdigest(),'git_sha1':g.hexdigest()}
    if not files: raise ValueError('Empty result folder')
    result['files']=files
print(json.dumps(result))
'''


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def hashes(path):
    h, g = hashlib.sha256(), hashlib.sha1()
    size = path.stat().st_size
    g.update(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            g.update(chunk)
    return {"size": size, "sha256": h.hexdigest(), "git_sha1": g.hexdigest()}


def validate_local(folder, expected):
    actual = {}
    for p in folder.rglob("*"):
        if p.is_symlink():
            raise ValueError("Local backup contains a symlink")
        if p.is_file():
            actual[p.relative_to(folder).as_posix()] = hashes(p)
    if actual != expected:
        raise ValueError("Local result copy differs from the Pod")


def verify_hf(repo, prefix, files, revision=None):
    from huggingface_hub import HfApi
    hub = HfApi()
    info = hub.repo_info(repo, repo_type="model")
    if info.private is not False:
        raise ValueError("Result repository must be public")
    revision = revision or info.sha
    remote = {entry.path: entry for entry in hub.list_repo_tree(
        repo, path_in_repo=prefix, repo_type="model", revision=revision, recursive=True)
        if hasattr(entry, "size")}
    expected = {prefix + "/" + name: item for name, item in files.items()}
    if set(remote) != set(expected):
        raise ValueError("HF result file inventory differs from the Pod")
    for path, item in expected.items():
        entry = remote[path]
        lfs = getattr(entry, "lfs", None)
        digest = lfs.sha256 if lfs else entry.blob_id
        if entry.size != item["size"] or digest != item["sha256" if lfs else "git_sha1"]:
            raise ValueError("HF result hash mismatch: " + path)
    return revision


def ssh_command(pod, identity):
    host, port = api.endpoint(pod)
    return ["ssh", "-i", str(identity), "-p", str(port), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "root@" + host]


def probe(pod, args, *, hashes=False):
    data = base64.b64encode(json.dumps({"root": args.remote_root, "job": args.job,
                                      "hashes": hashes}).encode()).decode()
    proc = subprocess.run(ssh_command(pod, args.identity) + ["python3 - " + shlex.quote(data)],
                          input=PROBE, text=True, capture_output=True, timeout=300)
    if proc.returncode:
        raise ValueError("Remote evaluation/result query failed (SSH or incomplete files)")
    return json.loads(proc.stdout)


def save(path, state, **fields):
    state.update(fields, updated_at=now())
    api.atomic_write(path, (json.dumps(state, indent=2) + "\n").encode())


def run(args):
    folder = ROOT / ".cache/runpod_finish"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (args.pod + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = folder / (args.pod + ".json")
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state and (state["job"] != args.job or state["repo"] != args.repo):
            raise ValueError("A different finalizer already owns this Pod")
        save(state_path, state, pod=args.pod, job=args.job, repo=args.repo)
        client = RunPodAPI()
        while True:
            try:
                pod = client.pod(args.pod)
                if pod["status"] != "RUNNING":
                    if state.get("verified_revision") and pod["status"] in {"STOPPED", "EXITED", "TERMINATED"}:
                        save(state_path, state, phase="stopped", error=None)
                        print(now(), "Backup verified; Pod stopped", args.pod, flush=True)
                        return
                    raise ValueError("Pod is not RUNNING; preserving backup state")
                result = probe(pod, args)
                if result["exit"] is None:
                    save(state_path, state, phase="waiting_for_eval", error=None)
                elif result["exit"] != "0":
                    save(state_path, state, phase="eval_failed", error="Eval exit " + result["exit"])
                elif result["active_jobs"]:
                    save(state_path, state, phase="waiting_for_other_jobs", error=None)
                else:
                    result = probe(pod, args, hashes=True)
                    files = result["files"]
                    signature = fingerprint(files)
                    relative = Path(result["output"]).relative_to(args.remote_root).as_posix()
                    target = folder / args.pod / "results" / relative
                    if state.get("uploaded_hash") != signature or not state.get("verified_revision"):
                        save(state_path, state, phase="copying_results", output=result["output"],
                             hf_path=relative, local_backup=str(target), files=files, error=None)
                        print(now(), "Copying result folder", len(files), "files", flush=True)
                        target.mkdir(parents=True, exist_ok=True)
                        ssh = ssh_command(pod, args.identity)
                        command = ["rsync", "-a", "--no-owner", "--no-group", "--checksum", "--protect-args",
                                   "-e", shlex.join(ssh[:-1]), ssh[-1] + ":" + result["output"] + "/", str(target) + "/"]
                        subprocess.run(command, check=True, timeout=3600)
                        validate_local(target, files)
                        from huggingface_hub import HfApi
                        if HfApi().repo_info(args.repo, repo_type="model").private is not False:
                            raise ValueError("Result repository must be public before upload")
                        save(state_path, state, phase="uploading_results")
                        # Uploads remain visible and managed in the existing HF Desk.
                        desk = HFDesk(args.desk_url, ROOT, None)
                        if desk.checkpoint_repo != args.repo:
                            raise ValueError("HF Desk public repository differs from the backup destination")
                        desk.publish(relative, eval_results=True, runpod_id=args.pod)
                        revision = verify_hf(args.repo, relative, files)
                        save(state_path, state, phase="backup_verified", uploaded_hash=signature,
                             verified_revision=revision, backup_verified_at=now())
                    else:
                        verify_hf(args.repo, relative, files, state["verified_revision"])
                    # Re-check the complete output and all scheduled jobs immediately
                    # before the destructive stop (container disk is ephemeral).
                    latest = probe(pod, args, hashes=True)
                    if latest["exit"] != "0" or latest["active_jobs"] or fingerprint(latest["files"]) != signature:
                        raise ValueError("Pod changed after backup; stop refused")
                    if not state.get("verified_revision") or state.get("uploaded_hash") != signature:
                        raise ValueError("Verified backup required before stop")
                    save(state_path, state, phase="stopping", error=None)
                    print(now(), "HF hashes verified; stopping Pod", args.pod, flush=True)
                    client.stop(args.pod)
            except Exception as error:
                message = re.sub(r"https?://\S+", "[URL]", str(error))[:500]
                save(state_path, state, error=message)
                print(now(), "Retrying without stopping:", message, flush=True)
            time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--repo", required=True, help="Existing public HF model repository")
    parser.add_argument("--desk-url", default="http://127.0.0.1:8765/hf")
    parser.add_argument("--remote-root", default="/workspace/Skill_Boundary_Detector")
    parser.add_argument("--identity", type=Path, default=Path.home() / ".ssh/id_ed25519")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("Polling interval must be at least 10 seconds")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.pod) or not re.fullmatch(r"[0-9]+", args.job):
        parser.error("Invalid Pod/job ID")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    if not os.environ.get("SSL_CERT_FILE"):
        for bundle in api.CA_BUNDLE_PATHS:
            if Path(bundle).is_file():
                os.environ["SSL_CERT_FILE"] = bundle
                break
    run(args)


if __name__ == "__main__":
    main()
