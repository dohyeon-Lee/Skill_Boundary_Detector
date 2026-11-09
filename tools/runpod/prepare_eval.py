#!/usr/bin/env python3
"""Publish portable SkillVLA eval inputs via HF Desk; consume them on RunPod.

The two-phase publication (immutable bundle, then a small named reference) means
a failed/interrupted Desk upload never replaces a usable evaluation reference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

# Match the project's RunPod SSH tooling: inherited loopback proxies on shared
# servers may not work here. Respect an explicitly configured CA bundle.
if __name__ == "__main__":
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    if not os.environ.get("SSL_CERT_FILE"):
        for bundle in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
            if Path(bundle).is_file():
                os.environ["SSL_CERT_FILE"] = bundle
                break
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RemoteEntryNotFoundError
import httpx

from hf_desk_client import connect

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = Path("lerobot/examples/libero/configs")
EVALS = {
    "ft": CONFIGS / "train_skillVLA/FT/eval/ft_eval_config.yaml",
    "pt": CONFIGS / "train_skillVLA/PT/eval/stage1_eval_config.yaml",
}
PREFIX = "dataset_eval_hf"
INPUT_FIELDS = (
    "policy_path", "external_skill_model", "external_predictor_model",
    "external_predictor_skill_model", "external_terminator_model", "fsq_path",
    "skill_dataset_dir", "eval_init_states_path", "skill_latents_path",
    "raw_dataset_dir", "focus_uv_path", "dino_model_path", "tokenizer_path",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".eval-hf-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def relative(value):
    if not isinstance(value, str):
        raise ValueError("경로는 문자열이어야 합니다.")
    path = PurePosixPath(value)
    if (not value or not path.parts or path.as_posix() != value or path.is_absolute() or ".." in path.parts
            or "\\" in value or any(part.startswith(".") for part in path.parts)):
        raise ValueError("허용되지 않는 상대 경로입니다.")
    return path


def asset_path(value):
    path = relative(value)
    first = path.parts[0]
    if not (first == "models" or first in {"dataset", "outputs"}
            or first.startswith(("dataset_", "outputs_"))
            or (path.is_relative_to(CONFIGS / "train_skillVLA")
                and path.suffix in {".npz", ".json"}
                and any(part in {"initial_state_banks", "references"} for part in path.parts))):
        raise ValueError(f"평가 입력으로 지원하지 않는 경로: {value}")
    return path


def retry(call):
    for attempt in range(3):
        try:
            return call()
        except (httpx.HTTPStatusError, HfHubHTTPError) as error:
            if error.response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (httpx.TransportError, ConnectionError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def downloaded(repo, filename, revision, *, repo_type="dataset", local_dir=None):
    return Path(retry(lambda: hf_hub_download(repo, filename=filename,
        revision=revision, repo_type=repo_type, local_dir=local_dir)))


class HashCache:
    """Only reuse a digest while the local file's full stat fingerprint is unchanged."""
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.values = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.values = {}

    @staticmethod
    def stamp(path):
        info = path.stat()
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]

    def hash(self, path):
        path = Path(path)
        stamp = self.stamp(path)
        old = self.values.get(str(path))
        if old and old.get("stat") == stamp:
            return old["sha256"], old["git_sha1"]
        sha, git = hashlib.sha256(), hashlib.sha1()
        git.update(b"blob " + str(stamp[2]).encode() + b"\0")
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                sha.update(block)
                git.update(block)
        if self.stamp(path) != stamp:
            raise ValueError(f"파일이 읽는 도중 변경됐습니다. 저장 완료 후 다시 실행하세요: {path.name}")
        self.values[str(path)] = {"stat": stamp, "sha256": sha.hexdigest(), "git_sha1": git.hexdigest()}
        return sha.hexdigest(), git.hexdigest()

    def save(self):
        atomic_write(self.path, encoded(self.values))


def configuration(root, path):
    """Use the existing evaluator's resolver, including its checkpoint contract checks."""
    sys.path.insert(0, str(root / "lerobot/src"))
    resolver_path = root / CONFIGS / "train_skillVLA/PT/eval/src/stage1_eval_config.py"
    spec = importlib.util.spec_from_file_location("sbd_eval_hf_resolver", resolver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    merged = module.load_config(path)
    settings = module.build_settings(merged)
    # Freeze shared functional defaults, while retaining the destination server's
    # scheduler, storage paths and machine profile.
    frozen = yaml.safe_load((root / CONFIGS / "global_config.yaml").read_text()) or {}
    frozen.update(yaml.safe_load(path.read_text()) or {})
    for key in ("dataset_root", "outputs_root"):
        frozen[key] = merged[key]
    for key in ("project_root", "server", "storage_volume", "storage_outputs", "scheduler"):
        frozen.pop(key, None)

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and value.startswith(str(root) + "/"):
            return value[len(str(root)) + 1:]
        return value

    frozen = portable(frozen)
    # Do not freeze a server-specific root outside the portable namespace.
    for key in ("dataset_root", "outputs_root"):
        asset_path(frozen[key])
    return settings, frozen


def collect(root, settings, cache):
    paths = set()
    models = json.loads(settings["models_json"])
    for model in models:
        for key in INPUT_FIELDS:
            if model.get(key):
                paths.add(Path(model[key]))
    if settings.get("initial_state_bank_path"):
        paths.add(Path(settings["initial_state_bank_path"]))
    files = {}

    def add(path, ancestors=()):
        logical = path.relative_to(root).as_posix()
        asset_path(logical)
        if path.is_dir():
            resolved = path.resolve()
            if resolved in ancestors:
                raise ValueError("평가 입력 폴더에 순환 링크가 있습니다.")
            for child in sorted(path.iterdir()):
                if not child.name.startswith(".") and child.name != "__pycache__":
                    add(child, (*ancestors, resolved))
        elif path.is_file():
            sha, git = cache.hash(path)
            files[logical] = {"path": logical, "size": path.stat().st_size,
                              "sha256": sha, "git_sha1": git}
        else:
            raise ValueError(f"평가 입력 파일이 없습니다: {logical}")

    for index, path in enumerate(sorted(paths), 1):
        if not path.is_absolute():
            path = root / path
        print(f"입력 확인 {index}/{len(paths)}: {path.name}", flush=True)
        add(path)
        cache.save()
    if not files:
        raise ValueError("평가 입력 파일이 없습니다.")
    return [files[name] for name in sorted(files)]


def same(remote, local):
    if remote is None or getattr(remote, "size", None) != local["size"]:
        return False
    lfs = getattr(remote, "lfs", None)
    return (lfs.sha256 == local["sha256"]) if lfs else (remote.blob_id == local["git_sha1"])


def refs_for_files(api, root, files, dataset_repo, checkpoint_repo, private_revision):
    specs = runpy.run_path(str(root / CONFIGS / "src/hf_sync.py"))["PRETRAINED_MODELS"]
    groups, matched = {}, {}
    for item in files:
        parts = PurePosixPath(item["path"]).parts
        if parts[0] == "models" and len(parts) >= 3 and parts[1] in specs:
            spec = specs[parts[1]]
            group = (spec["repo"], "model", spec.get("revision"))
            filename = "/".join(parts[2:])
        elif parts[0] == "outputs" or parts[0].startswith("outputs_"):
            group, filename = (checkpoint_repo, "model", None), item["path"]
        else:
            group, filename = (dataset_repo, "dataset", private_revision), item["path"]
        groups.setdefault(group, []).append((item, filename))
    for (repo, kind, rev), items in groups.items():
        revision = retry(lambda: api.repo_info(repo, repo_type=kind, revision=rev)).sha
        # 404 files are missing. Authentication and network failures must stop
        # preparation rather than trigger a potentially enormous re-upload.
        remote = {entry.path: entry for entry in retry(lambda: api.get_paths_info(
            repo, paths=[filename for _, filename in items], repo_type=kind, revision=revision))}
        for item, filename in items:
            if same(remote.get(filename), item):
                matched[item["path"]] = {"repo": repo, "repo_type": kind,
                                        "revision": revision, "filename": filename}
    if len(matched) < len(files):
        # Reuse prior managed bundles and the explicitly approved VSA benchmark
        # snapshot. Never search/modify the public checkpoint repository.
        candidates = []
        for prefix in (PREFIX + "/bundles", "runpod_transfer_benchmarks"):
            try:
                candidates.extend(retry(lambda: list(api.list_repo_tree(dataset_repo, path_in_repo=prefix,
                    recursive=True, repo_type="dataset", revision=private_revision))))
            except RemoteEntryNotFoundError:
                continue
        by_size = {}
        for entry in candidates:
            if hasattr(entry, "blob_id"):
                by_size.setdefault(entry.size, []).append(entry)
        for item in files:
            if item["path"] not in matched:
                other = next((entry for entry in by_size.get(item["size"], []) if same(entry, item)), None)
                if other:
                    matched[item["path"]] = {"repo": dataset_repo, "repo_type": "dataset",
                        "revision": private_revision, "filename": other.path}
    return matched


def read_json(repo, filename, revision):
    return json.loads(downloaded(repo, filename, revision).read_bytes())


def require_private(api, repo):
    info = retry(lambda: api.repo_info(repo, repo_type="dataset"))
    if info.private is not True:
        raise ValueError("새 평가 입력은 비공개 HF 데이터셋 저장소에만 업로드합니다. hf_dataset_repo를 확인하세요.")
    return info.sha


def prepare(root, config, slot, *, dry_run=False, desk_url=None):
    cache = HashCache(root / ".cache/eval_hf/hashes.json")
    settings, frozen = configuration(root, config)
    files = collect(root, settings, cache)
    recipe = {"schema": 1, "config": config.relative_to(root).as_posix(), "settings": frozen,
              "files": [{key: item[key] for key in ("path", "size", "sha256")} for item in files]}
    identifier = digest(encoded(recipe))
    bundle = f"{PREFIX}/bundles/{identifier}"
    repo = frozen["hf_dataset_repo"]
    api = HfApi()
    revision = require_private(api, repo)
    try:
        previous = read_json(repo, bundle + "/manifest.json", revision)
    except RemoteEntryNotFoundError:
        previous = None
    if previous is not None and previous.get("recipe") != recipe:
        raise ValueError("기존 HF 준비 목록이 예상 내용과 다릅니다.")
    matched = {} if previous else refs_for_files(api, root, files, repo,
        frozen["hf_checkpoint_repo"], revision)
    missing = [] if previous else [item for item in files if item["path"] not in matched]
    print(f"평가 입력: {len(files)}개 · {sum(item['size'] for item in files) / 2**30:.2f} GiB", flush=True)
    print(f"HF 재사용: {len(files) - len(missing)}개 · 새 업로드: {len(missing)}개 "
          f"({sum(item['size'] for item in missing) / 2**30:.2f} GiB)", flush=True)
    if dry_run:
        print("계획 확인 완료. Desk 업로드·평가 실행은 하지 않았습니다.")
        return
    desk = connect(root, repo, desk_url)
    # Never modify files which a previously detached Desk job may still read.
    desk.wait_pending(PREFIX)
    if previous:
        manifest = previous
    else:
        manifest = {"recipe": recipe, "files": [
            {**{key: item[key] for key in ("path", "size", "sha256")},
             "hf": matched.get(item["path"], {"source": "bundle",
                        "filename": bundle + "/assets/" + item["path"]})}
            for item in files]}
        stage = desk.base / bundle
        stage.mkdir(parents=True, exist_ok=True)
        for item in missing:
            source, target = root / item["path"], stage / "assets" / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source.resolve(), target)
                except OSError:
                    shutil.copyfile(source, target)
            # Detect changed or leftover staging before submitting an upload.
            if cache.hash(target)[0] != item["sha256"]:
                raise ValueError("준비 폴더 파일이 변경됐습니다. .cache/eval_hf와 Desk 작업을 확인하세요.")
        atomic_write(stage / "manifest.json", encoded(manifest))
        require_private(api, repo)
        desk.publish(bundle)
        revision = require_private(api, repo)
        actual = read_json(repo, bundle + "/manifest.json", revision)
        if actual != manifest:
            raise ValueError("HF 준비 목록 업로드 검증에 실패했습니다.")
        remote = {entry.path: entry for entry in retry(lambda: api.get_paths_info(repo,
            paths=[bundle + "/assets/" + item["path"] for item in missing],
            repo_type="dataset", revision=revision))} if missing else {}
        if any(not same(remote.get(bundle + "/assets/" + item["path"]), item) for item in missing):
            raise ValueError("업로드된 파일 해시가 원본과 다릅니다. 최신 평가 목록을 변경하지 않았습니다.")
    pointer = {"schema": 1, "bundle": bundle, "revision": revision,
               "manifest_sha256": digest(encoded(manifest))}
    ref = f"{PREFIX}/refs/{slot}.json"
    try:
        existing = read_json(repo, ref, require_private(api, repo))
    except RemoteEntryNotFoundError:
        existing = None
    if existing and existing.get("bundle") == bundle and existing.get("manifest_sha256") == pointer["manifest_sha256"]:
        print("같은 평가 준비 목록이 이미 HF에 있습니다.", flush=True)
    else:
        atomic_write(desk.base / ref, encoded(pointer))
        require_private(api, repo)
        desk.publish(ref)
        if read_json(repo, ref, require_private(api, repo)) != pointer:
            raise ValueError("HF 최신 평가 목록 검증에 실패했습니다.")
    # Only remove our immutable bundle after BOTH Desk jobs are complete.
    shutil.rmtree(desk.base / bundle, ignore_errors=True)
    cache.save()
    command = f"bash {config.parent.relative_to(root)}/submit_eval.sh"
    if config.name not in {"ft_eval_config.yaml", "stage1_eval_config.yaml"}:
        command = f"STAGE1_EVAL_CONFIG={recipe['config']} " + command
    print(f"준비 완료 ({slot}). RunPod에서 {command} 를 실행하세요.", flush=True)


def validate_manifest(manifest, config):
    recipe = manifest["recipe"]
    if recipe.get("schema") != 1 or recipe["config"] != config:
        raise ValueError("평가 준비 목록의 형식 또는 설정 경로가 다릅니다.")
    files = manifest["files"]
    if [{key: item[key] for key in ("path", "size", "sha256")} for item in files] != recipe["files"]:
        raise ValueError("평가 준비 목록의 파일 정보가 다릅니다.")
    names = set()
    for item in files:
        asset_path(item["path"])
        if item["path"] in names or not isinstance(item["size"], int) or item["size"] < 0:
            raise ValueError("평가 입력 파일 목록이 올바르지 않습니다.")
        names.add(item["path"])
        if not re.fullmatch("[0-9a-f]{64}", item["sha256"]):
            raise ValueError("평가 입력 해시가 올바르지 않습니다.")
        hf = item["hf"]
        relative(hf["filename"])
        if hf.get("source") != "bundle":
            if hf.get("repo_type") not in {"model", "dataset"} or not re.fullmatch("[0-9a-f]{40}", hf["revision"]):
                raise ValueError("HF 입력은 고정된 커밋으로 지정해야 합니다.")
    return files


def consume(root, config, slot, *, download_only=False, dry_run=False, workers=8):
    loader = runpy.run_path(str(root / CONFIGS / "src/global_config_loader.py"))["load_global_config"]
    repo = loader(root / CONFIGS / "global_config.yaml")["hf_dataset_repo"]
    revision = require_private(HfApi(), repo)
    try:
        pointer = read_json(repo, f"{PREFIX}/refs/{slot}.json", revision)
    except RemoteEntryNotFoundError:
        raise ValueError("HF에 준비된 평가 목록이 없습니다. 연구실 서버에서 같은 명령을 먼저 실행하세요.") from None
    bundle = pointer["bundle"]
    if (pointer.get("schema") != 1 or not re.fullmatch(PREFIX + r"/bundles/[0-9a-f]{64}", bundle)
            or not re.fullmatch("[0-9a-f]{40}", pointer["revision"])):
        raise ValueError("HF 평가 참조 형식이 올바르지 않습니다.")
    manifest = read_json(repo, bundle + "/manifest.json", pointer["revision"])
    if digest(encoded(manifest)) != pointer["manifest_sha256"]:
        raise ValueError("HF 평가 준비 목록 해시 검증에 실패했습니다.")
    files = validate_manifest(manifest, config.relative_to(root).as_posix())
    print(f"HF 평가 입력: {len(files)}개 · {sum(item['size'] for item in files) / 2**30:.2f} GiB", flush=True)
    if dry_run:
        print("계획 확인 완료. 평가 입력 다운로드·평가 실행은 하지 않았습니다.")
        return
    cache = HashCache(root / ".cache/eval_hf/hashes.json")
    started = time.monotonic()

    def one(item):
        target = root / item["path"]
        if target.exists():
            if target.is_file() and target.stat().st_size == item["size"] and cache.hash(target)[0] == item["sha256"]:
                return "cached"
            raise ValueError(f"로컬 파일이 HF 입력과 다릅니다. 파일을 별도로 옮긴 뒤 재실행하세요: {item['path']}")
        hf = item["hf"]
        if hf.get("source") == "bundle":
            if not hf["filename"].startswith(bundle + "/assets/"):
                raise ValueError("평가 번들 파일 경로가 다릅니다.")
            hf = {"repo": repo, "repo_type": "dataset", "revision": pointer["revision"], "filename": hf["filename"]}
        incoming = root / ".cache/eval_hf/downloads" / digest(encoded(hf))
        source = downloaded(hf["repo"], hf["filename"], hf["revision"],
                            repo_type=hf["repo_type"], local_dir=incoming)
        if source.stat().st_size != item["size"] or cache.hash(source)[0] != item["sha256"]:
            raise ValueError(f"HF 파일 해시가 원본과 다릅니다: {item['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Move out of a dedicated local_dir, avoiding a second full copy in the
        # shared HF cache. Cross-filesystem destinations need a temporary copy.
        fd, temp = tempfile.mkstemp(prefix=".eval-hf-", dir=target.parent)
        os.close(fd)
        try:
            try:
                os.replace(source, temp)
            except OSError:
                shutil.copyfile(source, temp)
            try:
                os.link(temp, target)
            except FileExistsError:
                if cache.hash(target)[0] != item["sha256"]:
                    raise ValueError("다운로드 중 대상 파일이 변경됐습니다.") from None
        finally:
            Path(temp).unlink(missing_ok=True)
        shutil.rmtree(incoming, ignore_errors=True)
        cache.hash(target)
        print(f"  완료: {PurePosixPath(item['path']).name}", flush=True)
        return "downloaded"

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, files))
    finally:
        cache.save()
    identifier = bundle.rsplit("/", 1)[-1]
    snapshot = config.parent / ".hf_configs" / identifier / config.name
    atomic_write(snapshot, yaml.safe_dump(manifest["recipe"]["settings"], allow_unicode=True, sort_keys=False).encode())
    print(f"입력 준비 완료: 다운로드 {results.count('downloaded')}개, 로컬 재사용 {results.count('cached')}개 "
          f"· {time.monotonic() - started:.1f}초", flush=True)
    if download_only:
        print(f"평가 설정: {snapshot}")
        return
    launcher = root / EVALS["pt" if slot == "pt" or "/PT/" in str(config) else "ft"].parent / "submit_eval.sh"
    env = os.environ.copy()
    env["STAGE1_EVAL_CONFIG"] = str(snapshot)
    env["SBD_EVAL_HF_READY"] = "1"
    # Retain the existing scheduler/local runner and GPU packing behavior.
    subprocess.run(["bash", str(launcher)], cwd=root, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="서버: HF Desk로 평가 준비 / RunPod: HF 다운로드 후 평가")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pt", action="store_true", help="PT 평가 (기본: FT)")
    group.add_argument("--config", type=Path, help="FT/eval 또는 PT/eval 안의 평가 YAML")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", "--prepare-runpod", action="store_true", help="서버에서 HF 준비만 수행")
    mode.add_argument("--download-only", action="store_true", help="HF 입력만 다운로드, 평가 실행 생략")
    parser.add_argument("--dry-run", action="store_true", help="필요 파일과 계획만 확인")
    parser.add_argument("--desk-url", help="기존 HF Desk 주소 (기본: 자동 검색/시작)")
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 33), metavar="1..32")
    args = parser.parse_args(argv)
    config = ROOT / (args.config or EVALS["pt" if args.pt else "ft"])
    # Absolute paths are fine only within the same repository's two eval trees.
    logical = config.relative_to(ROOT)
    if logical.suffix not in {".yaml", ".yml"} or not any(logical.is_relative_to(p.parent) for p in EVALS.values()):
        raise ValueError("--config는 SkillVLA FT/eval 또는 PT/eval 안의 YAML을 지정하세요.")
    if ".." in logical.parts:
        raise ValueError("설정 경로에 .. 을 사용할 수 없습니다.")
    slot = next((name for name, path in EVALS.items() if logical == path),
                "custom-" + digest(str(logical).encode())[:16])
    load_global = runpy.run_path(str(ROOT / CONFIGS / "src/global_config_loader.py"))["load_global_config"]
    pod = load_global(ROOT / CONFIGS / "global_config.yaml").get("server") == "runpod"
    if not args.prepare and (pod or args.download_only):
        lock = ROOT / ".cache/eval_hf/download.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            consume(ROOT, config, slot, download_only=args.download_only, dry_run=args.dry_run, workers=args.workers)
    else:
        lock = ROOT / ".cache/eval_hf/prepare.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("w") as stream:
            print("평가 준비 잠금 대기...", flush=True)
            fcntl.flock(stream, fcntl.LOCK_EX)
            prepare(ROOT, config, slot, dry_run=args.dry_run, desk_url=args.desk_url)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n대기를 중단했습니다. 이미 접수된 업로드는 HF Desk에서 계속됩니다. 같은 명령으로 재확인하세요.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, KeyError, httpx.HTTPError, EntryNotFoundError, subprocess.CalledProcessError) as error:
        # SDK exception text can contain signed URLs. Keep those out of CLI logs.
        message = str(error) if isinstance(error, ValueError) and not isinstance(error, httpx.HTTPError) else type(error).__name__
        print(f"ERROR: {message}\nHF 인증·네트워크 또는 HF Desk 작업 상세를 확인하세요.", file=sys.stderr)
        raise SystemExit(1)
