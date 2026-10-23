#!/usr/bin/env python3
"""Local browser UI for hf_sync.py; transfers run in cancellable child processes."""

from __future__ import annotations

import argparse
import errno
import itertools
import json
import os
import runpy
import secrets
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFolder
from huggingface_hub.utils import validate_repo_id

from fs_helper import relative_path
from hf_delete import checked_paths, preview_delete
from server import HERE, REPO, Handler, Transfers

SCRIPT = REPO / "lerobot/examples/libero/configs/src/hf_sync.py"
SYNC = runpy.run_path(str(SCRIPT))
CHECKPOINTS = runpy.run_path(str(SCRIPT.with_name("hf_checkpoints.py")))
ROUTES = {
    "dataset-pull": ("hub-datasets", "local-datasets", "데이터셋 다운로드"),
    "dataset-push": ("local-datasets", "hub-datasets", "데이터셋 업로드"),
    "models-pull": ("hub-models", "local-models", "사전학습 모델 다운로드"),
    "checkpoints-pull": ("hub-checkpoints", "local-checkpoints", "체크포인트 다운로드"),
    "watch": ("local-checkpoints", "hub-checkpoints", "체크포인트 자동 업로드"),
    "dataset-delete": ("hub-datasets", "hub-datasets", "HF 데이터셋 관리 · 삭제"),
    "checkpoints-delete": ("hub-checkpoints", "hub-checkpoints", "HF 체크포인트 관리 · 삭제"),
}


class HFBackend:
    command_label = "Hugging Face"
    completion_detail = "완료 · Hugging Face 동기화가 정상 종료되었습니다."

    def __init__(self, config, *, server=None, repo=None, checkpoint_repo=None, api=None):
        self.config, self.server = config, server
        self.base = SYNC["local_base"](config)
        self.project = Path(str(config["project_root"])).expanduser()
        self.repo = (repo or os.environ.get(SYNC["REPO_ENV"]) or str(config.get("hf_dataset_repo") or "")).strip()
        self.checkpoint_repo = (checkpoint_repo or os.environ.get(SYNC["CHECKPOINT_REPO_ENV"])
                                or str(config.get("hf_checkpoint_repo") or "")).strip()
        for value in (self.repo, self.checkpoint_repo):
            if value:
                validate_repo_id(value)
        self.api = api or HfApi()
        self.delete_previews = {}
        self.delete_lock = threading.Lock()

    def public_config(self):
        dataset_url = "https://huggingface.co/datasets/" + self.repo
        checkpoint_url = "https://huggingface.co/" + self.checkpoint_repo
        return {"mode": self.config.get("server", "local"), "authenticated": bool(get_token()),
            "routes": [{"id": key, "source": source, "destination": destination, "name": name}
                       for key, (source, destination, name) in ROUTES.items()],
            "endpoints": [
                {"id": "local-datasets", "name": "이 서버 · 데이터셋", "base": str(self.base)},
                {"id": "hub-datasets", "name": "Hugging Face · 비공개 데이터셋", "base": dataset_url},
                {"id": "hub-models", "name": "Hugging Face · 원본 모델 저장소", "base": "등록된 사전학습 모델"},
                {"id": "local-models", "name": "이 서버 · 사전학습 모델", "base": str(self.base / "models")},
                {"id": "hub-checkpoints", "name": "Hugging Face · 공개 체크포인트", "base": checkpoint_url},
                {"id": "local-checkpoints", "name": "이 서버 · 학습 결과", "base": str(self.project)},
            ]}

    @staticmethod
    def validate_path(path, kind, allow_root=False):
        path = relative_path(path, allow_root=allow_root)
        if not path:
            return path
        predicate = SYNC["is_dataset_root"] if kind == "dataset" else SYNC["is_outputs_root"]
        return SYNC["check_paths"]([path], predicate)[0]

    def local_path(self, path):
        path = self.validate_path(path, "dataset")
        # Dataset roots may themselves link to a mounted storage volume. Links below
        # that root must stay inside it, matching the CLI's supported volume layout.
        parts = Path(path).parts
        root = (self.base / parts[0]).resolve()
        target = (self.base / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("데이터셋 루트 밖을 가리키는 링크는 선택할 수 없습니다.")
        return target

    def browse(self, identifier, path="", offset=0, search=""):
        offset = max(0, min(100000, int(offset)))
        needle = search.strip().casefold()
        if identifier == "hub-models":
            if path:
                raise ValueError("모델은 폴더 단위로 선택하세요.")
            entries = [{"name": name, "path": name, "kind": "file", "size": None,
                        "description": spec["repo"]} for name, spec in SYNC["PRETRAINED_MODELS"].items()]
        elif identifier == "local-datasets":
            path = self.validate_path(path, "dataset", allow_root=True)
            directory = self.local_path(path) if path else self.base
            if not directory.is_dir():
                raise ValueError("폴더가 없습니다: " + str(directory))
            entries = []
            for child in directory.iterdir():
                if child.name.startswith(".") or (not path and not SYNC["is_dataset_root"](child.name)):
                    continue
                rel = path + "/" + child.name if path else child.name
                try:
                    target = self.local_path(rel)
                    kind = "directory" if target.is_dir() else "file" if target.is_file() else "special"
                    entries.append({"name": child.name, "path": rel, "kind": kind,
                                    "size": target.stat().st_size if kind == "file" else None})
                except (ValueError, FileNotFoundError):
                    continue
        elif identifier in {"hub-datasets", "hub-checkpoints"}:
            return self.browse_hub(identifier, path, offset, needle)
        else:
            raise ValueError("지원하지 않는 탐색 위치입니다.")
        entries = [entry for entry in entries if needle in entry["name"].casefold()]
        entries.sort(key=lambda entry: (entry["kind"] != "directory", entry["name"].casefold()))
        return {"path": path, "entries": entries[offset:offset + 200], "offset": offset,
                "limit": 200, "total": len(entries), "has_more": offset + 200 < len(entries)}

    def browse_hub(self, identifier, path, offset, needle):
        kind = "dataset" if identifier == "hub-datasets" else "model"
        path = self.validate_path(path, kind, allow_root=True)
        requested_path = path
        repo = self.repo if kind == "dataset" else self.checkpoint_repo
        if not repo:
            raise ValueError("저장소가 설정되지 않았습니다. --repo / --checkpoint-repo를 지정하세요.")
        predicate = SYNC["is_dataset_root"] if kind == "dataset" else SYNC["is_outputs_root"]
        while True:
            try:
                tree = self.api.list_repo_tree(repo, path_in_repo=path or None, repo_type=kind, recursive=False)
                matches = ({"name": entry.path.rsplit("/", 1)[-1], "path": entry.path,
                            "kind": "directory" if isinstance(entry, RepoFolder) else "file",
                            "size": getattr(entry, "size", None)} for entry in tree
                           if (path or predicate(entry.path)) and needle in entry.path.rsplit("/", 1)[-1].casefold())
                entries = list(itertools.islice(matches, offset, offset + 201))
                return {"path": path, "entries": entries[:200], "offset": offset, "limit": 200,
                        "total": offset + len(entries), "has_more": len(entries) > 200,
                        "recovered_from": requested_path if path != requested_path else None}
            except EntryNotFoundError as error:
                # Hub folders only exist while they contain files. Deleting the
                # last subtree can remove several ancestor folders at once.
                if not path:
                    raise ValueError("Hub 루트 목록을 찾을 수 없습니다: " + str(error)) from error
                path = path.rpartition("/")[0]
                offset, needle = 0, ""
            except Exception as error:
                raise ValueError("Hub 목록 조회 실패. hf auth login과 저장소 권한을 확인하세요: " + str(error)) from error

    def watch_options(self, values):
        if not isinstance(values, dict):
            raise ValueError("자동 업로드 설정이 올바르지 않습니다.")
        result = {}
        for key, default, minimum, maximum in (("interval", 300, 10, 86400), ("keep", 0, 0, 1000000)):
            value = values.get(key, default)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{key}: {minimum}~{maximum} 정수가 필요합니다.")
            result[key] = value
        root = values.get("root", "")
        if root:
            root = self.validate_path(root, "model")
            if "/" in root:
                raise ValueError("감시 루트는 outputs_filtered 같은 최상위 폴더여야 합니다.")
        result["root"] = root
        protect = values.get("protect", "")
        if not isinstance(protect, str) or len(protect) > 4096:
            raise ValueError("보호할 step은 쉼표로 구분한 숫자입니다.")
        parts = [part.strip() for part in protect.split(",") if part.strip()]
        if any(not part.isascii() or not part.isdigit() for part in parts):
            raise ValueError("보호할 step은 쉼표로 구분한 숫자입니다.")
        result["protect"] = ",".join(parts)
        for key in ("once", "prune_local", "squash"):
            value = values.get(key, False)
            if type(value) is not bool:
                raise ValueError(key + "는 true/false여야 합니다.")
            result[key] = value
        return result

    def plan(self, source_id, destination_id, paths, options=None, dry_run=False):
        if type(dry_run) is not bool:
            raise ValueError("dry_run은 true/false여야 합니다.")
        route = next((key for key, pair in ROUTES.items() if pair[:2] == (source_id, destination_id)), None)
        if route is None:
            raise ValueError("지원하지 않는 전송 방향입니다.")
        if not isinstance(paths, list) or not 1 <= len(paths) <= 200:
            raise ValueError("1~200개 항목을 선택하세요.")
        if route.endswith("-delete"):
            return self.plan_delete(route, paths, dry_run)
        if route == "watch":
            options = self.watch_options(options or {})
            paths = [options["root"] or "outputs*"]
        elif route == "models-pull":
            if any(not isinstance(path, str) or path not in SYNC["PRETRAINED_MODELS"] for path in paths):
                raise ValueError("등록된 사전학습 모델을 선택하세요.")
            options = {}
        else:
            kind = "model" if route == "checkpoints-pull" else "dataset"
            paths = [self.validate_path(path, kind) for path in paths]
            options = {}
        unique = []
        for path in sorted(set(paths), key=lambda value: (value.count("/"), value)):
            if not any(path.startswith(parent + "/") for parent in unique):
                unique.append(path)
        endpoint = {item["id"]: item for item in self.public_config()["endpoints"]}
        repo = self.checkpoint_repo if route in {"watch", "checkpoints-pull"} else self.repo
        if route != "models-pull" and not repo:
            raise ValueError("대상 Hugging Face 저장소가 설정되지 않았습니다.")
        plans = []
        for path in unique:
            if route == "dataset-push" and not self.local_path(path).exists():
                raise ValueError("출발 항목이 없습니다: " + path)
            if route in {"dataset-pull", "checkpoints-pull"}:
                try:
                    present = self.api.get_paths_info(repo, [path], repo_type="model" if route == "checkpoints-pull" else "dataset")
                except Exception as error:
                    raise ValueError("Hub 경로 확인 실패: " + str(error)) from error
                if not present:
                    raise ValueError("Hub에 항목이 없습니다: " + path)
            source = endpoint[source_id]["base"] + "/" + path
            destination = endpoint[destination_id]["base"] + "/" + path
            if route == "models-pull":
                source = "https://huggingface.co/" + SYNC["PRETRAINED_MODELS"][path]["repo"]
            if route == "watch":
                source = str(self.project / options["root"]) if options["root"] else str(self.project / "outputs*")
                destination = endpoint[destination_id]["base"]
            plans.append({"source_id": source_id, "destination_id": destination_id, "path": path,
                "source": source, "destination": destination, "destination_host": "",
                "source_name": endpoint[source_id]["name"], "destination_name": endpoint[destination_id]["name"],
                "route": route, "options": options, "dry_run": dry_run})
        notice = "같은 상대 경로로 전송합니다. 같은 이름의 파일은 갱신될 수 있습니다."
        if route == "dataset-push":
            notice += " 데이터셋 저장소는 최초 생성 시 비공개입니다."
        if route == "watch":
            notice = "체크포인트는 공개 저장소에 업로드됩니다. "
            notice += f"최근 {options['keep']}개 + 보호 step 외의 Hub 체크포인트를 삭제합니다. " if options["keep"] else "모든 step을 보관합니다. "
            if options["prune_local"] and options["keep"]:
                notice += "로컬의 오래된 step도 삭제합니다. "
            if options["squash"]:
                notice += "Hub 기록을 압축하며 되돌릴 수 없습니다. "
            notice += "Pod 자동 종료는 활성화하지 않습니다."
        if dry_run:
            notice = "미리보기 실행: 업로드·다운로드·삭제하지 않고 실행 계획만 로그에 출력합니다. " + notice
        return {"items": plans, "collapsed": len(paths) - len(unique), "notice": notice}

    def plan_delete(self, route, paths, dry_run):
        kind = "dataset" if route == "dataset-delete" else "model"
        repo = self.repo if kind == "dataset" else self.checkpoint_repo
        if not repo:
            raise ValueError("관리할 Hugging Face 저장소가 설정되지 않았습니다.")
        try:
            deletion = preview_delete(self.api, repo, kind, paths)
        except Exception as error:
            raise ValueError("삭제 미리보기 실패: " + str(error)) from error
        url = "https://huggingface.co/" + ("datasets/" if kind == "dataset" else "") + repo
        source = ROUTES[route][0]
        item = {"source_id": source, "destination_id": source, "route": route, "options": {},
            "source_name": "Hugging Face · " + repo, "destination_name": "Hub에서 삭제",
            "source": url, "destination": url, "destination_host": "", "dry_run": dry_run,
            "path": deletion["paths"][0] if len(deletion["paths"]) == 1 else f"선택한 {len(deletion['paths'])}개 항목 삭제",
            "paths": deletion["paths"], "repo": repo, "repo_type": kind, "revision": deletion["revision"],
            "file_count": deletion["file_count"], "size": deletion["size"], "sample_files": deletion["files"][:20]}
        plan = {"items": [item], "collapsed": len(paths) - len(deletion["paths"]),
            "notice": "선택한 파일·폴더를 Hugging Face에서 삭제합니다. 로컬 파일은 유지됩니다. "
                      "Hub의 이전 기록에 파일이 남아 저장 용량이 즉시 줄지 않을 수 있습니다."}
        if dry_run:
            plan["notice"] = "미리보기 실행: 삭제 계획만 출력하며 원격 파일을 삭제하지 않습니다. " + plan["notice"]
        with self.delete_lock:
            now = time.monotonic()
            self.delete_previews = {key: value for key, value in self.delete_previews.items() if value["expires"] > now}
            if len(self.delete_previews) >= 64:
                self.delete_previews.pop(next(iter(self.delete_previews)))
            token = secrets.token_urlsafe(24)
            self.delete_previews[token] = {"plan": plan, "expires": now + 600, "source": source,
                                           "paths": checked_paths(paths, kind), "dry_run": dry_run}
        return {**plan, "plan_id": token}

    def confirmed_delete(self, body):
        """Called under delete_lock: bind execution to the exact reviewed scope."""
        token = body.get("plan_id")
        preview = self.delete_previews.get(token) if isinstance(token, str) else None
        if preview is None or preview["expires"] <= time.monotonic():
            raise ValueError("삭제 미리보기가 만료됐습니다. 삭제 준비를 다시 눌러주세요.")
        if body.get("source") != preview["source"] or body.get("destination") != preview["source"] or (
            body.get("dry_run", False) is not preview["dry_run"]
        ) or checked_paths(body.get("paths"), preview["plan"]["items"][0]["repo_type"]) != preview["paths"]:
            raise ValueError("확인한 삭제 대상과 요청이 다릅니다. 삭제 준비를 다시 눌러주세요.")
        if not preview["dry_run"] and body.get("confirm_delete") is not True:
            raise ValueError("Hub 삭제 확인란을 체크하세요.")
        return preview["plan"]

    def commands(self, plan):
        if plan["route"].endswith("-delete"):
            command = [sys.executable, "-u", str(HERE / "hf_delete.py"), "--repo", plan["repo"],
                "--repo-type", plan["repo_type"], "--revision", plan["revision"],
                "--paths-json", json.dumps(plan["paths"], ensure_ascii=False)]
            return command + (["--dry-run"] if plan["dry_run"] else [])
        # Revalidate queued selections before launch; invoke existing CLI without a shell.
        item = self.plan(plan["source_id"], plan["destination_id"], [plan["path"]],
                         plan["options"], plan["dry_run"])["items"][0]
        route, path = item["route"], item["path"]
        command = [sys.executable, "-u", str(SCRIPT)]
        if route == "dataset-push":
            command += ["push", path]
        elif route == "dataset-pull":
            command += ["pull", path]
        elif route == "models-pull":
            command += ["pull", "--models", path]
        elif route == "checkpoints-pull":
            command += ["pull", "--checkpoints", path]
        else:
            options = item["options"]
            command += ["watch", "--interval", str(options["interval"]), "--keep", str(options["keep"]),
                        "--prune-local" if options["prune_local"] else "--no-prune-local"]
            if options["root"]:
                command += ["--root", options["root"]]
            if options["protect"]:
                command += ["--protect", options["protect"]]
            for flag in ("once", "squash"):
                if options[flag]:
                    command += ["--" + flag]
            if item["dry_run"]:
                command += ["--once"]
        if self.server:
            command += ["--server", self.server]
        if self.repo:
            command += ["--repo", self.repo]
        if self.checkpoint_repo:
            command += ["--checkpoint-repo", self.checkpoint_repo]
        command += ["--yes"]
        if item["dry_run"]:
            command += ["--dry-run"]
        return command


class HFTransfers(Transfers):
    def __init__(self, backend, concurrency=2):
        self.checkpoint_download = threading.Lock()
        super().__init__(backend, concurrency)

    def _execute(self, job):
        if job["route"] != "checkpoints-pull":
            return super()._execute(job)
        # The CLI updates shared checkpoints/last and .hf_pulled metadata after a
        # download. Serialize checkpoint pulls so those read/modify/writes cannot race.
        while not self.checkpoint_download.acquire(timeout=0.2):
            if job["cancel"]:
                return
        try:
            if not job["cancel"]:
                super()._execute(job)
        finally:
            self.checkpoint_download.release()


class HFHandler(Handler):
    def do_GET(self):
        try:
            self._guard()
            request = urlsplit(self.path)
            params = parse_qs(request.query)
            if request.path == "/api/config":
                self._reply(200, {**self.server.backend.public_config(), "token": self.server.token,
                                  "concurrency": self.server.manager.concurrency})
            elif request.path == "/api/browse":
                self._reply(200, self.server.backend.browse(params.get("endpoint", [""])[0],
                    params.get("path", [""])[0], int(params.get("offset", [0])[0]), params.get("search", [""])[0]))
            elif request.path in {"/", "/app.js", "/hf.css"}:
                name = {"/": "hf_index.html", "/app.js": "hf_app.js", "/hf.css": "hf.css"}[request.path]
                mime = {"/": "text/html", "/app.js": "text/javascript", "/hf.css": "text/css"}[request.path]
                self._reply(200, (HERE / "static" / name).read_bytes(), mime + "; charset=utf-8")
            else:
                super().do_GET()
        except (ValueError, OSError) as error:
            self._error(error)

    def do_POST(self):
        if urlsplit(self.path).path not in {"/api/plan", "/api/transfers"}:
            return super().do_POST()
        try:
            self._guard(mutation=True)
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("Request body must be 1~65536 bytes")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            is_deleting = any(route.endswith("-delete") and pair[:2] == (body.get("source"), body.get("destination"))
                              for route, pair in ROUTES.items())
            if urlsplit(self.path).path == "/api/transfers" and is_deleting:
                with self.server.backend.delete_lock:
                    plan = self.server.backend.confirmed_delete(body)
                    identifiers = self.server.manager.submit(plan["items"])
                    self.server.backend.delete_previews.pop(body["plan_id"])
                self._reply(202, {"ids": identifiers})
                return
            plan = self.server.backend.plan(body.get("source"), body.get("destination"), body.get("paths"),
                                            body.get("options"), body.get("dry_run", False))
            if urlsplit(self.path).path == "/api/plan":
                self._reply(200, plan)
            else:
                self._reply(202, {"ids": self.server.manager.submit(plan["items"])})
        except (ValueError, OSError) as error:
            self._error(error)


class HFHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend, manager):
        self.backend, self.manager = backend, manager
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, HFHandler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("HF_SYNC_UI_PORT", "8766")))
    parser.add_argument("--server", default=None, help="override server detection (like hf_sync.sh)")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--checkpoint-repo", default=None)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port=1~65535 required")
    backend = HFBackend(SYNC["load_config"](args.server), server=args.server, repo=args.repo,
                        checkpoint_repo=args.checkpoint_repo)
    manager = HFTransfers(backend, args.concurrency)
    try:
        server = HFHTTPServer(("127.0.0.1", args.port), backend, manager)
    except OSError as error:
        manager.close()
        if error.errno == errno.EADDRINUSE:
            parser.exit(2, f"포트 {args.port}를 이미 사용 중입니다. 실행 중인 UI를 열거나, "
                        f"bash hf_sync_ui.sh --port {args.port + 1 if args.port < 65535 else 8766} 로 다른 포트를 사용하세요.\n")
        raise
    print(f"SBD Hugging Face Desk · {backend.config.get('server', 'local')}", flush=True)
    print(f"Local URL: http://127.0.0.1:{args.port}", flush=True)
    print(f"VSCode Ports에서 {args.port}를 전달하세요. 로그인: hf auth login 또는 HF_TOKEN", flush=True)
    print("브라우저를 닫아도 계속 실행됩니다. Ctrl+C: 서버 종료 및 실행/대기 작업 취소", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
