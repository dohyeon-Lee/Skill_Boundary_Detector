#!/usr/bin/env python3
"""Preview/delete selected Hub files at a pinned revision; never touch local data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import PurePosixPath

from huggingface_hub import CommitOperationDelete, HfApi
from huggingface_hub.hf_api import RepoFolder
from huggingface_hub.utils import validate_repo_id


def checked_paths(paths, repo_type):
    if repo_type not in {"dataset", "model"}:
        raise ValueError("데이터셋/체크포인트 저장소만 관리할 수 있습니다.")
    if not isinstance(paths, list) or not 1 <= len(paths) <= 200:
        raise ValueError("삭제할 파일·폴더를 1~200개 선택하세요.")
    for path in paths:
        if not isinstance(path, str) or not path or len(path) > 4096 or any(
            part in {"", ".", ".."} for part in path.split("/")
        ) or any(char in path for char in "\x00*?\\"):
            raise ValueError("삭제할 경로는 루트 내부의 정확한 상대 경로여야 합니다.")
        root = PurePosixPath(path).parts[0]
        prefix = "dataset" if repo_type == "dataset" else "outputs"
        if root != prefix and not root.startswith(prefix + "_"):
            raise ValueError("데이터셋/체크포인트 폴더 안의 파일만 삭제할 수 있습니다.")
    result = []
    for path in sorted(set(paths), key=lambda value: (value.count("/"), value)):
        if not any(path.startswith(parent + "/") for parent in result):
            result.append(path)
    return result


def preview_delete(api, repo, repo_type, paths, revision=None):
    validate_repo_id(repo)
    paths = checked_paths(paths, repo_type)
    revision = revision or api.repo_info(repo, repo_type=repo_type, revision="main").sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("삭제 미리보기의 저장소 버전을 확인할 수 없습니다.")
    entries = api.get_paths_info(repo, paths, repo_type=repo_type, revision=revision)
    if {entry.path for entry in entries} != set(paths):
        raise ValueError("선택한 Hub 항목이 없거나 변경되었습니다. 목록을 새로고침하세요.")
    files = {}
    for entry in entries:
        children = api.list_repo_tree(repo, path_in_repo=entry.path, repo_type=repo_type,
                                      recursive=True, revision=revision) if isinstance(entry, RepoFolder) else [entry]
        for child in children:
            if isinstance(child, RepoFolder):
                continue
            if not any(child.path == path or child.path.startswith(path + "/") for path in paths):
                raise ValueError("선택 범위 밖의 Hub 파일이 반환되었습니다.")
            files[child.path] = child.size
    if not files:
        raise ValueError("선택한 경로에 삭제할 파일이 없습니다.")
    return {"repo": repo, "repo_type": repo_type, "paths": paths, "revision": revision,
            "files": sorted(files), "file_count": len(files), "size": sum(files.values())}


def delete_selected(api, repo, repo_type, paths, revision, *, dry_run=False):
    # Re-read the exact immutable preview revision. A later upload is never added
    # to this selection; parent_commit rejects a changed branch atomically.
    plan = preview_delete(api, repo, repo_type, paths, revision)
    print(f"Hub 삭제 대상: {repo} · {plan['file_count']}개 파일 · {plan['size'] / 1024**3:.2f} GiB", flush=True)
    for path in plan["paths"]:
        print("  " + path, flush=True)
    if dry_run:
        print("[dry-run] 원격 파일을 삭제하지 않았습니다.", flush=True)
        return None
    result = api.create_commit(repo_id=repo, repo_type=repo_type, revision="main", parent_commit=revision,
        operations=[CommitOperationDelete(path_in_repo=path, is_folder=False) for path in plan["files"]],
        commit_message=f"Delete {plan['file_count']} selected files via SBD Hugging Face Desk")
    print("Hub 삭제 완료: " + str(result.commit_url), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", required=True, choices=("dataset", "model"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--paths-json", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        delete_selected(HfApi(), args.repo, args.repo_type, json.loads(args.paths_json), args.revision, dry_run=args.dry_run)
    except Exception as error:
        sys.exit("Hub 삭제 실패: " + str(error) + "\n저장소가 변경됐다면 삭제 미리보기를 다시 준비하세요.")


if __name__ == "__main__":
    main()
