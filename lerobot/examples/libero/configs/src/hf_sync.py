#!/usr/bin/env python3
"""Pick dataset folders and push them to / pull them from a private Hugging Face dataset repo that
mirrors the project's dataset roots — sync_server.sh for the Hub.

    bash hf_sync.sh                          # interactive: push or pull, pick a dataset root, drill down
    bash hf_sync.sh push dataset_filtered/libero_90_full_full dataset_filtered/skillvla_dataset/libero_90_full_full/eval_init_states.npz
    bash hf_sync.sh pull dataset_filtered/libero_90_full_full --yes
    bash hf_sync.sh pull --models            # pretrained models from their original repos -> models/
    bash hf_sync.sh push --dry-run ...       # show the plan only

The repo keeps the dataset roots as top-level folders (``dataset_filtered/``, ``dataset_calvin/``,
...), so one repo holds every root and a pull lands exactly where the code expects it. Paths start
with such a root and are relative to the project root — or to ``storage_volume`` on servers with a
storage volume (RunPod: /workspace-global), after which the checkout links are refreshed.

Repo: ``hf_dataset_repo`` in configs/global_config.yaml (or $SBD_HF_DATASET_REPO); it is created
private on the first push. Log in first: ``hf auth login`` (or export HF_TOKEN). Re-running an
interrupted push is cheap: already-uploaded data is deduplicated and not sent again.

Pretrained models (PRETRAINED_MODELS) are never re-uploaded: pull fetches them from their original
repos into ``<base>/models/<folder>``. Gated ones (PaliGemma, DINOv3) need the terms accepted once
on their Hub page with the same account.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ENV = "SBD_HF_DATASET_REPO"
RULE = "━" * 50

# models/<folder> <- original Hub repo. Revisions pin the versions whose files match the copies on the
# Yonsei cluster (pi05_base weights and the PaliGemma tokenizer files compared by size); None = main.
PRETRAINED_MODELS = {
    "pi05_base": {"repo": "lerobot/pi05_base", "revision": "7de663972b7817d2c4cf2d84c821153dfea772e9"},
    "paligemma-3b-pt-224-tokenizer": {
        "repo": "google/paligemma-3b-pt-224",
        "revision": "35e4f46485b4d07967e7e9935bc3786aad50687c",
        "files": ["added_tokens.json", "config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
    },
    "dinov3-vits16": {"repo": "facebook/dinov3-vits16-pretrain-lvd1689m", "revision": None},
    "dinov3-vitl16": {"repo": "facebook/dinov3-vitl16-pretrain-lvd1689m", "revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51"},
}


def load_config(server: str | None = None) -> dict:
    load = runpy.run_path(str(_HERE / "global_config_loader.py"))["load_global_config"]
    return load(_HERE.parent / "global_config.yaml", server)


def local_base(config: dict) -> Path:
    """Where the dataset roots live: the storage volume on RunPod, else the project root."""
    storage = str(config.get("storage_volume", "") or "").strip()
    return Path(storage or str(config["project_root"])).expanduser()


def is_dataset_root(name: str) -> bool:
    return name == "dataset" or name.startswith("dataset_")


def check_paths(paths: list[str]) -> list[str]:
    """Normalise user paths; each must sit inside a dataset root (dataset, dataset_*)."""
    cleaned = []
    for rel in paths:
        if os.path.isabs(rel) or ".." in Path(rel).parts or not rel.strip("/"):
            raise ValueError(f"Paths must be relative to the project root: {rel!r}")
        rel = rel.strip("/")
        if not is_dataset_root(rel.split("/", 1)[0]):
            raise ValueError(f"Paths must start with a dataset root (dataset_filtered/...): {rel!r}")
        cleaned.append(rel)
    return cleaned


def patterns_for(paths: list[str], is_dir) -> list[str]:
    """allow_patterns for a pull: a folder takes everything below it (fnmatch ``*`` spans ``/``)."""
    return [f"{rel}/*" if is_dir(rel) else rel for rel in check_paths(paths)]


# ── interactive picking (same flow as sync_server.sh) ─────────────────────────────
def _menu(title: str, items: list[str], ask=input) -> int | None:
    print(f"\n{RULE}\n  {title}\n{RULE}")
    for index, item in enumerate(items, 1):
        print(f"  [{index}] {item}")
    print("  [q] cancel\n")
    while True:
        answer = ask("선택: ").strip()
        if answer.lower() in {"q", "quit", "exit"}:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return int(answer) - 1
        print(f"잘못된 선택입니다. 1-{len(items)} 또는 q를 입력하세요.")


def pick_paths(list_children, title: str, ask=input) -> list[str]:
    """Pick a dataset root, then drill down like sync_server.sh; repeat to collect several paths."""
    chosen: list[str] = []
    while True:
        rel = ""
        while True:
            children = list_children(rel)                  # [(name, is_dir)]
            if not rel:
                children = [(name, is_dir) for name, is_dir in children if is_dir and is_dataset_root(name)]
            if not children:
                print(f"{rel or '(데이터 루트)'} 아래에 항목이 없습니다.")
                break
            labels = [f"{name}/" if is_dir else name for name, is_dir in children]
            index = _menu(f"{title}: {rel or '데이터 루트 선택'}", labels, ask)
            if index is None:
                return chosen
            name, is_dir = children[index]
            rel = f"{rel}/{name}" if rel else name
            if not is_dir:
                chosen.append(rel)
                break
            print(f"\n현재 선택: {rel}/\n  [1] 이 폴더 전체\n  [2] 하위 항목 고르기\n  [q] cancel\n")
            action = ask("선택: ").strip()
            if action == "1":
                chosen.append(rel)
                break
            if action != "2":
                return chosen
        if chosen:
            print("\n지금까지 선택: " + ", ".join(chosen))
        if ask("다른 폴더/파일도 추가할까요? [y/N]: ").strip().lower() not in {"y", "yes"}:
            return chosen


def _local_children(base: Path):
    def children(rel: str):
        directory = base / rel
        entries = sorted(directory.iterdir(), key=lambda path: (not path.is_dir(), path.name)) if directory.is_dir() else []
        return [(path.name, path.is_dir()) for path in entries if not path.name.startswith(".")]
    return children


def _remote_children(api, repo: str):
    from huggingface_hub.hf_api import RepoFolder

    def children(rel: str):
        entries = api.list_repo_tree(repo, path_in_repo=rel or None, repo_type="dataset", recursive=False)
        items = [(entry.path.rsplit("/", 1)[-1], isinstance(entry, RepoFolder)) for entry in entries]
        return sorted(items, key=lambda item: (not item[1], item[0]))
    return children


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file() and ".cache" not in file.parts)


def _confirm(assume_yes: bool, ask=input) -> bool:
    return assume_yes or ask("진행할까요? [y/N]: ").strip().lower() in {"y", "yes"}


# ── push / pull ───────────────────────────────────────────────────────────────────
def push(repo: str, base: Path, paths: list[str], *, dry_run: bool, assume_yes: bool) -> None:
    paths = check_paths(paths)
    missing = [rel for rel in paths if not (base / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Not under {base}: {missing}")
    total = sum(_size(base / rel) for rel in paths)
    print(f"\n{RULE}\n  Push → https://huggingface.co/datasets/{repo} (private)\n  from: {base}")
    for rel in paths:
        print(f"    {rel}{'/' if (base / rel).is_dir() else ''}")
    print(f"  size: {total / 1e9:.2f} GB\n{RULE}")
    if dry_run:
        for rel in paths:
            kind = "upload_folder" if (base / rel).is_dir() else "upload_file"
            print(f"[dry-run] {kind}({base / rel} -> {repo}:{rel})")
        return
    if not _confirm(assume_yes):
        print("취소했습니다.")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for rel in paths:
        local = base / rel
        print(f"  uploading {rel} ...")
        # Only this folder is scanned; the repo path keeps the dataset root (dataset_filtered/...).
        if local.is_dir():
            api.upload_folder(repo_id=repo, folder_path=local, path_in_repo=rel, repo_type="dataset",
                              commit_message=f"Upload {rel}")
        else:
            api.upload_file(path_or_fileobj=local, path_in_repo=rel, repo_id=repo, repo_type="dataset",
                            commit_message=f"Upload {rel}")
    print("완료.")


def pull(repo: str, base: Path, paths: list[str], *, dry_run: bool, assume_yes: bool, config: dict,
         is_dir=None) -> None:
    if is_dir is None:
        from huggingface_hub import HfApi

        files = {entry.path for entry in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True)}
        is_dir = lambda rel: rel not in files  # noqa: E731 - a chosen path that is not a file is a folder
    patterns = patterns_for(paths, is_dir)
    print(f"\n{RULE}\n  Pull ← https://huggingface.co/datasets/{repo}\n  to  : {base}")
    for rel in check_paths(paths):
        print(f"    {rel}")
    print(RULE)
    if dry_run:
        print(f"[dry-run] snapshot_download(local_dir={base}, allow_patterns={patterns})")
        return
    if not _confirm(assume_yes):
        print("취소했습니다.")
        return
    from huggingface_hub import snapshot_download

    base.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, repo_type="dataset", local_dir=base, allow_patterns=patterns)
    print("완료.")


def pick_models(ask=input) -> list[str]:
    names = list(PRETRAINED_MODELS)
    print(f"\n{RULE}\n  사전학습 모델 (원본 저장소 → models/)\n{RULE}")
    for name in names:
        print(f"    {name:32s} ← {PRETRAINED_MODELS[name]['repo']}")
    print("\n  [1] 전부\n  [2] 골라서\n  [q] cancel\n")
    action = ask("선택: ").strip()
    if action == "1":
        return names
    chosen: list[str] = []
    while action == "2":
        index = _menu("받을 모델 선택", [name for name in names if name not in chosen], ask)
        if index is None:
            break
        chosen.append([name for name in names if name not in chosen][index])
        if len(chosen) == len(names) or ask("다른 모델도 추가할까요? [y/N]: ").strip().lower() not in {"y", "yes"}:
            break
    return chosen


def pull_models(base: Path, names: list[str], *, dry_run: bool, assume_yes: bool) -> list[str]:
    """Download pretrained models from their original repos into <base>/models/<folder>."""
    unknown = sorted(set(names) - set(PRETRAINED_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {list(PRETRAINED_MODELS)}.")
    print(f"\n{RULE}\n  Models ← original Hugging Face repos\n  to  : {base / 'models'}")
    for name in names:
        spec = PRETRAINED_MODELS[name]
        print(f"    models/{name}  ←  {spec['repo']}@{(spec['revision'] or 'main')[:8]}")
    print(RULE)
    if dry_run:
        for name in names:
            spec = PRETRAINED_MODELS[name]
            print(f"[dry-run] snapshot_download({spec['repo']}, revision={spec['revision']}, "
                  f"local_dir={base / 'models' / name}, allow_patterns={spec.get('files')})")
        return []
    if not _confirm(assume_yes):
        print("취소했습니다.")
        return []
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    failed = []
    for name in names:
        spec = PRETRAINED_MODELS[name]
        print(f"  downloading models/{name} ...")
        try:
            snapshot_download(spec["repo"], revision=spec["revision"], local_dir=base / "models" / name,
                              allow_patterns=spec.get("files"))
        except GatedRepoError:
            print(f"  ! {spec['repo']}: 이용 동의가 필요합니다. https://huggingface.co/{spec['repo']} 에서 동의 후 다시 실행하세요.")
            failed.append(name)
        except RepositoryNotFoundError:
            print(f"  ! {spec['repo']}: 저장소를 찾을 수 없습니다 (로그인: hf auth login).")
            failed.append(name)
    print("완료." if not failed else f"실패: {failed}")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=["push", "pull"], help="omit to choose interactively")
    parser.add_argument("paths", nargs="*", help="dataset_filtered/... paths (omit to pick)")
    parser.add_argument("--models", nargs="?", const="all", default=None,
                        help="pull pretrained models from their original repos: all (default) or a,b,...")
    parser.add_argument("--repo", default=None, help=f"override hf_dataset_repo / ${REPO_ENV}")
    parser.add_argument("--server", default=None, help="override server detection (like SBD_SERVER)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    config = load_config(args.server)
    repo = (args.repo or os.environ.get(REPO_ENV) or str(config.get("hf_dataset_repo", "") or "")).strip()
    base = local_base(config)
    print(f"server={config.get('server', '?')}  repo={repo or '(unset)'}  local={base}")

    mode = args.mode or ("pull" if args.models else None)
    if mode is None:
        index = _menu("작업 선택", ["push  (이 서버 → Hugging Face)", "pull  (Hugging Face → 이 서버)"])
        if index is None:
            return
        mode = ("push", "pull")[index]
    if mode == "push" and args.models:
        sys.exit("--models 는 pull 전용입니다: 사전학습 모델은 원본 저장소에서 받습니다.")

    paths = list(args.paths)
    models: list[str] = []
    if args.models:
        models = list(PRETRAINED_MODELS) if args.models == "all" else [name.strip() for name in args.models.split(",") if name.strip()]
    want_dataset = bool(paths) or not models
    if mode == "pull" and not paths and not models:
        index = _menu("무엇을 받을까요?", [
            f"내 데이터셋 ({repo or 'hf_dataset_repo 미설정'})",
            "사전학습 모델 (원본 저장소 → models/)",
            "둘 다",
        ])
        if index is None:
            return
        want_dataset = index in (0, 2)
        if index in (1, 2):
            models = pick_models()
    if want_dataset and not repo:
        sys.exit("hf_dataset_repo 가 비어 있습니다: configs/global_config.yaml 에 <아이디>/<저장소> 를 적거나 --repo 로 주세요.")
    if want_dataset and not paths:
        if mode == "push":
            paths = pick_paths(_local_children(base), "올릴 항목 선택")
        else:
            from huggingface_hub import HfApi

            paths = pick_paths(_remote_children(HfApi(), repo), "받을 항목 선택")
    if not paths and not models:
        print("선택한 항목이 없습니다.")
        return
    if mode == "push":
        push(repo, base, paths, dry_run=args.dry_run, assume_yes=args.yes)
        return
    if paths:
        pull(repo, base, paths, dry_run=args.dry_run, assume_yes=args.yes, config=config)
    if models:
        pull_models(base, models, dry_run=args.dry_run, assume_yes=args.yes)
    if not args.dry_run and str(config.get("storage_volume", "") or "").strip():
        runpy.run_path(str(_HERE / "link_storage.py"))["link_storage"](config)


if __name__ == "__main__":
    main()
