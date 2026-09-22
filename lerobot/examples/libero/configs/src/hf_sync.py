#!/usr/bin/env python3
"""Pick dataset folders and push them to / pull them from a private Hugging Face dataset repo that
mirrors the project's dataset roots — sync_server.sh for the Hub.

    bash hf_sync.sh                          # interactive: push or pull, pick a dataset root, drill down
    bash hf_sync.sh push dataset_filtered/libero_90_full_full dataset_filtered/skillvla_dataset/libero_90_full_full/eval_init_states.npz
    bash hf_sync.sh pull dataset_filtered/libero_90_full_full --yes
    bash hf_sync.sh pull --models            # pretrained models from their original repos -> models/
    bash hf_sync.sh pull --checkpoints outputs_filtered/pi05_PT/<run>/checkpoints/030000
    bash hf_sync.sh watch --keep 3 --protect 050000,100000   # tmux: upload new checkpoints (public repo)
    bash hf_sync.sh push --dry-run ...       # show the plan only

The repo keeps the dataset roots as top-level folders (``dataset_filtered/``, ``dataset_calvin/``,
...), so one repo holds every root and a pull lands exactly where the code expects it. Paths start
with such a root and are relative to the project root — or to ``storage_volume`` on servers with a
storage volume (RunPod: /workspace-global), after which the checkout links are refreshed.

Repo: ``hf_dataset_repo`` in configs/global_config.yaml (or $SBD_HF_DATASET_REPO); it is created
private on the first push. Log in first: ``hf auth login`` (or export HF_TOKEN). Re-running an
interrupted push is cheap: already-uploaded data is deduplicated and not sent again.

Checkpoints go to the public ``hf_checkpoint_repo`` (see src/hf_checkpoints.py for watch mode).

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
CHECKPOINT_REPO_ENV = "SBD_HF_CHECKPOINT_REPO"
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


def is_outputs_root(name: str) -> bool:
    return name == "outputs" or name.startswith("outputs_")


def check_paths(paths: list[str], is_root=is_dataset_root) -> list[str]:
    """Normalise user paths; each must sit inside a dataset root (or an outputs root for checkpoints)."""
    cleaned = []
    for rel in paths:
        if os.path.isabs(rel) or ".." in Path(rel).parts or not rel.strip("/"):
            raise ValueError(f"Paths must be relative to the project root: {rel!r}")
        rel = rel.strip("/")
        if not is_root(rel.split("/", 1)[0]):
            kind = "a dataset root (dataset_filtered/...)" if is_root is is_dataset_root else "an outputs root (outputs_filtered/...)"
            raise ValueError(f"Paths must start with {kind}: {rel!r}")
        cleaned.append(rel)
    return cleaned


def patterns_for(paths: list[str], is_dir, is_root=is_dataset_root) -> list[str]:
    """allow_patterns for a pull: a folder takes everything below it (fnmatch ``*`` spans ``/``)."""
    return [f"{rel}/*" if is_dir(rel) else rel for rel in check_paths(paths, is_root)]


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


def pick_paths(list_children, title: str, ask=input, is_root=is_dataset_root) -> list[str]:
    """Pick a dataset (or outputs) root, then drill down like sync_server.sh; repeat for several paths."""
    chosen: list[str] = []
    while True:
        rel = ""
        while True:
            children = list_children(rel)                  # [(name, is_dir)]
            if not rel:
                children = [(name, is_dir) for name, is_dir in children if is_dir and is_root(name)]
            if not children:
                print(f"{rel or '(데이터 루트)'} 아래에 항목이 없습니다.")
                break
            labels = [f"{name}/" if is_dir else name for name, is_dir in children]
            index = _menu(f"{title}: {rel or '루트 선택'}", labels, ask)
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


def _remote_children(api, repo: str, repo_type: str = "dataset"):
    from huggingface_hub.hf_api import RepoFolder

    def children(rel: str):
        entries = api.list_repo_tree(repo, path_in_repo=rel or None, repo_type=repo_type, recursive=False)
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
         is_dir=None, repo_type: str = "dataset") -> None:
    is_root = is_dataset_root if repo_type == "dataset" else is_outputs_root
    if is_dir is None:
        from huggingface_hub import HfApi

        files = set(HfApi().list_repo_files(repo, repo_type=repo_type))
        is_dir = lambda rel: rel not in files  # noqa: E731 - a chosen path that is not a file is a folder
    patterns = patterns_for(paths, is_dir, is_root)
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo}"
    print(f"\n{RULE}\n  Pull ← {url}\n  to  : {base}")
    for rel in check_paths(paths, is_root):
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
    snapshot_download(repo, repo_type=repo_type, local_dir=base, allow_patterns=patterns)
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


def _pick_what(ask=input) -> list[int]:
    items = ["내 데이터셋 (비공개, hf_dataset_repo)", "사전학습 모델 (원본 저장소 → models/)", "체크포인트 (공개, hf_checkpoint_repo)"]
    print(f"\n{RULE}\n  무엇을 받을까요? (여러 개면 쉼표로, 예: 1,3)\n{RULE}")
    for index, item in enumerate(items, 1):
        print(f"  [{index}] {item}")
    print("  [q] cancel\n")
    while True:
        answer = ask("선택: ").strip().lower()
        if answer in {"q", "quit", "exit"}:
            return []
        picks = [part.strip() for part in answer.split(",") if part.strip()]
        if picks and all(part.isdigit() and 1 <= int(part) <= len(items) for part in picks):
            return sorted({int(part) - 1 for part in picks})
        print(f"1-{len(items)} 중에서 쉼표로 골라주세요 (예: 1,2) 또는 q.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=["push", "pull", "watch"], help="omit to choose interactively")
    parser.add_argument("paths", nargs="*", help="dataset_filtered/... paths (omit to pick)")
    parser.add_argument("--models", nargs="?", const="all", default=None,
                        help="pull pretrained models from their original repos: all (default) or a,b,...")
    parser.add_argument("--checkpoints", default=None, help="pull: outputs_*/... checkpoint paths, comma separated")
    parser.add_argument("--repo", default=None, help=f"override hf_dataset_repo / ${REPO_ENV}")
    parser.add_argument("--checkpoint-repo", default=None, help=f"override hf_checkpoint_repo / ${CHECKPOINT_REPO_ENV}")
    parser.add_argument("--server", default=None, help="override server detection (like SBD_SERVER)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    watch_group = parser.add_argument_group("watch (upload new checkpoints)")
    watch_group.add_argument("--root", default=None, help="only this outputs root (default: every outputs*/)")
    watch_group.add_argument("--interval", type=int, default=300, help="seconds between scans")
    watch_group.add_argument("--once", action="store_true", help="one scan, then exit")
    watch_group.add_argument("--keep", type=int, default=0, help="keep the latest N steps per run on the Hub (0 = all)")
    watch_group.add_argument("--protect", default="", help="step numbers never deleted, e.g. 050000,100000")
    watch_group.add_argument("--squash", action="store_true", help="free Hub storage after deleting (irreversible)")
    watch_group.add_argument("--prune-local", action=argparse.BooleanOptionalAction, default=None,
                             help="also delete old local steps with the same --keep/--protect rule "
                                  "(default: hf_watch_prune_local of the server, on for RunPod)")
    args = parser.parse_args()

    config = load_config(args.server)
    repo = (args.repo or os.environ.get(REPO_ENV) or str(config.get("hf_dataset_repo", "") or "")).strip()
    checkpoint_repo = (args.checkpoint_repo or os.environ.get(CHECKPOINT_REPO_ENV)
                       or str(config.get("hf_checkpoint_repo", "") or "")).strip()
    base = local_base(config)
    print(f"server={config.get('server', '?')}  dataset repo={repo or '(unset)'}  "
          f"checkpoint repo={checkpoint_repo or '(unset)'}  local={base}")

    mode = args.mode or ("pull" if args.models or args.checkpoints else None)
    if mode is None:
        index = _menu("작업 선택", ["push  (데이터셋: 이 서버 → Hugging Face)", "pull  (Hugging Face → 이 서버)",
                                     "watch (새 체크포인트를 공개 저장소로 자동 업로드)"])
        if index is None:
            return
        mode = ("push", "pull", "watch")[index]

    if mode == "watch":
        if not checkpoint_repo:
            sys.exit("hf_checkpoint_repo 가 비어 있습니다: configs/global_config.yaml 에 적거나 --checkpoint-repo 로 주세요.")
        checkpoints_module = runpy.run_path(str(_HERE / "hf_checkpoints.py"))
        protect = {int(step) for step in args.protect.split(",") if step.strip()}
        prune_local = args.prune_local if args.prune_local is not None else (
            str(config.get("hf_watch_prune_local", False)).strip().lower() in {"1", "true", "yes"}
        )
        checkpoints_module["watch"](
            Path(str(config["project_root"])).expanduser(), checkpoint_repo, root=args.root,
            interval=args.interval, once=args.once, keep=args.keep, protect=protect, squash=args.squash,
            prune_local=prune_local, dry_run=args.dry_run, assume_yes=args.yes,
        )
        return
    if mode == "push" and (args.models or args.checkpoints):
        sys.exit("push 는 데이터셋 전용입니다: 모델은 원본 저장소에서 받고, 체크포인트는 watch 로 올립니다.")

    paths = list(args.paths)
    models: list[str] = []
    if args.models:
        models = list(PRETRAINED_MODELS) if args.models == "all" else [name.strip() for name in args.models.split(",") if name.strip()]
    checkpoints = [path.strip() for path in (args.checkpoints or "").split(",") if path.strip()]
    want_dataset = bool(paths) or (mode == "push") or (not models and not checkpoints and bool(args.mode))
    want_checkpoints = bool(checkpoints)
    if mode == "pull" and not paths and not models and not checkpoints:
        picks = _pick_what()
        if not picks:
            return
        want_dataset, want_checkpoints = 0 in picks, 2 in picks
        if 1 in picks:
            models = pick_models()
    if want_dataset and not repo:
        sys.exit("hf_dataset_repo 가 비어 있습니다: configs/global_config.yaml 에 <아이디>/<저장소> 를 적거나 --repo 로 주세요.")
    if want_checkpoints and not checkpoint_repo:
        sys.exit("hf_checkpoint_repo 가 비어 있습니다: configs/global_config.yaml 에 적거나 --checkpoint-repo 로 주세요.")
    if want_dataset and not paths:
        if mode == "push":
            paths = pick_paths(_local_children(base), "올릴 항목 선택")
        else:
            from huggingface_hub import HfApi

            paths = pick_paths(_remote_children(HfApi(), repo), "받을 데이터셋 선택")
    if want_checkpoints and not checkpoints:
        from huggingface_hub import HfApi

        checkpoints = pick_paths(_remote_children(HfApi(), checkpoint_repo, "model"), "받을 체크포인트 선택",
                                 is_root=is_outputs_root)
    if not paths and not models and not checkpoints:
        print("선택한 항목이 없습니다.")
        return
    if mode == "push":
        push(repo, base, paths, dry_run=args.dry_run, assume_yes=args.yes)
        return
    if paths:
        pull(repo, base, paths, dry_run=args.dry_run, assume_yes=args.yes, config=config)
    if models:
        pull_models(base, models, dry_run=args.dry_run, assume_yes=args.yes)
    if checkpoints:
        pull(checkpoint_repo, base, checkpoints, dry_run=args.dry_run, assume_yes=args.yes, config=config,
             repo_type="model")
    if not args.dry_run and str(config.get("storage_volume", "") or "").strip():
        runpy.run_path(str(_HERE / "link_storage.py"))["link_storage"](config)


if __name__ == "__main__":
    main()
