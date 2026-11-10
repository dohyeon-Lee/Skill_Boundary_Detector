"""Constrain public eval uploads to the existing PT/FT result folders."""
from pathlib import Path

BASES = tuple(f"lerobot/examples/libero/configs/train_skillVLA/{stage}/eval/outputs" for stage in ("PT", "FT"))


def result_relative(path, *, parents=False):
    if not isinstance(path, str) or len(path) > 4096 or "\x00" in path:
        raise ValueError("Invalid eval result path")
    if not path and parents:
        return ""
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError("Invalid eval result path")
    if parents and (not path or any(base == path or base.startswith(path + "/") for base in BASES)):
        return path
    if not any(path.startswith(base + "/") for base in BASES):
        raise ValueError("PT/eval/outputs 또는 FT/eval/outputs 안의 평가 결과를 선택하세요.")
    return path


def result_path(project, path, *, parents=False):
    path = result_relative(path, parents=parents)
    root = Path(project).resolve()
    target = root
    for part in Path(path).parts:
        target /= part
        if target.is_symlink():
            raise ValueError("평가 결과 경로의 링크는 허용하지 않습니다.")
    if not target.resolve().is_relative_to(root):
        raise ValueError("프로젝트 밖의 평가 결과는 허용하지 않습니다.")
    return target


def select_results(project, paths):
    selected = []
    for path in sorted(set(paths)):
        folder = result_path(project, path)
        if not folder.is_dir() or not (folder / "metrics/eval_info_merged.json").is_file():
            raise ValueError("최종 지표가 있는 평가 결과 폴더가 필요합니다: " + path)
        files = list(folder.rglob("*"))
        if any(p.is_symlink() for p in files):
            raise ValueError("평가 결과 내부 링크는 업로드하지 않습니다.")
        files = [p for p in files if p.is_file()]
        selected.append({"path": path, "file_count": len(files), "size": sum(p.stat().st_size for p in files)})
    return selected


def upload_results(project, repo, paths, *, dry_run=False, api=None):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(repo)
    selected = select_results(project, paths)
    if dry_run:
        return selected
    api = api or HfApi()
    if api.repo_info(repo, repo_type="model").private is not False:
        raise ValueError("평가 결과는 지정된 공개 모델 저장소에만 업로드합니다.")
    for item in selected:
        select_results(project, [item["path"]])
        print(f"평가 결과 업로드: {item['path']} · {item['file_count']}개 파일", flush=True)
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=Path(project) / item["path"],
                          path_in_repo=item["path"], commit_message="Archive eval results: " + item["path"])
    return selected
