#!/usr/bin/env python3
"""Upload explicitly selected completed checkpoints; no watch or retention."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

# Eval-only uploads run in their own Desk subprocess. Xet CAS timeouts on the
# lab network have left these processes alive with no data transfer. Use the
# SDK's bounded HTTP/LFS path here; downloads and normal uploads keep Xet.
# An explicit HF_HUB_DISABLE_XET=0 still opts into Xet for diagnostics.
if __name__ == '__main__' and '--pretrained-only' in sys.argv:
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id

from fs_helper import relative_path

REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS = runpy.run_path(str(REPO / 'lerobot/examples/libero/configs/src/hf_checkpoints.py'))


def configure_http_upload():
    """Bound inactive requests while allowing continuously streaming large files."""
    import httpx
    from huggingface_hub import set_client_factory
    from huggingface_hub.utils._http import default_client_factory

    def factory():
        client = default_client_factory()
        client.timeout = httpx.Timeout(connect=20, read=120, write=120, pool=20)
        return client

    set_client_factory(factory)


def local_output_path(project, path, *, allow_root=False):
    path = relative_path(path, allow_root=allow_root)
    if not path:
        return project
    parts = path.split('/')
    if not CHECKPOINTS['is_outputs_root'](parts[0]):
        raise ValueError('outputs 또는 outputs_* 안의 체크포인트를 선택하세요.')
    target = project / parts[0]
    root = target.resolve()
    for part in parts[1:]:
        target /= part
        if target.is_symlink():
            raise ValueError('last 링크 대신 실제 숫자 체크포인트 폴더를 선택하세요.')
    if not target.resolve().is_relative_to(root):
        raise ValueError('outputs 루트 밖의 경로는 선택할 수 없습니다.')
    return target


def select_checkpoints(project, paths, *, pretrained_only=False):
    project = Path(project).expanduser()
    if not isinstance(paths, list) or not 1 <= len(paths) <= 200:
        raise ValueError('체크포인트 또는 상위 폴더를 1~200개 선택하세요.')
    paths = sorted(set(relative_path(p) for p in paths))
    for path in paths:
        if not local_output_path(project, path).is_dir():
            raise ValueError('출발 폴더가 없습니다: ' + path)
    roots = sorted({p.split('/')[0] for p in paths})
    selected = {}
    matched = set()
    for _, run in CHECKPOINTS['find_runs'](project, roots):
        run_path = run.relative_to(project).as_posix()
        if not any(run_path == p or run_path.startswith(p + '/') or p.startswith(run_path + '/') for p in paths):
            continue
        for step in CHECKPOINTS['complete_steps'](run):
            relative = step.relative_to(project).as_posix()
            parents = {p for p in paths if relative == p or relative.startswith(p + '/')}
            if not parents:
                continue
            local_output_path(project, relative)
            upload = step / 'pretrained_model' if pretrained_only else step
            if not upload.is_dir():
                raise ValueError('평가 모델 폴더가 없습니다: ' + relative)
            files = list(upload.rglob('*'))
            if any(p.is_symlink() for p in files):
                raise ValueError('체크포인트 내부 링크는 업로드하지 않습니다: ' + relative)
            files = [p for p in files if p.is_file()]
            if not files:
                raise ValueError('빈 체크포인트입니다: ' + relative)
            selected[relative] = {'path': relative, 'file_count': len(files),
                                  'size': sum(p.stat().st_size for p in files)}
            matched.update(parents)
    if set(paths) - matched:
        raise ValueError('저장이 끝난 숫자 체크포인트 폴더 또는 그 상위 폴더를 선택하세요: '
                         + ', '.join(sorted(set(paths) - matched)))
    if len(selected) > 200:
        raise ValueError('한 번에 최대 200개 체크포인트를 업로드할 수 있습니다. 범위를 줄여주세요.')
    return [selected[p] for p in sorted(selected)]


def upload_selected(project, repo, paths, *, dry_run=False, api=None, pretrained_only=False):
    validate_repo_id(repo)
    project = Path(project).expanduser()
    selected = select_checkpoints(project, paths, pretrained_only=pretrained_only)
    for item in selected:
        print(f"체크포인트 업로드: {item['path']} · {item['file_count']}개 파일 · {item['size']/1024**3:.2f} GiB", flush=True)
    if dry_run:
        print('[dry-run] Hub에 업로드하지 않았습니다.', flush=True)
        return selected
    api = api or HfApi()
    CHECKPOINTS['ensure_card'](api, repo, project)
    for item in selected:
        # Retention can remove an older step while another selection is uploading.
        select_checkpoints(project, [item['path']], pretrained_only=pretrained_only)
        target = item['path'] + ('/pretrained_model' if pretrained_only else '')
        api.upload_folder(repo_id=repo, repo_type='model', folder_path=project / target,
                          path_in_repo=target, commit_message='Upload ' + target)
        print('업로드 완료: ' + item['path'], flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--paths-json', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pretrained-only', action='store_true', help='평가에 필요한 pretrained_model만 업로드')
    args = parser.parse_args()
    if args.pretrained_only:
        from huggingface_hub import constants
        print('평가 체크포인트 전송: ' + ('HTTP/LFS' if constants.HF_HUB_DISABLE_XET else 'Xet'), flush=True)
        if constants.HF_HUB_DISABLE_XET:
            configure_http_upload()
    upload_selected(args.project, args.repo, json.loads(args.paths_json), dry_run=args.dry_run,
                    pretrained_only=args.pretrained_only)


if __name__ == '__main__':
    main()
