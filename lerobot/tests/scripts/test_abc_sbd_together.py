"""Saved report comparison requires no policy, GPU or original build cache."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[2] / 'skillVLA/abc/configs/train_skills/DP/eval/src/together_report.py'
spec = importlib.util.spec_from_file_location('abc_sbd_together', SRC)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def source(root, name, episodes):
    folder = root / name
    (folder / 'src').mkdir(parents=True)
    cards = ''.join(f"<div class='card'><div class='cap'>task59 · episode {episode}</div>"
                    "<div class='episode-player' data-n-frames='60' data-fps='30'>"
                    "<video class='episode-video' data-src='../video%20file.mp4'></video>"
                    "</div><img src='data:image/svg+xml;base64,QQ=='></div>" for episode in episodes)
    script = "const episodePlayers=[];episodePlayers.forEach(other=>{const v=other.querySelector('.episode-video');if(v!==video)v.pause();});"
    (folder / 'index.html').write_text(f'<html><head><style>.card{{color:red}}</style></head><body>{cards}<script>{script}</script></body></html>')
    (folder / 'src/boundary_results.json').write_text(json.dumps({'episodes': [
        {'task_id': 59, 'episode_id': episode, 'n_frames': 60} for episode in episodes]}))
    (folder / 'src/evaluation_settings.json').write_text(json.dumps({
        'source_manifest': {'dataset_marker_sha256': 'same-dataset'},
        'probe_variants': [{'descriptor_temporal_length': 1, 'replan_interval': 8}],
        'cluster_variants': [{'k': 2}], 'boundary_variants': [{'nms_frames': 45, 'smooth_window': 8, 'polyorder': 4}],
    }))
    return folder / 'index.html'


def test_three_reports_align_common_episodes_reference_sources_without_copying(tmp_path):
    originals = [source(tmp_path, 'a', [1, 2]), source(tmp_path, 'b', [2, 3]), source(tmp_path, 'c', [2])]
    before = [path.read_bytes() for path in originals]
    (tmp_path / 'video file.mp4').touch()
    config = tmp_path / 'together.yaml'
    config.write_text(yaml.safe_dump({'eval_folders': ['b', 'a', 'c'], 'output_name': 'compare'}))
    path = report.build(config, outputs_root=tmp_path)
    content = path.read_text()
    assert 'const COMMON=new Set(["59:2"])' in content
    for name in ['b', 'a', 'c']:
        assert f'"src": "../../{name}/index.html"' in content
    assert 'data:image/svg' not in content and 'video%20file.mp4' not in content
    assert 'data-server-no-reload' in content and '<iframe title=' in content
    assert 'mirroredSeek=new WeakMap()' in content
    assert 'pageScrollAnchor(source)' in content and 'pageScrollTop(target,anchor)' in content
    assert 'card.dataset.comparisonKey=key' in content
    assert 'target.scrollTo(win.scrollX,win.scrollY)' not in content
    assert [path.read_bytes() for path in originals] == before
    manifest = json.loads((path.parent / 'src/manifest.json').read_text())
    assert manifest['eval_folders'] == ['b', 'a', 'c'] and manifest['episodes'] == 1
    assert sorted(p.name for p in path.parent.iterdir()) == ['index.html', 'src']
    assert report.build(config, outputs_root=tmp_path) == path


@pytest.mark.parametrize('folders', [['a'], ['a', 'a'], ['../a', 'b']])
def test_invalid_selection_creates_no_output(tmp_path, folders):
    config = tmp_path / 'together.yaml'
    config.write_text(yaml.safe_dump({'eval_folders': folders}))
    with pytest.raises(ValueError):
        report.build(config, outputs_root=tmp_path)
    assert not (tmp_path / 'together').exists()
