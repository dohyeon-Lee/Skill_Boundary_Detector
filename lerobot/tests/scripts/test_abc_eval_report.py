"""The report must preserve outcomes/media and be usable without a web backend."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'lerobot/skillVLA/abc/configs/train_pi05/PT/eval/src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from report import organize_existing, video_paths, panel_name  # noqa: E402
import run_eval  # noqa: E402


def test_skill_banner_and_termination_graph_share_frame_boundaries(tmp_path):
    from report import annotate_video, build_result, render_report
    from skill_trace_report import SKILL_OVERLAY_VERSION, active_skill, load_skill_trace, skill_color

    raw, target = tmp_path / 'raw.mp4', tmp_path / 'src/video.mp4'
    with imageio.get_writer(str(raw), fps=20, macro_block_size=1) as writer:
        for _ in range(5):
            writer.append_data(np.full((48, 256, 3), 100, dtype=np.uint8))
    trace = tmp_path / 'src/trace.json'
    trace.parent.mkdir()
    trace.write_text(json.dumps({'skills': [
        {'step': 0, 'end_step': 4, 'skill_order': 0, 'skill_code': 9, 'end_reason': 'terminator'},
        {'step': 4, 'end_step': 8, 'skill_order': 1, 'skill_code': 53, 'end_reason': 'timeout'}],
        'termination_trace': [[2, 0, .2], [4, 0, .8], [6, 1, .1], [8, 1, .3]]}))
    world = {'world_index': 0, 'seed': 1, 'steps': 8, 'success': True, 'prompt': 'grab target',
             'skill_trace_path': str(trace), 'video_every_n_steps': 2, 'video_path': str(target)}
    skills = load_skill_trace(world)
    assert active_skill(skills, 4)['code'] == 9
    assert active_skill(skills, 6)['code'] == 53
    world['video_overlay'] = annotate_video(raw, target, world)
    assert world['video_overlay']['version'] == SKILL_OVERLAY_VERSION
    assert world['video_overlay']['fps'] == 20
    with imageio.get_reader(str(target)) as reader:
        assert reader.count_frames() == 5
        # The colored skill bar follows the final camera row, before language.
        for frame, code in [(2, 9), (3, 53)]:
            pixel = reader.get_data(frame)[26 + 48 + 2, 5].astype(float)
            assert np.max(np.abs(pixel - skill_color(code))) < 15
    result = build_result({'type': 'skillvla', 'skillvla': {'termination_threshold': .7}}, 'abc', [
        {'task': 'grab_clutter', 'requested_task': 'grab_clutter', 'num_worlds': 1,
         'successes': 1, 'worlds': [world]}], expected=1)
    render_report(tmp_path, result)
    page = (tmp_path / 'index.html').read_text()
    assert 'class="termination-panel" hidden' in page
    assert '"threshold":0.7' in page and '"fps":20.0,"stride":2' in page
    assert 'video.currentTime * trace.fps * trace.stride' in page


def test_organize_preserves_raw_metrics_frames_and_relative_report_links(tmp_path):
    task = tmp_path / 'grab_clutter'
    task.mkdir()
    worlds, hashes = [], {}
    for index, success in enumerate([True, False]):
        video = task / f'world_{index:03d}.mp4'
        with imageio.get_writer(str(video), fps=30, macro_block_size=1) as writer:
            for _ in range(3):
                writer.append_data(np.full((48, 96, 3), 100, dtype=np.uint8))
        hashes[index] = hashlib.sha256(video.read_bytes()).hexdigest()
        world = {'world_index': index, 'seed': 1000 + index, 'prompt': 'grab <script>target</script>',
                 'success': success, 'final_success': success, 'steps': 2, 'reward': float(success),
                 'video_path': str(video)}
        worlds.append(world)
        (task / f'world_{index:03d}.json').write_text(json.dumps(world))
    saved = {'format': 'sbd_abc_sim_eval/v1', 'policy': {'type': 'pi05', 'checkpoint': '/model/checkpoints/050000/pretrained_model'},
             'dataset': 'abc_sim_grab_clutter', 'num_worlds': 2, 'num_tasks': 1, 'successes': 1, 'success_rate': 0.5,
             'tasks': [{'task': 'grab_clutter', 'requested_task': 'grab_clutter', 'num_worlds': 2,
                        'successes': 1, 'success_rate': 0.5, 'worlds': worlds}]}
    (tmp_path / 'summary.json').write_text(json.dumps(saved))
    (tmp_path / 'resolved_config.json').write_text('{}')
    result = organize_existing(tmp_path, workers=2)
    assert result['successes'] == 1 and result['success_rate'] == 0.5
    mtimes = []
    for world, original in zip(result['tasks'][0]['worlds'], worlds, strict=True):
        for key in original:
            if key != 'video_path':
                assert world[key] == original[key]
        raw, target = video_paths(tmp_path, '00_pi05_50k', 'grab_clutter', world['world_index'])
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == hashes[world['world_index']]
        reader = imageio.get_reader(str(target))
        try:
            assert reader.count_frames() == 3
            frame = reader.get_data(0)
            assert frame.shape[0] > 48  # Camera image remains intact between added banners.
            pixel = frame[0, 0]
            assert (pixel[1] > pixel[0]) if world['success'] else (pixel[0] > pixel[1])
        finally:
            reader.close()
        mtimes.append(target.stat().st_mtime_ns)
        assert world['video_overlay']['frames'] == 3
    assert not task.exists() and not (tmp_path / 'summary.json').exists()
    assert (tmp_path / 'metrics/summary.json').is_file()
    assert (tmp_path / 'metrics/resolved_config.json').is_file()
    assert (tmp_path / 'metrics/layout_originals.json').is_file()
    for name in ['task_success_rates.png', 'episode_steps.png']:
        assert (tmp_path / name).stat().st_size > 1000
    page = (tmp_path / 'index.html').read_text()
    assert '50.0%' in page and 'Success only' in page and 'Failure only' in page
    assert 'grab &lt;script&gt;target&lt;/script&gt;' in page
    assert 'src="panels/00_pi05_50k/videos/grab_clutter/world_000.mp4"' in page
    assert page.count('<video ') == 2
    organize_existing(tmp_path, workers=2)
    assert mtimes == [Path(w['video_path']).stat().st_mtime_ns for w in result['tasks'][0]['worlds']]


def test_annotation_failure_keeps_outcome_for_cpu_only_retry(tmp_path, monkeypatch):
    config = SimpleNamespace(save_video=True, output_dir=tmp_path, video_fps=30)
    metrics = tmp_path / 'metrics/tasks/grab_clutter'
    metrics.mkdir(parents=True)
    world = {'world_index': 0, 'seed': 123, 'success': True, 'steps': 24, 'prompt': 'grab target', 'video_path': None}
    def fail(*args):
        raise OSError('encoder failed')
    monkeypatch.setattr(run_eval, 'finish_video', fail)
    with pytest.raises(OSError, match='encoder failed'):
        run_eval.commit_world(config, '00_pi05_50k', 'grab_clutter', metrics, world)
    saved = json.loads((metrics / 'world_000.json').read_text())
    assert saved['success'] is True and saved['steps'] == 24 and saved['video_annotation_pending']
    def finish(raw, target, world, fps):
        world.update(video_path=str(target), video_overlay={'version': 'test'})
    monkeypatch.setattr(run_eval, 'finish_video', finish)
    run_eval.commit_world(config, '00_pi05_50k', 'grab_clutter', metrics, saved)
    assert 'video_annotation_pending' not in json.loads((metrics / 'world_000.json').read_text())


def test_finish_video_uses_diagnostic_source_and_preserves_clean_recording(tmp_path):
    from report import finish_video
    raw, marked, target = tmp_path/'raw.mp4', tmp_path/'marked.mp4', tmp_path/'final.mp4'
    for path, value in [(raw, 40), (marked, 140)]:
        with imageio.get_writer(str(path), fps=30, macro_block_size=1) as writer:
            for _ in range(2):
                writer.append_data(np.full((48, 96, 3), value, np.uint8))
    original = hashlib.sha256(raw.read_bytes()).hexdigest()
    world = {'world_index': 0, 'success': True, 'prompt': 'grab', 'camera_overlay_path': str(marked)}
    finish_video(raw, target, world, 30)
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == original
    assert world['raw_video_path'] == str(raw) and world['video_path'] == str(target)
    with imageio.get_reader(str(target)) as reader:
        assert reader.count_frames() == 2
        assert abs(float(reader.get_data(0)[40, 40].mean()) - 140) < 5


def test_evaluation_writes_metrics_and_refreshes_report(tmp_path, monkeypatch):
    from dataclasses import replace
    config = replace(run_eval.load_eval_config(), output_dir=tmp_path, num_worlds=2, save_video=False)
    spec = SimpleNamespace(name='grab_clutter', prompt='grab target', evaluator_name='target_in_box')
    monkeypatch.setitem(sys.modules, 'abc_sim', SimpleNamespace(get_task_spec=lambda task: spec))
    monkeypatch.setattr(run_eval, 'make_policy', lambda cfg: SimpleNamespace(describe=lambda: {'type': 'hold'}))
    def evaluate(cfg, requested, spec, policy, task_dir, on_progress):
        worlds = [{'world_index': i, 'seed': 1000+i, 'prompt': 'grab target', 'steps': 3,
                   'success': i == 0, 'video_path': None, 'reward': float(i == 0)} for i in range(2)]
        for world in worlds:
            run_eval.write_json(task_dir / f"world_{world['world_index']:03d}.json", world)
        summary = run_eval.task_summary(requested, spec, 'mujoco', worlds)
        on_progress(summary)
        return summary
    monkeypatch.setattr(run_eval, 'run_sequential_task', evaluate)
    result = run_eval.run(config)
    assert result['num_worlds'] == 2 and result['success_rate'] == 0.5
    assert not (tmp_path / 'summary.json').exists()
    assert (tmp_path / 'metrics/summary.json').is_file()
    assert (tmp_path / 'metrics/partial_summary.json').is_file()
    assert (tmp_path / 'metrics/tasks/grab_clutter/world_000.json').is_file()
    assert (tmp_path / 'index.html').read_text().count('<article ') == 2
