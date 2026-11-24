"""DROID SBD contracts, worker selection, and real CPU cache -> global report."""
import importlib.util
import json
import re
import os
import subprocess
from pathlib import Path
import sys

import av
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / 'lerobot/skillVLA/droid/configs/train_skills/DP/eval'
ABC = ROOT / 'lerobot/skillVLA/abc/configs/train_skills/DP/eval/src'
sys.path.insert(0, str(ABC))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = load(EVAL / 'src/config.py', 'droid_eval_config_test')
SOURCE = load(EVAL / 'src/dataset_source.py', 'droid_eval_source_test')


@pytest.fixture
def setup(tmp_path):
    dataset = tmp_path / 'dataset_DROID/tiny'
    (dataset / 'meta/episodes/chunk-000').mkdir(parents=True)
    (dataset / 'data/chunk-000').mkdir(parents=True)
    features = {'action': {'dtype': 'float32', 'shape': [8]},
                'observation.state': {'dtype': 'float32', 'shape': [8]}}
    cameras = ['image', 'wrist_image', 'image2']
    features.update({f'observation.images.{name}': {'dtype': 'video', 'shape': [24, 32, 3]} for name in cameras})
    (dataset / 'meta/info.json').write_text(json.dumps(dict(fps=15, features=features,
        video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')))
    (dataset / 'training_view_manifest.json').write_text(json.dumps(dict(fps=15,
        action='joint_velocity7_gripper_absolute1', state='joint_position7_gripper_state1')))
    pd.DataFrame({'task': ['Move cup', 'Open lid', 'Rare instruction'],
                  'task_index': [0, 1, 2]}).set_index('task').to_parquet(dataset / 'meta/tasks.parquet')
    rows = []
    for ep, task in enumerate([0, 0, 1, 1, 2]):
        row = dict(episode_index=ep, length=96, tasks=[['Move cup', 'Open lid', 'Rare instruction'][task]],
                   dataset_from_index=96*ep, dataset_to_index=96*(ep+1),
                   **{'data/chunk_index': 0, 'data/file_index': ep})
        for camera in cameras:
            key = f'observation.images.{camera}'
            row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': ep,
                        f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': 96/15})
            video = dataset / f'videos/{key}/chunk-000/file-{ep:03d}.mp4'
            video.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(video), 'w') as container:
                stream = container.add_stream('libx264', rate=15)
                stream.width, stream.height, stream.pix_fmt = 32, 24, 'yuv420p'
                stream.codec_context.thread_count = 1
                for t in range(96):
                    frame = av.VideoFrame.from_ndarray(np.full((24,32,3), (t+ep)%255, dtype=np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        rows.append(row)
        rng = np.random.default_rng(ep)
        pd.DataFrame([dict(episode_index=ep, frame_index=t,
            **{'observation.state': rng.normal(size=8).astype(np.float32),
               'action': np.r_[rng.normal(size=7), t%2].astype(np.float32)})
            for t in range(96)]).to_parquet(dataset / f'data/chunk-000/file-{ep:03d}.parquet')
    pd.DataFrame(rows).to_parquet(dataset / 'meta/episodes/chunk-000/file-000.parquet')
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text(json.dumps(dict(type='diffusion', state_only=True,
        n_obs_steps=30, horizon=18, action_sequence_mode='future_only', proprio_grounding='none',
        history_encoder='transformer', down_dims=[128,256,512],
        output_features={'action': {'shape': [8]}}, input_features={'observation.state': {'shape': [8]}})))
    for name in ('model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        (model / name).write_bytes(b'fixture: cached render must never load weights')
    cfg = yaml.safe_load((EVAL / 'eval_config.yaml').read_text())
    for key in ('dp_run_name', 'dp_checkpoint'):
        cfg.pop(key)
    cfg.update(project_root=str(tmp_path), dataset_root='dataset_DROID', target_dataset='tiny', score_power=2,
               policy_checkpoint=str(model), task_ids=[1,0], n_episodes=1, pca_scope='selected')
    path = tmp_path / 'eval.yaml'; path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def payload(path):
    return json.loads(CONFIG.build_settings(path)['dp_ablation_settings_json'])


def test_droid_uses_libero_standard_with_velocity_gripper_contract(setup):
    path, cfg = setup
    data = payload(path)
    assert data['analysis_domain'] == 'droid'
    manifest = data['source_manifest']
    assert manifest['evaluation_source'] == 'droid_subset'
    assert [e['episode_index'] for e in manifest['evaluation_episodes']] == [2,0]
    assert manifest['action']['mode'] == 'dataset'
    assert manifest['action']['gripper_indices'] == [7]
    assert manifest['action']['gripper_values'] == [0,1]
    assert manifest['action']['gripper_threshold'] == .5
    assert manifest['pca_episode_ids'] == [0,2]
    assert data['threshold_calibration'] == {'source': 'selected', 'expected_episodes': 2}
    probe, cluster, boundary, experiment = [data[k][0] for k in
        ('probe_variants','cluster_variants','boundary_variants','experiments')]
    assert (probe['count'], probe['eval_at_step'], probe['denoise_steps'], probe['denoise_output']) == (96,7,1,'pred_original_sample')
    assert (probe['gaussian_center'], probe['replan_interval'], probe['descriptor_temporal_length']) == ('current_gt',8,18)
    assert manifest['detector']['replan_interval'] == 8
    assert probe['mode'] == 'std' and probe['pca_variance'] == .95 and probe['pca_stride'] == 3
    assert cluster['k'] == 5 and cluster['covariance'] == 'diag' and cluster['weighted']
    assert experiment['metric'] == 'covariance_gated_cosine'
    assert (boundary['score_power'], boundary['smooth_window'], boundary['polyorder'],boundary['nms_frames']) == (2,7,4,23)
    assert boundary['initial_mask_frames'] == 30 and boundary['terminal_mask_frames'] == 24
    assert boundary['terminal_mask_mode'] == 'fixed' and boundary['threshold_scope'] == 'dataset'
    assert Path(data['report_root']).parent == EVAL / 'outputs'


def test_eef_delta_pose_source_matches_model_and_excludes_only_gripper(setup):
    path, cfg = setup
    dataset = Path(cfg['project_root']) / cfg['dataset_root'] / cfg['target_dataset']
    marker = dataset / 'training_view_manifest.json'
    marker.write_text(json.dumps(dict(fps=15,
        action='absolute_eef_pose6_gripper_absolute1', state='eef_pose6_gripper_state1')))
    info_path = dataset / 'meta/info.json'
    info = json.loads(info_path.read_text())
    for key in ('action', 'observation.state'):
        info['features'][key]['shape'] = [7]
    info_path.write_text(json.dumps(info))
    model = Path(cfg['policy_checkpoint']) / 'config.json'
    policy = json.loads(model.read_text())
    policy.update(use_delta_pose_actions=True, delta_pose_input='eef_pose')
    policy['output_features']['action']['shape'] = [7]
    policy['input_features']['observation.state']['shape'] = [7]
    model.write_text(json.dumps(policy))
    manifest = payload(path)['source_manifest']
    assert manifest['action']['mode'] == 'delta_pose'
    assert manifest['action']['dim'] == 7
    assert manifest['action']['gripper_indices'] == manifest['probe']['exclude_indices'] == [6]
    assert [ep['episode_index'] for ep in manifest['evaluation_episodes']] == [2, 0]
    policy['delta_pose_input'] = 'yam_joint'
    model.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='explicit EEF delta-pose'):
        payload(path)


def test_selection_caps_scarce_prompts_without_idle_or_frame_filtering(setup):
    path, cfg = setup
    cfg.update(task_ids='all', n_tasks='all', n_episodes=5, pca_scope='dataset')
    path.write_text(yaml.safe_dump(cfg));data=payload(path)
    assert [e['episode_index'] for e in data['source_manifest']['evaluation_episodes']] == [0,1,2,3,4]
    assert data['threshold_calibration']['expected_episodes'] == 5
    assert data['source_manifest']['pca_episode_ids'] == [0,1,2,3,4]
    assert all(e['length'] == 96 for e in data['source_manifest']['evaluation_episodes'])
    cfg.update(task_ids=[1], n_episodes=1)
    path.write_text(yaml.safe_dump(cfg));data=payload(path)
    assert data['source_manifest']['pca_episode_ids'] == [0,1,2,3,4]
    assert data['threshold_calibration']['expected_episodes'] == 1


@pytest.mark.parametrize('change,match', [
    ({'task_ids':[0,0]}, 'duplicated'), ({'task_ids':[99]}, 'absent'),
    ({'advanced':{'task_id_space':'suite'}}, 'dataset task'),
    ({'report_image_key':'observation.images.top'}, 'camera'),
    ({'sbd_prediction_horizon':19}, 'exceeds'), ({'n_episodes':0}, 'positive'),
])
def test_invalid_droid_inputs_fail_before_execution(setup,change,match):
    path,cfg=setup;cfg.update(change);path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match=match):payload(path)


def test_wrong_checkpoint_action_contract_is_rejected(setup):
    path,cfg=setup;model=Path(cfg['policy_checkpoint'])/'config.json'
    original=json.loads(model.read_text())
    for key,value in [('use_delta_joint_actions',True),('proprio_grounding','episode_start_xyz')]:
        model.write_text(json.dumps(dict(original,**{key:value})))
        with pytest.raises(ValueError,match='relative/delta|grounding'):payload(path)


def test_launcher_dry_run_resolves_existing_allocation_without_submission(setup):
    path, _ = setup
    proc = subprocess.run(['bash', str(EVAL / 'submit_eval.sh'), '--75000', '--config', str(path), '--dry-run'],
                          capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS='1'))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'Episodes: 2' in proc.stdout and 'prediction=18' in proc.stdout
    assert 'DROID_DP_SBD_eval' not in proc.stdout
    assert not (path.parent / '.config_snapshots').exists()


@pytest.mark.parametrize('cached', [False, True])
def test_gpu_guard_covers_eval_workers_but_cached_reports_stay_cpu(monkeypatch,cached):
    from lerobot.utils import gpu_memory_guard
    monkeypatch.setitem(sys.modules,'config',CONFIG)
    runner=load(EVAL/'src/run_eval.py','droid_guarded_eval_test')
    monkeypatch.setattr(sys,'argv',['run_eval.py','--config','frozen.yaml'])
    monkeypatch.setattr(runner,'build_settings',lambda path:{'dp_ablation_device':'cuda','dp_ablation_cache_ready':cached})
    seen=[]
    monkeypatch.setattr(gpu_memory_guard,'supervise_if_needed',lambda cfg:seen.append(cfg.policy.device) or (None if cached else 95))
    dispatched=[]
    monkeypatch.setattr(runner.runpy,'run_path',lambda path,**kw:dispatched.append(path))
    if cached:
        runner.main()
        assert seen==['cpu'] and len(dispatched)==1
    else:
        with pytest.raises(SystemExit) as error:runner.main()
        assert error.value.code==95 and seen==['cuda'] and not dispatched


def test_episode_workers_partition_droid_manifest_and_merge_globally_once(setup,tmp_path,monkeypatch):
    from run_parallel_ablation import ParallelAblation, load_evaluator
    path,cfg=setup;cfg['n_episodes']=2;path.write_text(yaml.safe_dump(cfg))
    settings=payload(path);artifact=tmp_path/'partial'
    settings.update(artifact_dir=str(artifact),skillset_dir=str(artifact/'source'))
    SOURCE.prepare_source(settings)
    evaluator=load_evaluator();runner=ParallelAblation(settings,tmp_path/'workers')
    shards={};global_calls=[]
    def worker(index,shard):
        shards[index]=json.loads(shard.read_text());return 0
    monkeypatch.setattr(runner,'_worker',worker)
    monkeypatch.setattr(evaluator,'run',lambda final:global_calls.append(final['threshold_calibration'].copy()))
    runner.run(evaluator)
    combined=[ep for shard in shards.values() for ep in shard]
    assert len(shards)==2 and len(combined)==len(set(combined))==4
    assert sorted(combined)==[0,1,2,3]
    assert global_calls==[{'source':'selected','expected_episodes':4}]


def test_cached_droid_report_uses_real_videos_and_one_global_threshold(setup,tmp_path,monkeypatch):
    from run_parallel_ablation import load_evaluator
    from cache_state import cache_status,sha256
    path,cfg=setup;settings=payload(path)
    artifact=tmp_path/'partial';report=tmp_path/'report'
    settings.update(artifact_dir=str(artifact),skillset_dir=str(artifact/'source'),
                    report_root=str(report),output_dir=str(report/'src'))
    SOURCE.prepare_source(settings)
    evaluator=load_evaluator()
    assert evaluator._selected_episode_ids(artifact/'source',task_ids=[1,0],
        task_id_space='dataset',target_task='droid',n_episodes=1)==[2,0]
    (artifact/'pca').mkdir(parents=True)
    pca=artifact/'pca/std_h18.npz'
    np.savez(pca,metadata=np.array(json.dumps(dict(source_tasks=[],action_mode='dataset',
        episode_ids=[0,2],action_dim=8,action_indices=list(range(8)),scale_mode='std'))))
    probe=settings['probe_variants'][0]
    for ep in [2,0]:
        raw=artifact/'raw'/probe['name']/f'ep{ep:07d}.npz';raw.parent.mkdir(parents=True,exist_ok=True)
        ts=np.arange(0,96,probe['replan_interval'])
        meta=dict(schema=2,pca_sha256=sha256(pca),descriptor_pca_sha256=sha256(pca),
            policy_path=settings['source_manifest']['policy_path'],seed=settings['seed'],
            eval_at_step=7,replan_interval=probe['replan_interval'],variant=probe)
        np.savez(raw,descriptors=np.random.default_rng(ep).normal(size=(len(ts),96,2)),
                 replan_ts=ts,n_frames=96,metadata_json=np.array(json.dumps(meta)))
    assert cache_status(settings,require_clusters=False)[0]
    import torch
    import skill_divider
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU cache rendering touched CUDA'))
    monkeypatch.setattr(skill_divider,'load_policy',lambda *a,**kw:pytest.fail('Cached render loaded policy weights'))
    before={p:sha256(p) for p in artifact.glob('raw/**/*.npz')}
    result=evaluator.render_cached_report(settings)
    assert 'DROID' in result.read_text() and len(re.findall(r'''class=['"]episode-player['"]''', result.read_text()))==2
    assert (report/'src/evaluation_complete.json').is_file()
    assert (report/'src/summary.json').is_file()
    assert (report/'src/threshold_statistics.json').is_file()
    assert before=={p:sha256(p) for p in before}
    assert cache_status(settings)[0]
    manifest=json.loads((report/'src/video_report_manifest.json').read_text())
    assert manifest['players']==2 and manifest['tasks']==2 and manifest['reencoded'] is False
    references=json.loads((report/'src/video_references.json').read_text())
    assert len(references)==2 and all(Path(p).is_file() for p in references)
    assert all('dataset_DROID/tiny/videos/' in p for p in references)
    results=json.loads((report/'src/boundary_results.json').read_text())
    assert {e['episode_id'] for e in results['episodes']}=={0,2}
    summary=json.loads((report/'src/summary.json').read_text())
    thresholds=[row[settings['experiments'][0]['name']]['threshold'] for row in summary.values()]
    assert len(thresholds)==2 and len(set(thresholds))==1
    statistics=json.loads((report/'src/threshold_statistics.json').read_text())
    global_mean=statistics[settings['experiments'][0]['name']]
    assert global_mean['groups']['dataset']['calibration_episodes']==2
    assert global_mean['calibration']['source']=='selected'
    np.testing.assert_allclose(thresholds,global_mean['groups']['dataset']['threshold'])
    clusters=list(artifact.glob('clusters/**/*.npz'))
    assert len(clusters)==2
