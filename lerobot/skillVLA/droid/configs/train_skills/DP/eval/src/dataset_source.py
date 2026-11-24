"""Select intact DROID episodes and validate the checkpoint's action coordinates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import runpy


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
spec = importlib.util.spec_from_file_location('droid_sbd_source_helpers',
    ROOT / 'lerobot/skillVLA/libero/configs/train_skills/DP/eval/src/dataset_source.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
prepare_source = helpers.prepare_source
GROUPS = runpy.run_path(str(ROOT / 'lerobot/skillVLA/droid/configs/src/dataset_groups.py'))


def dataset_path(cfg):
    return (Path(cfg['project_root']) / cfg['dataset_root'] / cfg['target_dataset']).resolve()


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def task_rows(cfg):
    import pandas as pd
    tasks = pd.read_parquet(dataset_path(cfg) / 'meta/tasks.parquet')
    if 'task' not in tasks.columns:
        tasks = tasks.reset_index()
    if tasks.task_index.duplicated().any() or tasks.task.duplicated().any():
        raise ValueError('DROID task metadata contains duplicate IDs or prompts')
    return tasks.sort_values('task_index')


def resolve_task_ids(cfg):
    tasks = task_rows(cfg)
    available = list(map(int, tasks.task_index))
    selection = GROUPS['select'](cfg, dataset_path(cfg))
    if selection:
        import pandas as pd
        parts = sorted((dataset_path(cfg) / 'meta/episodes').rglob('*.parquet'))
        rows = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        rows = rows[rows.episode_index.isin(selection['episode_ids'])]
        names = set(str(t) for values in rows.tasks for t in values)
        available = list(map(int, tasks[tasks.task.isin(names)].task_index))
    requested = cfg.get('task_ids', 'all')
    if requested == 'all':
        count = cfg.get('n_tasks', 10)
        result = available if count == 'all' else available[:positive(count, 'n_tasks')]
    else:
        if not isinstance(requested, list) or not requested or any(type(v) is not int or v < 0 for v in requested):
            raise ValueError('DROID task_ids must be all or a nonempty list of dataset integer IDs')
        result = requested
    if not result or len(result) != len(set(result)) or not set(result).issubset(available):
        raise ValueError('DROID task IDs are empty, duplicated, or absent from the dataset')
    return result


def make_source(cfg, settings):
    import pandas as pd
    if settings['task_id_space'] != 'dataset':
        raise ValueError('DROID uses dataset task IDs, not LIBERO suite IDs')
    dataset = dataset_path(cfg)
    view = json.loads((dataset / 'training_view_manifest.json').read_text())
    info = json.loads((dataset / 'meta/info.json').read_text())
    if view.get('action') == 'joint_velocity7_gripper_absolute1' and view.get('state') == 'joint_position7_gripper_state1':
        dimension, mode = 8, 'dataset'
    elif view.get('action') == 'absolute_eef_pose6_gripper_absolute1' and view.get('state') == 'eef_pose6_gripper_state1':
        dimension, mode = 7, 'delta_pose'
    else:
        raise ValueError('DROID eval requires a completed joint-velocity or EEF-pose training view')
    if info.get('fps') != 15 or view.get('fps') != 15:
        raise ValueError('DROID evaluation expects 15Hz; no implicit resampling')
    checkpoint = Path(settings['policy_checkpoint_override'])
    policy = json.loads((checkpoint / 'config.json').read_text())
    if policy.get('type') != 'diffusion' or not policy.get('state_only'):
        raise ValueError('DROID eval requires a state-only diffusion checkpoint')
    if any(policy.get(k) for k in ('use_relative_actions', 'use_delta_joint_actions')):
        raise ValueError('DROID joint velocities must not be converted to relative/delta actions')
    if mode == 'delta_pose':
        if not policy.get('use_delta_pose_actions') or policy.get('delta_pose_input') != 'eef_pose':
            raise ValueError('DROID EEF view requires an explicit EEF delta-pose checkpoint')
    elif policy.get('use_delta_pose_actions'):
        raise ValueError('A joint-velocity view cannot supply EEF delta-pose targets')
    if policy.get('proprio_grounding', 'none') != 'none':
        raise ValueError('DROID joint proprioception must not use XYZ grounding')
    for group, key in [('output_features', 'action'), ('input_features', 'observation.state')]:
        if policy.get(group, {}).get(key, {}).get('shape') != [dimension] or info['features'][key]['shape'] != [dimension]:
            raise ValueError(f'DROID checkpoint/dataset {key} must be {dimension}D')
    image = settings['report_image_key']
    if image not in {'observation.images.image', 'observation.images.wrist_image', 'observation.images.image2'} or image not in info['features']:
        raise ValueError('DROID report camera must be image, wrist_image, or image2 and exist in the dataset')
    metadata_files = sorted((dataset / 'meta/episodes').rglob('file-*.parquet'))
    metadata = pd.concat([pd.read_parquet(p) for p in metadata_files], ignore_index=True).sort_values('episode_index')
    if metadata.episode_index.duplicated().any():
        raise ValueError('DROID metadata contains duplicate episodes')
    selection = GROUPS['select'](cfg, dataset)
    if selection:
        metadata = metadata[metadata.episode_index.isin(selection['episode_ids'])]
    tasks = task_rows(cfg)
    names = dict(zip(tasks.task.astype(str), tasks.task_index.astype(int)))
    by_task = {task: [] for task in map(int, tasks.task_index)}
    all_episodes = []
    for row in metadata.to_dict('records'):
        indices = list(row.get('task_indices', [])) or [names[str(task)] for task in row['tasks']]
        if len(indices) != 1:
            raise ValueError('DROID subset selection requires exactly one prompt per episode')
        task, episode = int(indices[0]), int(row['episode_index'])
        by_task[task].append(dict(episode_index=episode, task_index=task,
                                 selection_task_id=task, length=int(row['length'])))
        all_episodes.append(episode)
    count = positive(settings['n_episodes'], 'n_episodes')
    selected = []
    for task in settings['task_ids']:
        if not by_task.get(task):
            raise ValueError(f'DROID task {task} has no episodes')
        selected.extend(by_task[task][:count])
    calibration = settings['threshold_calibration']
    if calibration['source'] != 'selected':
        raise ValueError('Partial DROID evaluation calibrates once on the selected episodes')
    if calibration['expected_episodes'] not in (None, len(selected)):
        raise ValueError('DROID threshold calibration count differs from selected episodes')
    calibration['expected_episodes'] = len(selected)
    scope = cfg.get('pca_scope', 'dataset')
    if scope not in {'dataset', 'selected'}:
        raise ValueError('DROID pca_scope must be dataset|selected')
    files = [dataset / 'meta/info.json', dataset / 'meta/tasks.parquet', dataset / 'training_view_manifest.json', *metadata_files]
    identity = {str(p.relative_to(dataset)): helpers._sha256(p) for p in files}
    ckpt_files = [checkpoint / name for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json')]
    ckpt_files += sorted(checkpoint.glob('policy_*processor*.safetensors'))
    return dict(schema_version=1, evaluation_source='droid_subset', dataset_name=dataset.name,
        dataset_dir=str(dataset), dataset_marker_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        source_tasks=[], policy_path=str(checkpoint), checkpoint_sha256={p.name: helpers._sha256(p) for p in ckpt_files},
        image_key=image, evaluation_episodes=selected, pca_scope=scope,
        pca_episode_ids=sorted(all_episodes if scope == 'dataset' else [row['episode_index'] for row in selected]),
        action=dict(dim=dimension, mode=mode, relative_exclude_joints=['gripper'], gripper_mode='discrete',
                    gripper_indices=[dimension-1], gripper_values=[0, 1], gripper_threshold=0.5),
        **({'dataset_selection': selection} if selection else {}),
        probe={'exclude_indices': [dimension-1]}, detector=dict(noise_scheduler_type='DDIM', num_inference_steps=10,
        eval_at_step=7, replan_interval=int(settings['probe_variants'][0]['replan_interval']),
        n_gmm_components=5, min_skill_len=10))
