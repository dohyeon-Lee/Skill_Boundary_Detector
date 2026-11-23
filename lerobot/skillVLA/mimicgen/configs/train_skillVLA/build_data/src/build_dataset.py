"""Encode Coffee skills and build standard SkillVLA annotations and camera targets.

Uses the frozen FSQ encoder's normalization, never re-fits it on Coffee.
Missing SBD is built in the same GPU allocation before FSQ encoding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
SHARED = ROOT / 'lerobot/skillVLA/libero/configs'
sys.path.insert(0, str(SHARED / 'train_skills/src'))
from train_skills_config import load_config
sys.path.insert(0, str(ROOT / "lerobot/skillVLA/mimicgen/configs/src"))
from model_paths import fsq_checkpoint, vocabulary_reference
from data_modes import data_mode, validate_skillset_mode
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbd_pipeline


def resolve(cfg):
    def path(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p
    raw = path(cfg['dataset_root']) / cfg['target_dataset']
    run = cfg['dataset_run'].format(skillvla_data_mode=data_mode(cfg))
    out = path(cfg['dataset_root']) / 'skillvla' / cfg['target_dataset'] / run
    if 'skillset_seg_name' in cfg:  # Explicit historical snapshots keep their selected input.
        skills = path(cfg['dataset_root']) / 'FSQ_dataset' / cfg['target_dataset'] / 'FSQ_inputs' / cfg['skillset_seg_name'] / 'skillset'
    else:
        skills = sbd_pipeline.settings(cfg)[1]['skillset_dir']
    paths = dict(raw=raw, output=out, skillset=skills, model=fsq_checkpoint(cfg))
    if cfg.get('fsq_snap_to_supported', False) and data_mode(cfg) != 'pt':
        reference = vocabulary_reference(cfg)
        paths.update(fsq_supported_freq_path=reference,
                     fsq_reference_model=reference.parent/'FSQ.pt',
                     fsq_reference_source=reference.parent/'fsq_source.json')
        if (reference.parent/'segmentation_contract.json').is_file():
            paths['fsq_reference_contract'] = reference.parent/'segmentation_contract.json'
    return paths


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def jitter_storage_pmax(cfg):
    """Match LIBERO's ISS capacity for percentage-based transition jitter."""
    max_length = int(cfg['max_length'])
    steps = int(cfg['jitter_max_steps'])
    percent = float(cfg['jitter_max_percent'])
    if max_length <= 0 or steps < 0:
        raise ValueError('max_length must be positive and jitter_max_steps non-negative')
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise ValueError('jitter_max_percent must be finite and in [0, 100]')
    return max(1, int(math.floor(max_length * percent / 100.0 + 0.5))) if percent > 0 else steps


def camera_targets(raw, output, latents):
    import numpy as np
    import pandas as pd
    from lerobot.policies.skillVLA.focus_projection import project_eef
    camera = json.loads((raw / 'meta/mimicgen_camera.json').read_text())
    width, height = int(camera['width']), int(camera['height'])
    projection = np.asarray(camera['top_world_to_pixel'], dtype=np.float64)
    # Shared SkillVLA projection functions express the LIBERO horizontal mirror.
    # Encode its inverse in the calibration, leaving Coffee videos untouched.
    flip = np.eye(4); flip[0, 0] = -1; flip[0, 2] = width - 1
    shared_projection = flip @ projection
    with np.load(latents, allow_pickle=False) as z:
        identity = {k: np.asarray(z[k]).copy() for k in ['episode_id','task_id','skill_index','frame_start','frame_end']}
    frame = pd.concat([pd.read_parquet(p, columns=['episode_index','frame_index','observation.state'])
                       for p in sorted((raw / 'data').rglob('*.parquet'))], ignore_index=True)
    groups = {int(ep): g.sort_values('frame_index') for ep, g in frame.groupby('episode_index')}
    n = len(identity['episode_id']); arrays = dict(identity)
    for prefix in ['', 'start_']:
        for name, shape, dtype in [('focus_uv',(n,2),np.float32),('focus_uv_pixels',(n,2),np.int32),
                                   ('focus_uv_raw_pixels',(n,2),np.float32),('focus_valid',(n,),bool),('focus_clipped',(n,),bool)]:
            arrays[prefix+name] = np.empty(shape, dtype=dtype)
    info_path = output / 'meta/info.json'; info = json.loads(info_path.read_text())
    grounding = info.get('proprio_grounding', 'none')
    if grounding not in {'none', 'episode_start_xyz'}: raise ValueError(f'Unsupported grounding {grounding}')
    transforms, episodes = [], []
    for ep in sorted(set(identity['episode_id'].tolist())):
        states = np.stack(groups[int(ep)]['observation.state']).astype(np.float64)
        transform = shared_projection.copy()
        if grounding == 'episode_start_xyz': transform[:,3] += transform[:,:3] @ states[0,:3]
        transforms.append(transform); episodes.append(ep)
        for row in np.flatnonzero(identity['episode_id'] == ep):
            start, end = int(identity['frame_start'][row]), min(int(identity['frame_end'][row]),len(states)-1)
            if not 0 <= start <= end < len(states): raise ValueError(f'Invalid skill range {ep}: {start}:{end}')
            for prefix, index in [('start_', start), ('', end)]:
                point = project_eef(states[index,:3], shared_projection, height=height,width=width)
                arrays[prefix+'focus_uv'][row] = point.normalized_xy
                arrays[prefix+'focus_uv_pixels'][row] = point.pixel_xy
                arrays[prefix+'focus_uv_raw_pixels'][row] = point.raw_xy
                arrays[prefix+'focus_valid'][row] = np.isfinite(point.normalized_xy).all()
                arrays[prefix+'focus_clipped'][row] = point.clipped
    target = output.parent / 'skill_focus_uv.npz'
    np.savez_compressed(target, **arrays, schema_version=np.int32(3),camera=np.asarray('agentview'),
        image_height=np.int32(height),image_width=np.int32(width),camera_episode_id=np.asarray(episodes,dtype=np.int64),
        camera_grounded_transform=np.asarray(transforms),startpoint_convention=np.asarray('frame_start'),
        endpoint_convention=np.asarray('min(frame_end, episode_length - 1)'))
    info.update(skill_focus_uv_path=str(target),skill_focus_uv_schema_version=3,skill_focus_uv_camera='agentview',
        skill_focus_uv_normalization='minus_one_to_one',skill_focus_uv_start='frame_start',
        skill_focus_uv_endpoint='min(frame_end, episode_length - 1)',skill_focus_uv_image_size=[height,width],
        skill_focus_uv_outside_policy='clip_to_nearest_image_point',wrist_camera_mirror_x=False,
        environment_type='mimicgen',mimicgen_camera_path=str(raw / 'meta/mimicgen_camera.json'))
    info_path.write_text(json.dumps(info,indent=2)+'\n')
    return n


def build(cfg):
    storage_pmax = jitter_storage_pmax(cfg)
    paths = resolve(cfg); raw, output, skills, model = (paths[k] for k in ['raw','output','skillset','model'])
    if not model.is_file(): raise FileNotFoundError(model)
    model_meta = model.parent / 'fsq_meta.json'
    meta = json.loads(model_meta.read_text())
    if list(meta.get('fsq_levels', [])) != list(cfg['fsq_levels']): raise ValueError('FSQ levels differ from checkpoint')
    snap = bool(cfg.get('fsq_snap_to_supported', False))
    snap_min_frequency = int(cfg.get('fsq_snap_min_code_freq', 1))
    if snap_min_frequency < 1:
        raise ValueError('fsq_snap_min_code_freq must be positive')
    external_snap = snap and data_mode(cfg) != 'pt'
    if external_snap:
        for key in ['fsq_supported_freq_path', 'fsq_reference_model', 'fsq_reference_source']:
            if not paths[key].is_file():
                raise FileNotFoundError(f'Original PT snap reference missing: {paths[key]}')
        if sha256(paths['fsq_reference_model']) != sha256(model):
            raise ValueError('The PT vocabulary reference uses a different FSQ checkpoint')
    if 'skillset_seg_name' not in cfg:
        sbd_pipeline.ensure(cfg, output)
    if not (skills / '.complete').exists(): raise FileNotFoundError(f'Complete SBD first: {skills}')
    manifest = json.loads((skills / 'skillset_manifest.json').read_text())
    if Path(manifest.get('dataset_dir', manifest.get('raw_dataset_dir', str(raw)))).resolve() != raw.resolve():
        raise ValueError('SBD manifest belongs to another dataset')
    validate_skillset_mode(cfg, manifest, skills)
    # Reusing a run folder with another encoder can silently invalidate skill IDs.
    contract = dict(schema_version=1,dataset=str(raw),fsq_sha256=sha256(model),
                    skillset_manifest_sha256=sha256(skills/'skillset_manifest.json'),
                    proprio_grounding=cfg['proprio_grounding'],max_length=int(cfg['max_length']),
                    jitter_max_steps=int(cfg['jitter_max_steps']),jitter_max_percent=float(cfg['jitter_max_percent']))
    if 'skillvla_data_mode' in cfg:
        contract['skillvla_data_mode'] = data_mode(cfg)
        if data_mode(cfg) == 'ft':
            contract['pt_threshold_sha256'] = sha256(cfg['skillset_global_threshold_source'])
    if snap:
        contract.update(fsq_snap_to_supported=True,
                        fsq_snap_min_code_freq=snap_min_frequency,
                        fsq_snap_reference_sha256=sha256(paths['fsq_supported_freq_path']) if external_snap else 'self')
    staging = output.parent / ('.build-' + hashlib.sha256(output.name.encode()).hexdigest()[:16])
    staging.mkdir(parents=True, exist_ok=True)
    marker = output / 'mimicgen_build.json'
    previous = marker if marker.exists() else staging / 'contract.json'
    if output.exists() and not previous.exists() and any(output.iterdir()):
        raise ValueError(f'Unowned output folder exists: {output}; select a new dataset_run')
    if previous.exists() and json.loads(previous.read_text()) != contract:
        raise ValueError('Build inputs changed; select a new dataset_run')
    required = ['skillvla/meta/info.json', 'skillvla/meta/stats.json', 'skill_initial_state.npz',
                'skill_focus_uv.npz', 'skill_latents.npz', 'FSQ.pt', 'fsq_meta.json', 'fsq_source.json', 'sbd_source.json']
    complete = output / '.complete'
    if (complete.is_file() and json.loads(complete.read_text()).get('contract') == contract
            and all((output/name).is_file() for name in required)):
        print(f'Verified SkillVLA build already complete: {output}', flush=True)
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    (staging / 'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    src = SHARED / 'train_skillVLA/build_data/src'
    latents = staging/'skill_latents.npz'
    def run(script, *args):
        subprocess.run([sys.executable,str(src/script),*map(str,args)],check=True)
    snap_args = []
    if snap:
        snap_args = ['--snap_to_supported', '--supported_freq_path', paths['fsq_supported_freq_path'] if external_snap else 'self',
                     '--min_code_freq', snap_min_frequency]
    if not latents.is_file() and marker.is_file() and (output/'skill_latents.npz').is_file():
        shutil.copy2(output/'skill_latents.npz', latents)
    cached = False
    if latents.is_file():
        import numpy as np
        try:
            with np.load(latents, allow_pickle=False) as z:
                n = len(z['episode_id'])
                cached = n > 0 and all(len(z[key]) == n for key in ['latents', 'tokens', 'frame_start', 'frame_end'])
        except (OSError, ValueError, KeyError):
            cached = False
    if cached:
        print('[2/3] Matching FSQ encoding already exists; reusing it', flush=True)
    else:
        print('[2/3] Encoding full SBD with FSQ', flush=True)
        run('encode_FSQ_skills.py','--skills_dir',skills/'skills','--model_path',model,
            '--output_path',latents,'--device',cfg['device'],*snap_args)
    print('[3/3] Building SkillVLA data / ISS / UV', flush=True)
    run('add_skill_latents_to_dataset.py','--src_dataset_dir',raw,'--dst_dataset_dir',output/'skillvla',
        '--dst_repo_id','mimicgen/coffee_d0_skillvla','--latents_path',latents,'--iss_npz_path',output/'skill_initial_state.npz',
        '--max_length',cfg['max_length'],'--fsq_levels',*cfg['fsq_levels'],
        '--pmax',storage_pmax,'--jitter_max_percent',cfg['jitter_max_percent'],
        '--jitter_precompute_gripper_validity','true','--proprio_grounding',cfg['proprio_grounding'])
    shutil.copy2(latents, output/'skill_latents.npz')
    marker.write_text(json.dumps(contract,indent=2)+'\n')
    count = camera_targets(raw,output/'skillvla',latents)
    shutil.copy2(model,output/'FSQ.pt')
    shutil.copy2(model_meta,output/'fsq_meta.json')
    (output/'fsq_source.json').write_text(json.dumps(dict(schema_version=1,source_fsq_run_name=model.parent.name,
       source_fsq_checkpoint=str(int(model.stem.removeprefix('FSQ_epoch'))) if model.stem.startswith('FSQ_epoch') else 'best',
       source_fsq_model_path=str(model),copied_fsq_path=str(output/'FSQ.pt')),indent=2)+'\n')
    threshold = Path(manifest['detector'].get('global_threshold_path') or skills/'global_boundary_threshold.json')
    (output/'sbd_source.json').write_text(json.dumps(dict(skillvla_data_mode=data_mode(cfg),
        skillset_dir=str(skills), global_threshold_path=str(threshold),
        global_threshold_sha256=sha256(threshold) if threshold.is_file() else None),indent=2)+'\n')
    (output/'.complete').write_text(json.dumps(dict(skills=count,contract=contract),indent=2)+'\n')
    print(f'Built {count} Coffee skills: {output}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dry-run',action='store_true');p.add_argument('--shell',action='store_true');p.add_argument('--freeze-sbd',action='store_true');a=p.parse_args()
    cfg=load_config(a.config)
    if a.freeze_sbd:
        import yaml
        path = Path(a.config)
        raw_cfg = yaml.safe_load(path.read_text())
        raw_cfg['sbd_recipe'] = sbd_pipeline.recipe(cfg)
        cfg['sbd_recipe'] = raw_cfg['sbd_recipe']
        resolve(cfg)  # Validate the frozen recipe and resource budget before submission.
        path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False, allow_unicode=True))
        print('Frozen SBD recipe: full SBD → FSQ → SkillVLA, one GPU allocation', flush=True)
    elif a.shell:
        from train_skills_config import print_shell
        paths=resolve(cfg)
        extras = {}
        if 'skillset_seg_name' not in cfg:
            extras['dp_policy_path'] = sbd_pipeline.settings(cfg)[1]['dp_policy_path']
        print_shell(dict(raw_dataset_dir=paths['raw'],skillset_dir=paths['skillset'],fsq_model_path=paths['model'],output_dir=paths['output'],
                         skillset_global_threshold_source=cfg.get('skillset_global_threshold_source',''),
                         **extras,
                         **{k:v for k,v in paths.items() if k.startswith('fsq_')}))
    elif a.dry_run: print(json.dumps({**{k:str(v) for k,v in resolve(cfg).items()},
        'skillvla_data_mode':data_mode(cfg), 'pt_threshold_source':cfg.get('skillset_global_threshold_source',''),
        'pipeline':'SBD → FSQ → SkillVLA' if 'skillset_seg_name' not in cfg else 'FSQ → SkillVLA',
        **({'sbd_workers':sbd_pipeline.settings(cfg)[1]['skillset_single_gpu_workers']} if 'skillset_seg_name' not in cfg else {})},indent=2))
    else: build(cfg)
if __name__=='__main__': main()
