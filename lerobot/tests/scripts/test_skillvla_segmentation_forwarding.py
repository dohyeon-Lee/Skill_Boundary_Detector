"""Run the real Bash launchers with a recording Python; no GPU or jobs used."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest
import tyro

from test_train_skillvla_build_config import _config, build_settings

CONFIGS = Path(__file__).resolve().parents[2] / 'examples/libero/configs'
SRC = CONFIGS / 'train_skillVLA/build_data/src'
sys.path.insert(0, str(CONFIGS / 'train_skills/build_data/src'))
sys.path.insert(0, str(CONFIGS / 'train_skills/src'))
from build_skill_dataset import Args, _curve_has_current_metrics, _load_global_threshold, _skillset_manifest
from compute_global_boundary_threshold import main as reduce_threshold, parse_args as reducer_args
from sbd_build_contract import (
    assert_subset, check_run, fingerprint, validate_args, validate_curve,
    validate_manifest, validate_reducer_args, validate_snap_reference, validate_threshold,
)


def settings_for(tmp_path, include_mean=False):
    cfg = _config(tmp_path, 'global_mean')
    fsq = tmp_path / 'outputs/FSQ' / cfg['fsq_run_name']
    p = fsq / 'fsq_meta.json'; meta = json.loads(p.read_text())
    meta['skillset_mode'] = 'gaussian_x0hat'; p.write_text(json.dumps(meta))
    manifest = tmp_path / 'dataset/FSQ_dataset/libero_90_full_full/FSQ_inputs/seg_libero_90_state_obs20_ck100000_std/skillset/skillset_manifest.json'
    d = json.loads(manifest.read_text())
    d['mode'] = 'gaussian_x0hat'
    d['probe'].update(generation='scheduler_gaussian', count=96, gaussian_sampling='iid',
                      gaussian_include_mean=include_mean, gaussian_seed=53, seed=17,
                      denoise_steps=2, denoise_output='pred_original_sample')
    d['detector'].update(score_metric='covariance_gated_cosine', gmm_covariance='diag',
                         gmm_n_init=13, gmm_max_iter=173, min_effective_samples=.2,
                         gmm_weighted=include_mean, score_power=2.7, terminal_mask_frames=25,
                         peak_nms=False, noise_scheduler_type='DDIM', num_inference_steps=12)
    manifest.write_text(json.dumps(d))
    # A stale duplicate in conversion YAML must not override the FSQ recipe.
    cfg['skillset_probe_generation'] = 'pca_offset'
    return build_settings(cfg)


def manifest_for(settings):
    c = json.loads(settings['skillset_contract_json'])
    d = copy.deepcopy(c['segmentation']); d.pop('policy')
    d['policy_path'] = str(settings['dp_policy_path'])
    return d


def record_launcher(tmp_path, settings, name, *, phase='', local=False, expected_exit=0):
    fake = tmp_path / 'fakebin'; fake.mkdir()
    recorder = fake / 'python'
    recorder.write_text(f'#!{sys.executable}\n' + '''
import json,os,pathlib,sys
target=pathlib.Path(sys.argv[2] if sys.argv[1]=='-u' else sys.argv[1]).name
args=sys.argv[3:] if sys.argv[1]=='-u' else sys.argv[2:]
if target=='verify_skillset.py':
 if '--print-missing-tasks' in args: print('3')
 elif '--check' in args: sys.exit(0 if pathlib.Path(os.environ['RETRIED']).exists() else 1)
elif target in ('build_skill_dataset.py','compute_global_boundary_threshold.py'):
 with open(os.environ['CAPTURE'],'a') as f:f.write(json.dumps({'target':target,'args':args})+'\\n')
 pathlib.Path(os.environ['RETRIED']).touch()
elif target=='build_skill_focus_uv.py' and '--check-current' in args:
 sys.exit(1)
elif target in ('ensure_episode_map.py','build_skill_focus_uv.py'):
 with open(os.environ['CAPTURE'],'a') as f:f.write(json.dumps({'target':target,'args':args})+'\\n')
 if target=='ensure_episode_map.py' and os.environ.get('FAIL_EPISODE_MAP')=='1': sys.exit(4)
''')
    recorder.chmod(0o755)
    (fake / 'nvidia-smi').write_text('#!/bin/sh\nexit 0\n'); (fake / 'nvidia-smi').chmod(0o755)
    root = Path(settings['project_root'])
    (root / '.venv/bin').mkdir(parents=True)
    (root / '.venv/bin/python').symlink_to(recorder)
    (root / '.venv/bin/activate').write_text('export PATH=' + shlex.quote(str(fake)) + ':"$PATH"\n')
    for file in ('runtime_env.sh', 'gpu_guard.sh'):
        p = root / 'lerobot/examples/libero/configs/src' / file; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('require_cuda_or_requeue() { :; }\n')
    Path(settings['dp_policy_path']).mkdir(parents=True)
    snapshot = tmp_path / 'env.sh'
    snapshot.write_text('\n'.join('export ' + k.upper() + '=' + shlex.quote(str(v).lower() if isinstance(v, bool) else str(v))
                                 for k, v in settings.items()))
    env = dict(os.environ, SKILLVLA_ENV_SNAPSHOT=str(snapshot), SLURM_SUBMIT_DIR=str(SRC.parent),
               CAPTURE=str(tmp_path/'argv.jsonl'), RETRIED=str(tmp_path/'retried'),
               PATH=str(fake)+os.pathsep+os.environ['PATH'], ALL_TASK_IDS='0 1 2 3 4')
    env.pop('SLURM_JOB_ID', None)
    if not local:env.update(SLURM_JOB_ID='12345', SLURM_ARRAY_TASK_ID='0')
    if phase:env[phase]='true'
    if name=='compute_global_boundary_threshold.sbatch':env['EXPECTED_EPISODES']='2'
    result = subprocess.run(['bash', str(SRC/name)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode==expected_exit, result.stdout+'\n'+result.stderr
    capture = tmp_path/'argv.jsonl'
    return [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []


@pytest.mark.parametrize('uv_enabled,map_fails', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('local', [False, True])
def test_final_worker_prepares_map_before_uv_only_when_needed(tmp_path, monkeypatch, uv_enabled, map_fails, local):
    s = settings_for(tmp_path)
    run = Path(s['skillvla_run_dir']); run.mkdir(parents=True)
    for key in ('iss_npz_path', 'fsq_copy_path'):
        Path(s[key]).touch()
    dataset = Path(s['skillvla_dataset_dir']); dataset.mkdir()
    (dataset/'existing_data').touch()
    s.update(focus_uv_enabled=str(uv_enabled).lower(), focus_uv_camera='agentview',
             focus_uv_suite='libero_90', focus_uv_path=run/'skill_focus_uv.npz',
             focus_uv_eval_init_states_path=run.parent/'eval_init_states.npz',
             focus_uv_original_dataset_dir=tmp_path/'original')
    if map_fails:
        monkeypatch.setenv('FAIL_EPISODE_MAP', '1')
    calls = record_launcher(tmp_path, s, 'build_skillvla.sbatch', local=local,
                            expected_exit=4 if map_fails else 0)
    targets = [call['target'] for call in calls]
    if not uv_enabled:
        assert targets == []
        return
    assert targets == (['ensure_episode_map.py'] if map_fails else ['ensure_episode_map.py', 'build_skill_focus_uv.py'])
    args = calls[0]['args']
    assert args[args.index('--source-dataset-dir')+1] == str(s['raw_dataset_dir'])
    assert args[args.index('--output')+1] == str(s['focus_uv_eval_init_states_path'])


@pytest.mark.parametrize('name,phase', [('build_skillset.sbatch',''), ('build_skillset.sbatch','CURVES_ONLY'),
                                      ('build_skillset.sbatch','USE_CACHED_CURVES'), ('verify_skillset.sbatch','')])
@pytest.mark.parametrize('include_mean', [False, True])
@pytest.mark.parametrize('local', [False, True])
def test_actual_launcher_argv_inherits_fsq_without_python_defaults(tmp_path, name, phase, include_mean, local):
    s = settings_for(tmp_path, include_mean)
    calls = record_launcher(tmp_path, s, name, phase=phase, local=local)
    call = next(c for c in calls if c['target']=='build_skill_dataset.py')
    parsed = tyro.cli(Args, args=call['args'])
    contract = json.loads(s['skillset_contract_json'])
    validate_args(parsed, contract)
    assert parsed.probe_generation=='scheduler_gaussian' and parsed.denoise_output=='pred_original_sample'
    assert parsed.gaussian_include_mean is include_mean and parsed.gmm_weighted is include_mean
    assert parsed.curves_only == (phase=='CURVES_ONLY')
    assert parsed.use_cached_curves == (phase=='USE_CACHED_CURVES' or name=='verify_skillset.sbatch')
    actual = _skillset_manifest(parsed, Path(parsed.dataset_dir), parsed.policy_path,
                               'observation.images.image', parsed.probe_mode, 7, None)
    p=tmp_path/'actual_manifest.json'; p.write_text(json.dumps(actual))
    validate_manifest(p, contract)
    # Removing a single forwarded option must be detected by parsed-CLI checks.
    parsed.probe_generation='pca_offset'
    with pytest.raises(ValueError, match='actual Python CLI.probe.generation'):validate_args(parsed, contract)


@pytest.mark.parametrize('local', [False, True])
def test_actual_threshold_launcher_records_complete_provenance(tmp_path, monkeypatch, local):
    s=settings_for(tmp_path)
    call=record_launcher(tmp_path,s,'compute_global_boundary_threshold.sbatch',local=local)[0]
    monkeypatch.setattr(sys,'argv',['reducer',*call['args']])
    args=reducer_args(); contract=json.loads(s['skillset_contract_json'])
    validate_reducer_args(args, contract)
    curves=Path(args.curves_dir); curves.mkdir(parents=True)
    (curves.parent/'skillset_manifest.json').write_text(json.dumps(manifest_for(s)))
    for i in range(2):
        np.savez(curves/f'ep{i:07d}.npz', episode_id=i, sg_vals=[1.,3.,999.],threshold_valid_count=2,
                 segmentation_contract_sha256=fingerprint(contract))
    reduce_threshold(); payload=json.loads(Path(args.output_path).read_text())
    assert payload['global_threshold']==2 and payload['n_replan_points']==4
    validate_threshold(payload, contract, args.output_path)
    assert payload['provenance']['gaussian_include_mean'] is False
    assert payload['provenance']['gmm_covariance']=='diag'
    args.probe_generation='pca_offset'
    with pytest.raises(ValueError,match='Threshold CLI'):validate_reducer_args(args,contract)


def test_existing_wrong_or_unproven_completed_outputs_are_preserved_and_rejected(tmp_path):
    s=settings_for(tmp_path);c=json.loads(s['skillset_contract_json']);skillset=Path(s['skillset_dir']);skillset.mkdir(parents=True)
    p=skillset/'skillset_manifest.json';d=manifest_for(s);d['probe']['generation']='pca_offset';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='probe.generation'):check_run(skillset,s['skillvla_run_dir'],c,'preflight')
    assert json.loads(p.read_text())['probe']['generation']=='pca_offset'
    p.write_text(json.dumps(manifest_for(s)));run=Path(s['skillvla_run_dir']);(run/'skillvla').mkdir(parents=True)
    keep=run/'skillvla/keep';keep.write_text('original')
    with pytest.raises(ValueError,match='no verified'):check_run(skillset,run,c,'preflight')
    assert keep.read_text()=='original'


def test_proof_survives_cleanup_but_cannot_hide_changed_settings(tmp_path):
    s=settings_for(tmp_path);c=json.loads(s['skillset_contract_json']);skillset=Path(s['skillset_dir']);skillset.mkdir(parents=True)
    (skillset/'skillset_manifest.json').write_text(json.dumps(manifest_for(s)))
    run=Path(s['skillvla_run_dir']);run.mkdir(parents=True);(run/'skill_latents.npz').touch()
    check_run(skillset,run,c,'record-encoded');check_run(skillset,run,c,'encoded')
    for n in ('FSQ.pt','skill_initial_state.npz'): (run/n).touch()
    (run/'skillvla').mkdir();check_run(skillset,run,c,'record-final')
    (skillset/'skillset_manifest.json').unlink();(run/'skill_latents.npz').unlink()
    check_run(skillset,run,c,'preflight');check_run(skillset,run,c,'final')
    changed=copy.deepcopy(c);changed['segmentation']['probe']['gaussian_include_mean']=True
    with pytest.raises(ValueError,match='gaussian_include_mean'):check_run(skillset,run,changed,'preflight')


def test_cached_curve_and_threshold_must_prove_the_recipe(tmp_path):
    s=settings_for(tmp_path);c=json.loads(s['skillset_contract_json']);p=tmp_path/'curve.npz'
    fields=dict(curve_schema_version=3,denoising_gain=[1],denoising_gain_sg=[1],delta_bic=[1],delta_bic_sg=[1],
                tail_excluded_frames=0,last_valid_anchor=0)
    np.savez(p,**fields)
    assert _curve_has_current_metrics(p) and not _curve_has_current_metrics(p,c)
    np.savez(p,**fields,segmentation_contract_sha256=fingerprint(c))
    assert _curve_has_current_metrics(p,c)
    changed=copy.deepcopy(c);changed['segmentation']['detector']['gmm_weighted']=True
    assert not _curve_has_current_metrics(p,changed)
    threshold={'segmentation_contract':c['segmentation']}
    ft=copy.deepcopy(c);ft['segmentation']['dataset_name']='libero_10_full_1'
    validate_threshold(threshold,ft,'PT reference')
    with pytest.raises(ValueError,match='contract mismatch'):validate_threshold({},ft,'old PT reference')
    with pytest.raises(ValueError,match='gmm_weighted'):validate_threshold(threshold,changed,'bad reference')


def test_incomplete_fsq_manifest_is_not_filled_with_legacy_gaussian_defaults(tmp_path):
    s=settings_for(tmp_path);p=Path(s['fsq_skillset_manifest_path']);d=json.loads(p.read_text())
    del d['probe']['generation'];p.write_text(json.dumps(d))
    # Recover the minimal fixture config without modifying any artifact.
    cfg={'project_root':str(tmp_path),'dataset_root':'dataset','outputs_root':'outputs',
         'source_dataset':s['source_dataset'],'skillvla_data_mode':'pt','fsq_run_name':s['fsq_run_name'],
         'fsq_checkpoint':'0','fsq_snap_to_supported':False}
    with pytest.raises(ValueError,match='provenance is incomplete'):build_settings(cfg)


def test_old_pt_snap_reference_cannot_seed_corrected_ft_labels(tmp_path):
    s=settings_for(tmp_path);c=json.loads(s['skillset_contract_json']);p=tmp_path/'pt/skill_latents.npz';p.parent.mkdir();p.touch()
    with pytest.raises(ValueError,match='no verified'):validate_snap_reference(str(p),c)
    (p.parent/'segmentation_contract.json').write_text(json.dumps({'contract':c,'stage':'encoded'}))
    ft=copy.deepcopy(c);ft['segmentation']['dataset_name']='libero_10_full_1';ft['segmentation']['detector']['boundary_threshold_mode']='episode_mean'
    validate_snap_reference(str(p),ft)
