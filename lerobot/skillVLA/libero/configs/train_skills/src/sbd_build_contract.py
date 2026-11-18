"""CPU-only contracts for SkillVLA's inherited FSQ segmentation settings.

Legacy FSQ builders keep their existing behavior. SkillVLA explicitly supplies
this contract, which must agree with actual CLI arguments and reused artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROBE_FIELDS = {
    'type': ('probe_type', str), 'count': ('probe_count', int),
    'alpha': ('probe_alpha', float), 'pca_variance': ('pca_variance', float),
    'pca_stride': ('pca_stride', int), 'pca_scale_mode': ('pca_scale_mode', str),
    'seed': ('seed', int),
}
GAUSSIAN_FIELDS = {
    'generation': ('probe_generation', str), 'gaussian_sampling': ('gaussian_sampling', str),
    'gaussian_include_mean': ('gaussian_include_mean', bool), 'gaussian_seed': ('gaussian_seed', int),
    'denoise_steps': ('denoise_steps', int), 'denoise_output': ('denoise_output', str),
}
DETECTOR_FIELDS = {
    'noise_scheduler_type': ('noise_scheduler_type', str), 'num_inference_steps': ('num_inference_steps', int),
    'eval_at_step': ('dn_step', int), 'replan_interval': ('replan_interval', int),
    'n_gmm_components': ('n_gmm', int), 'smooth_window': ('smooth_window', int),
    'savgol_polyorder': ('savgol_polyorder', int), 'peak_nms': ('peak_nms', bool),
    'nms_dist': ('nms_dist', int), 'min_skill_len': ('min_skill_len', int),
    'min_skills': ('min_skills', int), 'boundary_threshold_mode': ('boundary_threshold_mode', str),
    'boundary_threshold_scale': ('boundary_threshold_scale', float),
}
GMM_FIELDS = {
    'score_metric': ('score_metric', str), 'gmm_covariance': ('gmm_covariance', str),
    'gmm_n_init': ('gmm_n_init', int), 'gmm_max_iter': ('gmm_max_iter', int),
    'min_effective_samples': ('min_effective_samples', float), 'gmm_weighted': ('gmm_weighted', bool),
    'score_power': ('score_power', float), 'terminal_mask_frames': ('terminal_mask_frames', int),
}


def csv(value, cast):
    return [cast(x.strip()) for x in str(value).split(',') if x.strip()]


def typed(value, cast):
    if cast is bool and isinstance(value, str):
        if value.lower() not in ('true', 'false'):
            raise ValueError(f'Invalid boolean contract value: {value!r}')
        return value.lower() == 'true'
    return cast(value)


def from_settings(settings):
    def fields(mapping):
        return {key: typed(settings['skillset_' + name], cast) for key, (name, cast) in mapping.items()}
    probe, detector = fields(PROBE_FIELDS), fields(DETECTOR_FIELDS)
    probe['exclude_indices'] = csv(settings['skillset_probe_exclude_indices'], int)
    if settings['skillset_mode'] == 'gaussian_x0hat':
        probe.update(fields(GAUSSIAN_FIELDS)); detector.update(fields(GMM_FIELDS))
    return {
        'segmentation': {
            'mode': settings['skillset_mode'], 'dataset_name': settings['source_dataset'],
            'policy': {'run': settings['dp_policy_name'], 'checkpoint': str(settings['dp_checkpoint']).zfill(6)},
            'action': {
                'mode': settings['skillset_action_mode'],
                'relative_exclude_joints': csv(settings['skillset_relative_exclude_joints'], str),
                'gripper_mode': settings['skillset_gripper_mode'],
                'gripper_indices': csv(settings['skillset_gripper_indices'], int),
                'gripper_values': csv(settings['skillset_gripper_values'], float),
                'gripper_threshold': float(settings['skillset_gripper_threshold']),
            },
            'probe': probe, 'detector': detector,
        },
        'fsq': {'run': settings['fsq_run_name'], 'checkpoint': str(settings['fsq_checkpoint']),
                'snap_to_supported': typed(settings['fsq_snap_to_supported'], bool),
                'snap_min_code_freq': int(settings['fsq_snap_min_code_freq'])},
    }


def fingerprint(contract):
    segmentation = contract.get('segmentation', contract)
    return hashlib.sha256(json.dumps(segmentation, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assert_subset(expected, observed, where='contract'):
    for key, value in expected.items():
        actual = observed.get(key) if isinstance(observed, dict) else None
        if isinstance(value, dict):
            assert_subset(value, actual, where + '.' + key)
        elif actual != value:
            raise ValueError(f'Segmentation contract mismatch at {where}.{key}: expected={value!r}, actual={actual!r}. '
                             'Existing data is preserved. Use a separate build location; do not resume incompatible artifacts.')


def policy_identity(path):
    p = Path(path)
    if p.name == 'pretrained_model':
        p = p.parent
    if p.parent.name != 'checkpoints':
        raise ValueError(f'Policy checkpoint has no stable run/step identity: {path}')
    return {'run': p.parent.parent.name, 'checkpoint': p.name.zfill(6)}


def validate_manifest(path, contract, *, allow_absent=False):
    path = Path(path)
    if not path.is_file():
        if allow_absent and not any((path.parent / name).exists() and any((path.parent / name).rglob('*'))
                                    for name in ('skills', 'curves')):
            return
        raise ValueError(f'Missing segmentation provenance: {path}. Existing artifacts cannot be safely reused.')
    actual = json.loads(path.read_text())
    actual['policy'] = policy_identity(actual['policy_path'])
    # Historical scale=1 manifests omit this field; the value is unambiguous.
    actual['detector'].setdefault('boundary_threshold_scale', 1.0)
    assert_subset(contract['segmentation'], actual, str(path))


def validate_args(args, contract):
    """Validate the arguments Python actually parsed, not just shell exports."""
    aliases = {'dn_step': 'eval_at_step', 'n_gmm': 'n_gmm_components'}
    observed = {'mode': args.probe_mode, 'dataset_name': Path(args.dataset_dir).name,
                'policy': policy_identity(args.policy_path), 'action': {}, 'probe': {}, 'detector': {}}
    for dest, mapping in [('probe', {**PROBE_FIELDS, **GAUSSIAN_FIELDS}),
                          ('detector', {**DETECTOR_FIELDS, **GMM_FIELDS})]:
        for key, (name, cast) in mapping.items():
            observed[dest][key] = typed(getattr(args, aliases.get(name, name)), cast)
    observed['probe']['exclude_indices'] = csv(args.probe_exclude_indices, int)
    for key in ('mode', 'gripper_mode', 'gripper_threshold'):
        observed['action'][key] = getattr(args, 'action_mode' if key == 'mode' else key)
    for key, cast in [('relative_exclude_joints', str), ('gripper_indices', int), ('gripper_values', float)]:
        observed['action'][key] = csv(getattr(args, key), cast)
    assert_subset(contract['segmentation'], observed, 'actual Python CLI')


def validate_curve(curve, contract, path):
    observed = str(curve['segmentation_contract_sha256'].item()) if 'segmentation_contract_sha256' in curve else None
    if observed != fingerprint(contract):
        raise ValueError(f'Cached curve contract mismatch/missing: {path}. Regenerate curves with the intended settings.')


def validate_threshold(payload, contract, path):
    expected = dict(contract['segmentation']); expected.pop('dataset_name')
    actual = dict(payload.get('segmentation_contract') or {}); actual.pop('dataset_name', None)
    # A PT threshold can deliberately be shared by FT, but its detector/probe
    # recipe must match. An empty legacy provenance is not proof of agreement.
    assert_subset(expected, actual, str(path))


def validate_reducer_args(args, contract):
    expected = contract['segmentation']
    checks = {
        'dp_run_name': expected['policy']['run'], 'dp_checkpoint': expected['policy']['checkpoint'],
        'probe_mode': expected['mode'], 'probe_count': expected['probe']['count'],
        **{k: expected['detector'][k] for k in ('eval_at_step', 'n_gmm_components', 'replan_interval',
                                               'smooth_window', 'savgol_polyorder')},
    }
    if expected['mode'] == 'gaussian_x0hat':
        checks.update(probe_generation=expected['probe']['generation'],
                      denoise_output=expected['probe']['denoise_output'],
                      **{k: expected['detector'][k] for k in ('score_metric', 'score_power', 'terminal_mask_frames')})
    for name, value in checks.items():
        if str(getattr(args, name)) != str(value):
            raise ValueError(f'Threshold CLI contract mismatch for {name}: {getattr(args, name)!r} != {value!r}')
    if args.threshold_scale != expected['detector']['boundary_threshold_scale']:
        raise ValueError('Threshold scale does not match the inherited contract')


def validate_snap_reference(path, contract):
    if not path or path == 'self':
        return
    marker = Path(path).parent / 'segmentation_contract.json'
    if not marker.is_file():
        raise ValueError(f'Snap reference has no verified segmentation contract: {marker}. '
                         'Build the corrected PT reference before FT; existing reference is preserved.')
    source = json.loads(marker.read_text())['contract']
    assert_subset({k: contract['fsq'][k] for k in ('run', 'checkpoint')}, source['fsq'], str(marker))
    expected = json.loads(json.dumps(contract['segmentation']))
    expected.pop('dataset_name')
    # FT-own/episode thresholds may intentionally differ from PT. Probe and
    # score generation must still match the FSQ recipe used by the PT labels.
    for key in ('boundary_threshold_mode', 'boundary_threshold_scale'):
        expected['detector'].pop(key)
    assert_subset(expected, source['segmentation'], str(marker))


def check_run(skillset, run_dir, contract, stage):
    run_dir, skillset = Path(run_dir), Path(skillset)
    marker = run_dir / 'segmentation_contract.json'
    record = json.loads(marker.read_text()) if marker.is_file() else None
    if record:
        assert_subset(contract, record['contract'], str(marker))
    if stage in ('preflight', 'skillset', 'record-encoded'):
        validate_manifest(skillset / 'skillset_manifest.json', contract, allow_absent=stage == 'preflight')
    products = ['skill_latents.npz', 'skill_initial_state.npz', 'FSQ.pt', 'skillvla']
    if stage == 'preflight' and not record and any((run_dir / n).exists() for n in products):
        raise ValueError(f'Existing SkillVLA outputs have no verified segmentation contract: {run_dir}. '
                         'Preserved; select a new output location before rebuilding.')
    if stage in ('encoded', 'final', 'record-final') and not record:
        raise ValueError(f'Missing verified encoding provenance: {marker}; refusing to reuse old outputs.')
    if stage == 'final' and record.get('stage') != 'final':
        raise ValueError(f'Final build provenance is incomplete: {marker}')
    if stage in ('record-encoded', 'record-final'):
        required = ['skill_latents.npz'] if stage == 'record-encoded' else ['skillvla', 'skill_initial_state.npz', 'FSQ.pt']
        if not all((run_dir / n).exists() for n in required):
            raise ValueError(f'Cannot record provenance for missing products: {run_dir}')
        run_dir.mkdir(parents=True, exist_ok=True)
        temp = marker.with_suffix('.tmp')
        temp.write_text(json.dumps({'schema_version': 1, 'contract': contract,
                                    'stage': 'final' if stage == 'record-final' else 'encoded'}, indent=2) + '\n')
        temp.replace(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-json', required=True)
    parser.add_argument('--skillset-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stage', choices=['preflight', 'skillset', 'encoded', 'final', 'record-encoded', 'record-final'], required=True)
    parser.add_argument('--threshold-reference', default='')
    parser.add_argument('--snap-reference', default='')
    args = parser.parse_args()
    contract = json.loads(args.contract_json)
    if args.threshold_reference:
        path = Path(args.threshold_reference)
        validate_threshold(json.loads(path.read_text()), contract, path)
    validate_snap_reference(args.snap_reference, contract)
    check_run(args.skillset_dir, args.run_dir, contract, args.stage)


if __name__ == '__main__':
    main()
