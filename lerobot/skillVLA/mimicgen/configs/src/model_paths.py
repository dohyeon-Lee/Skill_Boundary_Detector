"""Resolve concise model references without repeating workflow folder names."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def project_path(cfg, value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path(cfg['project_root']) / path


def model_root(cfg, override=None):
    return project_path(cfg, override or cfg['outputs_root'])


def expand_warm_start(cfg, resolve_checkpoint):
    """Resolve the active component only; other shared FT references may be remote."""
    warm = cfg.get('warm_start')
    if not isinstance(warm, dict):
        return cfg
    cfg = deepcopy(cfg)
    warm = cfg['warm_start']
    outputs = model_root(cfg, warm.get('outputs_root'))
    active = cfg.get('stage1_component')
    for key, component in [('vsa_checkpoint', 'VSA'), ('predictor_checkpoint', 'Predictor'),
                           ('terminator_checkpoint', 'Terminator')]:
        if active in {'VSA', 'Predictor', 'Terminator'} and component != active:
            warm.pop(key, None)
            continue
        selector = warm.get(key)
        if isinstance(selector, dict) and 'run' in selector:
            warm[key] = str(resolve_checkpoint(outputs, component, selector, field='warm_start.' + key))
        elif isinstance(selector, list):
            warm[key] = [str(resolve_checkpoint(outputs, component, item, field='warm_start.' + key))
                         if isinstance(item, dict) else item for item in selector]
    return cfg


def fsq_checkpoint(cfg):
    """FSQ directory and epoch filename are implied by run + checkpoint."""
    run = cfg.get('fsq_run_name')
    if not run:
        return project_path(cfg, cfg['fsq_checkpoint'])  # frozen legacy config
    if Path(str(run)).name != run or run in {'.', '..'}:
        raise ValueError('fsq_run_name must be a model folder name')
    directory = model_root(cfg, cfg.get('fsq_outputs_root')) / 'FSQ' / run
    step = str(cfg['fsq_checkpoint'])
    if step in {'best', '0'}:
        return directory / 'FSQ.pt'
    if not step.isdigit() or int(step) < 1:
        raise ValueError('fsq_checkpoint must be a positive epoch or best')
    return directory / f'FSQ_epoch{int(step):04d}.pt'


def vocabulary_reference(cfg):
    reference = cfg['fsq_snap_reference']
    if not isinstance(reference, dict):
        return project_path(cfg, reference)  # frozen legacy config
    root = project_path(cfg, reference.get('dataset_root', cfg['dataset_root']))
    return root / reference.get('skillvla_root', 'skillvla') / reference['source'] / reference['run'] / 'skill_latents.npz'
