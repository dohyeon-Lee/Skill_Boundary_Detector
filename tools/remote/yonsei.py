#!/usr/bin/env python3
"""Freeze a workflow, stage its inputs, and launch it on the Yonsei login host.

SSH never connects to a compute node. Code is copied into a private per-launch
checkout, and data/checkpoints are reused from the destination workspace. Missing
inputs are copied through staging directories; existing inputs are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[2]
TREE = Path('lerobot/skillVLA')
SHARED = TREE / 'libero/configs'
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
CODE_SUFFIXES = {'.py', '.sh', '.sbatch', '.yaml', '.yml', '.json', '.toml',
                 '.md', '.txt', '.cfg', '.ini', '.xml', '.urdf', '.obj', '.stl', '.mjcf', '.csv'}
SKIP_PARTS = {'.git', '.venv', '__pycache__', '.cache', '.config_snapshots', '.hf_configs',
              'checkpoints', 'partial_build', 'logs', 'logs_train', 'wandb', 'outputs',
              'outputs_filtered', 'outputs_ABC', 'dataset', 'dataset_filtered', 'dataset_ABC', 'models'}
SELECTORS = {'TRAIN_DATA', 'SOURCE_DATA', 'STAGE1_ARCHITECTURE_OVERRIDE',
             'TERMINATOR_ARCHITECTURE_OVERRIDE', 'SKILL_AUX_PREDICTOR_MODE_OVERRIDE',
             'EVAL_MAX_CONCURRENT', 'STAGE1_EVAL_JOB_NAME', 'EVAL_NODE_LOCAL_VENV'}


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def load_profile(root=ROOT):
    import runpy
    loader = runpy.run_path(str(root / SHARED / 'src/global_config_loader.py'))
    return loader['load_global_config'](root / SHARED / 'global_config.yaml', server='yonsei')


def endpoint(profile, env):
    host = env.get('YONSEI_HOST') or profile.get('remote_host', 'mdorazi@165.132.142.208')
    base = env.get('YONSEI_PROJECT_ROOT') or env.get('YONSEI_SCRATCH2_BASE') or profile.get(
        'remote_project_root', '/scratch2/mdorazi/Skill_Boundary_Detector')
    if not re.fullmatch(r'[\w.@:\[\]-]+', host) or host.startswith('-'):
        raise ValueError('Invalid Yonsei SSH host')
    if not Path(base).is_absolute() or Path(base) == Path('/') or '..' in Path(base).parts:
        raise ValueError('Yonsei project root must be an absolute workspace path')
    return host, str(Path(base))


def selectors(env):
    # Configs and experiment selectors only; never credentials, CUDA/Slurm state,
    # or a caller's scheduler/partition/venv overrides.
    selected = {k: v for k, v in env.items() if k in SELECTORS or
            (k.endswith(('_CONFIG', '_CONFIG_RESOLVER', '_DRY_RUN')) and
             k.startswith(('ABC_', 'TRAIN_', 'STAGE1_', 'SKILL_', 'NEWTASK_',
                           'PI05_', 'LIBERO_', 'FSQ_', 'FT_', 'XYZ_', 'ATTENTION_',
                           'FILTERED_', 'ORIGINAL_', 'GROOT_', 'MOLMOACT2_')))}
    for key, value in list(selected.items()):
        if key.endswith('_CONFIG') and '.config_snapshots' in Path(value).parts:
            print(f'Ignoring inherited automatic snapshot: {key}={value}; using the current workflow YAML.',
                  file=sys.stderr, flush=True)
            selected.pop(key)
            selected.pop(key.removesuffix('_CONFIG') + '_JOB_NAME', None)
    return selected


def parse_flags(args):
    forwarded, plan, check = [], False, False
    enabled = True
    for arg in args:
        if enabled and arg == '--':
            enabled = False
            forwarded.append(arg)
        elif enabled and arg in {'--yonsei', '--yonsei-plan', '--yonsei-check'}:
            plan |= arg == '--yonsei-plan'
            check |= arg == '--yonsei-check'
        else:
            forwarded.append(arg)
    if plan and check:
        raise ValueError('Choose --yonsei-plan or --yonsei-check, not both')
    return forwarded, plan, check


def config_and_resolver(script, args, env, root=ROOT):
    """Use the same config resolvers as native launchers, without submitting jobs."""
    rel = script.relative_to(root).as_posix()
    family = 'abc' if '/abc/' in rel else 'libero'
    cfg = TREE / family / 'configs'
    parent = script.parent
    variable, default, resolver, extra = '', '', None, []
    if '/generate_training_dataset/visualized_dataset/' in rel and family == 'abc':
        default = 'visualized_dataset_config.yaml'
        resolver = parent / 'src/visualized_dataset_config.py'
    elif '/train_skillVLA/' in rel:
        common = root / SHARED / 'train_skillVLA'
        if '/eval/' in rel and script.name == 'submit_eval.sh':
            if family == 'libero':
                variable = 'STAGE1_EVAL_CONFIG'
                default = 'ft_eval_config.yaml' if '/FT/' in rel else 'pt_eval_config.yaml'
                resolver = common / 'PT/eval/src/stage1_eval_config.py'
            else:
                variable, default = 'ABC_SKILLVLA_EVAL_CONFIG', 'pt_eval_config.yaml'
                resolver = root / cfg / 'train_pi05/PT/eval/src/eval_config.py'
        elif '/FT/eval/' in rel and script.name == 'submit_pred_eval.sh':
            variable, default = 'FT_PRED_EVAL_CONFIG', 'ft_pred_eval_config.yaml'
            resolver = common / 'FT/eval/src/ft_pred_eval_config.py'
        elif '/FT/VSA/' in rel:
            variable, default = 'NEWTASK_FT_VSA_CONFIG', 'vsa_ft_config.yaml'
            resolver = common / 'FT/VSA/src/newtask_ft_vsa_config.py'
        elif '/PT/VSA/' in rel:
            variable, default = 'STAGE1_TRAIN_CONFIG', 'vsa_train_config.yaml'
            resolver = common / 'PT/src/stage1_train_config.py'
        elif any('/' + part + '/' in rel for part in ('Predictor', 'Terminator')):
            variable = 'SKILL_AUX_TRAIN_CONFIG'
            component = 'predictor' if '/Predictor/' in rel else 'terminator'
            default = component + ('_ft_config.yaml' if '/FT/' in rel else '_train_config.yaml')
            resolver = common / 'PT/Terminator/src/terminator_train_config.py'
        elif '/FT/Joint/' in rel:
            variable, default = 'NEWTASK_FT_JOINT_CONFIG', 'joint_ft_config.yaml'
            resolver = common / 'FT/Joint/src/newtask_ft_joint_config.py'
        elif '/build_data/' in rel:
            variable = 'ABC_SKILLVLA_CONFIG' if family == 'abc' else 'TRAIN_SKILLVLA_CONFIG'
            default = 'train_skillVLA_config.yaml'
            resolver = root / cfg / 'train_skillVLA/build_data/src/train_skillVLA_config.py'
    elif '/train_skills/' in rel:
        common = root / cfg / 'train_skills'
        if '/DP/eval/' in rel:
            variable = 'ABC_DP_EVAL_CONFIG' if family == 'abc' else 'LIBERO_DP_EVAL_CONFIG'
            default, resolver = 'eval_config.yaml', common / 'DP/eval/src/config.py'
        elif '/DP/' in rel:
            variable = 'ABC_DP_CONFIG' if family == 'abc' else 'TRAIN_SKILLS_CONFIG'
            default = 'dp_config.yaml'
            resolver = common / ('DP/src/train_dp_config.py' if family == 'abc' else 'src/train_skills_config.py')
            extra = [] if family == 'abc' else ['--stage', 'dp']
        elif '/FSQ/eval/' in rel:
            variable, default = 'ABC_FSQ_EVAL_CONFIG', 'eval_config.yaml'
            resolver = root / SHARED / 'train_skills/skill_eval/fsq_codebook/src/config.py'
        elif '/FSQ/' in rel and script.name == 'submit_train_fsq.sh':
            variable = 'ABC_FSQ_CONFIG' if family == 'abc' else 'TRAIN_SKILLS_CONFIG'
            default, resolver = 'fsq_config.yaml', common / 'src/train_skills_config.py'
            if family == 'abc':
                extra = ['--stage', 'fsq']
        elif '/build_data/' in rel:
            variable = 'ABC_BUILD_DATA_CONFIG' if family == 'abc' else 'TRAIN_SKILLS_CONFIG'
            default, resolver = 'build_data_config.yaml', common / 'src/train_skills_config.py'
            extra = ['--stage', 'build_data']
    elif '/train_pi05/' in rel:
        if '/eval/' in rel:
            variable = 'ABC_SIM_EVAL_CONFIG' if family == 'abc' else 'PI05_EVAL_CONFIG'
            default = 'abc_sim_eval_config.yaml' if family == 'abc' else 'pi05_eval_config.yaml'
            resolver = root / cfg / ('train_pi05/PT/eval/src/eval_config.py' if family == 'abc'
                                     else 'train_pi05/PT/eval/src/pi05_eval_config.py')
        else:
            variable, default = 'TRAIN_PI05_CONFIG', 'pi05_FT_config.yaml' if '/FT/' in rel else 'pi05_PT_config.yaml'
            resolver = root / cfg / 'train_pi05/PT/src/train_pi05_config.py'
    elif '/train_cyclic/pi05/' in rel:
        is_ft = '/FT/' in rel
        if '/eval/' in rel:
            variable, default = 'PI05_EVAL_CONFIG', 'pi05_eval_config.yaml'
            resolver = root / cfg / 'train_pi05/PT/eval/src/pi05_eval_config.py'
        else:
            variable = 'TRAIN_CYCLIC_FT_CONFIG' if is_ft else 'TRAIN_CYCLIC_CONFIG'
            mode = next((a for a in args if a in {'iid','cyclic'}), 'iid')
            default = 'ft_config.yaml' if is_ft else mode + '_config.yaml'
            resolver = root / cfg / ('train_cyclic/pi05/src/ft_config.py' if is_ft
                                     else 'train_cyclic/pi05/src/config.py')
    elif any('/train_' + name + '/' in rel for name in ('groot','molmoact2')):
        name = 'groot' if '/train_groot/' in rel else 'molmoact2'
        if '/eval/' in rel:
            variable = name.upper() + ('_FT_EVAL_CONFIG' if '/FT/' in rel else '_EVAL_CONFIG')
            default = name + '_eval_config.yaml'
            resolver = root / cfg / 'src/baseline_eval/config.py'
            extra = ['--work-dir', str(parent)]
        elif script.name == 'submit_train.sh':
            variable = name.upper() + '_TRAIN_CONFIG'
            default = name + ('_FT_config.yaml' if '/FT/' in rel else '_PT_config.yaml')
            resolver = root / cfg / ('train_' + name + '/PT/src/' + name + '_config.py')
    if script.name == 'make_together.sh':
        default = 'together_config.yaml'
    elif script.name == 'run_visualize.sh':
        default = 'visualize_config.yaml'
    if not default and (parent/'config.yaml').is_file():
        default = 'config.yaml'
    # Explicit config arguments win, just as in the launcher itself.
    explicit = None
    for index, arg in enumerate(args):
        if arg == '--config' and index + 1 < len(args):
            explicit = args[index + 1]
        elif arg.startswith('--config='):
            explicit = arg.split('=', 1)[1]
    default_path = parent/default if default else None
    if default_path and not default_path.is_file():
        for ancestor in list(parent.parents)[:3]:
            if (ancestor/default).is_file():
                default_path = ancestor/default
                break
    config = explicit or env.get(variable) or (str(default_path) if default_path else None)
    path = Path(config).absolute() if config else None
    if path and not path.is_file():
        raise FileNotFoundError('Evaluation/training config not found: ' + str(path))
    if family == 'libero' and env.get('TRAIN_DATA') and '/train_skills/' in rel:
        extra += ['--dataset', env['TRAIN_DATA']]
    return path, resolver, extra


def resolve_settings(script, args, env, root=ROOT):
    config, resolver, extra = config_and_resolver(script, args, env, root)
    if resolver is None:
        return config, {}
    if not resolver.is_file():
        raise FileNotFoundError('Native workflow config resolver not found: ' + str(resolver))
    command = [str(root / '.venv/bin/python'), str(resolver), '--config', str(config), *extra, '--shell']
    resolver_env = {**os.environ, **env,
        'PYTHONPATH': str(root / 'lerobot/src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    if resolver.name == 'stage1_eval_config.py' and script.name == 'submit_eval.sh':
        # Reproduce the PT/FT shell wrapper rather than inheriting its caller's directory.
        resolver_env['STAGE1_EVAL_WORK_DIR'] = str(script.parent)
    result = run(command, cwd=root, env=resolver_env, capture_output=True)
    exports = {}
    for line in result.stdout.splitlines():
        words = shlex.split(line)
        if len(words) == 2 and words[0] == 'export' and '=' in words[1]:
            key, value = words[1].split('=', 1)
            try:
                exports[key.lower()] = json.loads(value)
            except (ValueError, TypeError):
                exports[key.lower()] = value
    return config, exports


def configured_file_inputs(config, root=ROOT):
    """Explicit paths in utility/report YAMLs and extra config arguments."""
    found = set()
    if not config:
        return found
    def walk(value):
        if isinstance(value, dict):
            for k,v in value.items():
                if k.endswith(('_root', '_root_override')) or any(word in k.lower() for word in ('output','archive')):
                    continue
                walk(v)
        elif isinstance(value,list):
            for v in value: walk(v)
        elif isinstance(value,str) and value not in {'', '.', '..'}:
            for base in (config.parent, config.parent.parent, root):
                p = Path(value) if Path(value).is_absolute() else base/value
                if p.exists() and p.absolute().is_relative_to(root):
                    if p.suffix not in CODE_SUFFIXES or (p.is_dir() and p != root):
                        found.add(p.absolute())
                    break
    walk(yaml.safe_load(config.read_text()))
    return found


def asset_relative(path, root=ROOT):
    """Keep lexical workspace aliases, including checkpoints actually on data2."""
    path = Path(path).absolute()
    try:
        return path.relative_to(root)
    except ValueError:
        pass
    for part in ('models', 'dataset', 'dataset_filtered', 'dataset_ABC', 'outputs',
                 'outputs_filtered', 'outputs_ABC', 'libero_original_dataset', 'third_party'):
        base = root / part
        if base.is_symlink():
            try:
                return Path(part) / path.relative_to(base.resolve())
            except ValueError:
                pass
        if part in path.parts:
            return Path(*path.parts[path.parts.index(part):])
    # External output storage uses the same group/run layout as outputs_filtered.
    for group in ('skillVLA_PT', 'skillVLA_FT', 'DP', 'FSQ', 'pi05_PT', 'pi05_FT'):
        if group in path.parts:
            candidate = Path('outputs_filtered') / Path(*path.parts[path.parts.index(group):])
            if (root / candidate).exists() and (root / candidate).resolve() == path.resolve():
                return candidate
    raise ValueError('Input outside the portable project layout: ' + str(path))


def input_paths(settings, root=ROOT):
    paths = set()
    def walk(value, key=''):
        if isinstance(value, dict):
            for name, child in value.items():
                walk(child, name.lower())
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif isinstance(value, str):
            if value.startswith(('{', '[')):
                try:
                    walk(json.loads(value), key)
                except ValueError:
                    pass
                return
            if any(term in key for term in ('output', 'archive', 'log_', 'repo_', 'project_root')) or key.endswith('_root'):
                return
            p = Path(value)
            if not p.is_absolute() or not p.exists():
                return
            try:
                relative = asset_relative(p, root)
            except ValueError:
                return
            first = relative.parts[0]
            if first in {'models', 'libero_original_dataset'} or first.startswith('dataset'):
                paths.add(p)
            elif first.startswith('outputs') and ('checkpoints' in relative.parts or p.is_file()):
                paths.add(p)
            elif relative.suffix in {'.npz', '.json'} and any(
                    part in relative.parts for part in ('initial_state_banks', 'references')):
                paths.add(p)
    walk(settings)
    # Dataset contracts often validate provenance adjacent to skillvla/.
    for path in tuple(paths):
        if path.name == 'skillvla' and path.is_dir():
            for name in ('FSQ.pt', 'fsq_source.json', 'skill_focus_uv.npz',
                         'skill_latents.npz', 'initial_state.npz', 'eval_init_states.npz'):
                if (path.parent / name).is_file():
                    paths.add(path.parent / name)
        if path.is_dir() and path.name == 'pretrained_model' and (path/'config.json').is_file():
            contract = json.loads((path/'config.json').read_text())
            for key in ('fsq_path','skill_dataset_dir'):
                raw = contract.get(key)
                if not raw: continue
                candidate = Path(raw)
                if not candidate.is_absolute(): candidate = root/candidate
                if not candidate.exists():
                    try: candidate = root/asset_relative(candidate, root)
                    except ValueError: continue
                if candidate.is_file(): paths.add(candidate)
                elif candidate.is_dir():
                    for name in ('meta/info.json','meta/stats.json'):
                        if (candidate/name).is_file(): paths.add(candidate/name)
                    for name in ('FSQ.pt','fsq_source.json','skill_focus_uv.npz'):
                        if (candidate.parent/name).is_file(): paths.add(candidate.parent/name)
    selected = []
    for p in sorted(paths, key=lambda p: len(p.parts)):
        if not any(p.is_relative_to(parent) for parent in selected if parent.is_dir()):
            selected.append(p)
    return selected


def portable(value, root, destination, profile, key=''):
    if isinstance(value, dict):
        out = {k: portable(v, root, destination, profile, k) for k, v in value.items()}
        for name in ('train_partition', 'train_qos', 'train_nodelist', 'train_exclude_nodes',
                     'cpu_partition', 'cpu_qos', 'cpu_account', 'cpu_exclude_nodes'):
            if name in out:
                out[name] = profile.get(name, '' if not name.endswith('nodes') else [])
        if 'slurm_submit_mode' in out:
            out['slurm_submit_mode'] = profile.get('slurm_submit_mode', 'sbatch')
        if 'scheduler' in out:
            out['scheduler'] = 'slurm'
        if 'server' in out:
            out['server'] = 'yonsei'
        if key == 'slurm':
            for old, canonical in [('partition', 'train_partition'), ('qos', 'train_qos'),
                                   ('nodelist', 'train_nodelist'), ('exclude_nodes', 'train_exclude_nodes')]:
                if old in out:
                    out[old] = profile.get(canonical, '')
        return out
    if isinstance(value, list):
        return [portable(v, root, destination, profile, key) for v in value]
    if not isinstance(value, str):
        return value
    if key == 'checkpoint_archive':
        return ''  # Yonsei must never offload/delete checkpoints to itself.
    if (key == 'gres' or key.endswith('_gres')) and value.startswith('gpu'):
        base = str(profile.get('train_gres', 'gpu:1')).rsplit(':', 1)[0]
        count = value.rsplit(':', 1)[-1]
        return base + ':' + (count if count.isdigit() else '1')
    if value == str(root) or value.startswith(str(root) + '/'):
        return destination + value[len(str(root)):]
    if key in {'project_root', 'dataset_root', 'outputs_root'} and value.startswith('/'):
        if key == 'project_root':
            return destination
        return str(Path(destination) / str(profile.get(key, 'dataset_filtered' if key == 'dataset_root' else 'outputs_filtered')))
    if value.startswith('/') and Path(value).exists():
        try:
            return str(Path(destination)/asset_relative(Path(value),root))
        except ValueError:
            pass
    return value


def code_files(root=ROOT):
    raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
    files = set()
    for name in raw.decode().split('\0'):
        p = Path(name)
        if not name or any(part in SKIP_PARTS or part.startswith('.') for part in p.parts):
            continue
        f = root / p
        if f.is_file() and not f.is_symlink() and f.suffix in CODE_SUFFIXES:
            files.add(p)
    return files


RPC = r'''
import json, os, subprocess, sys, tempfile
from pathlib import Path
r = json.load(sys.stdin)
base = Path(r['base'])
if not base.is_dir() or not (base/'.venv/bin/python').is_file():
    raise RuntimeError('Yonsei workspace/.venv is missing: ' + str(base))
if r['action'] == 'probe':
    print(json.dumps({'host':os.uname().nodename,'python':str(base/'.venv/bin/python'),
        'missing':[p for p in r.get('inputs',[]) if not (base/p).exists()],
        'sbatch':subprocess.check_output(['which','sbatch'],universal_newlines=True).strip()}))
elif r['action'] == 'init':
    work = Path(r['work']);work.mkdir(parents=True,exist_ok=False)
    print(json.dumps({'work':str(work)}))
elif r['action'] == 'input':
    target=base/r['relative']
    stage=Path(r['stage'])
    import fcntl
    target.parent.mkdir(parents=True,exist_ok=True)
    with (target.parent/'.sbd-yonsei-input.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():os.rename(str(stage),str(target))
    print(json.dumps({'input':str(target)}))
elif r['action'] == 'stage':
    target=base/r['relative']
    target.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'stage':str(target.parent/('.sbd-yonsei-'+r['identifier']))}))
elif r['action'] == 'start':
    work=Path(r['work']); project=work/'project'
    subprocess.check_call(['git','init','-q',str(project)])
    for name in r['links']:
        link=project/name
        if not link.exists() and not link.is_symlink():
            target=base/name
            if name.startswith(('dataset','outputs')):target.mkdir(parents=True,exist_ok=True)
            if target.exists():
                link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target,target_is_directory=target.is_dir())
        elif link.is_dir() and not link.is_symlink() and (base/name).is_dir():
            # Code snapshots contain source/config files but omit binary simulator
            # assets. Keep frozen code; fill absent entries from shared dependencies.
            pending=[(link,base/name)]
            while pending:
                destination,source=pending.pop()
                for child in source.iterdir():
                    entry=destination/child.name
                    if not entry.exists() and not entry.is_symlink():
                        entry.symlink_to(child,target_is_directory=child.is_dir())
                    elif entry.is_dir() and not entry.is_symlink() and child.is_dir():
                        pending.append((entry,child))
    for name in r['manifest'].get('runtime_inputs',[]):
        if not (project/name).is_dir():
            raise RuntimeError('Required runtime directory missing before job submission: '+str(project/name))
    evaluation=r['manifest'].get('evaluation')
    if evaluation and evaluation.get('storage_project'):
        relative=Path(evaluation['output'])
        if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
            raise ValueError('Invalid evaluation output path')
        target=base/relative
        target.resolve().relative_to(base.resolve())
        link=project/relative
        # Keep source frozen while publishing reports in the normal workspace.
        if link.exists() or link.is_symlink():
            raise RuntimeError('Snapshot evaluation output already exists: '+str(link))
        target.mkdir(parents=True,exist_ok=True)
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(target,target_is_directory=True)
    manifest=work/'launch.json';manifest.write_text(json.dumps(r['manifest']))
    with (work/'console.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([str(base/'.venv/bin/python'),str(project/'tools/remote/yonsei.py'),
            'run',str(manifest)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    (work/'process.json').write_text(json.dumps({'pid':child.pid,
        'birth':Path('/proc/{}/stat'.format(child.pid)).read_text().rsplit(')',1)[1].split()[19]}))
    print(json.dumps({'pid':child.pid,'work':str(work),'log':str(work/'console.log')}))
elif r['action'] == 'status':
    work=Path(r['work']);p=work/'status.json'
    status=json.loads(p.read_text()) if p.exists() else {}
    process=json.loads((work/'process.json').read_text())
    try:
        fields=Path('/proc/{}/stat'.format(process['pid'])).read_text().rsplit(')',1)[1].split()
        alive=fields[0]!='Z' and fields[19]==process['birth']
    except OSError:alive=False
    if not alive and status.get('state') not in ('COMPLETED','FAILED'):
        status={'state':'FAILED','exit_code':1,'reason':'Remote launcher exited before recording completion'}
    print(json.dumps({'status':status,
                     'log':(work/'console.log').read_text(errors='replace')[-16000:] if (work/'console.log').exists() else ''}))
'''


def rpc(host, base, action, **values):
    result = run(['ssh', *SSH_OPTIONS, host, 'python3 -c ' + shlex.quote(RPC)],
                 input=json.dumps({'base': base, 'action': action, **values}), capture_output=True)
    return json.loads(result.stdout)


def transfer(host, source, destination, directory=False):
    # --copy-links is intentional for local storage aliases; no remote deletion.
    command = ['rsync', '-rltL', '--protect-args', '--partial-dir=.sbd-yonsei-partial',
               '--info=progress2' if sys.stdout.isatty() else '--stats',
               '--timeout=120', '-e', shlex.join(['ssh', *SSH_OPTIONS]),
               str(source) + ('/' if directory else ''), host + ':' + destination + ('/' if directory else '')]
    run(command)


def freeze(root, destination, profile, selected, config, extra_files):
    files = code_files(root) | set(extra_files)
    if config:
        try:
            files.add(config.relative_to(root))
        except ValueError:
            raise ValueError('Keep remote launch YAML inside the project so its shared defaults can be frozen')
    for rel in files:
        src, dst = root / rel, selected / rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in {'.yaml', '.yml'} and TREE in rel.parents and 'servers' not in rel.parts:
            data = yaml.safe_load(src.read_text())
            dst.write_text(yaml.safe_dump(portable(data, root, destination, profile),sort_keys=False,allow_unicode=True))
        else:
            shutil.copy2(src, dst)
    return len(files)


def evaluation_spec(script, settings, root):
    """Record only this evaluation's output, never the entire outputs tree."""
    if '/eval/' not in script.as_posix():
        return None
    output = next((settings.get(k) for k in ('eval_out_dir', 'eval_output_dir', 'output_dir',
                                             'abc_eval_output_dir', 'dp_ablation_output_dir')
                   if settings.get(k)), None)
    if not output:
        return None
    output = Path(output)
    relative = output.absolute().relative_to(root)
    if relative.name in {'outputs', 'attention_outputs', '.', '..'}:
        raise ValueError('Remote results must identify one evaluation run directory')
    def decoded(value):
        return json.loads(value) if isinstance(value, str) else value
    models = decoded(settings.get('models_json', []))
    tasks = decoded(settings.get('task_ids', []))
    labels = [m['label'] for m in models] if isinstance(models, list) else []
    spec = {'output': relative.as_posix(), 'tasks': tasks if isinstance(tasks, list) else [],
            'labels': labels or ['eval'], 'episodes': int(settings.get('n_episodes') or 0)}
    if settings.get('abc_eval_output_dir'):
        spec['kind'] = 'abc'
    return spec


def launch(script, args, root=ROOT, env=None):
    env = dict(os.environ if env is None else env)
    script = Path(script).absolute()
    relative = script.relative_to(root)
    if not relative.is_relative_to(TREE) or script.suffix != '.sh' or not script.is_file():
        raise ValueError('Remote entry must be a .sh script below lerobot/skillVLA')
    args, plan_only, check_only = parse_flags(args)
    profile = load_profile(root)
    host, base = endpoint(profile, env)
    exported = selectors(env)
    config, settings = resolve_settings(script, args, exported, root)
    inputs = input_paths(settings, root) + list(configured_file_inputs(config, root))
    runtime_inputs = []
    if ('/libero/' in relative.as_posix() and '/eval/' in relative.as_posix()
            and script.name == 'submit_eval.sh' and settings.get('models_json')):
        benchmark = root / 'tools/lerobot-libero/libero/libero'
        for name in ('init_files', 'bddl_files', 'assets'):
            dependency = benchmark / name
            if not dependency.is_dir():
                raise FileNotFoundError('Required LIBERO runtime input: ' + str(dependency))
            inputs.append(dependency)
            runtime_inputs.append(dependency.relative_to(root).as_posix())
    # Explicit file arguments and custom config selectors travel with the recipe.
    files = {relative}
    for value in [*args, *exported.values()]:
        candidate = value.split('=',1)[-1] if value.startswith('--') and '=' in value else value
        p = Path(candidate)
        if p.is_file():
            if p.suffix in {'.yaml', '.yml', '.py', '.sh'}:
                files.add(p.absolute().relative_to(root))
            else:
                inputs.append(p.absolute())
    inputs = sorted(set(inputs), key=str)
    inputs = [p for p in inputs if not any(p != q and q.is_dir() and p.is_relative_to(q) for q in inputs)]
    names = [asset_relative(p, root).as_posix() for p in inputs]
    summary = {'host': host, 'project': base, 'script': str(relative), 'args': args,
               'config': str(config) if config else None, 'inputs': names,
               'slurm_profile': profile.get('slurm_profile'),
               'slurm_submit_mode': profile.get('slurm_submit_mode', 'sbatch'),
               'partition': profile.get('train_partition'), 'qos': profile.get('train_qos'),
               'gres': profile.get('train_gres', 'gpu:1')}
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if plan_only:
        print('Plan only: no SSH, transfer, or job submission.')
        return 0
    probe = rpc(host, base, 'probe', inputs=names)
    if check_only:
        print(json.dumps(probe, indent=2, ensure_ascii=False))
        print('Check only: no files changed and no jobs submitted.')
        return 0
    identifier = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:10]
    work = base + '/.cache/yonsei_launches/' + identifier
    destination = work + '/project'
    audit = root / '.cache/yonsei_launches' / identifier
    audit.mkdir(parents=True)
    rpc(host, base, 'init', work=work)
    with tempfile.TemporaryDirectory(prefix='sbd-yonsei-') as temporary:
        packet = Path(temporary)
        freeze(root, destination, profile, packet, config, files)
        transfer(host, packet, destination, directory=True)
    missing = set(probe['missing'])
    for path, name in zip(inputs, names):
        if name not in missing:
            print('Reuse Yonsei input: ' + name, flush=True)
            continue
        stage = rpc(host,base,'stage',relative=name,identifier=uuid.uuid4().hex)['stage']
        print('Copy missing Yonsei input: ' + name, flush=True)
        transfer(host, path, stage, directory=path.is_dir())
        rpc(host, base, 'input', relative=name, stage=stage)
    def remap(text):
        if text == str(root) or text.startswith(str(root) + '/'):
            return destination + text[len(str(root)):]
        return text
    forwarded = [remap(a) if not a.startswith('--') or '=' not in a else
                 a.split('=', 1)[0] + '=' + remap(a.split('=', 1)[1]) for a in args]
    exported = {k: remap(v) for k,v in exported.items()}
    cwd = Path.cwd()
    cwd = destination + '/' + str(cwd.relative_to(root)) if cwd.is_relative_to(root) else str(Path(destination) / relative.parent)
    manifest = {'source': str(root), 'host': host, 'project': destination, 'script': str(relative),
                'cwd': cwd, 'args': forwarded, 'environment': exported,
                'profile': summary, 'inputs': names,
                'runtime_inputs': runtime_inputs,
                'created_at': time.time(),
                'evaluation': evaluation_spec(script, settings, root),
                'dry_run': any(k.endswith('_DRY_RUN') and v == '1' for k,v in exported.items())
                           or '--dry-run' in forwarded}
    if not manifest['evaluation'] and 'train' in script.stem:
        raw = (yaml.safe_load(config.read_text()) or {}) if config else {}
        protected = (settings.get('protect_checkpoints') or raw.get('training', {}).get('protect_checkpoints') or
                     raw.get('protect_checkpoints') or raw.get('dp_protect_checkpoints') or [])
        if isinstance(protected, str):
            protected = json.loads(protected)
        if not isinstance(protected, list) or any(type(n) is not int or n < 0 for n in protected):
            raise ValueError('protect_checkpoints must be a list of non-negative integers')
        manifest['training'] = {'checkpoint_sync': True, 'protected': protected}
        for key in ('dp_steps', 'steps', 'fsq_num_epochs'):
            if settings.get(key):
                manifest['training']['total'] = int(settings[key])
                manifest['training']['unit'] = 'epoch' if key == 'fsq_num_epochs' else 'step'
                break
        dataset = next((settings.get(k) for k in ('skillvla_dataset_dir', 'dp_dataset_dir', 'dataset_dir')
                        if settings.get(k)), None)
        if dataset:
            manifest['training']['dataset'] = asset_relative(Path(dataset), root).as_posix()
    if manifest['evaluation'] and not manifest['dry_run']:
        manifest['evaluation']['storage_project'] = base
    (audit/'launch.json').write_text(json.dumps(manifest,indent=2))
    links = ['.venv', '.venv_groot_lerobot', '.venv_molmoact2', 'models', 'third_party', 'libero_original_dataset',
             'tools/lerobot-libero', 'tools/openpi', 'lerobot/libero']
    links += sorted({n.split('/')[0] for n in names if n.startswith(('dataset', 'outputs'))})
    links += ['dataset_filtered','dataset_ABC','outputs_filtered','outputs_ABC']
    # Small banks/references live alongside code, outside the root data links.
    links += [n for n in names if not n.startswith(('dataset','outputs','models','third_party','libero_original_dataset'))]
    started = rpc(host, base, 'start', work=work, links=links, manifest=manifest)
    (audit/'remote.json').write_text(json.dumps(started,indent=2))
    # Collection is supervised locally and survives terminal/UI closure.
    if (manifest['evaluation'] or manifest.get('training')) and not manifest['dry_run']:
        sys.path.insert(0, str(root))
        from tools.remote.desk import ensure_daemon
        ensure_daemon(root)
    print('Yonsei launch: ' + work, flush=True)
    print('Log: ssh ' + shlex.quote(host) + ' ' + shlex.quote('tail -f ' + shlex.quote(started['log'])), flush=True)
    # Surface immediate configuration/submission failures without attaching the SSH
    # session to a long preprocessing task or a pending srun allocation.
    for _ in range(8):
        result = rpc(host, base, 'status', work=work)
        (audit/'status.json').write_text(json.dumps(result['status'],indent=2))
        if result['status'].get('state') in {'COMPLETED','FAILED'}:
            print(result['log'])
            return int(result['status'].get('exit_code', 1))
        time.sleep(1)
    print('Remote launcher is running; follow the log above.')
    return 0


SBATCH_RECORDER = '''#!/usr/bin/env python3
import fcntl, json, os, re, subprocess, sys, shlex, uuid
from pathlib import Path
args = sys.argv[1:]
array = None
for i, value in enumerate(args):
    if value.startswith('--array='): array = value.split('=',1)[1]
    elif value in {'--array','-a'} and i+1<len(args): array = args[i+1]
tasks = [None]
if array:
    tasks=[]
    for item in array.split('%',1)[0].split(','):
        m=re.fullmatch(r'(\\d+)(?:-(\\d+)(?::(\\d+))?)?',item)
        if not m: raise ValueError('Unsupported Slurm array specification')
        start=int(m[1]);stop=int(m[2] or m[1]);step=int(m[3] or 1)
        if step<1 or stop-start>10000: raise ValueError('Invalid Slurm array size')
        tasks.extend(range(start,stop+1,step))
# Durable per-array-element exits are independent of slurmdbd/controller retention.
status_dir=os.environ.get('SBD_YONSEI_JOB_STATUS_DIR')
if status_dir:
    folder=Path(status_dir);folder.mkdir(parents=True,exist_ok=True)
    for i,value in enumerate(args):
        if value.startswith('-') or not Path(value).is_file(): continue
        original=Path(value).absolute()
        text=original.read_text()
        if not text.startswith('#!'): continue
        wrapper=folder/('batch-'+uuid.uuid4().hex+'.sh')
        directives=''.join(line+'\\n' for line in text.splitlines() if line.startswith('#SBATCH'))
        body="""#!/usr/bin/env bash
"""+directives+"""
record_exit() {
  result=$?
  trap - EXIT
  master=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}
  task=${SLURM_ARRAY_TASK_ID:-solo}
  if [[ "$master" =~ ^[0-9]+$ && "$task" =~ ^([0-9]+|solo)$ ]]; then
    marker="${SBD_YONSEI_JOB_STATUS_DIR}/${master}_${task}.exit"
    printf '%s\\n' "$result" > "${marker}.$$"
    mv -- "${marker}.$$" "$marker"
  fi
  exit "$result"
}
trap record_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash """+shlex.quote(str(original))+""" "$@"
"""
        wrapper.write_text(body);wrapper.chmod(0o755);args[i]=str(wrapper);break
result = subprocess.run([os.environ['SBD_YONSEI_REAL_SBATCH'], *args],
                        stdout=subprocess.PIPE, text=True)
sys.stdout.write(result.stdout)
sys.stdout.flush()
if result.returncode == 0:
    match = re.search(r'(?:Submitted batch job |^)([0-9]+)(?:;[^\\n]*)?\\s*$', result.stdout)
    if not match:
        print('Yonsei Desk: could not identify submitted job', file=sys.stderr)
        sys.exit(2)
    with open(os.environ['SBD_YONSEI_JOB_FILE'], 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({'job_id': match[1], 'tasks': tasks}) + '\\n')
sys.exit(result.returncode)
'''


def execute(manifest_path):
    path = Path(manifest_path)
    data = json.loads(path.read_text())
    status = path.parent / 'status.json'
    status.write_text(json.dumps({'state':'RUNNING','started_at':time.time()}))
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBD_'))
                   and k not in {'CUDA_VISIBLE_DEVICES','VIRTUAL_ENV','PYTHONPATH'}}
    environment.update(data['environment'], SBD_SERVER='yonsei', SBD_SCHEDULER='slurm',
                       SBD_REMOTE_ACTIVE='1', SBD_SLURM_SUBMIT_MODE=data['profile'].get('slurm_submit_mode','sbatch'))
    if data.get('training', {}).get('checkpoint_sync'):
        environment.update(SBD_YONSEI_CHECKPOINT_OUTBOX=str(path.parent / 'checkpoint_events'),
            SBD_YONSEI_LAUNCH_ID=path.parent.name, SBD_YONSEI_PROJECT=data['project'],
            SBD_YONSEI_WORKSPACE=data['profile']['project'],
            SBD_YONSEI_PROTECTED_CHECKPOINTS=json.dumps(data['training']['protected']),
            SBD_YONSEI_DATASET_RELATIVE=data['training'].get('dataset', ''))
    # The reused venv's editable lerobot install points at the old main checkout.
    # Resolve every bootstrap import from this launch's frozen source instead.
    # Slurm exports this path to workers as well as the submission-time resolver.
    environment['PYTHONPATH'] = str(Path(data['project']) / 'lerobot/src')
    real_sbatch = shutil.which('sbatch', path=environment.get('PATH'))
    if real_sbatch:
        bin_dir = path.parent / 'bin'
        bin_dir.mkdir(exist_ok=True)
        recorder = bin_dir / 'sbatch'
        recorder.write_text(SBATCH_RECORDER)
        recorder.chmod(0o755)
        environment.update(SBD_YONSEI_REAL_SBATCH=real_sbatch,
                           SBD_YONSEI_JOB_FILE=str(path.parent / 'jobs.jsonl'),
                           SBD_YONSEI_JOB_STATUS_DIR=str(path.parent / 'job_status'),
                           PATH=str(bin_dir) + os.pathsep + environment.get('PATH', ''))
    try:
        code = subprocess.call(['bash',str(Path(data['project'])/data['script']),*data['args']],
                               cwd=data['cwd'], env=environment)
    except Exception as error:
        print(str(error),file=sys.stderr)
        code = 1
    status.write_text(json.dumps({'state':'COMPLETED' if code == 0 else 'FAILED',
                                 'exit_code':code,'ended_at':time.time()}))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    p = sub.add_parser('launch');p.add_argument('--script',required=True);p.add_argument('args',nargs=argparse.REMAINDER)
    p = sub.add_parser('run');p.add_argument('manifest')
    args = parser.parse_args()
    if args.action == 'run':
        return execute(args.manifest)
    forwarded = args.args[1:] if args.args[:1] == ['--'] else args.args
    return launch(args.script,forwarded)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        print('Yonsei dispatch failed: ' + str(error),file=sys.stderr)
        if isinstance(error,subprocess.CalledProcessError) and error.stderr:
            print(error.stderr,file=sys.stderr)
        raise SystemExit(1)
