"""Coffee evaluation contract using the common SkillVLA panels and metrics."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'lerobot/src/lerobot').is_dir())
SHARED=ROOT/'lerobot/skillVLA/libero/configs'
sys.path.insert(0,str(SHARED/'train_skills/src'))
from train_skills_config import load_config,load_stage1_component_config,print_shell,stage1_run_dir
sys.path.insert(0,str(ROOT/"lerobot/skillVLA/mimicgen/configs/src"))
from model_paths import model_root


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def build_settings(cfg):
    if cfg.get('target_task')!='Coffee_D0': raise ValueError('This dataset and instruction are for Coffee_D0')
    if cfg.get('oracle',{}).get('episode_exact') or cfg.get('initial_state_bank'):
        raise ValueError('Coffee data has no demo-exact simulator state bank')
    models=[dict(row) for row in cfg['models']]
    train_config=cfg.get('vsa_training_config')
    if train_config:
        path=(ROOT/'lerobot/skillVLA/mimicgen/configs/train_skillVLA/FT/VSA/vsa_ft_config.yaml'
              if train_config == 'auto' else ROOT/train_config)
        resolver=module(SHARED/'train_skillVLA/FT/VSA/src/newtask_ft_vsa_config.py','coffee_ft_resolver')
        settings=resolver.build_settings(load_stage1_component_config(path))
        models[0]['vsa_dir']=str(settings['output_dir'])
    aux_configs=cfg.get('aux_training_configs', {})
    for component, config_path in aux_configs.items():
        if component not in {'predictor','terminator'}:
            raise ValueError('aux_training_configs supports predictor|terminator')
        resolver=module(SHARED/'train_skillVLA/PT/Terminator/src/terminator_train_config.py','coffee_aux_'+component)
        path=(ROOT/f'lerobot/skillVLA/mimicgen/configs/train_skillVLA/FT/{component.title()}/{component}_ft_config.yaml'
              if config_path == 'auto' else ROOT/config_path)
        settings=resolver.build_settings(load_stage1_component_config(path))
        models[0][component+'_dir']=str(settings['output_dir'])
    # The compact shared UI only accepts run names; translate explicit cross-domain
    # references to its established per-model root / external checkpoint selectors.
    resolved=[]
    for row in models:
        row=dict(row)
        outputs=model_root(cfg,row.pop('outputs_root',cfg.get('model_defaults',{}).get('outputs_root')))
        vsa=Path(row.pop('vsa_dir')).expanduser()
        if len(vsa.parts)==1:vsa=stage1_run_dir(outputs,str(vsa),'VSA')
        if not vsa.is_absolute():vsa=ROOT/vsa
        row['model_dir']=vsa.name;row['outputs_root']=str(vsa.parents[2])
        row['checkpoint']=row.pop('vsa_checkpoint')
        for component in ['predictor','terminator']:
            source=Path(row.pop(component+'_dir')).expanduser()
            if len(source.parts)==1:source=stage1_run_dir(outputs,str(source),component.title())
            if not source.is_absolute():source=ROOT/source
            step=str(row.pop(component+'_checkpoint'))
            if step.isdigit():step=step.zfill(6)
            row['external_'+component+'_model']=str(source/'checkpoints'/step/'pretrained_model')
        resolved.append(row)
    cfg={**cfg,'models':resolved}
    return module(SHARED/'train_skillVLA/PT/eval/src/stage1_eval_config.py','coffee_eval_shared').build_settings(cfg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--shell',action='store_true');a=p.parse_args()
    os.environ.setdefault('STAGE1_EVAL_WORK_DIR',str(a.config.resolve().parent if '.config_snapshots' not in a.config.parts else next(p for p in a.config.parents if p.name=='.config_snapshots').parent))
    settings=build_settings(load_config(a.config))
    if a.shell: print_shell(settings)
    else:
        for k,v in settings.items():print(f'{k}: {v}')
if __name__=='__main__':main()
