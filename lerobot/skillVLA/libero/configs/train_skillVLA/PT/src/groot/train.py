#!/usr/bin/env python3
"""Resolve existing SkillVLA YAMLs and run the shared trainer in the GR00T venv."""
from __future__ import annotations

import argparse
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
sys.path.insert(0, str(ROOT / 'lerobot/src'))


def resolved(path, phase, component):
    workflow = ROOT / 'lerobot/skillVLA/libero/configs/train_skillVLA'
    file = (workflow / 'FT/VSA/src/newtask_ft_vsa_config.py' if phase == 'ft' and component == 'VSA'
            else workflow / 'PT/src/stage1_train_config.py' if component == 'VSA'
            else workflow / 'PT/src/auxiliary/auxiliary_train_config.py')
    spec = importlib.util.spec_from_file_location('_skill_groot_resolver', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_stage1_component_config(path)
    return config, module.build_settings(config)


def decode(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and (value.startswith(('[', '{')) or value in ('true', 'false', 'null')):
        return json.loads(value)
    return value


def pipeline(recipe, settings, phase, component):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
    from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
    cls = SkillExpertConfig if component == 'VSA' else SkillAuxConfig
    names = {field.name for field in fields(cls)}
    params = {key: decode(value) for key, value in settings.items() if key in names}
    checkpoint = settings.get('vsa_checkpoint_path' if component == 'VSA' else 'predictor_checkpoint_path')
    if phase == 'ft':
        source = json.loads((Path(checkpoint) / 'config.json').read_text())
        # Architecture/quantiles/reader geometry are PT-owned; only documented FT
        # adaptation and training controls below may replace source fields.
        inherited = {key: value for key, value in source.items() if key in names}
        inherited.update({key: value for key, value in params.items() if key.startswith(('transition_jitter_', 'skill_predictor_'))})
        for key in ('skillvla_backbone', 'groot_base_path', 'tokenizer_path', 'fsq_path', 'gradient_checkpointing',
                    'vlm_checkpointing', 'vision_checkpointing', 'action_expert_checkpointing', 'skill_planner_checkpointing',
                    'training_batch_size', 'dataset_source_lineage', 'run_suffix_lineage'):
            if key in params:
                inherited[key] = params[key]
        params = inherited
        if component == 'VSA':
            params.update(newtask_ft_enabled=True,
                newtask_ft_freeze_expert=settings['freeze_expert'],
                newtask_ft_skill_flow_loss=settings['skill_flow_loss'],
                newtask_ft_visual_aux_loss=settings['visual_aux_loss'],
                vlm_termination_loss_weight=settings['vlm_termination_loss_weight'])
    params.update(device='cuda', push_to_hub=False, optimizer_lr=settings['lr'],
                  pretrained_path=str(checkpoint or settings['pi_base']),
                  scheduler_mode=settings['scheduler_mode'], scheduler_warmup_steps=settings['scheduler_warmup_steps'],
                  scheduler_decay_steps=settings['scheduler_decay_steps'])
    if component == 'VSA':
        params['dtype'] = 'float32'
    params['input_features'] = {
        'observation.state': PolicyFeature(FeatureType.STATE, (8,)),
        **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in
           decode(settings.get('skill_image_keys', '["observation.images.image","observation.images.wrist_image"]'))}}
    params['output_features'] = {'action': PolicyFeature(FeatureType.ACTION, (7,))}
    cfg = cls(**params)
    output = Path(settings.get('pt_output_dir', settings.get('output_dir')))
    episodes = decode(settings.get('dataset_episodes', 'null'))
    schedule = recipe.get('training', {}).get('schedule', {})
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=settings['repo_id'], root=str(settings['skillvla_dataset_dir']), episodes=episodes, video_backend='pyav'),
        policy=cfg, output_dir=output, job_name=output.name, steps=settings['steps'],
        batch_size=settings['batch_size'], num_workers=settings['num_workers'],
        dataloader_persistent_workers=settings.get('dataloader_persistent_workers', settings['num_workers'] > 0),
        dataloader_multiprocessing_context='spawn' if settings['num_workers'] else None,
        dataloader_prefetch_factor=settings.get('dataloader_prefetch_factor', 2),
        dataloader_pin_memory=settings.get('dataloader_pin_memory', False),
        save_freq=settings['save_freq'], log_freq=settings['log_freq'], eval_freq=0,
        save_steps=sorted(set(int(x) for x in schedule.get('save_steps', []))),
        keep_checkpoints=settings.get('keep_checkpoints', -1),
        protect_checkpoints=decode(settings.get('protect_checkpoints', '[]')),
        checkpoint_archive=settings.get('checkpoint_archive', ''),
        wandb=WandBConfig(enable=settings['wandb_enable'], project=settings['wandb_project']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--phase', choices=['pt', 'ft'], required=True)
    parser.add_argument('--component', choices=['VSA', 'Predictor'], required=True)
    parser.add_argument('--print-only', action='store_true')
    args = parser.parse_args()
    recipe, settings = resolved(args.config, args.phase, args.component)
    cfg = pipeline(recipe, settings, args.phase, args.component)
    if args.print_only:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    last = cfg.output_dir / 'checkpoints/last/pretrained_model/train_config.json'
    if last.is_file():
        source = json.loads(last.read_text())
        if source['policy'].get('skillvla_backbone', 'pi05') != recipe['skillvla_backbone']:
            raise ValueError('Existing output belongs to another backbone')
        # Shared trainer restores optimizer/RNG/processors as well as weights.
        sys.argv = [sys.argv[0], '--config_path=' + str(last), '--resume=true']
        cfg = TrainPipelineConfig.from_pretrained(last, cli_args=['--resume=true'])
    else:
        sys.argv = [sys.argv[0]]
    from lerobot.scripts.lerobot_train import train
    train(cfg)


if __name__ == '__main__':
    main()
