"""The real LIBERO recipes share a dataset but keep backend checkpoints distinct."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / 'skillVLA/libero/configs/train_skillVLA'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load(WORKFLOW / 'PT/src/groot/train.py', 'groot_workflow_runtime')


@pytest.mark.parametrize('component', ['VSA', 'Predictor'])
def test_pt_backend_selection_and_serialized_training_config(component, tmp_path):
    recipe = WORKFLOW / ('PT/VSA/vsa_train_config.yaml' if component == 'VSA'
                         else 'PT/Predictor/predictor_train_config.yaml')
    original, _ = runtime.resolved(recipe, 'pt', component)
    resolver = load(WORKFLOW / ('PT/src/stage1_train_config.py' if component == 'VSA'
                                else 'PT/src/auxiliary/auxiliary_train_config.py'), 'backend_pt_' + component)
    from skillvla_backend import prepare
    # The live common YAML may currently select either backend. Start this
    # three-way comparison from an explicit pi0.5 recipe, not an already
    # expanded GR00T warm-start path.
    original['skillvla_backbone'] = 'pi05'
    original['warm_start']['pi_base'] = str(ROOT.parent / 'models/pi05_base')
    baseline = resolver.build_settings(original)
    outputs = []
    for selected in ['pi05', 'groot_integrate', 'groot_seperate']:
        config = copy.deepcopy(original)
        config['skillvla_backbone'] = selected
        config = prepare(config, recipe)
        settings = resolver.build_settings(config)
        pipeline = runtime.pipeline(config, settings, 'pt', component)
        destination = pipeline.output_dir
        pipeline.output_dir = tmp_path / selected
        pipeline.validate()
        pipeline.output_dir = destination
        assert pipeline.dataset.root == str(baseline['skillvla_dataset_dir'])
        assert pipeline.policy.skillvla_backbone == selected
        assert pipeline.to_dict()['policy']['type'] == ('skill_expert' if component == 'VSA' else 'skill_aux')
        assert pipeline.policy.transition_jitter_pmax_percent == 25
        outputs.append(pipeline.output_dir)
        if selected == 'pi05':
            assert pipeline.output_dir == baseline.get('pt_output_dir', baseline.get('output_dir'))
        else:
            assert selected in pipeline.output_dir.name
            assert str(pipeline.policy.pretrained_path).endswith('models/groot_n1_7_base')
    assert len(set(outputs)) == 3


@pytest.mark.parametrize('component', ['VSA', 'Predictor'])
@pytest.mark.parametrize('selected', ['groot_integrate', 'groot_seperate'])
def test_ft_inherits_matching_pt_contract(tmp_path, component, selected):
    from skillvla_backend import prepare
    pt_path = WORKFLOW / ('PT/VSA/vsa_train_config.yaml' if component == 'VSA'
                          else 'PT/Predictor/predictor_train_config.yaml')
    config, settings = runtime.resolved(pt_path, 'pt', component)
    config['skillvla_backbone'] = selected
    config = prepare(config, pt_path)
    settings['skillvla_backbone'] = selected
    settings['groot_base_path'] = str(ROOT.parent / 'models/groot_n1_7_base')
    pt = runtime.pipeline(config, settings, 'pt', component)
    checkpoint = tmp_path / 'skillVLA_PT' / component / 'matching' / 'checkpoints/100000/pretrained_model'
    checkpoint.mkdir(parents=True)
    pt.policy.save_pretrained(checkpoint)
    (checkpoint / 'model.safetensors').touch()  # resolver validates existence; no weights needed here
    (checkpoint / 'policy_preprocessor.json').write_text('{}')
    ft_path = WORKFLOW / ('FT/VSA/vsa_ft_config.yaml' if component == 'VSA'
                          else 'FT/Predictor/predictor_ft_config.yaml')
    resolver = load(WORKFLOW / ('FT/VSA/src/newtask_ft_vsa_config.py' if component == 'VSA'
                                else 'PT/src/auxiliary/auxiliary_train_config.py'), 'backend_ft_' + component)
    ft = resolver.load_stage1_component_config(ft_path)
    ft['skillvla_backbone'] = selected
    ft['outputs_root'] = str(tmp_path)
    key = 'vsa_checkpoint' if component == 'VSA' else 'predictor_checkpoint'
    ft['warm_start']['backend_sources'] = {selected: {key: {'run': 'matching', 'checkpoint': '100000'}}}
    ft = prepare(ft, ft_path)
    resolved = resolver.build_settings(ft)
    pipeline = runtime.pipeline(ft, resolved, 'ft', component)
    pipeline.validate()
    assert pipeline.policy.skillvla_backbone == selected
    assert str(pipeline.policy.pretrained_path) == str(checkpoint)
    assert pipeline.policy.transition_jitter_pmax_percent == pt.policy.transition_jitter_pmax_percent
    assert len(pipeline.dataset.episodes) == 10
    if component == 'VSA':
        assert pipeline.policy.newtask_ft_freeze_expert
        assert pipeline.policy.newtask_ft_skill_flow_loss
        assert pipeline.policy.newtask_ft_visual_aux_loss
        assert pipeline.policy.bspline_gripper_action == pt.policy.bspline_gripper_action
    else:
        assert pipeline.policy.skill_predictor_mode == pt.policy.skill_predictor_mode == 'mode2'
        assert pipeline.policy.dataset_source_lineage[-1] == 'libero_10_full_full'
    # A wrong backbone must fail before loading tensors or submitting a job.
    saved = json.loads((checkpoint / 'config.json').read_text())
    saved['skillvla_backbone'] = 'pi05'
    (checkpoint / 'config.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='belongs to pi05'):
        prepare(ft, ft_path)


@pytest.mark.parametrize('component', ['VSA', 'Predictor'])
def test_ft_default_automatically_selects_matching_groot_pt_run(tmp_path, monkeypatch, component):
    from skillvla_backend import prepare
    import train_skills_config
    recipe = WORKFLOW / ('FT/VSA/vsa_ft_config.yaml' if component == 'VSA'
                         else 'FT/Predictor/predictor_ft_config.yaml')
    resolver = load(WORKFLOW / ('FT/VSA/src/newtask_ft_vsa_config.py' if component == 'VSA'
                                else 'PT/src/auxiliary/auxiliary_train_config.py'), 'auto_ft_' + component)
    config = resolver.load_stage1_component_config(recipe)
    config['skillvla_backbone'] = 'groot_seperate'
    source = tmp_path / 'pretrained_model'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps({'skillvla_backbone': 'groot_seperate'}))
    selected = []
    def checkpoint(root, comp, selector, **kwargs):
        selected.append((comp, selector))
        return source
    monkeypatch.setattr(train_skills_config, 'resolve_run_checkpoint', checkpoint)
    resolved = prepare(config, recipe)
    assert selected[0][0] == component
    assert 'groot_seperate' in selected[0][1]['run']
    assert selected[0][1]['checkpoint'] == '100000'
    key = 'vsa_checkpoint' if component == 'VSA' else 'predictor_checkpoint'
    assert resolved['warm_start'][key] == str(source)


def test_submission_snapshot_pins_auto_source_without_touching_live_yaml(tmp_path, monkeypatch):
    from skillvla_backend import freeze_ft_source
    import train_skills_config
    import yaml
    folder = tmp_path / '.config_snapshots' / 'submission'
    folder.mkdir(parents=True)
    path = folder / 'vsa_ft_config.yaml'
    path.write_text('newtask_ft_common: true\nstage1_component: VSA\n')
    checkpoint = tmp_path / 'outputs/skillVLA_PT/VSA/groot_pt_run/checkpoints/100000/pretrained_model'
    monkeypatch.setattr(train_skills_config, 'load_stage1_component_config', lambda p: {
        'skillvla_backbone': 'groot_integrate', 'stage1_component': 'VSA',
        'warm_start': {'vsa_checkpoint': str(checkpoint)}})
    freeze_ft_source(path)
    assert yaml.safe_load(path.read_text())['warm_start']['backend_sources']['groot_integrate'] == {
        'vsa_checkpoint': {'run': 'groot_pt_run', 'checkpoint': '100000'}}
    with pytest.raises(ValueError, match='private submission snapshots'):
        freeze_ft_source(tmp_path / 'live.yaml')
