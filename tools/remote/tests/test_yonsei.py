import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest
import yaml

SOURCE = Path(__file__).resolve().parents[1] / 'yonsei.py'
SPEC = importlib.util.spec_from_file_location('yonsei_remote_test', SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_remote_selector_preserves_arguments_and_separator():
    args, plan, check = M.parse_flags(['--yonsei', '--12345', '--config', 'a b.yaml',
                                      '$(touch /tmp/no)', '--', '--yonsei'])
    assert args == ['--12345', '--config', 'a b.yaml', '$(touch /tmp/no)', '--', '--yonsei']
    assert not plan and not check
    assert M.parse_flags(['--yonsei-plan'])[1:]==(True,False)


def test_only_recipe_selectors_travel_not_credentials_or_source_allocation():
    env = {'STAGE1_EVAL_CONFIG':'/src/custom.yaml','HF_TOKEN':'secret','WANDB_API_KEY':'secret',
           'SLURM_JOB_ID':'74311','CUDA_VISIBLE_DEVICES':'4','SBD_SLURM_SUBMIT_MODE':'srun_existing',
           'TRAIN_DATA':'libero_90','NEWTASK_FT_VSA_CONFIG':'a.yaml'}
    assert M.selectors(env)=={k:env[k] for k in ('STAGE1_EVAL_CONFIG','TRAIN_DATA','NEWTASK_FT_VSA_CONFIG')}


def test_abc_dp_uses_native_training_resolver_without_build_stage_flags():
    script = M.ROOT / 'lerobot/skillVLA/abc/configs/train_skills/DP/submit_train_dp.sh'
    config, resolver, extra = M.config_and_resolver(script, [], {})
    assert config == script.parent / 'dp_config.yaml'
    assert resolver == script.parent / 'src/train_dp_config.py'
    assert extra == []


def test_droid_dp_uses_its_own_resolver_and_uppercase_stage():
    script = M.ROOT / 'lerobot/skillVLA/droid/configs/train_skills/DP/submit_train_dp.sh'
    config, resolver, extra = M.config_and_resolver(script, [], {})
    assert config == script.parent / 'dp_config.yaml'
    assert resolver == script.parent.parent / 'src/train_skills_config.py'
    assert extra == ['--stage', 'DP']


def test_droid_fsq_uses_native_resolver_and_transfers_both_inputs(tmp_path):
    script = M.ROOT / 'lerobot/skillVLA/droid/configs/train_skills/FSQ/submit_train_fsq.sh'
    config, resolver, extra = M.config_and_resolver(script, [], {})
    assert config == script.parent / 'fsq_config.yaml'
    assert resolver == script.parent.parent / 'src/train_skills_config.py'
    assert extra == ['--stage', 'FSQ']
    dataset = tmp_path / 'dataset_DROID/converted_joint_velocity'
    skills = tmp_path / 'dataset_DROID/FSQ_dataset/converted_joint_velocity/FSQ_inputs/run/skillset'
    output = tmp_path / 'outputs_DROID/FSQ/FSQ3332_delta_pose'
    for path in (dataset, skills, output):
        path.mkdir(parents=True)
    settings = {'dataset_dir': str(dataset), 'skills_dir': str(skills),
                'droid_output_dir': str(output), 'fsq_num_epochs': 2000}
    assert set(M.input_paths(settings, tmp_path)) == {dataset, skills}


def test_droid_fsq_eval_native_resolver_and_frozen_last(tmp_path, monkeypatch):
    script = M.ROOT / 'lerobot/skillVLA/droid/configs/train_skills/FSQ/eval/submit_eval.sh'
    config, resolver, extra = M.config_and_resolver(script, [], {})
    assert resolver == script.parent / 'src/config.py' and extra == []
    assert config == script.parent / 'eval_config.yaml'
    assert M.selectors({'DROID_FSQ_EVAL_CONFIG': str(config)}) == {'DROID_FSQ_EVAL_CONFIG': str(config)}
    monkeypatch.setattr(M, 'code_files', lambda root: set())
    settings = {'droid_fsq_models_json': [{'run_name': 'joint', 'checkpoint': 100},
                 {'run_name': 'pose', 'checkpoint': 200}]}
    M.freeze(M.ROOT, '/remote/frozen', {}, tmp_path, config, set(), settings=settings)
    copied = yaml.safe_load((tmp_path / config.relative_to(M.ROOT)).read_text())
    assert copied['models'] == [{'run_name': 'joint', 'checkpoints': [100]},
                                {'run_name': 'pose', 'checkpoints': [200]}]


def test_droid_builtin_group_is_in_code_snapshot(tmp_path, monkeypatch):
    group = Path('lerobot/skillVLA/droid/configs/dataset_groups/without_deformable.v1.json')
    # A generic dataset_*/ ignore previously dropped this configuration folder.
    assert group in M.code_files(M.ROOT)
    monkeypatch.setattr(M, 'code_files', lambda root: {group})
    M.freeze(M.ROOT, '/remote/frozen', {}, tmp_path, None, set())
    assert (tmp_path / group).read_bytes() == (M.ROOT / group).read_bytes()


def test_droid_dp_transfers_converted_corpus_not_external_raw_or_derived_view(tmp_path):
    configs = M.ROOT / 'lerobot/skillVLA/droid/configs'
    trainer = runpy.run_path(str(configs / 'train_skills/src/train_skills_config.py'))
    config = trainer['load_config'](configs / 'train_skills/DP/dp_config.yaml')
    source = tmp_path / 'dataset_DROID/converted'
    source.mkdir(parents=True)
    (source / 'conversion_manifest.json').write_text('{}')
    # An existing local derived view must not be sent: its source path is local.
    derived = source.with_name('converted_joint_velocity')
    derived.mkdir()
    config.update(project_root=str(tmp_path), source_dataset='converted',
        target_dataset=derived.name, dataset_root='dataset_DROID')
    config['droid_dataset']['group'] = 'all'
    plan = trainer['resolve'](config, 'DP')
    settings = {'dataset_dir': plan['source_dataset'], 'droid_output_dir': plan['output']}
    assert M.input_paths(settings, tmp_path) == [source]
    assert M.input_paths({'raw_root': '/data4/yeonsumia/raw',
                         'calibration_root': '/data4/yeonsumia/calibration'}, tmp_path) == []


def test_droid_resume_contract_survives_new_frozen_checkout_links(tmp_path):
    configs = M.ROOT / 'lerobot/skillVLA/droid/configs'
    trainer = runpy.run_path(str(configs / 'train_skills/src/train_skills_config.py'))
    dataset = tmp_path / 'persistent/dataset'
    dataset.mkdir(parents=True)
    (dataset / 'training_view_manifest.json').write_text('{"source_sha256":"same-corpus"}')
    aliases = [tmp_path / name for name in ('old-launch', 'new-launch')]
    for path in aliases:
        path.symlink_to(dataset, target_is_directory=True)
    plan = dict(stage='DP', dataset=str(aliases[0]), idle_filter={}, fps=15,
                action_dim=8, state_dim=8, output=str(tmp_path / 'outputs/run'))
    trainer['_contract']({}, plan)
    plan['dataset'] = str(aliases[1])
    trainer['_contract']({}, plan)


def test_mimicgen_eval_ignores_inherited_libero_work_dir(tmp_path, monkeypatch):
    script = tmp_path / 'lerobot/skillVLA/mimicgen/configs/train_skillVLA/FT/eval/submit_eval.sh'
    script.parent.mkdir(parents=True)
    resolver = script.parent / 'config.py'
    resolver.write_text('')
    config = script.parent / 'ft_eval_config.yaml'
    config.write_text('{}')
    monkeypatch.setenv('STAGE1_EVAL_WORK_DIR', '/old/libero/FT/eval')
    monkeypatch.setattr(M, 'config_and_resolver', lambda *a: (config, resolver, []))
    def resolve(command, **kwargs):
        assert kwargs['env']['STAGE1_EVAL_WORK_DIR'] == str(script.parent)
        return subprocess.CompletedProcess(command, 0, stdout='')
    monkeypatch.setattr(M, 'run', resolve)
    assert M.resolve_settings(script, [], {}, tmp_path) == (config, {})


def test_yonsei_resources_replace_local_gpu_and_scheduler_without_changing_model():
    source={'server':'rllab','project_root':'/src','train_nodelist':'node200',
            'train_partition':['debug'],'train_qos':'base_qos','slurm_submit_mode':'srun_existing',
            'slurm':{'gres':'gpu:B200:2','cpus':16,'memory':'128G'},
            'training':{'gradient_checkpointing':True,'checkpoint_archive':'scratch2'},
            'checkpoint':'005000','oracle':{'episode_exact':False},'initial_state_bank':''}
    profile={'train_partition':['a6000'],'train_qos':'big_qos','train_nodelist':'',
             'train_gres':'gpu:1'}
    out=M.portable(source,Path('/src'),'/remote/frozen',profile)
    assert out['slurm']=={'gres':'gpu:2','cpus':16,'memory':'128G'}
    assert out['train_partition']==['a6000'] and out['train_nodelist']==''
    assert out['train_qos']=='big_qos' and out['slurm_submit_mode']=='sbatch'
    assert out['project_root']=='/remote/frozen' and out['server']=='yonsei'
    assert out['checkpoint']=='005000' and out['oracle']==source['oracle']
    assert out['training']=={'gradient_checkpointing':True,'checkpoint_archive':''}


def test_inputs_select_checkpoint_and_dataset_not_output_or_whole_roots(tmp_path):
    dataset=tmp_path/'dataset_filtered/train';dataset.mkdir(parents=True)
    output=tmp_path/'outputs_filtered/VSA/run';output.mkdir(parents=True)
    cp=output/'checkpoints/005000/pretrained_model';cp.mkdir(parents=True)
    settings={'dataset_root':str(dataset.parent),'skillvla_dataset_dir':str(dataset),
              'output_dir':str(output),'models_json':json.dumps([{'policy_path':str(cp)}])}
    assert set(M.input_paths(settings,tmp_path))=={dataset,cp}
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump({'dataset_root':'dataset_filtered',
                                                          'output_dir':str(output)}))
    assert not M.configured_file_inputs(cfg,tmp_path)


def test_skillvla_inputs_include_required_iss_beside_the_dataset(tmp_path):
    run = tmp_path / 'dataset_mimicgen/skillvla/coffee/run'
    dataset = run / 'skillvla'
    dataset.mkdir(parents=True)
    iss = run / 'skill_initial_state.npz'
    iss.touch()
    assert set(M.input_paths({'skillvla_dataset_dir': str(dataset)}, tmp_path)) == {dataset, iss}


def test_all_skillvla_shells_dispatch_early_and_parse():
    paths=subprocess.check_output(['rg','--files','lerobot/skillVLA','-g','*.sh'],cwd=M.ROOT,text=True).splitlines()
    for rel in paths:
        path=M.ROOT/rel
        subprocess.run(['bash','-n',str(path)],check=True,capture_output=True)
        if path.name=='remote.sh':continue
        code=path.read_text()
        assert '# SkillVLA remote dispatch' in code,rel
        assert code.index('# SkillVLA remote dispatch')<code.index('set -') if 'set -' in code else True


def test_source_helpers_keep_arguments_and_do_not_dispatch(tmp_path):
    script=tmp_path/'caller.sh'
    hook=M.ROOT/'lerobot/skillVLA/remote.sh'
    script.write_text('set -- --yonsei value\nsource '+str(hook)+'\nprintf "%s\\n" "$@"\n')
    # BASH_SOURCE[1] != $0 when sourced from another source helper.
    result=subprocess.run(['bash','-c','source "$1"','bash',str(script)],text=True,capture_output=True,check=True)
    assert result.stdout.splitlines()==['--yonsei','value']


@pytest.mark.parametrize('training', [False, True])
def test_real_staging_and_detached_runner_with_fake_transport(tmp_path,monkeypatch,training):
    root=tmp_path/'source';root.mkdir()
    remote=tmp_path/'remote';remote.mkdir()
    (remote/'.venv').symlink_to(sys.prefix,target_is_directory=True)
    entry=Path('lerobot/skillVLA/libero/configs/work') / ('submit_train.sh' if training else 'submit.sh')
    script=root/entry;script.parent.mkdir(parents=True)
    script.write_text('#!/bin/bash\nprintf "%s\\n" "$SBD_SERVER" "$@"\n')
    tool=root/'tools/remote/yonsei.py';tool.parent.mkdir(parents=True);shutil.copy2(SOURCE,tool)
    config=script.with_name('config.yaml');config.write_text('slurm: {gres: "gpu:B200:1"}\n')
    data=root/'dataset_filtered/train/data.txt';data.parent.mkdir(parents=True);data.write_text('tiny training input')
    dep=Path('tools/lerobot-libero/libero/libero')
    (root/dep).mkdir(parents=True)
    (root/dep/'__init__.py').write_text('FROZEN_CODE=True\n')
    (remote/dep/'init_files').mkdir(parents=True)
    (remote/dep/'init_files/fixture.pt').write_bytes(b'runtime states')
    (remote/dep/'__init__.py').write_text('OLD_CODE=True\n')
    unrelated=remote/'untouched.txt';unrelated.write_text('existing remote work')
    profile={'remote_host':'fake','remote_project_root':str(remote),'train_partition':['base'],
             'train_qos':'base_qos','train_nodelist':'','train_gres':'gpu:1'}
    monkeypatch.setattr(M,'load_profile',lambda root:profile)
    monkeypatch.setattr(M,'code_files',lambda root:{entry,tool.relative_to(root),config.relative_to(root),dep/'__init__.py'})
    monkeypatch.setattr(M,'resolve_settings',lambda *a:(config,{'dataset_dir':str(data.parent)}))
    from tools.remote import desk
    watchers=[]
    monkeypatch.setattr(desk,'ensure_daemon',lambda root:watchers.append(root))
    calls=[]
    def rpc(host,base,action,**values):
        calls.append(action)
        result=subprocess.run([sys.executable,'-c',M.RPC],input=json.dumps({'base':base,'action':action,**values}),
                              text=True,capture_output=True,check=True)
        return json.loads(result.stdout)
    def transfer(host,source,target,directory=False):
        if directory:shutil.copytree(source,target,dirs_exist_ok=True)
        else:shutil.copy2(source,target)
    monkeypatch.setattr(M,'rpc',rpc);monkeypatch.setattr(M,'transfer',transfer)
    monkeypatch.chdir(root)
    assert M.launch(script,['--yonsei','a b','$(not a command)'],root,env={})==0
    audit=next((root/'.cache/yonsei_launches').iterdir())
    manifest=json.loads((audit/'launch.json').read_text())
    work=Path(manifest['project']).parent
    assert (work/'status.json').exists()
    assert json.loads((work/'status.json').read_text())['state']=='COMPLETED'
    assert 'yonsei\na b\n$(not a command)' in (work/'console.log').read_text()
    assert (remote/'dataset_filtered/train/data.txt').read_text()=='tiny training input'
    assert unrelated.read_text()=='existing remote work'
    assert (Path(manifest['project']) / 'outputs_DROID').is_symlink()
    assert (Path(manifest['project']) / 'outputs_DROID').resolve() == remote / 'outputs_DROID'
    frozen=yaml.safe_load((Path(manifest['project'])/config.relative_to(root)).read_text())
    assert frozen['slurm']['gres']=='gpu:1'
    assert (Path(manifest['project'])/dep/'__init__.py').read_text()=='FROZEN_CODE=True\n'
    assert (Path(manifest['project'])/dep/'init_files/fixture.pt').read_bytes()==b'runtime states'
    assert 'start' in calls
    assert watchers == ([root] if training else [])
    if training:
        assert manifest['evaluation'] is None
        assert manifest['training']['checkpoint_sync'] is True
        assert manifest['training']['dataset'] == 'dataset_filtered/train'


def test_check_only_does_not_copy_or_launch(tmp_path,monkeypatch):
    script=tmp_path/'lerobot/skillVLA/a.sh';script.parent.mkdir(parents=True);script.write_text('exit 0')
    monkeypatch.setattr(M,'load_profile',lambda _: {})
    monkeypatch.setattr(M,'resolve_settings',lambda *a:(None,{}))
    monkeypatch.setattr(M,'rpc',lambda host,base,action,**kw: {'missing':[],'action':action})
    monkeypatch.setattr(M,'transfer',lambda *a,**kw:pytest.fail('must not copy'))
    assert M.launch(script,['--yonsei-check'],tmp_path,env={})==0
    assert not (tmp_path/'.cache/yonsei_launches').exists()


def test_remote_start_links_eval_results_to_normal_workspace(tmp_path):
    base = tmp_path / 'remote'
    base.mkdir()
    (base / '.venv').symlink_to(sys.prefix, target_is_directory=True)
    work = base / '.cache/yonsei_launches/test'
    project = work / 'project'
    runner = project / 'tools/remote/yonsei.py'
    runner.parent.mkdir(parents=True)
    runner.write_text('')  # No scheduler or GPU work in this staging check.
    relative = 'lerobot/skillVLA/libero/configs/train_skillVLA/FT/eval/outputs/run'
    target = base / relative
    target.mkdir(parents=True)
    (target / 'existing.txt').write_text('preserved')
    manifest = {'evaluation': {'output': relative, 'storage_project': str(base)}}
    request = {'base': str(base), 'action': 'start', 'work': str(work),
               'links': [], 'manifest': manifest}
    subprocess.run([sys.executable, '-c', M.RPC], input=json.dumps(request),
                   text=True, capture_output=True, check=True)
    link = project / relative
    assert link.is_symlink() and link.resolve() == target
    (link / 'index.html').write_text('report')
    assert (target / 'index.html').read_text() == 'report'
    assert (target / 'existing.txt').read_text() == 'preserved'


@pytest.mark.parametrize('relative', [
    'abc/configs/train_skillVLA/PT/eval/submit_eval.sh',
    'abc/configs/train_pi05/PT/eval/submit_eval.sh',
    'abc/configs/train_skills/DP/submit_train_dp.sh',
    'abc/configs/train_skills/FSQ/eval/submit_eval.sh',
    'abc/configs/train_skillVLA/PT/Predictor/submit_train.sh',
    'libero/configs/train_skillVLA/FT/VSA/submit_train.sh',
    'libero/configs/train_skillVLA/PT/src/auxiliary/submit_auxiliary.sh',
    'libero/configs/train_skillVLA/build_data/src/submit_build_skillset.sh',
    'libero/configs/train_skills/build_data/src/submit_build_skillset.sh',
    'libero/configs/train_cyclic/pi05/FT/submit_train.sh',
    'libero/configs/train_groot/FT/submit_train.sh',
    'libero/configs/train_molmoact2/FT/eval/submit_eval.sh',
])
def test_native_workflow_config_discovery(relative):
    entry=M.ROOT/'lerobot/skillVLA'/relative
    config,resolver,extra=M.config_and_resolver(entry,[],{})
    assert config.is_file(),config
    assert resolver.is_file(),resolver


def test_component_selector_is_not_a_directory_input(tmp_path):
    vsa = tmp_path/'FT/VSA'
    vsa.mkdir(parents=True)
    config = vsa/'vsa_ft_config.yaml'
    config.write_text('stage1_component: VSA\n')
    assert M.configured_file_inputs(config, tmp_path) == set()
