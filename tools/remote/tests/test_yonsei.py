import importlib.util
import json
import os
from pathlib import Path
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
    'libero/configs/train_skillVLA/PT/Terminator/src/submit_auxiliary.sh',
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
