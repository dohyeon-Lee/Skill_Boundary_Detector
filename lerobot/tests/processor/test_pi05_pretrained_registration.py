"""Load saved PI05 preprocessing in a fresh interpreter, without policy weights."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('saved_postprocessor', [False, True])
def test_saved_pi05_processor_registers_policy_step_before_loading(tmp_path, saved_postprocessor):
    script = r'''
import json,sys
from pathlib import Path
import torch
import lerobot.policies.factory as factory
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline, ProcessorStepRegistry
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
root=Path(sys.argv[1]);saved=sys.argv[2]=='True'
key='pi05_prepare_state_tokenizer_processor_step'
assert key not in ProcessorStepRegistry._registry, 'Test must start with a cold registry'
(root/'policy_preprocessor.json').write_text(json.dumps({'name':'policy_preprocessor','steps':[
 {'registry_name':key,'config':{}},
 {'registry_name':'device_processor','config':{'device':'cpu'}}]}))
if saved:
 (root/'policy_postprocessor.json').write_text(json.dumps({'name':'policy_postprocessor','steps':[
  {'registry_name':'device_processor','config':{'device':'cpu'}}]}))
fallback=[]
def make_fallback(config,dataset_stats):
 fallback.append(True)
 return None,PolicyProcessorPipeline(steps=[DeviceProcessorStep(device='cpu')],
  to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
factory._make_processors_from_policy_config=make_fallback
pre,post=factory.make_pre_post_processors(PI05Config(device='cpu'),pretrained_path=root,dataset_stats={})
assert ProcessorStepRegistry.get(key).__name__=='Pi05PrepareStateTokenizerProcessorStep'
batch=pre({'observation.state':torch.zeros(1,14),'task':['grab_clutter']})
assert batch['task'][0].startswith('Task: grab clutter, State: ')
action=torch.zeros(1,14)
assert torch.equal(post(action),action)
assert bool(fallback)==(not saved)
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2] / 'src')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path), str(saved_postprocessor)],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
