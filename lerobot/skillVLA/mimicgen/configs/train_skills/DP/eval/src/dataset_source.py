"""Coffee uses dataset task IDs, with the shared v3 episode selector."""
import importlib.util
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'lerobot/src/lerobot').is_dir())
path=root/'lerobot/skillVLA/libero/configs/train_skills/DP/eval/src/dataset_source.py'
spec=importlib.util.spec_from_file_location('mimicgen_v3_source',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
prepare_source=module.prepare_source

def make_source(cfg,settings):
 if settings['task_id_space']!='dataset': raise ValueError('MimicGen uses dataset task IDs')
 result=module.make_source(cfg,settings)
 result['evaluation_source']='mimicgen_subset'
 return result
