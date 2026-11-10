import sys
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/build_data/src"
sys.path.insert(0, str(_SRC))
import predictor_eval_utils as eval_utils


def test_preprocessor_overrides_relocate_every_joint_tokenizer(tmp_path):
    config = {"steps": [
        {"registry_name": "device_processor", "config": {"device": "cuda:7"}},
        {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "/old/server/tokenizer"}},
        {"registry_name": "skill_vla_terminator_predictor_tokenizer_processor_step", "config": {}},
        {"registry_name": "normalizer_processor", "config": {"stats": "unchanged"}},
    ]}
    path = tmp_path / "policy_preprocessor.json"
    path.write_text(eval_utils.json.dumps(config))
    assert eval_utils._preprocessor_overrides(tmp_path, Path("/local/tokenizer"), "cuda") == {
        "device_processor": {"device": "cuda"},
        "tokenizer_processor": {"tokenizer_name": "/local/tokenizer"},
        "skill_vla_terminator_predictor_tokenizer_processor_step": {"tokenizer_name": "/local/tokenizer"},
    }
    assert eval_utils.json.loads(path.read_text()) == config


def test_predictor_filters_training_only_prompts_but_preserves_normalizer():
    from lerobot.policies.skillVLA.processor_skillVLA import SkillVLATerminatorPredictorTokenizerProcessorStep

    training_prompt = object.__new__(SkillVLATerminatorPredictorTokenizerProcessorStep)
    normalizer, tokenizer = object(), object()
    pipeline = SimpleNamespace(steps=[normalizer, training_prompt, tokenizer, training_prompt])
    assert eval_utils._prepare_predictor_preprocessor(pipeline) is pipeline
    assert pipeline.steps == [normalizer, tokenizer]

