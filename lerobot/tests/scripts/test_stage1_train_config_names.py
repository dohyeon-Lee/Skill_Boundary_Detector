import sys
from pathlib import Path


_STAGE1_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/stage1/src"
sys.path.insert(0, str(_STAGE1_SRC))

from stage1_train_config import _anchor_weight_run_suffix, _lora_targets_run_suffix


def test_attention_only_targets_keep_historical_run_name() -> None:
    assert _lora_targets_run_suffix("q,k,v,o") == ""


def test_expanded_targets_and_relaxed_anchor_are_tagged() -> None:
    assert _lora_targets_run_suffix("q,k,v,o,mlp,action_out") == "_ltq-k-v-o-mlp-actionout"
    assert _anchor_weight_run_suffix(1.0) == ""
    assert _anchor_weight_run_suffix(0.5) == "_aw0p5"
