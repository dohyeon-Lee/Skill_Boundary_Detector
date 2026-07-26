import json
import sys
from pathlib import Path


_STAGE1_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/stage1/src"
sys.path.insert(0, str(_STAGE1_SRC))

from stage1_train_config import _read_dataset_contract


def test_stage1_reads_skill_space_from_dataset_metadata(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "meta"
    metadata_dir.mkdir()
    (metadata_dir / "info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "skill_pmax": 15,
                "skill_jitter_distribution": "half-normal",
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )

    contract = _read_dataset_contract(tmp_path, "FSQ333_example")

    assert contract == {
        "levels": [3, 3, 3],
        "state_dim": 8,
        "action_dim": 7,
        "jitter_pmax": 15,
        "jitter_distribution": "half_normal",
    }
