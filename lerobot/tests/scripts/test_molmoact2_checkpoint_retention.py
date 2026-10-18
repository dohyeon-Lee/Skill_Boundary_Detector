"""Pruning never deletes protected, partial, symlinked or unrelated artifacts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/libero/configs/train_molmoact2/PT/src"))
from checkpoint_retention import prune_checkpoints
from test_molmoact2_train_config import make_training_checkpoint


def test_latest_two_union_protected_and_last_link(tmp_path):
    run = tmp_path / "run"
    for step in range(10000, 100001, 10000):
        ckpt = run / "checkpoints" / f"{step:06d}"
        make_training_checkpoint(ckpt)
        link = ckpt.parent / "last"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(ckpt.name)
        prune_checkpoints(run, 2, [30000, 50000, 100000])
    assert {int(p.name) for p in (run / "checkpoints").iterdir() if p.name.isdigit()} == {30000, 50000, 90000, 100000}
    assert (run / "checkpoints/last").resolve().name == "100000"


def test_keep_all_and_incomplete_or_foreign_folders(tmp_path):
    for step in [10000, 20000, 30000]:
        make_training_checkpoint(tmp_path / "checkpoints" / f"{step:06d}")
    incomplete = tmp_path / "checkpoints/040000"
    incomplete.mkdir()
    other = tmp_path / "unrelated"
    other.mkdir()
    (tmp_path / "checkpoints/999999").symlink_to(other)
    assert prune_checkpoints(tmp_path, -1, []) == []
    removed = prune_checkpoints(tmp_path, 2, [])
    assert [p.name for p in removed] == ["010000"]
    assert incomplete.is_dir() and other.is_dir()
    assert (tmp_path / "checkpoints/999999").is_symlink()
