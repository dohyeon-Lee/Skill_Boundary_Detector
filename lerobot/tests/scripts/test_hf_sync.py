"""hf_sync.py: dataset-root picking and push/pull planning (no network)."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

SYNC = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/libero/configs/src/hf_sync.py"))


def test_paths_keep_their_dataset_root() -> None:
    patterns = SYNC["patterns_for"](
        ["dataset_filtered/libero_90_full_full/", "dataset_filtered/skill/x/eval.npz"], lambda rel: not rel.endswith(".npz")
    )
    assert patterns == ["dataset_filtered/libero_90_full_full/*", "dataset_filtered/skill/x/eval.npz"]
    for bad in ("../outside", "/abs/path", "", "outputs_filtered/run", "models/pi05_base"):
        with pytest.raises(ValueError, match="relative|dataset root"):
            SYNC["check_paths"]([bad])


def test_base_follows_the_server_storage(tmp_path: Path) -> None:
    assert SYNC["local_base"]({"project_root": "/repo"}) == Path("/repo")
    (tmp_path / "volume").mkdir()
    assert SYNC["local_base"]({"project_root": "/repo", "storage_volume": str(tmp_path / "volume")}) == tmp_path / "volume"
    # A pod without the Global volume: pull straight into the checkout (container disk only).
    assert SYNC["local_base"]({"project_root": "/repo", "storage_volume": str(tmp_path / "absent")}) == Path("/repo")


def test_picking_starts_at_the_dataset_roots_and_collects_several_paths() -> None:
    tree = {
        "": [("dataset_calvin", True), ("dataset_filtered", True), ("outputs_filtered", True), ("setup_env.sh", False)],
        "dataset_filtered": [("libero_90_full_full", True), ("skillvla_dataset", True)],
        "dataset_filtered/skillvla_dataset": [("run_fov", True), ("eval_init_states.npz", False)],
    }
    answers = iter([
        "2", "2", "1", "1", "y",     # dataset_filtered -> drill -> libero_90_full_full -> whole folder -> more
        "2", "2", "2", "2", "2",     # dataset_filtered -> drill -> skillvla_dataset -> drill -> the .npz file
        "n",
    ])
    chosen = SYNC["pick_paths"](lambda rel: tree.get(rel, []), "pick", ask=lambda prompt: next(answers))
    assert chosen == ["dataset_filtered/libero_90_full_full", "dataset_filtered/skillvla_dataset/eval_init_states.npz"]
    assert SYNC["pick_paths"](lambda rel: tree.get(rel, []), "pick", ask=lambda prompt: "q") == []


def test_push_and_pull_dry_runs_touch_nothing(tmp_path: Path, capsys) -> None:
    (tmp_path / "dataset_filtered/libero/data").mkdir(parents=True)
    (tmp_path / "dataset_filtered/libero/data/file.parquet").write_bytes(b"x" * 1000)
    (tmp_path / "dataset_filtered/eval.npz").write_bytes(b"y")
    SYNC["push"]("me/repo", tmp_path, ["dataset_filtered/libero", "dataset_filtered/eval.npz"], dry_run=True, assume_yes=False)
    out = capsys.readouterr().out
    assert "upload_folder(" in out and "me/repo:dataset_filtered/libero)" in out
    assert "upload_file(" in out and "me/repo:dataset_filtered/eval.npz)" in out
    with pytest.raises(FileNotFoundError):
        SYNC["push"]("me/repo", tmp_path, ["dataset_filtered/missing"], dry_run=True, assume_yes=False)

    target = tmp_path / "pulled"
    SYNC["pull"]("me/repo", target, ["dataset_filtered/libero"], dry_run=True, assume_yes=False, config={},
                 is_dir=lambda rel: True)
    assert "allow_patterns=['dataset_filtered/libero/*']" in capsys.readouterr().out
    assert not target.exists()


def test_models_come_from_their_original_repos_into_models(tmp_path: Path, capsys) -> None:
    models = SYNC["PRETRAINED_MODELS"]
    assert set(models) == {"pi05_base", "paligemma-3b-pt-224-tokenizer", "dinov3-vits16", "dinov3-vitl16"}
    assert models["pi05_base"]["repo"] == "lerobot/pi05_base"
    assert models["paligemma-3b-pt-224-tokenizer"]["files"] == [
        "added_tokens.json", "config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    ]
    SYNC["pull_models"](tmp_path, ["pi05_base", "dinov3-vits16"], dry_run=True, assume_yes=False)
    out = capsys.readouterr().out
    assert f"local_dir={tmp_path / 'models' / 'pi05_base'}" in out and "lerobot/pi05_base" in out
    assert f"local_dir={tmp_path / 'models' / 'dinov3-vits16'}" in out
    assert not (tmp_path / "models").exists()
    with pytest.raises(ValueError, match="Unknown model"):
        SYNC["pull_models"](tmp_path, ["gemma_300m"], dry_run=True, assume_yes=False)


def test_model_picking_all_or_some() -> None:
    assert SYNC["pick_models"](ask=lambda prompt: "1") == list(SYNC["PRETRAINED_MODELS"])
    answers = iter(["2", "3", "y", "1", "n"])            # pick -> dinov3-vits16, then pi05_base
    assert SYNC["pick_models"](ask=lambda prompt: next(answers)) == ["dinov3-vits16", "pi05_base"]
    assert SYNC["pick_models"](ask=lambda prompt: "q") == []
