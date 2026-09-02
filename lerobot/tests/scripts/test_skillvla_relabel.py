import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/build_data/src"
)
sys.path.insert(0, str(_SRC))

import relabel_skillvla as relabel_module  # noqa: E402
from relabel_skillvla import (  # noqa: E402
    discover_segments,
    normalized_latents_from_codes,
    rewrite_skill_latents,
    rewrite_skill_sequences,
)


def _tiny_skillvla(tmp_path: Path) -> Path:
    dataset = tmp_path / "skillvla"
    data = dataset / "data/chunk-000"
    data.mkdir(parents=True)
    rows = 7
    table = pa.table(
        {
            "index": list(range(rows)),
            "episode_index": [0, 0, 0, 0, 1, 1, 1],
            "task_index": [3, 3, 3, 3, 4, 4, 4],
            "frame_index": [0, 1, 2, 3, 0, 1, 2],
            "skill_index": [0, 0, 1, 1, 0, 0, 0],
            "skill_sequence": [
                [1, 2, 27, 28],
                [1, 2, 27, 28],
                [1, 2, 27, 28],
                [1, 2, 27, 28],
                [27, 28, 28, 28],
                [27, 28, 28, 28],
                [27, 28, 28, 28],
            ],
            "skill_sequence_len": [3, 3, 3, 3, 1, 1, 1],
            "skill_initial_frame": [
                [0, 2, -1, -1],
                [0, 2, -1, -1],
                [0, 2, -1, -1],
                [0, 2, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            "skill_ds": [0, 1, 0, 1, 0, 0, 0],
            "untouched": np.arange(rows, dtype=np.float32),
        }
    )
    pq.write_table(table, data / "file-000.parquet")
    return dataset


def test_rewrite_preserves_boundaries_and_zero_filled_episode(tmp_path: Path) -> None:
    dataset = _tiny_skillvla(tmp_path)
    segments = discover_segments(dataset, num_embeddings=27)

    assert [(item.episode_index, item.skill_index) for item in segments] == [
        (0, 0),
        (0, 1),
    ]

    rewrite_skill_sequences(
        dataset,
        segments=segments,
        predictions=np.asarray([4, 5]),
    )
    rewritten = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pydict()
    assert rewritten["skill_sequence"][:4] == [[4, 5, 27, 28]] * 4
    assert rewritten["skill_sequence"][4:] == [[27, 28, 28, 28]] * 3
    assert rewritten["skill_index"] == [0, 0, 1, 1, 0, 0, 0]
    assert rewritten["skill_ds"] == [0, 1, 0, 1, 0, 0, 0]
    assert rewritten["untouched"] == list(np.arange(7, dtype=np.float32))


def test_rewrite_skill_latents_uses_fsq_grid_coordinates(tmp_path: Path) -> None:
    dataset = _tiny_skillvla(tmp_path)
    segments = discover_segments(dataset, num_embeddings=27)
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    np.savez_compressed(
        source,
        episode_id=np.asarray([0, 0]),
        skill_index=np.asarray([0, 1]),
        frame_start=np.asarray([0, 2]),
        tokens=np.asarray([1, 2]),
        latents=np.zeros((2, 3), dtype=np.float32),
        length=np.asarray([2, 2]),
    )

    rewrite_skill_latents(
        source,
        output,
        segments=segments,
        predictions=np.asarray([0, 26]),
        levels=[3, 3, 3],
    )

    with np.load(output) as archive:
        np.testing.assert_array_equal(archive["tokens"], [0, 26])
        np.testing.assert_allclose(
            archive["latents"],
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        )


def test_fsq_code_coordinates_are_little_endian() -> None:
    np.testing.assert_allclose(
        normalized_latents_from_codes(np.asarray([1, 3, 9]), [3, 3, 3]),
        [[0.0, -1.0, -1.0], [-1.0, 0.0, -1.0], [-1.0, -1.0, 0.0]],
    )


def test_complete_relabel_build_is_atomic_and_keeps_original_code_space(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_run = tmp_path / "FSQ333_original"
    dataset = _tiny_skillvla(source_run)
    meta = dataset / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        '{"repo_id":"skillvla/source","skill_fsq_levels":[3,3,3]}'
    )
    (source_run / "FSQ.pt").write_bytes(b"fsq")
    np.savez_compressed(source_run / "skill_initial_state.npz", state=[1])
    np.savez_compressed(
        source_run / "skill_latents.npz",
        episode_id=np.asarray([0, 0]),
        skill_index=np.asarray([0, 1]),
        frame_start=np.asarray([0, 2]),
        tokens=np.asarray([1, 2]),
        latents=np.zeros((2, 3), dtype=np.float32),
    )
    # This large optional artifact intentionally must not be copied with stale
    # skill codes; Stage3 rebuilds it from relabeled parquet if requested.
    (source_run / "transitions.npz").write_bytes(b"stale")

    predictor = tmp_path / "predictor"
    predictor.mkdir()
    (predictor / "config.json").write_text(
        '{"skill_fsq_levels":[3,3,3],'
        '"skill_code_space_id":"FSQ333_original"}'
    )
    (predictor / "model.safetensors").touch()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    monkeypatch.setattr(
        relabel_module,
        "_predict",
        lambda **_: np.asarray([4, 5], dtype=np.int64),
    )
    output_run = tmp_path / "FSQ333_original_relabeled"
    args = Namespace(
        source_run_dir=str(source_run),
        output_run_dir=str(output_run),
        predictor_path=str(predictor),
        predictor_model="predictor_run",
        predictor_checkpoint="070000",
        tokenizer_path=str(tokenizer),
        code_space_id="FSQ333_original",
        batch_size=4,
    )

    relabel_module.build_relabeled_dataset(args)

    output_info = relabel_module.json.loads(
        (output_run / "skillvla/meta/info.json").read_text()
    )
    assert output_info["repo_id"] == "skillvla/source"
    assert output_info["skill_code_space_id"] == "FSQ333_original"
    assert output_info["skill_dataset_variant"] == "predictor_relabeled"
    assert not (output_run / "transitions.npz").exists()
    assert (source_run / "transitions.npz").read_bytes() == b"stale"
    with np.load(output_run / "skill_relabel.npz") as audit:
        np.testing.assert_array_equal(audit["original_code"], [1, 2])
        np.testing.assert_array_equal(audit["predicted_code"], [4, 5])
