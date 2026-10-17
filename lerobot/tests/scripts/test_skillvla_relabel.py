import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


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


@pytest.mark.parametrize("different_run_tags", [False, True])
def test_complete_relabel_build_is_atomic_and_keeps_original_code_space(
    tmp_path: Path,
    monkeypatch,
    different_run_tags,
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
    predictor = tmp_path / "predictor"
    predictor.mkdir()
    predictor_config = {"skill_fsq_levels": [3, 3, 3], "skill_code_space_id": "FSQ333_original"}
    if different_run_tags:
        provenance = {"source_fsq_run_name": "shared_model", "source_fsq_checkpoint": "2000"}
        (source_run / "fsq_source.json").write_text(relabel_module.json.dumps(provenance))
        predictor_config.update(skill_code_space_id="PT_original", fsq_path=str(source_run / "FSQ.pt"))
    (predictor / "config.json").write_text(relabel_module.json.dumps(predictor_config))
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
    with np.load(output_run / "skill_relabel.npz") as audit:
        np.testing.assert_array_equal(audit["original_code"], [1, 2])
        np.testing.assert_array_equal(audit["predicted_code"], [4, 5])
    # Linked metadata/parquet must be replaced atomically, not edited through
    # the hard link back into the immutable source.
    assert "skill_dataset_variant" not in relabel_module.json.loads((meta / "info.json").read_text())
    original = pq.read_table(dataset / "data/chunk-000/file-000.parquet")
    assert original["skill_sequence"].to_pylist()[0] == [1, 2, 27, 28]
    with np.load(source_run / "skill_latents.npz") as archive:
        np.testing.assert_array_equal(archive["tokens"], [1, 2])


@pytest.mark.parametrize("architecture", ["standalone", "both_3", "direct_patch_2", "residual_patch_2_dGD"])
@pytest.mark.parametrize("start_proprio", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_runtime_predictions_use_checkpoint_normalized_boundary_state(tmp_path, architecture, start_proprio, batch_size):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    segments = discover_segments(_tiny_skillvla(tmp_path), num_embeddings=27)
    accesses = []

    class Dataset:
        def __getitem__(self, index):
            accesses.append(index)
            return {
                "observation.images.image": torch.full((3, 2, 2), float(index)),
                "observation.images.wrist_image": torch.full((3, 2, 2), float(index + 10)),
                OBS_STATE: torch.full((8,), float(index + 100)),
                "task": "pick object",
                "skill_sequence": "GT labels must not enter the predictor",
            }

    def preprocessor(batch):
        assert set(batch) == {"observation.images.image", "observation.images.wrist_image", OBS_STATE, "task"}
        indices = batch["observation.images.image"][:, 0, 0, 0].long()
        return {
            **batch, OBS_STATE: (batch[OBS_STATE] - 100) / 10,
            OBS_LANGUAGE_TOKENS: indices[:, None],
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(len(indices), 1, dtype=torch.bool),
        }

    class Predictor:
        config = SimpleNamespace(architecture_label=architecture, skill_predictor_start_proprio=start_proprio)

        def predict(self, images, tokens, mask, **kwargs):
            assert not torch.is_grad_enabled()
            assert mask.all()
            torch.testing.assert_close(images[1][:, 0, 0, 0], tokens[:, 0].float() + 10)
            if start_proprio:
                assert set(kwargs) == {"start_state"}
                expected = (tokens[:, :1].float() / 10).expand(-1, 8)
                torch.testing.assert_close(kwargs["start_state"], expected)
            else:
                assert kwargs == {}  # Do not feed state into an old adapter-free predictor.
            return tokens[:, 0]

    predictions = relabel_module._predict_skill_starts(
        predictor=Predictor(), dataset=Dataset(), preprocessor=preprocessor,
        segments=segments, batch_size=batch_size, device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(predictions, [0, 2])
    assert accesses == [0, 2]
    assert predictions.dtype == np.int64


def test_preprocessor_overrides_relocate_every_joint_tokenizer(tmp_path):
    config = {"steps": [
        {"registry_name": "device_processor", "config": {"device": "cuda:7"}},
        {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "/old/server/tokenizer"}},
        {"registry_name": "skill_vla_terminator_predictor_tokenizer_processor_step", "config": {}},
        {"registry_name": "normalizer_processor", "config": {"stats": "unchanged"}},
    ]}
    path = tmp_path / "policy_preprocessor.json"
    path.write_text(relabel_module.json.dumps(config))
    assert relabel_module._preprocessor_overrides(tmp_path, Path("/local/tokenizer"), "cuda") == {
        "device_processor": {"device": "cuda"},
        "tokenizer_processor": {"tokenizer_name": "/local/tokenizer"},
        "skill_vla_terminator_predictor_tokenizer_processor_step": {"tokenizer_name": "/local/tokenizer"},
    }
    assert relabel_module.json.loads(path.read_text()) == config


def test_relabel_filters_training_only_prompts_but_preserves_normalizer():
    from lerobot.policies.skillVLA.processor_skillVLA import SkillVLATerminatorPredictorTokenizerProcessorStep

    training_prompt = object.__new__(SkillVLATerminatorPredictorTokenizerProcessorStep)
    normalizer, tokenizer = object(), object()
    pipeline = SimpleNamespace(steps=[normalizer, training_prompt, tokenizer, training_prompt])
    assert relabel_module._prepare_relabel_preprocessor(pipeline) is pipeline
    assert pipeline.steps == [normalizer, tokenizer]


@pytest.mark.parametrize("kind,joint,proprio", [("skill_aux", False, False), ("skill_aux", False, True), ("skill_expert", True, True)])
def test_predict_loads_only_the_saved_predictor_and_its_processors(tmp_path, monkeypatch, kind, joint, proprio):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import factory
    from lerobot.policies.skill_expert import modeling_skill_expert, modeling_skill_predictor
    from lerobot.policies.skillVLA.processor_skillVLA import SkillVLATerminatorPredictorTokenizerProcessorStep

    predictor_path = tmp_path / "checkpoint"
    predictor_path.mkdir()
    saved = {"steps": [
        {"registry_name": "device_processor"}, {"registry_name": "tokenizer_processor"},
        {"registry_name": "skill_vla_terminator_predictor_tokenizer_processor_step"},
    ]}
    (predictor_path / "policy_preprocessor.json").write_text(relabel_module.json.dumps(saved))
    dataset_dir = tmp_path / "skillvla"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text('{"repo_id":"skillvla/test"}')
    config = SimpleNamespace(type=kind, train_skill_predictor=not joint, newtask_joint_enabled=joint,
                             skill_predictor_start_proprio=proprio, dtype="bfloat16")
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(lambda cls, *args, **kw: config))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loaded = []

    class Predictor:
        def __init__(self, cfg):
            self.config = cfg

        def to(self, **kwargs):
            return self  # CPU-only test; never allocate CUDA or full-size VLM weights.

        def requires_grad_(self, enabled):
            assert not enabled
            return self

        def eval(self):
            return self

    monkeypatch.setattr(modeling_skill_predictor, "FrozenVLMSkillPredictor", Predictor)
    monkeypatch.setattr(modeling_skill_expert, "_load_complete_predictor_parameters", lambda predictor, path: loaded.append(path))
    normalizer = object()
    pipeline = SimpleNamespace(steps=[normalizer, object.__new__(SkillVLATerminatorPredictorTokenizerProcessorStep)])

    def processors(**kwargs):
        assert kwargs["pretrained_path"] == str(predictor_path)
        assert "dataset_stats" not in kwargs  # Must not recompute/replace checkpoint statistics.
        assert kwargs["preprocessor_overrides"]["skill_vla_terminator_predictor_tokenizer_processor_step"] == {"tokenizer_name": str(tmp_path / "tokenizer")}
        return pipeline, None

    monkeypatch.setattr(factory, "make_pre_post_processors", processors)
    dataset = object()
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", lambda **kwargs: dataset)

    def predict_starts(**kwargs):
        assert kwargs["predictor"].config is config
        assert kwargs["dataset"] is dataset
        assert kwargs["preprocessor"].steps == [normalizer]
        return np.array([4, 5], dtype=np.int64)

    monkeypatch.setattr(relabel_module, "_predict_skill_starts", predict_starts)
    result = relabel_module._predict(dataset_dir=dataset_dir, predictor_path=predictor_path,
                                    tokenizer_path=tmp_path / "tokenizer", segments=[], batch_size=4)
    np.testing.assert_array_equal(result, [4, 5])
    assert loaded == [predictor_path]
