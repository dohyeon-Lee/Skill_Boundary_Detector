from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]
_LIBERO_EXAMPLES = _ROOT / "lerobot/examples/libero"
_EVAL_SRC = _ROOT / "lerobot/examples/libero/configs/train_skills/FSQ_eval/src"
sys.path.insert(0, str(_LIBERO_EXAMPLES))
sys.path.insert(0, str(_EVAL_SRC))

from FSQ import VSAFlowExpert  # noqa: E402

_ORACLE_SPEC = importlib.util.spec_from_file_location(
    "fsq_eval_oracle_data", _EVAL_SRC / "oracle_data.py"
)
assert _ORACLE_SPEC is not None and _ORACLE_SPEC.loader is not None
oracle_data = importlib.util.module_from_spec(_ORACLE_SPEC)
_ORACLE_SPEC.loader.exec_module(oracle_data)

_REPORT_SPEC = importlib.util.spec_from_file_location(
    "fsq_eval_report", _LIBERO_EXAMPLES / "fsq_eval.py"
)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
fsq_eval_report = importlib.util.module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(fsq_eval_report)


def test_sample_actions_forwards_broadcast_scale_on_every_step() -> None:
    calls = []
    fake = SimpleNamespace(chunk_size=2, max_action_dim=3)

    def velocity(self, x_t, time, state, z_norm, **kwargs):
        calls.append(kwargs)
        return torch.ones_like(x_t)

    fake.velocity = MethodType(velocity, fake)
    result = VSAFlowExpert.sample_actions(
        fake,
        torch.zeros(1, 5),
        torch.zeros(1, 3),
        noise=torch.zeros(1, 2, 3),
        num_steps=2,
        broadcast_scale=0.3,
    )

    torch.testing.assert_close(result, -torch.ones_like(result))
    assert len(calls) == 2
    assert all(call["broadcast_scale"] == 0.3 for call in calls)


def test_skill_broadcast_applies_scale_only_in_broadcast_mode() -> None:
    projection = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    fake = SimpleNamespace(
        state_cond_mode="broadcast",
        skill_proj=projection,
        working_dtype=torch.float32,
    )
    z_norm = torch.tensor([[2.0, 4.0]])

    result = VSAFlowExpert._skill_broadcast(fake, z_norm, 0.25)

    torch.testing.assert_close(result, torch.tensor([[0.5, 1.0, 1.5]]))
    fake.state_cond_mode = "state"
    assert VSAFlowExpert._skill_broadcast(fake, z_norm, 0.25) is None


def test_report_compare_reuses_noise_and_aux_outputs() -> None:
    class FakeActionExpert:
        @staticmethod
        def sample_noise(shape, device):
            return torch.arange(1, np.prod(shape) + 1, device=device).reshape(shape).float()

    class FakeModel:
        chunk_size = 2
        action_dim = 1
        cfg = SimpleNamespace(max_action_dim=1)
        action_expert = FakeActionExpert()

        def __init__(self):
            self.noise_ptrs = []

        def sample_action_chunks(
            self, z_q, raw_states, *, noise, broadcast_scale=1.0, **kwargs
        ):
            self.noise_ptrs.append(noise.data_ptr())
            skill = z_q[:, :1, None, None]
            return noise[..., :1].unsqueeze(1) * float(broadcast_scale) + skill

        def decode(self, z_q, raw_states, third, wrist, *, noise, **kwargs):
            actions = self.sample_action_chunks(z_q, raw_states, noise=noise)
            batch = z_q.shape[0]
            return actions, torch.zeros(batch, 1), torch.zeros(batch, 1)

    model = FakeModel()
    raw_dataset = [
        {
            "observation.images.image": torch.zeros(3, 4, 4),
            "observation.images.wrist_image": torch.zeros(3, 4, 4),
        }
        for _ in range(2)
    ]
    base, progress, term, compare, random_action = fsq_eval_report._batched_decode_impl(
        model,
        np.zeros((1, 2), dtype=np.float32),
        [np.zeros((2, 3), dtype=np.float32)],
        [{"dataset_from_index": 0, "frame_start": 0}],
        raw_dataset,
        [2],
        "cpu",
        2,
        broadcast_compare_scale=0.5,
        random_skill_latents=np.ones((1, 2), dtype=np.float32),
    )

    assert len(set(model.noise_ptrs)) == 1
    np.testing.assert_allclose(compare[0], base[0] * 0.5)
    np.testing.assert_allclose(random_action[0], base[0] + 1.0)
    np.testing.assert_allclose(progress[0], 0.0)
    np.testing.assert_allclose(term[0], 0.5)


def test_far_skill_selection_uses_farthest_active_pool() -> None:
    latents = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    selected, tokens, distances = fsq_eval_report.select_far_active_skill_latents(
        latents,
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        [0],
        [3, 3],
        seed=7,
        far_fraction=0.25,
    )

    assert tokens.tolist() == [3]
    np.testing.assert_allclose(selected, [[1.0, 1.0]])
    np.testing.assert_allclose(distances, [np.sqrt(8.0)])


def test_fsq_episode_data_sorts_skills(tmp_path, monkeypatch) -> None:
    latents = tmp_path / "skill_latents_eval.npz"
    np.savez(
        latents,
        tokens=np.asarray([8, 3], dtype=np.int32),
        episode_id=np.asarray([7, 7], dtype=np.int64),
        task_id=np.asarray([2, 2], dtype=np.int64),
        skill_index=np.asarray([1, 0], dtype=np.int64),
        frame_start=np.asarray([20, 5], dtype=np.int64),
        frame_end=np.asarray([30, 20], dtype=np.int64),
        length=np.asarray([10, 15], dtype=np.int64),
    )
    init_states = tmp_path / "eval_init_states.npz"
    np.savez(
        init_states,
        episode_index=np.asarray([7]),
        init_states=np.asarray([np.asarray([1.0, 2.0])], dtype=object),
        scene_file=np.asarray(["SCENE_demo.hdf5"]),
    )
    monkeypatch.setattr(oracle_data, "build_task_name_to_id", lambda _: {"SCENE": 2})
    monkeypatch.setattr(
        "lerobot.datasets.io_utils.load_episodes",
        lambda _: [
            {
                "episode_index": 7,
                "dataset_from_index": 100,
                "dataset_to_index": 140,
            }
        ],
    )

    result = oracle_data.load_fsq_episode_data(
        latents, init_states, tmp_path / "raw", "libero_90"
    )

    record = result[2][0]
    assert record["episode_index"] == 7
    assert [skill["token"] for skill in record["skills"]] == [3, 8]
    assert [skill["gt_length"] for skill in record["skills"]] == [15, 10]
