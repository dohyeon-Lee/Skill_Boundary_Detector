from __future__ import annotations

import base64
import sys
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
_LIBERO_EXAMPLES = _ROOT / "lerobot/examples/libero"
_FSQ_NEW_SRC = _ROOT / "lerobot/examples/libero/configs/train_skills/FSQ_new/src"
_EVAL_SRC = _ROOT / "lerobot/examples/libero/configs/train_skills/FSQ_new_eval/src"
sys.path.insert(0, str(_LIBERO_EXAMPLES))
sys.path.insert(0, str(_FSQ_NEW_SRC))
sys.path.insert(0, str(_EVAL_SRC))

from FSQ_new import VSAFlowExpert  # noqa: E402

_ORACLE_SPEC = importlib.util.spec_from_file_location(
    "fsq_new_eval_oracle_data", _EVAL_SRC / "oracle_data.py"
)
assert _ORACLE_SPEC is not None and _ORACLE_SPEC.loader is not None
oracle_data = importlib.util.module_from_spec(_ORACLE_SPEC)
_ORACLE_SPEC.loader.exec_module(oracle_data)

_REPORT_SPEC = importlib.util.spec_from_file_location(
    "fsq_new_eval_report", _LIBERO_EXAMPLES / "fsq_eval.py"
)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
fsq_eval_report = importlib.util.module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(fsq_eval_report)


def test_sample_actions_forwards_context_on_every_denoising_step() -> None:
    calls = []
    fake = SimpleNamespace(chunk_size=2, max_action_dim=3)

    def velocity(self, x_t, time, state, z_norm, **kwargs):
        calls.append(kwargs)
        return torch.ones_like(x_t)

    fake.velocity = MethodType(velocity, fake)
    context = torch.randn(1, 4, 8)
    result = VSAFlowExpert.sample_actions(
        fake,
        torch.zeros(1, 5),
        torch.zeros(1, 3),
        noise=torch.zeros(1, 2, 3),
        num_steps=2,
        skill_scale=0.5,
        image_context=context,
        image_scale=0.75,
    )

    torch.testing.assert_close(result, -torch.ones_like(result))
    assert len(calls) == 2
    assert all(call["image_context"] is context for call in calls)
    assert all(call["skill_scale"] == 0.5 and call["image_scale"] == 0.75 for call in calls)


def test_skill_eval_abc_shares_noise_and_terminator() -> None:
    class FakeActionExpert:
        def __init__(self):
            self.noise_ptrs = []

        @staticmethod
        def sample_noise(shape, device):
            return torch.ones(shape, device=device)

    class FakeTerminator:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def image_features(third, wrist):
            return third.mean(dim=(-2, -1)), wrist.mean(dim=(-2, -1))

        def __call__(self, z_norm, state, third, wrist, image_features=None):
            self.calls += 1
            batch = state.shape[0]
            return torch.zeros(batch), torch.zeros(batch)

    class FakeFSQ:
        @staticmethod
        def normalized(z_q):
            return z_q

    class FakeModel:
        chunk_size = 2
        action_dim = 1
        cfg = SimpleNamespace(
            max_action_dim=1,
            a_skill_scale=1.0,
            b_skill_scale=0.5,
            c_skill_scale=0.5,
            b_image_scale=1.0,
            c_image_scale=0.5,
            c_goal_scale=1.0,
        )

        def __init__(self):
            self.action_expert = FakeActionExpert()
            self.terminator = FakeTerminator()
            self.fsq = FakeFSQ()

        @staticmethod
        def image_context(third, wrist):
            return torch.stack([third[:, 0], wrist[:, 0]], dim=1)

        @staticmethod
        def goal_context(third, wrist):
            return torch.stack([third[:, 0], wrist[:, 0]], dim=1)

        def sample_action_chunks(
            self,
            z_q,
            raw_states,
            *,
            noise,
            skill_scale,
            image_scale=0.0,
            goal_scale=0.0,
            **kwargs,
        ):
            self.action_expert.noise_ptrs.append(noise.data_ptr())
            factor = float(skill_scale + image_scale + goal_scale + z_q.mean())
            return noise[..., :1].unsqueeze(1) * factor

    model = FakeModel()
    raw_dataset = [
        {
            "observation.images.image": torch.full((3, 4, 4), float(index)),
            "observation.images.wrist_image": torch.full((3, 4, 4), float(index + 1)),
        }
        for index in range(2)
    ]
    actions, progress, term = fsq_eval_report.batched_decode_fsq_new_abc(
        model,
        np.zeros((1, 2), dtype=np.float32),
        [np.zeros((2, 3), dtype=np.float32)],
        [{"dataset_from_index": 0, "frame_start": 0, "frame_end": 2}],
        raw_dataset,
        [2],
        "cpu",
        2,
        random_skill_latents=np.ones((1, 2), dtype=np.float32),
    )

    assert len(set(model.action_expert.noise_ptrs)) == 1
    assert model.terminator.calls == 1
    np.testing.assert_allclose(actions["A"][0], 1.0)
    np.testing.assert_allclose(actions["B"][0], 1.5)
    np.testing.assert_allclose(actions["C"][0], 2.0)
    np.testing.assert_allclose(actions["C + far skill"][0], 3.0)
    np.testing.assert_allclose(progress[0], 0.0)
    np.testing.assert_allclose(term[0], 0.5)


def test_skill_eval_plot_renders_abc_action_variants() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    base = np.zeros((2, 2, 1), dtype=np.float32)
    encoded = fsq_eval_report.make_sample_plot(
        image,
        image,
        base,
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([0.1, 0.9], dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        2,
        ["grip"],
        1,
        0.5,
        action_variants=[
            ("A", base, "#B71C1C", "--"),
            ("B", base + 0.1, "#00897B", ":"),
            ("C", base + 0.2, "#7B1FA2", "-."),
        ],
    )

    assert len(base64.b64decode(encoded)) > 1_000


def test_fsq_episode_data_sorts_skills_and_joins_goal_frames(tmp_path, monkeypatch) -> None:
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
    assert [skill["goal_index"] for skill in record["skills"]] == [119, 129]
