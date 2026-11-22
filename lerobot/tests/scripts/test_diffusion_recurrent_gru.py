"""Streaming GRU rollout, episode reset, IID windows and episodic TBPTT."""
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.recurrent_sampler import RecurrentEpisodeBatchSampler, recurrent_episode_ranges
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionHistoryEncoder, DiffusionPolicy
from lerobot.policies.diffusion.recurrent import EpisodeGRUContext
from lerobot.processor.converters import batch_to_transition, transition_to_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    values = dict(
        device="cpu", state_only=True, history_encoder="gru", history_encoder_dim=8,
        history_gru_layers=2, history_gru_stateful=True, n_obs_steps=4,
        horizon=8, n_action_steps=3, down_dims=(8, 16), n_groups=4,
        diffusion_step_embed_dim=8, num_train_timesteps=8, num_inference_steps=2,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    values.update(kwargs)
    return DiffusionConfig(**values)


def test_single_steps_and_sparse_episode_anchors_match_full_prefix():
    policy = DiffusionPolicy(config()).eval()
    states = torch.randn(2, 11, 3)
    consumed = []
    context = EpisodeGRUContext(
        policy.diffusion, lambda t: (consumed.append(t) or states[:, t])
    )
    with torch.no_grad():
        for frame in [1, 4, 9]:
            actual = context.at(frame)
            _, expected_hidden = policy.diffusion.history_encoder.encoder(states[:, :frame + 1])
            expected = policy.diffusion.history_encoder.output_norm(expected_hidden[-1])
            torch.testing.assert_close(actual, expected)
        context.at(9)
        assert consumed == list(range(10))
        assert context.hidden.shape == (2, 2, 8)
        fresh = EpisodeGRUContext(policy.diffusion, lambda t: states[:, t])
        expected, _ = policy.diffusion.gru_step(states[:, 0])
        torch.testing.assert_close(fresh.at(0), expected)
    with pytest.raises(ValueError, match="chronological"):
        context.at(3)


def test_rollout_updates_hidden_while_executing_cached_actions_and_resets(monkeypatch):
    policy = DiffusionPolicy(config()).eval()
    conditions = []
    def sample(batch_size, global_cond, noise=None):
        conditions.append(global_cond.clone())
        return global_cond.new_zeros(batch_size, 8, 2)
    monkeypatch.setattr(policy.diffusion, "conditional_sample", sample)
    states = torch.randn(2, 8, 3)
    for frame in range(8):
        policy.select_action({"observation.state": states[:, frame]})
        with torch.no_grad():
            _, expected = policy.diffusion.history_encoder.encoder(states[:, :frame + 1])
        torch.testing.assert_close(policy._gru_hidden, expected)
    assert len(conditions) == 3  # replanning at frames 0, 3, 6, not every step
    assert len(policy._queues["observation.state"]) == 0
    policy.reset()
    policy.select_action({"observation.state": states[:, 0]})
    torch.testing.assert_close(conditions[0], conditions[-1])


def test_iid_training_restarts_hidden_and_ignores_previous_rollout():
    policy = DiffusionPolicy(config())
    history = torch.randn(2, 4, 3)
    before = policy.diffusion._prepare_global_conditioning({"observation.state": history})
    policy.select_action({"observation.state": torch.randn(2, 3)})
    after = policy.diffusion._prepare_global_conditioning({"observation.state": history})
    torch.testing.assert_close(before, after)


def test_epi_carries_hidden_without_padding_or_cross_update_gradients():
    encoder = DiffusionHistoryEncoder(config(history_gru_training="epi"), input_dim=3)
    states = torch.randn(1, 7, 3)
    first_window = torch.cat([states[:, :1], states[:, :3]], dim=1)
    first = encoder.episode_training_condition(first_window, torch.tensor([8]), torch.tensor([2]))
    first.square().mean().backward()
    encoder.zero_grad()
    cache = encoder._episode_training_states[True][8][1]
    assert not cache.requires_grad and cache.grad_fn is None
    second = encoder.episode_training_condition(states[:, 3:], torch.tensor([8]), torch.tensor([6]))
    _, expected = encoder.encoder(states)
    torch.testing.assert_close(second, encoder.output_norm(expected[-1]))
    second.square().mean().backward()  # no backward-through-freed-graph error
    restart = encoder.episode_training_condition(first_window, torch.tensor([8]), torch.tensor([2]))
    torch.testing.assert_close(restart, first)
    with pytest.raises(ValueError, match="chronological"):
        encoder.episode_training_condition(states[:, 3:], torch.tensor([8]), torch.tensor([9]))


@pytest.mark.parametrize("mode", ["iid", "epi"])
def test_real_diffusion_loss_and_optimizer_steps(mode):
    policy = DiffusionPolicy(config(history_gru_training=mode))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    for anchor in [3, 7]:
        batch = {
            "observation.state": torch.randn(2, 4, 3), "action": torch.randn(2, 8, 2),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
            "episode_index": torch.tensor([1, 2]), "frame_index": torch.tensor([anchor, anchor]),
        }
        batch = transition_to_batch(batch_to_transition(batch))
        loss, _ = policy.forward(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert policy.diffusion.history_encoder.encoder.weight_ih_l0.grad is not None
        optimizer.step()
        optimizer.zero_grad()


@pytest.mark.parametrize('mode', ['iid', 'epi'])
def test_libero_format_episode_subset_trains_real_gru(tmp_path, mode):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / 'libero'
    writer = LeRobotDataset.create(
        repo_id='test/libero', root=root, fps=20,
        features={
            'observation.state': {'dtype': 'float32', 'shape': (3,), 'names': ['x', 'y', 'z']},
            'action': {'dtype': 'float32', 'shape': (2,), 'names': ['a', 'b']},
        },
    )
    for episode, length in enumerate([17, 21, 25]):
        for frame in range(length):
            writer.add_frame({
                'observation.state': np.array([episode, frame, frame * .1], dtype=np.float32),
                'action': np.array([frame * .01, episode], dtype=np.float32), 'task': 'test',
            })
        writer.save_episode()
    writer.finalize()
    dataset = LeRobotDataset(
        repo_id='test/libero', root=root, episodes=[1, 2],
        delta_timestamps={'observation.state': [-3/20, -2/20, -1/20, 0],
                          'action': [i/20 for i in range(8)]},
    )
    ranges = recurrent_episode_ranges(dataset, drop_n_last_frames=7)
    assert ranges == [(0, 14), (21, 18)]
    sampler = RecurrentEpisodeBatchSampler(ranges, 2, 4)
    policy = DiffusionPolicy(config(history_gru_training=mode))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    for batch in loader:
        assert not batch['action_is_pad'].any()
        assert set(batch['episode_index'].tolist()) == {1, 2}
        batch = transition_to_batch(batch_to_transition(batch))
        loss, _ = policy.forward(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert policy.diffusion.history_encoder.encoder.weight_ih_l1.grad is not None
        optimizer.step()
        optimizer.zero_grad()
    if mode == 'epi':
        assert set(policy.diffusion.history_encoder._episode_training_states[True]) == {1, 2}


def test_episode_sampler_is_chronological_and_varies_chunk_boundaries():
    ranges = [(0, 25), (25, 17), (42, 2)]
    sampler = RecurrentEpisodeBatchSampler(ranges, batch_size=2, chunk_length=4)
    phases, last, seen_episodes = set(), {}, set()
    for _ in range(8):
        batches = list(iter(sampler))
        assert len(batches) == len(sampler)
        for batch in batches:
            assert len(batch) == 2
            episodes = []
            for index in batch:
                episode, (start, length) = next((i, span) for i, span in enumerate(ranges)
                                               if span[0] <= index < sum(span))
                frame = index - start
                episodes.append(episode)
                seen_episodes.add(episode)
                if episode not in last or frame != last[episode] + 4:
                    assert frame < min(length, 4)
                    phases.add(frame)
                last[episode] = frame
            assert len(episodes) == len(set(episodes))
    assert phases == {0, 1, 2, 3}
    assert seen_episodes == {0, 1, 2}


def test_episode_sampler_keeps_long_tail_across_passes_and_rejects_too_few_episodes():
    sampler = RecurrentEpisodeBatchSampler([(0, 41), (41, 1), (42, 1)], 2, 4)
    long_frames = []
    for _ in range(4):
        for batch in sampler:
            assert len(batch) == len(set(batch)) == 2
            long_frames.extend(index for index in batch if index < 41)
    assert max(long_frames) >= 37  # The long tail is retained, not dropped.
    for previous, current in zip(long_frames, long_frames[1:]):
        assert current == previous + 4 or current < 4
    with pytest.raises(ValueError, match="at least batch_size"):
        RecurrentEpisodeBatchSampler([(0, 5)], 2, 4)


@pytest.mark.parametrize("workers", [0, 1])
def test_episode_sampler_survives_accelerate_preparation(workers):
    from accelerate import Accelerator
    sampler = RecurrentEpisodeBatchSampler([(0, 7), (7, 9)], 2, 4)
    reference = RecurrentEpisodeBatchSampler([(0, 7), (7, 9)], 2, 4)
    loader = torch.utils.data.DataLoader(
        list(range(16)), batch_sampler=sampler, num_workers=workers,
        persistent_workers=bool(workers), multiprocessing_context="spawn" if workers else None,
    )
    prepared = Accelerator(cpu=True).prepare(loader)
    try:
        for _ in range(3):
            assert [batch.tolist() for batch in prepared] == list(reference)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_epi_full_batches_keep_hidden_across_loader_passes():
    encoder = DiffusionHistoryEncoder(config(history_gru_training="epi"), input_dim=3)
    lengths = [29, 9, 2]
    states = [torch.randn(length, 3) for length in lengths]
    ranges = [(0, 29), (29, 9), (38, 2)]
    sampler = RecurrentEpisodeBatchSampler(ranges, 2, 4)
    with torch.no_grad():
        for _ in range(3):
            for batch in sampler:
                histories, episodes, frames, expected = [], [], [], []
                for index in batch:
                    episode, (start, _) = next((i, span) for i, span in enumerate(ranges)
                                              if span[0] <= index < sum(span))
                    frame = index - start
                    histories.append(states[episode][torch.arange(frame - 3, frame + 1).clamp_min(0)])
                    episodes.append(episode)
                    frames.append(frame)
                    _, hidden = encoder.encoder(states[episode][:frame + 1].unsqueeze(0))
                    expected.append(encoder.output_norm(hidden[-1]))
                actual = encoder.episode_training_condition(
                    torch.stack(histories), torch.tensor(episodes), torch.tensor(frames),
                )
                torch.testing.assert_close(actual, torch.cat(expected))


def test_old_windowed_gru_contract_stays_available():
    cfg = config(history_gru_stateful=False, history_gru_layers=1)
    assert cfg.history_gru_training == "iid"
    policy = DiffusionPolicy(cfg)
    policy.select_action({"observation.state": torch.randn(1, 3)})
    assert policy._gru_hidden is None


@pytest.mark.parametrize("descriptor_length", [1, 8])
@pytest.mark.parametrize("center", ["previous_action", "pure_noise"])
def test_sbd_streams_intermediate_frames_and_restarts_each_episode(monkeypatch, descriptor_length, center):
    import importlib
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd

    source = Path(__file__).resolve().parents[3] / "lerobot/skillVLA/libero/configs/train_skills/build_data/src"
    monkeypatch.syspath_prepend(str(source))
    divider = importlib.import_module("skill_divider")
    manifold = importlib.import_module("action_manifold")
    policy = DiffusionPolicy(config()).eval()
    states = torch.randn(15, 3)
    states[:, 0] = torch.arange(15)
    episode = pd.DataFrame({"observation.state": list(states.numpy()),
                            "action": list(np.zeros((15, 2), dtype=np.float32))})
    pca = manifold.ActionPCA(
        mean=np.zeros(2), scale=np.ones(2), components=np.eye(2),
        explained_variance=np.ones(2), explained_variance_ratio=np.full(2, .5),
        sample_count=15, metadata={},
    )
    conditions, consumed = [], []
    def query(_policy, conditioning, chunks, *args, **kwargs):
        conditions.append(conditioning.clone())
        return chunks.detach().cpu().numpy()
    def preprocess(obs):
        consumed.append(int(obs["observation.state"][0]))
        return {key: value.unsqueeze(0) for key, value in obs.items()}
    monkeypatch.setattr(divider, "_query_vf_chunks", query)
    if center == "pure_noise":
        def trajectory(_policy, conditioning, chunks, *args):
            step7 = query(_policy, conditioning, chunks)
            return step7, step7 + 1
        monkeypatch.setattr(divider, "_sample_pure_noise_trajectory", trajectory)
        def no_gt_center(*args, **kwargs):
            pytest.fail("pure_noise must not construct forward-noised GT probes")
        monkeypatch.setattr(divider, "_make_scheduler_gaussian_probes", no_gt_center)
    for _ in range(2):
        conditions.clear()
        consumed.clear()
        diagnostics = {} if center == "pure_noise" else None
        result = divider.run_vf_analysis(
            policy, preprocess, episode, {}, [], (7 if center == "pure_noise" else 0), 3,
            probe_type=manifold.PROBE_PCA_ACTION, action_pca=pca,
            action_normalizer=manifold.NumpyActionNormalizer("IDENTITY", {}),
            probe_directions=np.zeros((4, 8, 2), dtype=np.float32),
            probe_generation="scheduler_gaussian", scheduler_gaussian_center=center,
            descriptor_temporal_length=descriptor_length, compute_legacy_divergence=False,
            metric_diagnostics=diagnostics,
        )
        assert result[0] == ([0, 3, 6] if center == "pure_noise" else [3, 6])
        if center == "pure_noise":
            np.testing.assert_allclose(diagnostics["final_descriptors"] - result[1], 1)
        assert consumed == list(range(7))
        for anchor, actual in zip(result[0], conditions, strict=True):
            with torch.no_grad():
                _, hidden = policy.diffusion.history_encoder.encoder(states[None, :anchor + 1])
                expected = policy.diffusion.history_encoder.output_norm(hidden[-1])
            torch.testing.assert_close(actual, expected)
