from types import SimpleNamespace

import lerobot.datasets.factory as dataset_factory
import lerobot.policies.skillVLA.dataset_skillVLA as skill_dataset_module


def test_stage2_skill_only_dataset_includes_canonical_actions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSkillVLADataset:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.meta = SimpleNamespace(camera_keys=[], stats={})

    monkeypatch.setattr(
        dataset_factory,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(features={}, camera_keys=[]),
    )
    monkeypatch.setattr(
        skill_dataset_module, "SkillVLADataset", FakeSkillVLADataset
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            repo_id="local/test",
            root="/tmp/test",
            revision=None,
            image_transforms=SimpleNamespace(enable=False),
            streaming=False,
            episodes=None,
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        policy=SimpleNamespace(
            type="skill_vla_stage2",
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_supervision="skill_only",
            skill_flow_enabled=False,
            skill_flow_target="canonical",
            skill_flow_max_length=0,
            transition_jitter_pmax=10,
            transition_jitter_early_start_pmax=10,
            transition_jitter_late_start_pmax=5,
            transition_jitter_early_end_pmax=10,
            transition_jitter_late_end_pmax=5,
            use_dino_features=False,
            state_only=False,
            state_only_auxiliary=False,
        ),
        tolerance_s=1e-4,
    )

    dataset = dataset_factory.make_dataset(cfg)

    assert isinstance(dataset, FakeSkillVLADataset)
    assert captured["include_canonical_skill_actions"] is True
    assert captured["canonical_skill_action_max_length"] is None

