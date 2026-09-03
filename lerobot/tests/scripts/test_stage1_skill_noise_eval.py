from __future__ import annotations

import sys
from pathlib import Path


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_skill_eval/src"
)
sys.path.insert(0, str(_SRC))

import run_skill_noise_eval as noise_eval  # noqa: E402
import stage1_skill_eval_config as eval_config  # noqa: E402
import stage1_skill_noise_eval_config as noise_config  # noqa: E402


def test_neighbor_and_opposite_corner_includes_shells_and_antipodal_corner() -> None:
    assert noise_eval._evaluated_tokens(
        0, [3, 3, 3], probe_mode="neighbor_and_opposite"
    ) == [0, 1, 3, 9, 17, 23, 25, 26]


def test_neighbor_and_opposite_center_uses_all_eight_tied_farthest_codes() -> None:
    assert noise_eval._evaluated_tokens(
        13, [3, 3, 3], probe_mode="neighbor_and_opposite"
    ) == [13, 4, 10, 12, 14, 16, 22, 0, 2, 6, 8, 18, 20, 24, 26]


def test_neighbor_and_opposite_config_alias() -> None:
    assert (
        noise_config._code_probe_mode("neighbor+opposite")
        == "neighbor_and_opposite"
    )


def test_noise_output_name_always_includes_probe_mode() -> None:
    assert (
        noise_config._probe_suffixed_output_name(
            "FSQ333_skills", "30k", "neighbor_and_opposite"
        )
        == "FSQ333_skills_30k_neighbor_and_opposite"
    )
    assert (
        noise_config._probe_suffixed_output_name("FSQ333_skills", "30k", "off")
        == "FSQ333_skills_30k_off"
    )
    assert (
        noise_config._probe_suffixed_output_name(
            "FSQ333_skills",
            "30k",
            "off",
            skill_only_rollout_probe=True,
        )
        == "FSQ333_skills_30k_off_skillonly"
    )


def test_output_name_starts_with_episode_layout_mode() -> None:
    assert (
        eval_config._episode_mode_prefixed_output_name(
            "FSQ333_skills", episode_exact=True
        )
        == "exact_FSQ333_skills"
    )
    assert (
        eval_config._episode_mode_prefixed_output_name(
            "FSQ333_skills", episode_exact=False
        )
        == "random_FSQ333_skills"
    )


def test_episode_layout_prefix_is_canonical_and_not_duplicated() -> None:
    assert (
        eval_config._episode_mode_prefixed_output_name(
            "exact_FSQ333_skills", episode_exact=True
        )
        == "exact_FSQ333_skills"
    )
    assert (
        eval_config._episode_mode_prefixed_output_name(
            "exact_FSQ333_skills", episode_exact=False
        )
        == "random_FSQ333_skills"
    )


def test_noise_checkpoint_suffix_is_compact() -> None:
    assert noise_config._compact_checkpoint("030000") == "30k"
    assert noise_config._compact_checkpoint("002500") == "2p5k"
    assert noise_config._compact_checkpoint("last") == "last"


def test_neighbor_and_opposite_roles_are_explicit() -> None:
    roles = noise_eval._evaluated_token_roles(
        0, [3, 3, 3], probe_mode="neighbor_and_opposite"
    )
    assert roles[0] == "original"
    assert {token for token, role in roles.items() if role == "neighbor"} == {
        1,
        3,
        9,
    }
    assert {token for token, role in roles.items() if role == "opposite"} == {
        17,
        23,
        25,
        26,
    }


def test_skill_only_seed_matches_each_training_noise_contract() -> None:
    base = 123
    assert noise_eval._rollout_sampling_seed(
        base, rollout_path="main", skill_flow_target="canonical"
    ) == base
    assert noise_eval._rollout_sampling_seed(
        base, rollout_path="skill_only", skill_flow_target="extended_chunk"
    ) == base
    assert noise_eval._rollout_sampling_seed(
        base, rollout_path="skill_only", skill_flow_target="canonical"
    ) != base
