from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_skill_eval/src/merge_results.py"
)
SPEC = importlib.util.spec_from_file_location("stage1_skill_eval_merge_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_report_counts_one_occurrence_and_two_policy_evaluations(
    tmp_path: Path,
) -> None:
    base = {
        "token": 1,
        "task_id": 0,
        "episode_id": 4,
        "frame_start": 10,
        "branches": [],
    }
    manifest = {
        "model_label": "2 policies",
        "signature": {
            "policies": [{"label": "A"}, {"label": "B"}],
            "main_terminator": {"label": "FSQ_INIT", "variant": "fsq_initial"},
            "terminator_models": [{"label": "TERM", "variant": "state_image"}],
            "target_task": "libero_90",
            "selected_episodes": {"0": [4]},
            "time_shift_offset": 15,
        },
        "records": {
            "model_00__occ": {
                **base,
                "uid": "model_00__occ",
                "occurrence_uid": "occ",
                "model_index": 0,
                "branches": [
                    {
                        "name": "gt",
                        "unavailable_reason": None,
                        "green_tint": True,
                    },
                    {
                        "name": "policy",
                        "unavailable_reason": None,
                        "green_tint": True,
                    },
                    {
                        "name": "policy_alt_noise",
                        "unavailable_reason": None,
                        "green_tint": False,
                    },
                    {
                        "name": "policy_early",
                        "unavailable_reason": None,
                        "green_tint": True,
                    },
                    {
                        "name": "policy_late",
                        "unavailable_reason": "invalid shift",
                        "green_tint": False,
                    },
                ],
            },
            "model_01__occ": {
                **base,
                "uid": "model_01__occ",
                "occurrence_uid": "occ",
                "model_index": 1,
                "branches": [
                    {
                        "name": "policy",
                        "unavailable_reason": None,
                        "green_tint": False,
                    },
                    {
                        "name": "policy_alt_noise",
                        "unavailable_reason": None,
                        "green_tint": True,
                    },
                    {
                        "name": "policy_early",
                        "unavailable_reason": None,
                        "green_tint": False,
                    },
                    {
                        "name": "policy_late",
                        "unavailable_reason": None,
                        "green_tint": True,
                    },
                ],
            },
        },
    }

    payload = MODULE.report_payload(manifest, levels=[3, 3, 3])

    assert payload["occurrence_count"] == 1
    assert payload["evaluation_count"] == 2
    assert len(payload["review_id"]) == 20
    assert payload["models"] == [{"label": "A"}, {"label": "B"}]
    assert payload["main_terminator"]["label"] == "FSQ_INIT"
    assert payload["terminator_models"][0]["label"] == "TERM"
    assert payload["model_skill_success"] == [
        {
            "model_index": 0,
            "label": "A",
            "id": {
                "success_count": 1,
                "total_count": 2,
                "success_rate": 0.5,
                "rank": 1,
            },
            "ood": {
                "success_count": 1,
                "total_count": 1,
                "success_rate": 1.0,
                "rank": 1,
            },
        },
        {
            "model_index": 1,
            "label": "B",
            "id": {
                "success_count": 1,
                "total_count": 2,
                "success_rate": 0.5,
                "rank": 1,
            },
            "ood": {
                "success_count": 1,
                "total_count": 2,
                "success_rate": 0.5,
                "rank": 2,
            },
        },
    ]
    assert payload["skills"][0]["model_skill_success"] == payload["model_skill_success"]

    html = MODULE.write_html_report(tmp_path, payload).read_text(encoding="utf-8")
    assert "ID: exact + different noise" in html
    assert "OOD: early + late" in html
    assert 'successMetric("ID",stat.id)' in html
    assert 'successMetric("OOD",stat.ood)' in html
    assert 'class="success-metric${rankClass}"' in html
    assert 'class="success-rank">#${rank}' in html
    assert ".success-metric.rank-1" in html
    assert ".success-metric.rank-2" in html
    assert "Human review: click a policy panel" in html
    assert "stage1_skill_eval_human_review_v1" in html
    assert "function toggleCorrection(occ,branch)" in html
    assert "function successStats(occurrences)" in html
    assert "Export corrections" in html
    assert "Import corrections" in html
    assert "Clear corrections" in html
    assert "MANUAL SUCCESS" in html
    assert "MANUAL FAIL" in html
    assert "localStorage.setItem(STORAGE_KEY" in html
    assert 'fetch("./api/corrections"' in html
    assert "Server autosave connected" in html
    assert "connectReviewServer();" in html
    assert "GT is excluded from human-review success rates." in html
    assert "Token #${skill.token} skill success" in html
    assert "successCards(successStats(skill.occurrences))" in html
    assert "boundary-values" not in html
    assert "FIRED BEFORE MAIN" not in html


def test_each_token_has_its_own_success_rates() -> None:
    manifest = {
        "model_label": "1 policy",
        "signature": {
            "policies": [{"label": "A"}],
            "target_task": "libero_90",
            "selected_episodes": {"0": [4]},
            "time_shift_offset": 15,
        },
        "records": {
            "token_1": {
                "uid": "token_1",
                "occurrence_uid": "occ_1",
                "model_index": 0,
                "token": 1,
                "task_id": 0,
                "episode_id": 4,
                "frame_start": 10,
                "branches": [
                    {
                        "name": "policy",
                        "unavailable_reason": None,
                        "green_tint": True,
                    }
                ],
            },
            "token_2": {
                "uid": "token_2",
                "occurrence_uid": "occ_2",
                "model_index": 0,
                "token": 2,
                "task_id": 0,
                "episode_id": 4,
                "frame_start": 20,
                "branches": [
                    {
                        "name": "policy",
                        "unavailable_reason": None,
                        "green_tint": False,
                    }
                ],
            },
        },
    }

    payload = MODULE.report_payload(manifest, levels=[3, 3, 3])
    skills = {skill["token"]: skill for skill in payload["skills"]}

    assert payload["model_skill_success"][0]["id"]["success_rate"] == 0.5
    assert skills[1]["model_skill_success"][0]["id"]["success_rate"] == 1.0
    assert skills[2]["model_skill_success"][0]["id"]["success_rate"] == 0.0


def test_id_and_ood_are_ranked_independently_with_dense_ties() -> None:
    manifest = {
        "signature": {
            "policies": [
                {"label": "A"},
                {"label": "B"},
                {"label": "C"},
                {"label": "D"},
            ]
        },
        "records": {},
    }
    branch_results = {
        0: {"policy": True, "policy_early": False},
        1: {"policy": True, "policy_early": True},
        2: {"policy": False, "policy_early": True},
        3: {"policy": False},
    }
    for model_index, results in branch_results.items():
        manifest["records"][str(model_index)] = {
            "model_index": model_index,
            "branches": [
                {
                    "name": name,
                    "unavailable_reason": None,
                    "green_tint": success,
                }
                for name, success in results.items()
            ],
        }

    stats = MODULE._model_skill_success(manifest)

    assert [stat["id"]["rank"] for stat in stats] == [1, 1, 2, 2]
    assert [stat["ood"]["rank"] for stat in stats] == [2, 1, 1, None]
