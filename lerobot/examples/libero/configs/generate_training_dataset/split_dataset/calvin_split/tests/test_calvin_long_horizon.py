from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


CALVIN_SPLIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALVIN_SPLIT_DIR / "src"))

from calvin_long_horizon import (  # noqa: E402
    Annotation,
    Candidate,
    Event,
    Occurrence,
    _build_html,
    _successors,
    candidate_key,
    candidate_counts_by_min_occurrences,
    choose_visualized_candidates,
    discover_candidates,
    independent_occurrences,
    merge_overlapping_same_task,
)
from calvin_long_horizon_config import load_config, settings  # noqa: E402
from calvin_task_split import (  # noqa: E402
    _selected_occurrences,
    load_candidate_report,
    make_split_units,
)
from calvin_task_split_config import (  # noqa: E402
    SelectedCandidate,
    load_settings as load_split_settings,
    output_names,
)


def _event(recording: int, start: int, end: int, task: str, index: int) -> Event:
    return Event(
        recording_index=recording,
        start=start,
        end=end,
        task_id=task,
        annotation_indexes=(index,),
        languages=(f"prompt {index}",),
    )


def _candidate(steps: int, count: int, name: str) -> Candidate:
    occurrences = []
    for index in range(count):
        events = tuple(
            _event(index, step * 80, step * 80 + 64, f"{name}_{step}", step)
            for step in range(steps)
        )
        occurrences.append(
            Occurrence(
                recording_index=index,
                start=events[0].start,
                end=events[-1].end,
                task_ids=tuple(event.task_id for event in events),
                events=events,
                gaps=tuple(15 for _ in range(steps - 1)),
            )
        )
    return Candidate(
        task_ids=tuple(f"{name}_{step}" for step in range(steps)),
        raw_occurrence_count=count,
        occurrences=tuple(occurrences),
    )


def test_default_config_requires_repeated_long_continuous_sequences() -> None:
    resolved = settings(load_config())

    assert resolved.search.sequence_steps == (2, 3)
    assert resolved.search.min_total_frames == 120
    assert resolved.search.max_total_frames == 450
    assert resolved.search.max_gap_frames == 90
    assert resolved.search.min_occurrences == (3, 5)
    assert resolved.source.source_dir.name == "training"


def test_candidate_key_is_stable_and_rank_independent() -> None:
    assert candidate_key(("open_drawer", "lift_blue_block_drawer")) == (
        "2step__open_drawer__lift_blue_block_drawer"
    )


def test_overlapping_same_task_annotations_merge_but_paraphrases_remain() -> None:
    annotations = [
        Annotation(0, 0, 10, 74, "open_drawer", "open it"),
        Annotation(1, 0, 20, 84, "open_drawer", "pull the drawer open"),
        Annotation(2, 0, 90, 140, "open_drawer", "open it again"),
        Annotation(3, 0, 30, 80, "turn_on_led", "turn on the led"),
    ]

    events = merge_overlapping_same_task(annotations)
    drawer = [event for event in events if event.task_id == "open_drawer"]

    assert len(drawer) == 2
    assert drawer[0].start == 10 and drawer[0].end == 84
    assert drawer[0].annotation_indexes == (0, 1)
    assert drawer[0].languages == ("open it", "pull the drawer open")


def test_successor_is_earliest_nonoverlapping_event_not_overlapping_colabel() -> None:
    events = [
        _event(0, 0, 50, "task_a", 0),
        _event(0, 20, 60, "overlapping_label", 1),
        _event(0, 70, 120, "task_b", 2),
        _event(0, 80, 130, "task_c", 3),
    ]

    assert _successors(events, 0, max_gap_frames=90) == [2]


def test_semantic_sequence_needs_five_independent_occurrences() -> None:
    resolved = settings(load_config())
    resolved = replace(
        resolved,
        search=replace(resolved.search, sequence_steps=(2,), min_occurrences=(5,)),
    )
    events = []
    for recording in range(5):
        offset = recording * 1000
        events.extend(
            [
                _event(recording, offset, offset + 64, "open_drawer", recording * 2),
                _event(
                    recording,
                    offset + 80,
                    offset + 150,
                    "place_in_drawer",
                    recording * 2 + 1,
                ),
            ]
        )

    candidates, by_steps = discover_candidates(events, resolved)

    assert by_steps == {"2": 1}
    assert len(candidates) == 1
    assert candidates[0].task_ids == ("open_drawer", "place_in_drawer")
    assert candidates[0].occurrence_count == 5


def test_multiple_occurrence_thresholds_use_one_candidate_list() -> None:
    resolved = settings(load_config())
    resolved = replace(
        resolved,
        search=replace(
            resolved.search,
            sequence_steps=(2,),
            min_occurrences=(3, 5),
        ),
    )
    events = []
    for recording in range(5):
        offset = recording * 1000
        events.extend(
            [
                _event(recording, offset, offset + 64, "five_a", recording * 4),
                _event(recording, offset + 80, offset + 150, "five_b", recording * 4 + 1),
            ]
        )
        if recording < 3:
            events.extend(
                [
                    _event(recording, offset + 300, offset + 364, "three_a", recording * 4 + 2),
                    _event(recording, offset + 380, offset + 450, "three_b", recording * 4 + 3),
                ]
            )

    candidates, _ = discover_candidates(events, resolved)
    counts = candidate_counts_by_min_occurrences(
        candidates, resolved.search.min_occurrences
    )

    assert [candidate.task_ids for candidate in candidates] == [
        ("five_a", "five_b"),
        ("three_a", "three_b"),
    ]
    assert counts == {
        "3": {"candidate_count": 2, "candidate_count_by_steps": {"2": 2}},
        "5": {"candidate_count": 1, "candidate_count_by_steps": {"2": 1}},
    }


def test_occurrence_count_discards_overlapping_duplicate_demos() -> None:
    event_a = _event(0, 0, 60, "a", 0)
    event_b = _event(0, 80, 140, "b", 1)
    event_c = _event(0, 200, 260, "a", 2)
    event_d = _event(0, 280, 340, "b", 3)
    rows = [
        Occurrence(0, 0, 140, ("a", "b"), (event_a, event_b), (19,)),
        Occurrence(0, 20, 120, ("a", "b"), (event_a, event_b), (19,)),
        Occurrence(0, 200, 340, ("a", "b"), (event_c, event_d), (19,)),
    ]

    selected = independent_occurrences(rows)

    assert [(row.start, row.end) for row in selected] == [(20, 120), (200, 340)]


def test_max_candidates_is_applied_per_sequence_length() -> None:
    resolved = settings(load_config())
    resolved = replace(
        resolved,
        visualization=replace(resolved.visualization, max_candidates=1),
    )
    candidates = [
        _candidate(2, 7, "two_first"),
        _candidate(2, 6, "two_second"),
        _candidate(3, 5, "three_first"),
        _candidate(3, 4, "three_second"),
    ]

    selected = choose_visualized_candidates(candidates, resolved)

    assert [(rank, len(candidate.task_ids)) for rank, candidate in selected] == [
        (1, 2),
        (3, 3),
    ]


def test_html_has_summary_index_links_and_step_sections(tmp_path: Path) -> None:
    resolved = settings(load_config())
    two_step = _candidate(2, 5, "two")
    three_step = _candidate(3, 3, "three")
    rendered = [
        (1, two_step, [(two_step.occurrences[0], tmp_path / "two.mp4")]),
        (2, three_step, [(three_step.occurrences[0], tmp_path / "three.mp4")]),
    ]

    document = _build_html(
        rendered,
        tmp_path / "index.html",
        [two_step, three_step],
        resolved,
    )

    assert "Eligibility summary" in document
    assert 'href="#candidate-001"' in document
    assert 'id="candidate-001"' in document
    assert 'id="step-2"' in document
    assert 'id="step-3"' in document
    assert "2-step candidates (1 shown / 1 eligible)" in document
    assert "2step__two_0__two_1" in document
    assert "Copy key" in document


def test_split_config_uses_automatic_three_way_output_names() -> None:
    resolved = load_split_settings()

    assert resolved.selected_candidates == ()
    assert output_names(resolved) == {
        "play_pretrain": "calvin_D_play_pretrain_full_full",
        "language_pretrain": "calvin_D_pretrain_full_full",
        "heldout": "calvin_D_heldout_full_full",
    }


def test_candidate_report_is_selected_by_stable_key(tmp_path: Path) -> None:
    key = "2step__open_drawer__lift_blue_block_drawer"
    report_path = tmp_path / "candidates.json"
    report_path.write_text(
        """{
          "source_dir": "/tmp/source",
          "candidates": [{
            "candidate_key": "2step__open_drawer__lift_blue_block_drawer",
            "task_ids": ["open_drawer", "lift_blue_block_drawer"],
            "occurrences": [{
              "recording_index": 0,
              "source_start": 5,
              "source_end": 15
            }]
          }]
        }""",
        encoding="utf-8",
    )
    _, indexed = load_candidate_report(report_path)

    occurrences, metadata = _selected_occurrences(
        indexed,
        (SelectedCandidate(key, "open the drawer, then lift the blue block"),),
        [(0, 20)],
    )

    assert occurrences == [
        {
            "candidate_key": key,
            "language": "open the drawer, then lift the blue block",
            "candidate_occurrence_index": 0,
            "recording_index": 0,
            "recording_start": 0,
            "recording_end": 20,
            "start": 5,
            "end": 15,
        }
    ]
    assert metadata == [
        {
            "candidate_key": key,
            "language": "open the drawer, then lift the blue block",
        }
    ]


def test_three_way_units_remove_only_selected_occurrence_spans() -> None:
    annotation = {
        "annotations": np.asarray(["a", "b", "c", "d"]),
        "task_ids": np.asarray(["task_a", "task_b", "task_c", "task_d"]),
        "embeddings": np.zeros((4, 3), dtype=np.float32),
        "intervals": np.asarray([[0, 2], [4, 6], [10, 12], [17, 19]]),
    }
    selected = [
        {
            "candidate_key": "2step__task_b__task_c",
            "language": "do b and then c",
            "candidate_occurrence_index": 0,
            "recording_index": 0,
            "recording_start": 0,
            "recording_end": 20,
            "start": 5,
            "end": 15,
        }
    ]

    units, removed_intervals, removed_annotations = make_split_units(
        annotation, [(0, 20)], selected
    )

    assert removed_intervals == [(5, 15)]
    assert [(row["start"], row["end"]) for row in units["play_pretrain"]] == [
        (0, 4),
        (16, 20),
    ]
    assert removed_annotations == [1, 2]
    assert [row["source_unit_index"] for row in units["language_pretrain"]] == [0, 3]
    assert units["heldout"][0]["task_id"] == "2step__task_b__task_c"
    assert units["heldout"][0]["language"] == "do b and then c"
