import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hf_eval_results import BASES, BOUNDARY_BASE, result_path, result_relative, select_results, upload_results
from hf_server import HFBackend


@pytest.fixture
def result(tmp_path):
    relative = BASES[0] + "/example"
    folder = tmp_path / relative
    (folder / "metrics").mkdir(parents=True)
    (folder / "metrics/eval_info_merged.json").write_text("{}")
    (folder / "video.mp4").write_bytes(b"video")
    return tmp_path, relative


def test_public_upload_preserves_exact_result_path(result):
    project, relative = result
    calls = []
    hub = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(private=False),
                          upload_folder=lambda **kw: calls.append(kw))
    selected = upload_results(project, "me/checkpoints", [relative], api=hub)
    assert selected[0]["file_count"] == 2
    assert calls[0]["repo_type"] == "model"
    assert calls[0]["path_in_repo"] == relative
    assert calls[0]["folder_path"] == project / relative


def test_private_destination_and_symlinks_rejected(result):
    project, relative = result
    hub = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(private=True),
                          upload_folder=lambda **kw: pytest.fail("Must not upload"))
    with pytest.raises(ValueError, match="공개"):
        upload_results(project, "me/private", [relative], api=hub)
    (project / relative / "secret-link").symlink_to(__file__)
    with pytest.raises(ValueError, match="링크"):
        select_results(project, [relative])
    for path in ("../secret", "outputs_filtered/run", "lerobot/src", BASES[0]):
        with pytest.raises(ValueError):
            result_relative(path)


def test_desk_stages_pod_results_without_touching_existing_local_eval(result):
    project, relative = result
    stage = project / ".cache/runpod_finish/pod123/results"
    metrics = stage / relative / "metrics/eval_info_merged.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"pod": true}')
    backend = HFBackend({"project_root": str(project), "hf_dataset_repo": "me/data",
                         "hf_checkpoint_repo": "me/checkpoints"})
    plan = backend.plan("local-eval-results", "hub-eval-results", [relative],
                        options={"runpod_id": "pod123"}, operation="eval-push")["items"][0]
    assert plan["source"] == str(stage / relative)
    assert plan["destination"] == "https://huggingface.co/me/checkpoints/" + relative
    command = backend.commands(plan)
    assert command[command.index("--project") + 1] == str(stage)
    assert "--eval-results" in command
    assert (project / relative / "metrics/eval_info_merged.json").read_text() == "{}"
    with pytest.raises(ValueError):
        backend.eval_project({"runpod_id": "../escape"})


def test_boundary_report_requires_completion_marker_and_preserves_path(tmp_path):
    relative = BOUNDARY_BASE + "/libero_90/run"
    folder = tmp_path / relative
    folder.mkdir(parents=True)
    for name in ("index.html", "summary.json", "threshold_statistics.json"):
        (folder / name).write_text("{}")
    with pytest.raises(ValueError):
        select_results(tmp_path, [relative])
    (folder / "evaluation_complete.json").write_text(json.dumps({"kind": "skill_boundary_detection"}))
    assert select_results(tmp_path, [relative])[0]["file_count"] == 4
