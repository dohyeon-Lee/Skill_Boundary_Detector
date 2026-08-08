from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_skill_eval/src/review_server.py"
)
SPEC = importlib.util.spec_from_file_location("stage1_skill_eval_review_server", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _report(tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "report"
    (output / "metrics").mkdir(parents=True)
    (output / "index.html").write_text("<h1>report</h1>", encoding="utf-8")
    manifest = {
        "signature": {
            "format": "test",
            "policies": [{"label": "A"}],
            "target_task": "libero_90",
            "selected_episodes": {"0": [1]},
            "time_shift_offset": 15,
        },
        "model_label": "A",
        "levels": [3, 3, 3],
        "records": {
            "record": {
                "uid": "model_00__occ",
                "occurrence_uid": "occ",
                "model_index": 0,
                "token": 1,
                "task_id": 0,
                "episode_id": 1,
                "frame_start": 10,
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
                        "name": "policy_early",
                        "unavailable_reason": None,
                        "green_tint": False,
                    },
                    {
                        "name": "policy_late",
                        "unavailable_reason": "invalid shift",
                        "green_tint": False,
                    },
                ],
            }
        },
    }
    (output / "metrics" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return output, manifest


def _payload(store, corrections: dict) -> dict:
    return {
        "schema": MODULE.REVIEW_SCHEMA,
        "report_id": store.report_id,
        "updated_at": "2026-08-07T12:00:00+00:00",
        "corrections": corrections,
    }


def test_review_store_validates_and_atomically_saves_overrides(tmp_path: Path) -> None:
    output, manifest = _report(tmp_path)
    store = MODULE.ReviewStore(output)

    assert store.report_id == MODULE.review_id_for_signature(manifest["signature"])
    assert store.load()["corrections"] == {}

    saved = store.save(
        _payload(
            store,
            {
                "model_00__occ::policy": {"success": False},
                "model_00__occ::policy_early": {"success": True},
            },
        )
    )

    assert saved["updated_at"] == "2026-08-07T12:00:00+00:00"
    assert saved["corrections"] == {
        "model_00__occ::policy": {
            "success": False,
            "updated_at": "2026-08-07T12:00:00+00:00",
        },
        "model_00__occ::policy_early": {
            "success": True,
            "updated_at": "2026-08-07T12:00:00+00:00",
        },
    }
    assert json.loads(store.corrections_path.read_text()) == saved

    # Saving the model's original state removes that override.
    assert store.save(
        _payload(store, {"model_00__occ::policy": {"success": True}})
    )["corrections"] == {}

    with pytest.raises(ValueError, match="non-reviewable"):
        store.save(_payload(store, {"model_00__occ::gt": {"success": False}}))
    with pytest.raises(ValueError, match="different evaluation"):
        store.save({**_payload(store, {}), "report_id": "wrong"})


def test_refresh_html_rebuilds_review_enabled_report(tmp_path: Path) -> None:
    output, _ = _report(tmp_path)

    html_path = MODULE.refresh_html(output)
    html = html_path.read_text(encoding="utf-8")

    assert "Stage-1 multi-policy skill evaluation" in html
    assert "connectReviewServer();" in html
    assert "Server autosave connected" in html


def test_review_http_server_serves_report_and_persists_post(tmp_path: Path) -> None:
    output, _ = _report(tmp_path)
    try:
        server = MODULE.make_server(output, port=0)
    except PermissionError:
        pytest.skip("This sandbox does not permit localhost socket creation.")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(f"{base}/", timeout=3) as response:
            assert b"report" in response.read()
        with urlopen(f"{base}/api/corrections", timeout=3) as response:
            initial = json.loads(response.read())
        assert initial["corrections"] == {}

        body = json.dumps(
            _payload(
                server.store,
                {"model_00__occ::policy_early": {"success": True}},
            )
        ).encode()
        request = Request(
            f"{base}/api/corrections",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            saved = json.loads(response.read())
        assert saved["corrections"]["model_00__occ::policy_early"]["success"] is True
        assert server.store.load() == saved
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
