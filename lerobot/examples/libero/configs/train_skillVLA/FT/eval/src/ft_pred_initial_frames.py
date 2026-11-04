"""Prediction-only initial scene reports. No simulator or GT labels are loaded."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np

CONTRACT = "libero_processed_rgb_state_v1"


def capture_settings(root, block):
    """Resolve a complete input capture once; snapshots pin the concrete directory."""
    value = str(block.get("capture_dir", "latest"))
    task_ids = block.get("task_ids", list(range(10)))
    episodes = int(block.get("episodes_per_task", 10))
    scale = float(block.get("random_range", 2.0))
    if not task_ids or any(type(t) is not int or t < 0 for t in task_ids) or episodes < 1:
        raise ValueError("initial_frames needs nonnegative task_ids and positive episodes_per_task")
    base = Path(root) / "lerobot/examples/libero/configs/train_pi05/FT/eval/initial_frames"
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = Path(root) / directory
    candidates = sorted(base.glob("*/manifest.json"), reverse=True) if value == "latest" else [directory / "manifest.json"]
    for path in candidates:
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        settings = manifest.get("settings", {})
        if not manifest.get("complete") or settings.get("model_input_contract") != CONTRACT:
            continue
        if float(settings.get("random_range", 1.0)) != scale:
            continue
        counts = {task: len([r for r in manifest["frames"] if r["task_id"] == task]) for task in task_ids}
        if any(n < episodes for n in counts.values()):
            continue
        return {"capture_dir": str(path.parent.resolve()), "random_range": scale,
                "task_ids": task_ids, "episodes_per_task": episodes,
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    raise FileNotFoundError(
        "No complete matching initial-frame capture with wrist/proprio inputs. "
        "Re-run pi05 FT/eval/capture_initial_frames.sh --random-range 2.0 --label predictor_inputs "
        "(CPU capture only), or set initial_frames.capture_dir/task_ids/episodes_per_task."
    )


def load_initial_samples(settings):
    source = settings["initial_frames"]
    root = Path(source["capture_dir"])
    path = root / "manifest.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["manifest_sha256"]:
        raise ValueError("Initial-frame capture changed after configuration was resolved")
    manifest = json.loads(path.read_text())
    samples = []
    for task in source["task_ids"]:
        rows = sorted((r for r in manifest["frames"] if r["task_id"] == task), key=lambda r: r["episode"])
        selected = rows[:source["episodes_per_task"]]
        if len({row["episode"] for row in selected}) != len(selected):
            raise ValueError("Duplicate task/episode in initial-frame capture")
        for row in selected:
            payload = (root / row["model_input"]).resolve()
            if not payload.is_relative_to(root.resolve()) or not payload.is_file():
                raise ValueError("Missing/invalid model_input; never substitute zero wrist/proprio")
            if not row.get("task_description"):
                raise ValueError("Capture lacks the task language instruction")
            samples.append({"uid": f"task{task:02d}_ep{row['episode']:05d}_initial",
                            "task_id": task, "episode_index": row["episode"], "skill_index": 0,
                            "language": row["task_description"], "model_input": str(payload),
                            "model_input_sha256": hashlib.sha256(payload.read_bytes()).hexdigest()})
    return samples


def read_input(sample, grounding):
    import torch

    path = Path(sample["model_input"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != sample["model_input_sha256"]:
        raise ValueError("Captured model input changed during prediction")
    with np.load(path, allow_pickle=False) as data:
        if str(data["contract"].item()) != CONTRACT:
            raise ValueError("Unknown initial observation orientation/state contract")
        top, wrist, state = data["top"].copy(), data["wrist"].copy(), data["state_world"].astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 8D captured EEF axis-angle / gripper proprio")
    if grounding == "episode_start_xyz":
        state[:3] = 0.  # This is frame zero: its own EEF XYZ is the episode origin.
    elif grounding != "none":
        raise ValueError(f"Unsupported initial-frame grounding: {grounding}")
    def image_tensor(array):
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError("Expected captured uint8 HWC RGB image")
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float() / 255
    return {"observation.images.image": image_tensor(top), "observation.images.wrist_image": image_tensor(wrist),
            "observation.state": torch.from_numpy(state), "task": sample["language"]}


def signature(settings, panel, samples):
    files = list(Path(panel["path"]).glob("*.safetensors")) + [Path(panel["path"]) / name for name in ("config.json", "policy_preprocessor.json")]
    payload = {"version": 1, "input_mode": "initial_frames", "previous_skill": -1, "panel": panel,
               "samples": samples, "device": settings["device"], "seed": settings["seed"],
               "grounding": settings["input_grounding"], "visualization": settings["visualization"],
               "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def overlay(path, raw, record):
    from PIL import Image, ImageDraw
    from ft_pred_eval import _rgb, project_uv

    top = Image.fromarray(_rgb(raw["observation.images.image"]))
    wrist = Image.fromarray(_rgb(raw["observation.images.wrist_image"]))
    canvas = Image.new("RGB", (top.width * 2, top.height + 65), "white")
    canvas.paste(top, (0, 0)); canvas.paste(wrist.resize(top.size), (top.width, 0))
    draw = ImageDraw.Draw(canvas)
    projection = project_uv(record["pred_uv"], top.height, top.width)
    if projection["pixel_xy"] is not None:
        x, y = projection["pixel_xy"]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="red", width=3)
    draw.text((6, top.height + 4), f"Task {record['task_id']} / ep {record['episode_index']} | predicted skill {record['pred_skill']} | BOS=-1", fill="black")
    draw.text((6, top.height + 24), f"UV={record['pred_uv']} | GT unavailable (no accuracy/error scoring)", fill="black")
    wrist_label = "wrist input" if record["wrist_input_enabled"] else "wrist (NOT used)"
    draw.text((6, top.height + 44), f"Right: {wrist_label} | UV inside view: {projection['visible']}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return projection


def write_report(folder, panel, rows):
    from ft_pred_eval import atomic_json, atomic_text

    overall = {"samples": len(rows), "episodes": len(rows), "gt_available": False,
               "skill_accuracy_pct": None, "uv_l2_mean_px": None}
    atomic_json(folder / "metrics.json", {"mode": "prediction_only_initial_frames", "overall": overall})
    atomic_text(folder / "predictions.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    buffer = io.StringIO()
    columns = ("task_id", "episode_index", "language", "pred_skill", "pred_uv", "previous_skill_input", "gt_available")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    atomic_text(folder / "predictions.csv", buffer.getvalue())
    cards = []
    for row in rows:
        image = f'<img loading="lazy" src="{row["image"]}" style="max-width:100%">' if row.get("image") else ""
        cards.append(f'<article><h3>Task {row["task_id"]} / episode {row["episode_index"]}: skill {row["pred_skill"]}</h3>'
                     f'<p>{html.escape(row["language"])} · UV {row["pred_uv"]}</p>{image}</article>')
    atomic_text(folder / "index.html", '<!doctype html><meta charset="utf-8"><style>body{font-family:system-ui;max-width:1000px;margin:auto}article{border-top:1px solid #ccc}</style>'
                f'<h1>{html.escape(panel["label"])}</h1><p>Initial-frame prediction only. Every episode starts with BOS=-1. '
                'Red = predicted UV. No GT labels, accuracy, UV error or success rate.</p>' + "".join(cards))


def update_overview(settings, samples):
    from ft_pred_eval import atomic_json, atomic_text, report_lock

    root = Path(settings["output_dir"])
    with report_lock(root / ".locks/overview.lock"):
        sample_path = root / "selected_samples.json"
        if sample_path.exists() and json.loads(sample_path.read_text()) != samples:
            raise ValueError("Initial-frame samples changed; use another output_name")
        atomic_json(sample_path, samples); atomic_json(root / "resolved_config.json", settings)
        links = []
        for panel in settings["models"]:
            folder = root / panel["folder"]
            completion = folder / "completion.json"
            ready = completion.exists() and json.loads(completion.read_text()).get("signature") == signature(settings, panel, samples)
            links.append(f'<li><a href="{panel["folder"]}/index.html">{html.escape(panel["label"])}</a>: '
                         f'{"complete" if ready else "pending / running"}</li>')
        atomic_text(root / "index.html", '<!doctype html><meta charset="utf-8"><h1>Initial-frame Predictor preview</h1>'
                    '<p>Prediction only; no GT accuracy / UV error / task success rate.</p><ul>' + "".join(links) + '</ul>')


def run_initial_frames(settings, *, worker_index=0, worker_count=1):
    import torch
    from torch.utils.data._utils.collate import default_collate
    from ft_pred_eval import atomic_json, balanced_chunks, fsq_coordinates, infer_batch, load_predictor, report_lock

    if not 1 <= worker_count <= len(settings["models"]) or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid initial-frame worker index/count")
    samples = load_initial_samples(settings)
    update_overview(settings, samples)
    for index in balanced_chunks(list(range(len(settings["models"]))), worker_count)[worker_index]:
        panel = settings["models"][index]
        folder = Path(settings["output_dir"]) / panel["folder"]
        with report_lock(Path(settings["output_dir"]) / ".locks" / f"{panel['folder']}.lock"):
            current_signature = signature(settings, panel, samples)
            manifest = folder / "manifest.json"
            if manifest.exists() and json.loads(manifest.read_text())["signature"] != current_signature:
                raise ValueError("Initial-frame preview changed; use another output_name")
            if folder.exists() and any(folder.iterdir()) and not manifest.exists():
                raise ValueError("Existing preview folder has no matching manifest")
            atomic_json(manifest, {"signature": current_signature, "settings": settings, "panel": panel})
            predictor, preprocessor, rows = None, None, []
            try:
                torch.manual_seed(settings["seed"])
                for position, sample in enumerate(samples):
                    record_path = folder / "records" / f"{sample['uid']}.json"
                    wants_image = settings["visualization"]["enabled"] and (not settings["visualization"]["max_samples"] or position < settings["visualization"]["max_samples"])
                    if settings["resume"] and record_path.exists():
                        old = json.loads(record_path.read_text())
                        if not wants_image or (old.get("image") and (folder / old["image"]).exists()):
                            rows.append(old); continue
                    if predictor is None:
                        predictor, preprocessor, _ = load_predictor(settings, panel)
                    raw = read_input(sample, settings["input_grounding"])
                    preprocessor.reset()
                    processed = preprocessor(default_collate([raw]))
                    processed["skill_previous_code"] = torch.tensor([-1], device=settings["device"], dtype=torch.long)
                    codes, no_gt, uv = infer_batch(predictor, processed)
                    if no_gt is not None or not np.isfinite(uv[0]).all():
                        raise ValueError("Unexpected GT branch or nonfinite UV in prediction-only mode")
                    code = int(codes[0])
                    if not 0 <= code < int(np.prod(predictor.config.skill_fsq_levels)):
                        raise ValueError("Predicted skill is outside the FSQ vocabulary")
                    record = {**sample, "pred_skill": code, "pred_uv": uv[0].astype(float).tolist(),
                              "pred_skill_coordinates": fsq_coordinates(code, predictor.config.skill_fsq_levels),
                              "previous_skill_input": -1, "gt_available": False,
                              "wrist_input_enabled": getattr(predictor, "uses_wrist", True)}
                    if wants_image:
                        record["image"] = f"images/{sample['uid']}.png"
                        record["projection"] = overlay(folder / record["image"], raw, record)
                    atomic_json(record_path, record); rows.append(record)
                    print(f"[{panel['label']}] {position + 1}/{len(samples)} initial frames", flush=True)
                write_report(folder, panel, rows)
                atomic_json(folder / "completion.json", {"signature": current_signature})
            finally:
                del predictor, preprocessor
                if settings["device"] == "cuda":
                    torch.cuda.empty_cache()
        update_overview(settings, samples)
    print(f"HTML report: {Path(settings['output_dir']) / 'index.html'}", flush=True)
