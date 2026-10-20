"""Teacher-forced Predictor diagnosis on recorded skill starts; never runs a simulator."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import html
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from ft_pred_eval_config import DEFAULT_CONFIG_PATH, balanced_chunks, build_settings, load_config
from relabel_skillvla import (  # build_data/src is installed by the config module
    _prepare_relabel_preprocessor,
    _preprocessor_overrides,
    discover_segments,
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


@contextmanager
def report_lock(path: Path):
    """Serialize report writers across processes/nodes sharing this output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_demo_map(path: Path) -> dict[int, tuple[str, str]]:
    """Episode numbers are local to each dataset; the original demo identity is not."""
    with np.load(path, allow_pickle=True) as data:
        result = {}
        for episode, scene, demo in zip(data["episode_index"], data["scene_file"], data["demo"], strict=True):
            episode = int(episode)
            identity = (_text(scene), _text(demo))
            if episode in result or not all(identity):
                raise ValueError(f"Invalid/duplicate exact demo mapping in {path}: {episode}.")
            result[episode] = identity
    return result


def select_segments(
    segments,
    demo_map: dict,
    exclude_maps: list[dict],
    *,
    task_ids: list[int],
    episodes_per_task: int,
    seed: int,
):
    excluded = {identity for mapping in exclude_maps for identity in mapping.values()}
    by_task = {}
    eligible = []
    requested = set(task_ids)
    for segment in segments:
        if requested and segment.task_index not in requested:
            continue
        if segment.episode_index not in demo_map:
            raise ValueError(f"Episode {segment.episode_index} lacks an exact demo identity.")
        if demo_map[segment.episode_index] in excluded:
            continue
        by_task.setdefault(segment.task_index, set()).add(segment.episode_index)
        eligible.append(segment)
    missing = requested - set(by_task)
    if missing:
        raise ValueError(f"No held-out episodes remain for dataset tasks {sorted(missing)}.")
    rng = np.random.default_rng(seed)
    selected = set()
    for task, ids in sorted(by_task.items()):
        ids = sorted(ids)
        if episodes_per_task and len(ids) < episodes_per_task:
            raise ValueError(f"Task {task} has {len(ids)} held-out demos, requested {episodes_per_task}.")
        chosen = ids if not episodes_per_task else rng.choice(ids, episodes_per_task, replace=False)
        selected.update(int(x) for x in chosen)
    result = [s for s in eligible if s.episode_index in selected]
    if not result:
        raise ValueError("No offline evaluation samples remain after training-demo exclusion.")
    identities = {demo_map[s.episode_index] for s in result}
    if identities & excluded:
        raise AssertionError("Training demo leaked into offline evaluation.")
    return result


def load_samples(settings: dict) -> tuple[list[dict], dict]:
    """Use the existing canonical-boundary reader and exact endpoint convention."""
    import pyarrow.parquet as pq

    run = Path(settings["run_dir"])
    info = json.loads((run / "skillvla/meta/info.json").read_text())
    levels = info["skill_fsq_levels"]
    segments = discover_segments(run / "skillvla", num_embeddings=int(np.prod(levels)))
    demo_map = load_demo_map(Path(settings["init_map"]))
    exclusions = [load_demo_map(Path(p)) for p in settings["exclude_maps"]]
    training_maps = [load_demo_map(Path(p)) for p in settings.get("training_maps", [])]
    training_pairs = {identity for mapping in training_maps for identity in mapping.values()}
    selected = select_segments(
        segments,
        demo_map,
        exclusions,
        task_ids=settings["task_ids"],
        episodes_per_task=settings["episodes_per_task"],
        seed=settings["seed"],
    )
    wanted = {s.episode_index for s in selected}
    frames = {}
    for path in sorted((run / "skillvla/data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"])
        values = table.to_pydict()
        for ep, frame, state in zip(
            *(values[k] for k in ("episode_index", "frame_index", "observation.state")), strict=True
        ):
            ep, frame = int(ep), int(frame)
            if ep in wanted:
                if frame in frames.setdefault(ep, {}):
                    raise ValueError(f"Duplicate episode/frame {ep}/{frame} in {path}.")
                frames[ep][frame] = np.asarray(state, dtype=np.float32)
    with np.load(run / "skill_latents.npz", allow_pickle=False) as archive:
        boundaries = {}
        for ep, skill, start, end, token in zip(
            *(archive[key] for key in ("episode_id", "skill_index", "frame_start", "frame_end", "tokens")),
            strict=True,
        ):
            key = (int(ep), int(skill))
            if key in boundaries:
                raise ValueError(f"Duplicate latent occurrence {key}.")
            boundaries[key] = (int(start), int(end), int(token))
    all_codes = {(s.episode_index, s.skill_index): s.original_code for s in segments}
    samples = []
    for segment in selected:
        ep, skill = segment.episode_index, segment.skill_index
        start, end, token = boundaries[(ep, skill)]
        if start != segment.frame_start or token != segment.original_code:
            raise ValueError(f"Parquet/skill_latents disagree at episode {ep}, skill {skill}.")
        episode = frames[ep]
        if set(episode) != set(range(len(episode))) or not 0 <= start < end <= len(episode):
            raise ValueError(f"Invalid contiguous episode or skill interval for episode {ep}.")
        endpoint = min(end, len(episode) - 1)  # same as build_skill_focus_uv / closed-loop GT
        xyz = episode[endpoint][:3]
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"Invalid endpoint XYZ for episode {ep}, skill {skill}.")
        scene, demo = demo_map[ep]
        samples.append(
            {
                "uid": f"task{segment.task_index:02d}_ep{ep:05d}_skill{skill:03d}",
                "dataset_index": segment.dataset_index,
                "episode_index": ep,
                "task_id": segment.task_index,
                "skill_index": skill,
                "frame_start": start,
                "frame_end": end,
                "target_frame": endpoint,
                "gt_skill": token,
                "previous_gt_skill": -1 if skill == 0 else all_codes[(ep, skill - 1)],
                "gt_xyz": xyz.astype(float).tolist(),
                "scene_file": scene,
                "demo": demo,
                "is_training_demo": (scene, demo) in training_pairs if training_maps else None,
            }
        )
    # Different local episode numbers cannot accidentally count the same original demo twice.
    episode_pairs = [(ep, demo_map[ep]) for ep in sorted(wanted)]
    if len({identity for _, identity in episode_pairs}) != len(episode_pairs):
        raise ValueError("Multiple evaluation episode IDs refer to the same original demo.")
    return samples, info


def fsq_coordinates(code: int, levels: list[int]) -> list[int]:
    result, stride = [], 1
    for level in levels:
        result.append((int(code) // stride) % int(level))
        stride *= int(level)
    return result


def infer_batch(predictor, processed: dict, gt_codes):
    """Share one VLM pass; diagnose XYZ both with GT skill and the actual prediction."""
    import torch

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    images = [processed[k] for k in ("observation.images.image", "observation.images.wrist_image")]
    with torch.inference_mode():
        memory, ignore = predictor.reader_memory(
            images, processed[OBS_LANGUAGE_TOKENS], processed[OBS_LANGUAGE_ATTENTION_MASK]
        )
        start_state = processed[OBS_STATE] if predictor.config.skill_predictor_start_proprio else None
        previous = processed["skill_previous_code"]
        hidden = predictor.skill_reader_hidden(memory, ignore, start_state, previous)
        codes = predictor.head.decode(hidden).reshape(-1).long()
        gt_codes = torch.as_tensor(gt_codes, device=codes.device, dtype=torch.long)
        xyz_gt = predictor.predict_end_state_from_memory(memory, ignore, gt_codes, start_state)
        xyz_pred = predictor.predict_end_state_from_memory(memory, ignore, codes, start_state)
    return (codes.cpu().numpy(), xyz_gt.float().cpu().numpy()[:, :3], xyz_pred.float().cpu().numpy()[:, :3])


def load_predictor_config(checkpoint: Path, device: str, tokenizer: str):
    """Override device BEFORE decoding: saved CUDA configs otherwise reject CPU hosts."""
    import draccus

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
    from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig

    values = json.loads((checkpoint / "config.json").read_text())
    classes = {"skill_aux": SkillAuxConfig, "skill_expert": SkillExpertConfig}
    cls = classes[values["type"]]
    values = cls._migrate_legacy_config(values)
    values.update(device=device, tokenizer_path=tokenizer)
    if device == "cpu":
        values.update(dtype="float32", use_amp=False)
    return draccus.decode(PreTrainedConfig, values)


def load_predictor(settings: dict, panel: dict):
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.skill_expert.modeling_skill_expert import _load_complete_predictor_parameters
    from lerobot.policies.skill_expert.modeling_skill_predictor import FrozenVLMSkillPredictor

    device = settings["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Submit on a GPU or select device: cpu.")
    torch.set_num_threads(settings["cpu_threads"])
    config = load_predictor_config(Path(panel["path"]), device, settings["tokenizer_path"])
    # CPU kernels do not universally implement the saved BF16 policy's operators.
    dtype = (
        torch.float32
        if device == "cpu"
        else (torch.bfloat16 if config.dtype == "bfloat16" else torch.float32)
    )
    predictor = FrozenVLMSkillPredictor(config).to(dtype=dtype)
    _load_complete_predictor_parameters(predictor, panel["path"])
    predictor.to(device=device).requires_grad_(False).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=panel["path"],
        preprocessor_overrides=_preprocessor_overrides(
            Path(panel["path"]), Path(settings["tokenizer_path"]), device
        ),
    )
    preprocessor = _prepare_relabel_preprocessor(preprocessor)
    # Stored SkillVLA states are already episode-grounded. Do not reset the
    # origin to each boundary (or subtract an origin a second time).
    from lerobot.policies.skill_expert.processor_skill_expert import EpisodeStartXYZGroundingProcessorStep

    preprocessor.steps = [
        s for s in preprocessor.steps if not isinstance(s, EpisodeStartXYZGroundingProcessorStep)
    ]
    info = json.loads((Path(settings["run_dir"]) / "skillvla/meta/info.json").read_text())
    dataset = LeRobotDataset(
        repo_id=str(info["repo_id"]), root=Path(settings["run_dir"]) / "skillvla", video_backend="pyav"
    )
    return predictor, preprocessor, dataset


class CameraProjector:
    """Use saved calibration, or XML + the builder's action-aligned original EEF origin."""

    def __init__(self, settings: dict, info: dict):
        self.run = Path(settings["run_dir"])
        self.original = Path(settings["original_dataset_dir"])
        self.grounding = info.get("proprio_grounding", "none")
        self.matrices = {}
        self.filtered_actions = None
        focus = self.run / "skill_focus_uv.npz"
        self.size = None
        if focus.is_file():
            with np.load(focus, allow_pickle=False) as data:
                if "camera_grounded_transform" in data.files:
                    if _text(data["camera"].item()) != "agentview":
                        raise ValueError("Offline image overlay requires agentview camera calibration.")
                    self.size = (int(data["image_height"]), int(data["image_width"]))
                    self.matrices = {
                        int(ep): np.asarray(mat)
                        for ep, mat in zip(
                            data["camera_episode_id"], data["camera_grounded_transform"], strict=True
                        )
                    }

    def matrix(self, sample: dict, height: int, width: int) -> np.ndarray:
        ep = sample["episode_index"]
        if ep in self.matrices:
            if self.size is not None and self.size != (height, width):
                raise ValueError("Recorded camera resolution differs from the decoded top image.")
            return self.matrices[ep]
        import h5py
        import pyarrow.parquet as pq

        from lerobot.policies.skillVLA.focus_projection import camera_transform_from_recorded_xml

        with h5py.File(self.original / sample["scene_file"], "r") as handle:
            demo = handle[f"data/{sample['demo']}"]
            xml = demo.attrs["model_file"]
            matrix = camera_transform_from_recorded_xml(
                _text(xml), camera_name="agentview", height=height, width=width
            )
            if self.grounding == "episode_start_xyz":
                from skill_episode_data import SkillEvaluationDataset

                if self.filtered_actions is None:
                    self.filtered_actions = {}
                    for path in sorted((self.run / "skillvla/data").rglob("*.parquet")):
                        rows = pq.read_table(
                            path, columns=["episode_index", "frame_index", "action"]
                        ).to_pydict()
                        for episode, frame, action in zip(
                            *(rows[k] for k in ("episode_index", "frame_index", "action")), strict=True
                        ):
                            self.filtered_actions.setdefault(int(episode), []).append((int(frame), action))
                actions = np.asarray([a for _, a in sorted(self.filtered_actions[ep])], dtype=np.float32)
                indices, _, _ = SkillEvaluationDataset.align_actions(actions, np.asarray(demo["actions"]))
                origin = np.asarray(demo["obs/ee_states"][int(indices[0]), :3], dtype=np.float64)
                # The preconverted raw observation's frame zero may differ very
                # slightly from original ee_states. Use exactly the build-time
                # origin rather than introducing even a subpixel displacement.
                matrix[:, 3] += matrix[:, :3] @ origin
            elif self.grounding != "none":
                raise ValueError(f"Unsupported projection grounding: {self.grounding}")
        self.matrices[ep] = matrix
        return matrix


def project_xyz(xyz, matrix, height: int, width: int) -> dict:
    from lerobot.policies.skillVLA.focus_projection import project_eef

    try:
        projection = project_eef(np.asarray(xyz), matrix, height=height, width=width)
        return {
            "raw_xy": list(projection.raw_xy),
            "pixel_xy": list(projection.pixel_xy),
            "visible": bool(projection.valid),
            "error": None,
        }
    except ValueError as error:
        # A prediction behind the camera must remain an explicit invalid result,
        # not disappear from XYZ metrics or look like a correct clipped point.
        return {"raw_xy": None, "pixel_xy": None, "visible": False, "error": str(error)}


def _rgb(image):
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected one RGB image, got {array.shape}.")
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255, 0, 255).round().astype(np.uint8)
    return array


def draw_overlay(path: Path, top, wrist, record: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont

    top, wrist = Image.fromarray(_rgb(top)), Image.fromarray(_rgb(wrist))
    height, width = top.height, top.width
    canvas = Image.new("RGB", (width * 2, height + 94), "white")
    canvas.paste(top, (0, 0))
    canvas.paste(wrist.resize((width, height)), (width, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    colors = {"gt": "#00dd55", "pred": "#ff3030", "gt_skill": "#ffb000"}
    for name, point in record["projection"].items():
        pixel = point["pixel_xy"]
        if pixel is None:
            continue
        x, y = pixel
        color = colors[name]
        # Correct skills (and all mode2 cases) make the two XYZ predictions
        # coincide. Distinct marker shapes keep all three signals visible.
        if name == "gt":
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
        elif name == "pred":
            draw.line((x - 5, y - 5, x + 5, y + 5), fill=color, width=2)
            draw.line((x - 5, y + 5, x + 5, y - 5), fill=color, width=2)
        else:
            draw.rectangle((x - 10, y - 10, x + 10, y + 10), outline=color, width=2)
        text_y = y + {"gt": -16, "pred": 0, "gt_skill": 16}[name]
        draw.text(
            (max(0, min(x + 12, width - 90)), max(0, min(text_y, height - 14))),
            name + (" (offscreen)" if not point["visible"] else ""),
            fill=color,
            font=font,
        )
    draw.text(
        (5, height + 3),
        f"GT skill {record['gt_skill']} {record['gt_skill_coordinates']} | "
        f"Pred {record['pred_skill']} {record['pred_skill_coordinates']} | "
        f"{'CORRECT' if record['skill_correct'] else 'WRONG'}",
        fill="black",
        font=font,
    )
    draw.text(
        (5, height + 24), "green: GT endpoint | red: XYZ given predicted skill", fill="black", font=font
    )
    draw.text((5, height + 44), "orange: XYZ given GT skill | right: wrist input", fill="black", font=font)
    draw.text(
        (5, height + 65),
        f"XYZ L2: pred-skill {record['xyz_error_m'] * 100:.2f} cm | "
        f"GT-skill {record['xyz_gt_skill_error_m'] * 100:.2f} cm",
        fill="black",
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_record(sample: dict, code: int, xyz_gt, xyz_pred, levels: list[int], language: str) -> dict:
    target = np.asarray(sample["gt_xyz"], dtype=np.float64)
    xyz_gt, xyz_pred = np.asarray(xyz_gt, dtype=np.float64), np.asarray(xyz_pred, dtype=np.float64)
    if not 0 <= int(code) < int(np.prod(levels)) or any(
        value.shape != (3,) or not np.isfinite(value).all() for value in (xyz_gt, xyz_pred)
    ):
        raise ValueError("Predictor emitted an invalid skill code or non-finite XYZ.")
    return {
        **sample,
        "language": language,
        "pred_skill": int(code),
        "gt_skill_coordinates": fsq_coordinates(sample["gt_skill"], levels),
        "pred_skill_coordinates": fsq_coordinates(code, levels),
        "skill_correct": int(code) == sample["gt_skill"],
        "pred_xyz": xyz_pred.tolist(),
        "xyz_given_gt_skill": xyz_gt.tolist(),
        "xyz_error_m": float(np.linalg.norm(xyz_pred - target)),
        "xyz_gt_skill_error_m": float(np.linalg.norm(xyz_gt - target)),
        "xyz_abs_error_m": np.abs(xyz_pred - target).tolist(),
        "image": None,
    }


def summarize(records: list[dict]) -> dict:
    def metrics(rows):
        errors = np.asarray([r["xyz_error_m"] for r in rows])
        return {
            "samples": len(rows),
            "episodes": len({r["episode_index"] for r in rows}),
            "skill_accuracy_pct": 100 * sum(r["skill_correct"] for r in rows) / len(rows),
            "xyz_l2_mean_m": float(errors.mean()),
            "xyz_l2_median_m": float(np.median(errors)),
            "xyz_l2_p95_m": float(np.quantile(errors, 0.95)),
            "xyz_given_gt_skill_l2_mean_m": float(np.mean([r["xyz_gt_skill_error_m"] for r in rows])),
            "xyz_mae_per_axis_m": np.mean([r["xyz_abs_error_m"] for r in rows], axis=0).tolist(),
        }

    if not records:
        raise ValueError("Cannot summarize an empty offline evaluation.")
    by_task = {
        str(task): metrics([r for r in records if r["task_id"] == task])
        for task in sorted({r["task_id"] for r in records})
    }
    confusion = Counter((r["gt_skill"], r["pred_skill"]) for r in records)
    splits = {}
    for flag, name in ((True, "training_demo"), (False, "held_out_demo"), (None, "unknown")):
        rows = [r for r in records if r.get("is_training_demo") is flag]
        if rows:
            splits[name] = metrics(rows)
    return {
        "overall": metrics(records),
        "per_task": by_task,
        "by_demo_split": splits,
        "confusion": [
            {"gt_skill": a, "pred_skill": b, "count": n} for (a, b), n in sorted(confusion.items())
        ],
        "interpretation": "Offline skill-start accuracy / XYZ error, NOT simulator success rate. "
        "GT previous skill is used by preskill models; XYZ is in the dataset's grounded frame.",
    }


def write_report(folder: Path, panel: dict, records: list[dict], metrics: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "metrics.json", metrics)
    (folder / "predictions.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    fields = [k for k in records[0] if k != "projection"]
    with (folder / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    k: json.dumps(record[k]) if isinstance(record[k], (dict, list)) else record[k]
                    for k in fields
                }
            )
    overall = metrics["overall"]
    cards = []
    for r in records:
        badge = "CORRECT" if r["skill_correct"] else "WRONG"
        split = (
            "unknown"
            if r.get("is_training_demo") is None
            else "training"
            if r["is_training_demo"]
            else "held_out"
        )
        split_text = {
            "training": "Training demo",
            "held_out": "Held-out demo",
            "unknown": "Training membership unknown",
        }[split]
        image = (
            f'<img loading="lazy" src="{html.escape(r["image"], quote=True)}">'
            if r["image"]
            else "<p>Image not requested.</p>"
        )
        cards.append(f'''<article data-task="{r["task_id"]}" data-correct="{str(r["skill_correct"]).lower()}" data-split="{split}">
<h3>Task {r["task_id"]} · episode {r["episode_index"]} · skill {r["skill_index"]} <span class="{badge}">{badge}</span></h3>
<p>{html.escape(r["language"])}</p><p>{split_text} · {html.escape(r["scene_file"])} / {html.escape(r["demo"])} · frame {r["frame_start"]}</p>
{image}<p><b>GT skill: {r["gt_skill"]} {r["gt_skill_coordinates"]}</b> → predicted: {r["pred_skill"]} {r["pred_skill_coordinates"]}</p>
<pre>GT XYZ:               {r["gt_xyz"]}
XYZ given pred skill: {r["pred_xyz"]}
XYZ given GT skill:   {r["xyz_given_gt_skill"]}</pre>
<p>XYZ error: {r["xyz_error_m"] * 100:.2f} cm (pred skill), {r["xyz_gt_skill_error_m"] * 100:.2f} cm (GT skill)</p>
</article>''')
    task_rows = "".join(
        f"<tr><td>Task {task}</td><td>{v['samples']}</td><td>{v['skill_accuracy_pct']:.1f}%</td>"
        f"<td>{v['xyz_l2_mean_m'] * 100:.2f}</td><td>{v['xyz_given_gt_skill_l2_mean_m'] * 100:.2f}</td></tr>"
        for task, v in metrics["per_task"].items()
    )
    options = "".join(f'<option value="{t}">Task {t}</option>' for t in metrics["per_task"])
    split_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{v['episodes']}</td><td>{v['samples']}</td>"
        f"<td>{v['skill_accuracy_pct']:.1f}%</td><td>{v['xyz_l2_mean_m'] * 100:.2f}</td>"
        f"<td>{v['xyz_given_gt_skill_l2_mean_m'] * 100:.2f}</td></tr>"
        for name, v in metrics["by_demo_split"].items()
    )
    title = html.escape(panel["label"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;background:#f4f6f7;color:#182329;max-width:1100px;margin:25px auto;padding:16px}}
article,header{{background:white;border:1px solid #ddd;border-radius:12px;margin:18px 0;padding:20px}}
img{{max-width:100%;width:800px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}td,th{{padding:8px;text-align:left}}
.CORRECT{{color:#158344}}.WRONG{{color:#cf3030}}select,label{{margin:10px}}article[hidden]{{display:none}}</style>
<header><a href="../index.html">All checkpoints</a><h1>{title}</h1>
<p>Recorded skill-start observations only; no simulator. These are NOT rollout success rates.</p>
<p>Weights and preprocessing source: <code>{html.escape(panel["path"])}</code></p>
<p>{overall["samples"]} skill starts / {overall["episodes"]} demos · skill accuracy <b>{overall["skill_accuracy_pct"]:.1f}%</b></p>
<p>Mean XYZ L2: <b>{overall["xyz_l2_mean_m"] * 100:.2f} cm</b> given predicted skill;
<b>{overall["xyz_given_gt_skill_l2_mean_m"] * 100:.2f} cm</b> given GT skill.</p>
<p>For preskill models, previous skill is GT (BOS at episode start). In mode2 the XYZ head ignores the skill, so both XYZ predictions agree.</p>
<p>Green circle: GT endpoint. Red cross: XYZ conditioned on predicted skill. Orange square: XYZ conditioned on GT skill.
The overlay is on the skill-start agentview image; wrist is the other input. Offscreen/behind-camera predictions are not hidden from XYZ metrics.</p>
<a href="predictions.csv">CSV</a> · <a href="predictions.jsonl">JSONL</a> · <a href="metrics.json">Metrics</a>
<p>Training/held-out membership refers to FT-reference demo occurrences, not PT training membership or whether an FSQ code was seen.</p>
<table><tr><th>Demo split</th><th>Demos</th><th>Samples</th><th>Skill accuracy</th><th>XYZ L2 (cm)</th><th>XYZ with GT skill (cm)</th></tr>{split_rows}</table>
<table><tr><th>Task</th><th>Samples</th><th>Skill accuracy</th><th>XYZ L2 (cm)</th><th>XYZ with GT skill (cm)</th></tr>{task_rows}</table>
<select id="task"><option value="all">All tasks</option>{options}</select>
<select id="split"><option value="all">All demos</option><option value="training">Training demos</option><option value="held_out">Held-out demos</option><option value="unknown">Unknown</option></select>
<label><input id="wrong" type="checkbox">Wrong skill only</label></header>
{"".join(cards)}<script>function filter(){{let t=document.getElementById('task').value,w=document.getElementById('wrong').checked;
let s=document.getElementById('split').value;
document.querySelectorAll('article').forEach(a=>a.hidden=(t!=='all'&&a.dataset.task!==t)||(w&&a.dataset.correct==='true')||(s!=='all'&&a.dataset.split!==s));}}
document.getElementById('task').onchange=filter;document.getElementById('split').onchange=filter;document.getElementById('wrong').onchange=filter;</script></html>"""
    (folder / "index.html").write_text(document)


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def panel_signature(settings: dict, panel: dict, samples: list[dict]) -> str:
    run = Path(settings["run_dir"])
    files = [
        Path(panel["path"]) / n for n in ("model.safetensors", "config.json", "policy_preprocessor.json")
    ]
    files.extend((Path(panel["path"])).glob("*preprocessor*.safetensors"))
    files += [run / "skill_latents.npz", run / "skillvla/meta/info.json", Path(settings["init_map"])]
    files.extend(Path(p) for p in settings["exclude_maps"])
    files.extend(Path(p) for p in settings.get("training_maps", []))
    files.extend(sorted((run / "skillvla/data").rglob("*.parquet")))
    if (run / "skill_focus_uv.npz").is_file():
        files.append(run / "skill_focus_uv.npz")
    payload = {
        "version": 1,
        "files": [_fingerprint(p) for p in files],
        "samples": samples,
        "device": settings["device"],
        "visualization": settings["visualization"],
        "previous_skill": "gt",
        "tokenizer": settings["tokenizer_path"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_panel(settings: dict, panel: dict, samples: list[dict], info: dict) -> dict:
    # A resubmission with a different GPU count can overlap an older attempt.
    # Only one process may evaluate/write any particular checkpoint report.
    lock = Path(settings["output_dir"]) / ".locks" / f"{panel['folder']}.lock"
    with report_lock(lock):
        return _run_panel(settings, panel, samples, info)


def _run_panel(settings: dict, panel: dict, samples: list[dict], info: dict) -> dict:
    import torch
    from torch.utils.data._utils.collate import default_collate

    folder = Path(settings["output_dir"]) / panel["folder"]
    signature = panel_signature(settings, panel, samples)
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing["signature"] != signature:
            raise ValueError(f"Refusing to mix different evaluations at {folder}; use another output_name.")
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output exists without a matching manifest: {folder}")
    atomic_json(manifest_path, {"signature": signature, "panel": panel, "settings": settings})
    visual = settings["visualization"]
    max_images = visual["max_samples"] or len(samples)
    records, pending = {}, []
    for index, sample in enumerate(samples):
        path = folder / "records" / f"{sample['uid']}.json"
        old = json.loads(path.read_text()) if path.is_file() and settings["resume"] else None
        wants_image = visual["enabled"] and index < max_images
        if old is not None and (not wants_image or (old.get("image") and (folder / old["image"]).is_file())):
            records[sample["uid"]] = old
        else:
            pending.append((sample, wants_image))
    if pending:
        predictor, preprocessor, dataset = load_predictor(settings, panel)
        projector = CameraProjector(settings, info) if visual["enabled"] else None
        keys = ("observation.images.image", "observation.images.wrist_image", "observation.state", "task")
        for offset in range(0, len(pending), settings["batch_size"]):
            chunk = pending[offset : offset + settings["batch_size"]]
            raw = []
            for sample, _ in chunk:
                value = dataset[sample["dataset_index"]]
                if (
                    int(value["episode_index"]) != sample["episode_index"]
                    or int(value["frame_index"]) != sample["frame_start"]
                ):
                    raise ValueError("LeRobot indexing does not point at the canonical skill-start frame.")
                raw.append({key: value[key] for key in keys})
            processed = preprocessor(default_collate(raw))
            processed["skill_previous_code"] = torch.tensor(
                [s["previous_gt_skill"] for s, _ in chunk], dtype=torch.long, device=settings["device"]
            )
            codes, gt_xyz, pred_xyz = infer_batch(predictor, processed, [s["gt_skill"] for s, _ in chunk])
            for (sample, wants_image), value, code, xyz_gt, xyz_pred in zip(
                chunk, raw, codes, gt_xyz, pred_xyz, strict=True
            ):
                record = make_record(
                    sample, int(code), xyz_gt, xyz_pred, info["skill_fsq_levels"], value["task"]
                )
                if wants_image:
                    top, wrist = value[keys[0]], value[keys[1]]
                    h, w = _rgb(top).shape[:2]
                    matrix = projector.matrix(sample, h, w)
                    record["projection"] = {
                        name: project_xyz(xyz, matrix, h, w)
                        for name, xyz in (
                            ("gt", record["gt_xyz"]),
                            ("pred", record["pred_xyz"]),
                            ("gt_skill", record["xyz_given_gt_skill"]),
                        )
                    }
                    record["image"] = f"images/{sample['uid']}.png"
                    draw_overlay(folder / record["image"], top, wrist, record)
                atomic_json(folder / "records" / f"{sample['uid']}.json", record)
                records[sample["uid"]] = record
            print(f"[{panel['label']}] {len(records)}/{len(samples)} skill starts", flush=True)
        del predictor, preprocessor, dataset
        if settings["device"] == "cuda":
            torch.cuda.empty_cache()
    ordered = [records[s["uid"]] for s in samples]
    metrics = summarize(ordered)
    write_report(folder, panel, ordered, metrics)
    # Publish completion only after CSV, JSONL, images and HTML are available.
    atomic_json(folder / "completion.json", {"signature": signature})
    return metrics


def update_overview(settings: dict, samples: list[dict]) -> None:
    """Rebuild from every completed panel, not just this worker's local list."""
    root = Path(settings["output_dir"])
    with report_lock(root / ".locks/overview.lock"):
        selected_path = root / "selected_samples.json"
        if selected_path.is_file() and json.loads(selected_path.read_text()) != samples:
            raise ValueError(
                f"Refusing to mix different evaluation samples at {root}; use another output_name."
            )
        atomic_json(selected_path, samples)
        atomic_json(root / "resolved_config.json", settings)
        links, completed = [], 0
        for panel in settings["models"]:
            folder = root / panel["folder"]
            completion = folder / "completion.json"
            ready = (
                completion.is_file()
                and all(
                    (folder / name).is_file()
                    for name in ("metrics.json", "index.html", "predictions.csv", "predictions.jsonl")
                )
                and json.loads(completion.read_text()).get("signature")
                == panel_signature(settings, panel, samples)
            )
            label = html.escape(panel["label"])
            if not ready:
                links.append(f"<li>{label}: pending / running</li>")
                continue
            overall = json.loads((folder / "metrics.json").read_text())["overall"]
            completed += 1
            links.append(
                f'<li><a href="{html.escape(panel["folder"], quote=True)}/index.html">{label}</a>: '
                f"skill accuracy {overall['skill_accuracy_pct']:.1f}%; XYZ L2 {overall['xyz_l2_mean_m'] * 100:.2f} cm</li>"
            )
        atomic_text(
            root / "index.html",
            '<!doctype html><meta charset="utf-8"><title>Offline Predictor evaluation</title>'
            "<h1>Offline Predictor evaluation</h1><p>Recorded demonstrations; no simulator success rate.</p>"
            f"<p>Completed checkpoints: {completed}/{len(settings['models'])}</p><ul>"
            + "".join(links)
            + "</ul>",
        )


def run(settings: dict, *, worker_index: int = 0, worker_count: int = 1) -> None:
    if not 1 <= worker_count <= len(settings["models"]) or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid Predictor worker index/count for the number of checkpoint panels.")
    indices = balanced_chunks(list(range(len(settings["models"]))), worker_count)[worker_index]
    samples, info = load_samples(settings)
    update_overview(settings, samples)
    print(
        f"Offline evaluation worker {worker_index + 1}/{worker_count}: "
        f"panels {indices}; {len(samples)} skill starts in {len({s['episode_index'] for s in samples})} demos.",
        flush=True,
    )
    for index in indices:
        run_panel(settings, settings["models"][index], samples, info)
        update_overview(settings, samples)
    print(f"HTML report: {Path(settings['output_dir']) / 'index.html'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    sys.path.insert(0, str(Path(settings["project_root"]) / "lerobot/src"))
    run(settings, worker_index=args.worker_index, worker_count=args.worker_count)


if __name__ == "__main__":
    main()
