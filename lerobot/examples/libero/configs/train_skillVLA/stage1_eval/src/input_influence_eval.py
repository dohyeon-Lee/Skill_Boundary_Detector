#!/usr/bin/env python3
"""Paired, offline input-influence evaluation for Stage-1 SkillVLA policies.

For every dataset batch, the baseline and all counterfactual variants use the
same flow-matching noise. Each variant swaps exactly one input with another
in-distribution sample, so output changes are attributable to that input rather
than to sampling noise or an obviously out-of-distribution zero tensor.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

from stage1_eval_config import build_settings, load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("input_influence_eval")

STATE_KEY = "observation.state"
TOP_IMAGE_KEY = "observation.images.image"
WRIST_IMAGE_KEY = "observation.images.wrist_image"
SUPPORTED_PERTURBATIONS = (
    "state_swap",
    "top_image_swap",
    "wrist_image_swap",
    "image_swap",
    "skill_swap",
    "all_inputs_swap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def _analysis_settings(config: dict) -> dict:
    analysis = config.get("analysis") or {}
    perturbations = analysis.get("perturbations", list(SUPPORTED_PERTURBATIONS))
    if not isinstance(perturbations, list) or not perturbations:
        raise ValueError("analysis.perturbations must be a non-empty YAML list.")
    unknown = sorted(set(perturbations) - set(SUPPORTED_PERTURBATIONS))
    if unknown:
        raise ValueError(
            f"Unknown analysis perturbations {unknown}; supported={SUPPORTED_PERTURBATIONS}."
        )
    if len(perturbations) != len(set(perturbations)):
        raise ValueError("analysis.perturbations must not contain duplicates.")
    settings = {
        "n_frames": int(analysis.get("n_frames", 512)),
        "batch_size": int(analysis.get("batch_size", 16)),
        "num_workers": int(analysis.get("num_workers", 4)),
        "seed": int(analysis.get("seed", 1000)),
        "num_inference_steps": analysis.get("num_inference_steps"),
        "save_per_sample": bool(analysis.get("save_per_sample", True)),
        "perturbations": perturbations,
    }
    if settings["n_frames"] < 2 or settings["batch_size"] < 2:
        raise ValueError("n_frames and batch_size must both be at least 2 for swapping.")
    if settings["num_workers"] < 0:
        raise ValueError("analysis.num_workers must be non-negative.")
    if settings["num_inference_steps"] is not None:
        settings["num_inference_steps"] = int(settings["num_inference_steps"])
        if settings["num_inference_steps"] <= 0:
            raise ValueError("analysis.num_inference_steps must be positive or null.")
    return settings


def _safe_dir_name(index: int, label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-_") or "model"
    return f"{index:02d}_{clean}"


def _masked_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return standard per-element MSE and MAE for each sample."""
    valid = (~padding).to(prediction.dtype).unsqueeze(-1)
    difference = (prediction - target) * valid
    denominator = (valid.sum(dim=(1, 2)) * target.shape[-1]).clamp_min(1)
    mse = difference.square().sum(dim=(1, 2)) / denominator
    mae = difference.abs().sum(dim=(1, 2)) / denominator
    return mse, mae


def _masked_step_l2(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    valid = (~padding).to(prediction.dtype)
    per_step = (prediction - reference).norm(dim=-1) * valid
    return per_step.sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _different_skill_codes(
    true_codes: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose another batch member's code, with code+1 only as a fallback."""
    batch_size = true_codes.numel()
    wrong = true_codes.clone()
    changed = torch.zeros(batch_size, dtype=torch.bool, device=true_codes.device)
    for shift in range(1, batch_size):
        candidate = true_codes.roll(shift, dims=0)
        use = (~changed) & (candidate != true_codes)
        wrong[use] = candidate[use]
        changed |= use
    fallback = ~changed
    if fallback.any():
        if vocab_size < 2:
            raise ValueError("skill_vocab_size must be at least 2 for skill_swap.")
        wrong[fallback] = (true_codes[fallback] + 1) % vocab_size
    return wrong, fallback


def _counterfactual_batch(
    batch: dict,
    condition: str,
    true_codes: torch.Tensor,
    vocab_size: int,
) -> tuple[dict, dict[str, float]]:
    """Build a shallow-copy batch with only the requested semantic input changed."""
    batch_size = true_codes.numel()
    if batch_size < 2:
        raise ValueError("Counterfactual swapping requires a batch of at least two.")
    permutation = torch.arange(batch_size, device=true_codes.device).roll(1)
    variant = dict(batch)
    metadata: dict[str, float] = {}

    if condition in {"state_swap", "all_inputs_swap"}:
        variant[STATE_KEY] = batch[STATE_KEY].index_select(0, permutation)
    if condition in {"top_image_swap", "image_swap", "all_inputs_swap"}:
        variant[TOP_IMAGE_KEY] = batch[TOP_IMAGE_KEY].index_select(0, permutation)
    if condition in {"wrist_image_swap", "image_swap", "all_inputs_swap"}:
        variant[WRIST_IMAGE_KEY] = batch[WRIST_IMAGE_KEY].index_select(0, permutation)
    if condition == "skill_swap":
        wrong_codes, fallback = _different_skill_codes(true_codes, vocab_size)
        metadata["skill_fallback_fraction"] = float(fallback.float().mean().item())
        variant["skill_code"] = wrong_codes
        variant["skill_sequence"] = wrong_codes[:, None]
        variant["skill_index"] = torch.zeros_like(wrong_codes)
    elif condition == "all_inputs_swap":
        wrong_codes = true_codes.index_select(0, permutation)
        same = wrong_codes == true_codes
        if same.any():
            wrong_codes = wrong_codes.clone()
            wrong_codes[same] = (true_codes[same] + 1) % vocab_size
        metadata["skill_fallback_fraction"] = float(same.float().mean().item())
        variant["skill_code"] = wrong_codes
        variant["skill_sequence"] = wrong_codes[:, None]
        variant["skill_index"] = torch.zeros_like(wrong_codes)
    return variant, metadata


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _sem(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def _condition_summary(
    condition: str,
    values: dict[str, list[float]],
    baseline_mse: np.ndarray,
) -> dict:
    gt_mse = np.asarray(values["gt_mse"], dtype=np.float64)
    gt_mae = np.asarray(values["gt_mae"], dtype=np.float64)
    output_delta = np.asarray(values["output_delta_l2"], dtype=np.float64)
    normalized_delta = np.asarray(values["normalized_output_delta"], dtype=np.float64)
    if condition == "baseline":
        mse_increase = np.zeros_like(gt_mse)
        baseline_better_rate = None
    else:
        mse_increase = gt_mse - baseline_mse
        baseline_better_rate = float(np.mean(baseline_mse < gt_mse))
    baseline_mean = float(np.mean(baseline_mse))
    return {
        "condition": condition,
        "n_frames": int(gt_mse.size),
        "gt_mse": float(np.mean(gt_mse)),
        "gt_mse_sem": _sem(gt_mse.tolist()),
        "gt_mae": float(np.mean(gt_mae)),
        "output_delta_l2": float(np.mean(output_delta)),
        "output_delta_l2_sem": _sem(output_delta.tolist()),
        "normalized_output_delta": float(np.mean(normalized_delta)),
        "gt_mse_increase": float(np.mean(mse_increase)),
        "gt_mse_ratio": float(np.mean(gt_mse) / max(baseline_mean, 1e-12)),
        "baseline_better_rate": baseline_better_rate,
        "skill_fallback_fraction": _mean(values.get("skill_fallback_fraction", []))
        if values.get("skill_fallback_fraction")
        else None,
    }


@torch.no_grad()
def evaluate_model(config_path: Path, model_index: int) -> Path:
    config = load_config(config_path)
    resolved = build_settings(config)
    models = json.loads(resolved["models_json"])
    if not 0 <= model_index < len(models):
        raise IndexError(f"model-index={model_index} outside [0, {len(models) - 1}].")
    analysis = _analysis_settings(config)
    spec = models[model_index]
    output_dir = Path(resolved["eval_out_dir"]) / "input_influence" / _safe_dir_name(
        model_index, spec["label"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = analysis["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_path = Path(spec["policy_path"])
    dataset_dir = Path(spec["skill_dataset_dir"])
    dataset_info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(dataset_info["fps"])

    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path
    policy_config.eval_legacy_vsa = bool(spec.get("eval_legacy_vsa", False))
    if not policy_config.eval_legacy_vsa:
        expected_mode = str(
            spec.get("vision_conditioning_mode", "residual_cross_attention")
        )
        actual_mode = str(
            getattr(
                policy_config,
                "vision_conditioning_mode",
                "residual_cross_attention",
            )
        )
        if actual_mode != expected_mode:
            raise RuntimeError(
                "Checkpoint vision_conditioning_mode changed during input-influence eval: "
                f"resolved={expected_mode}, loaded={actual_mode}."
            )
    policy_config.num_visual_latents_per_camera = int(
        spec.get(
            "num_visual_latents_per_camera",
            8 if policy_config.eval_legacy_vsa else 32,
        )
    )
    policy_config.device = str(device)
    policy_config.gradient_checkpointing = False
    policy_config.compile_model = False
    chunk_size = int(policy_config.chunk_size)
    max_action_dim = int(policy_config.max_action_dim)
    vocab_size = int(policy_config.skill_vocab_size)

    dataset = LeRobotDataset(
        repo_id=f"local/input_influence_{model_index}",
        root=dataset_dir,
        delta_timestamps={"action": [index / fps for index in range(chunk_size)]},
    )
    n_frames = min(analysis["n_frames"], len(dataset))
    if n_frames < 2:
        raise ValueError(f"Dataset needs at least two frames, got {len(dataset)}.")
    batch_size = min(analysis["batch_size"], n_frames)
    rng = np.random.default_rng(seed)
    frame_ids = rng.choice(len(dataset), size=n_frames, replace=False).tolist()
    loader = DataLoader(
        Subset(dataset, frame_ids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=analysis["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=analysis["num_workers"] > 0,
    )

    policy = make_policy(cfg=policy_config, ds_meta=dataset.meta)
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    inference_kwargs = {}
    if analysis["num_inference_steps"] is not None:
        inference_kwargs["num_steps"] = analysis["num_inference_steps"]

    conditions = ["baseline", *analysis["perturbations"]]
    collected: dict[str, dict[str, list[float]]] = {
        condition: defaultdict(list) for condition in conditions
    }
    per_sample: list[dict] = []
    evaluated_frames = 0
    determinism_max_delta = None

    log.info(
        "[%s] policy=%s frames=%d batch=%d conditions=%s",
        spec["label"],
        policy_path,
        n_frames,
        batch_size,
        conditions,
    )
    for batch_index, raw_batch in enumerate(loader):
        current_batch = int(raw_batch["action"].shape[0])
        if current_batch < 2:
            log.warning("Skipping final singleton batch; swapping needs two samples.")
            continue
        gt_actions = raw_batch["action"].clone()
        padding = raw_batch.get(
            "action_is_pad",
            torch.zeros(gt_actions.shape[:2], dtype=torch.bool),
        ).bool()
        processed = preprocessor(raw_batch)
        missing = [
            key
            for key in (STATE_KEY, TOP_IMAGE_KEY, WRIST_IMAGE_KEY)
            if key not in processed
        ]
        if missing:
            raise KeyError(f"Preprocessed dataset batch is missing Stage-1 inputs: {missing}.")
        true_codes = policy._skill_code(processed).clone()  # noqa: SLF001
        noise = torch.randn(
            current_batch,
            chunk_size,
            max_action_dim,
            device=device,
            dtype=torch.float32,
        )
        baseline = postprocessor(
            policy.predict_action_chunk(processed, noise=noise, **inference_kwargs)
        ).cpu()
        baseline_mse, baseline_mae = _masked_errors(baseline, gt_actions, padding)
        gt_step_norm = _masked_step_l2(gt_actions, torch.zeros_like(gt_actions), padding)

        if batch_index == 0:
            repeated = postprocessor(
                policy.predict_action_chunk(processed, noise=noise, **inference_kwargs)
            ).cpu()
            determinism_max_delta = float((repeated - baseline).abs().max().item())
            log.info("[%s] same-noise determinism max |delta|=%.3e", spec["label"], determinism_max_delta)

        baseline_values = collected["baseline"]
        baseline_values["gt_mse"].extend(baseline_mse.tolist())
        baseline_values["gt_mae"].extend(baseline_mae.tolist())
        baseline_values["output_delta_l2"].extend([0.0] * current_batch)
        baseline_values["normalized_output_delta"].extend([0.0] * current_batch)

        batch_results = {
            "baseline": {
                "gt_mse": baseline_mse,
                "gt_mae": baseline_mae,
                "output_delta_l2": torch.zeros_like(baseline_mse),
                "normalized_output_delta": torch.zeros_like(baseline_mse),
            }
        }
        for condition in analysis["perturbations"]:
            variant, metadata = _counterfactual_batch(
                processed, condition, true_codes, vocab_size
            )
            prediction = postprocessor(
                policy.predict_action_chunk(variant, noise=noise, **inference_kwargs)
            ).cpu()
            gt_mse, gt_mae = _masked_errors(prediction, gt_actions, padding)
            output_delta = _masked_step_l2(prediction, baseline, padding)
            normalized_delta = output_delta / gt_step_norm.clamp_min(1e-8)
            values = collected[condition]
            values["gt_mse"].extend(gt_mse.tolist())
            values["gt_mae"].extend(gt_mae.tolist())
            values["output_delta_l2"].extend(output_delta.tolist())
            values["normalized_output_delta"].extend(normalized_delta.tolist())
            for key, value in metadata.items():
                values[key].append(value)
            batch_results[condition] = {
                "gt_mse": gt_mse,
                "gt_mae": gt_mae,
                "output_delta_l2": output_delta,
                "normalized_output_delta": normalized_delta,
            }

        if analysis["save_per_sample"]:
            for sample_index in range(current_batch):
                record = {
                    "sample": evaluated_frames + sample_index,
                    "gt_action_step_norm": float(gt_step_norm[sample_index]),
                    "conditions": {
                        condition: {
                            key: float(value[sample_index])
                            for key, value in metrics.items()
                        }
                        for condition, metrics in batch_results.items()
                    },
                }
                per_sample.append(record)
        evaluated_frames += current_batch
        log.info("[%s] batch %d/%d complete", spec["label"], batch_index + 1, len(loader))

    if evaluated_frames == 0:
        raise RuntimeError("No complete counterfactual batches were evaluated.")
    baseline_mse = np.asarray(collected["baseline"]["gt_mse"], dtype=np.float64)
    condition_summaries = [
        _condition_summary(condition, collected[condition], baseline_mse)
        for condition in conditions
    ]
    result = {
        "model_index": model_index,
        "label": spec["label"],
        "policy_path": str(policy_path),
        "dataset_dir": str(dataset_dir),
        "architecture": spec.get("architecture"),
        "architecture_revision": spec.get("architecture_revision"),
        "vision_conditioning_mode": spec.get("vision_conditioning_mode"),
        "visual_crossattn_queries": spec.get("visual_crossattn_queries"),
        "analysis": analysis,
        "evaluated_frames": evaluated_frames,
        "same_noise_determinism_max_delta": determinism_max_delta,
        "metric_definition": {
            "output_delta_l2": "mean valid-step L2(pred_corrupt - pred_baseline)",
            "normalized_output_delta": "output_delta_l2 / mean valid-step ||GT action||2",
            "gt_mse_increase": "paired MSE(corrupt, GT) - MSE(baseline, GT)",
            "baseline_better_rate": "fraction of frames where baseline MSE < corrupt-input MSE",
        },
        "conditions": condition_summaries,
    }
    if analysis["save_per_sample"]:
        result["per_sample"] = per_sample
    result_path = output_dir / "input_influence.json"
    result_path.write_text(json.dumps(result, indent=2))
    log.info("[%s] saved %s", spec["label"], result_path)
    for row in condition_summaries:
        log.info(
            "[%s] %-18s delta=%.5f mse=%.6f increase=%+.6f win=%s",
            spec["label"],
            row["condition"],
            row["output_delta_l2"],
            row["gt_mse"],
            row["gt_mse_increase"],
            row["baseline_better_rate"],
        )
    return result_path


def aggregate_results(config_path: Path) -> Path:
    config = load_config(config_path)
    resolved = build_settings(config)
    models = json.loads(resolved["models_json"])
    root = Path(resolved["eval_out_dir"]) / "input_influence"
    results = []
    missing = []
    for index, spec in enumerate(models):
        path = root / _safe_dir_name(index, spec["label"]) / "input_influence.json"
        if path.is_file():
            results.append(json.loads(path.read_text()))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing model input-influence results: " + ", ".join(missing))

    rows = []
    for result in results:
        for condition in result["conditions"]:
            rows.append(
                {
                    "model": result["label"],
                    "condition": condition["condition"],
                    "n_frames": condition["n_frames"],
                    "gt_mse": condition["gt_mse"],
                    "gt_mse_increase": condition["gt_mse_increase"],
                    "gt_mse_ratio": condition["gt_mse_ratio"],
                    "output_delta_l2": condition["output_delta_l2"],
                    "normalized_output_delta": condition["normalized_output_delta"],
                    "baseline_better_rate": condition["baseline_better_rate"],
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps({"models": results, "rows": rows}, indent=2))

    headers = list(rows[0])
    table_rows = []
    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.6g}"
            elif value is None:
                value = "-"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    html_path = root / "summary.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Stage1 input influence</title>"
        "<style>body{font-family:sans-serif;margin:32px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:6px 9px;text-align:right}"
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}"
        "tr:hover{background:#f5f5f5}</style>"
        "<h1>Stage1 paired input-influence evaluation</h1>"
        "<p>Each corrupted condition keeps the same flow noise and swaps an input "
        "with another dataset sample. Large output delta means the model uses that input; "
        "positive GT-MSE increase and high baseline-better rate mean it uses it usefully.</p>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
    )
    log.info("Aggregated %d models -> %s", len(results), html_path)
    return html_path


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate_results(args.config)
        return
    if args.model_index is None:
        raise ValueError("--model-index is required unless --aggregate is used.")
    evaluate_model(args.config, args.model_index)


if __name__ == "__main__":
    main()
