#!/usr/bin/env python3
"""Visualize the exact spline/normalization tensors used by FSQ training."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


LIBERO_ROOT = Path(__file__).resolve().parents[4]
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from FSQ import (  # noqa: E402
    load_fsq_model,
    prepare_encoder_trajectory,
    spline_decode,
    spline_encode,
)


DIM_NAMES = ("EE x", "EE y", "EE z", "rotvec x", "rotvec y", "rotvec z", "grip 0", "grip 1")
POSITION_COLORS = ("#ef476f", "#06d6a0", "#118ab2")
ROTATION_COLORS = ("#f78c6b", "#9b5de5", "#00bbf9")
GRIPPER_COLORS = ("#ff9f1c", "#4361ee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--start-run", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path, required=True)
    parser.add_argument("--start-checkpoint", default="latest")
    parser.add_argument("--zero-checkpoint", default="latest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    return parser.parse_args()


def scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def cfg_value(cfg: Any, name: str) -> Any:
    return cfg[name] if isinstance(cfg, dict) else getattr(cfg, name)


def load_stats(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "action_stats.npz", allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def resolve_checkpoint(run_dir: Path, requested: str) -> Path:
    if requested != "latest":
        path = run_dir / requested
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = list(run_dir.glob("FSQ_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoint in {run_dir}")
    return max(candidates, key=lambda path: int(path.stem.removeprefix("FSQ_epoch")))


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def skill_states(bundle: dict[str, np.ndarray], index: int) -> np.ndarray:
    lengths = bundle["states_len"].astype(np.int64)
    start = int(lengths[:index].sum())
    return bundle["states_cat"][start : start + int(lengths[index])].astype(np.float32)


def select_diverse_samples(bundle: dict[str, np.ndarray], count: int) -> list[int]:
    """Greedy farthest-point sampling over absolute start pose, with unique tasks."""
    lengths = bundle["states_len"].astype(np.int64)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1])))
    start_pose = bundle["states_cat"][starts, :6].astype(np.float64)
    center = np.median(start_pose, axis=0)
    scale = np.quantile(np.abs(start_pose - center), 0.75, axis=0)
    scale = np.where(scale > 1e-8, scale, np.std(start_pose, axis=0) + 1e-8)
    standardized = (start_pose - center) / scale
    seed = int(np.argmin(np.linalg.norm(standardized, axis=1)))
    selected = [seed]
    tasks = bundle["meta_task_id"].astype(np.int64)
    selected_tasks = {int(tasks[seed])}
    distance = np.linalg.norm(standardized - standardized[seed], axis=1)
    while len(selected) < min(count, len(start_pose)):
        score = distance.copy()
        repeated_task = np.isin(tasks, list(selected_tasks))
        score[repeated_task] *= 0.1
        score[selected] = -np.inf
        next_index = int(np.argmax(score))
        selected.append(next_index)
        selected_tasks.add(int(tasks[next_index]))
        new_distance = np.linalg.norm(standardized - standardized[next_index], axis=1)
        distance = np.minimum(distance, new_distance)
    return selected


def normalized_grid(
    states: np.ndarray,
    mode: str,
    minimum: np.ndarray,
    maximum: np.ndarray,
    n_control: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    grounded = prepare_encoder_trajectory(states, mode)
    grid, _ = spline_encode(states, n_control, degree, input_mode=mode)
    normalized = 2.0 * (grid - minimum) / (maximum - minimum + 1e-8) - 1.0
    return grounded, normalized.astype(np.float32)


@torch.inference_mode()
def model_values(
    model: Any,
    normalized_input: np.ndarray,
    length: int,
) -> dict[str, Any]:
    ctrl = torch.from_numpy(normalized_input).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.long)
    z_e = model.encoder.encode_continuous(ctrl, lengths, normalized=True)
    u_cont = model.fsq.bound(z_e)
    z_q, token = model.fsq(z_e)
    z_norm = model.fsq.normalized(z_q)
    predicted, predicted_length = model.reconstructor(z_norm, start_state=None)
    prediction_physical = model.sample_control_points(z_q)[0].cpu().numpy()
    return {
        "z_e": z_e[0].cpu().numpy(),
        "u_cont": u_cont[0].cpu().numpy(),
        "z_q": z_q[0].cpu().numpy(),
        "z_norm": z_norm[0].cpu().numpy(),
        "token": int(token.item()),
        "prediction": predicted[0].cpu().numpy(),
        "prediction_physical": prediction_physical,
        "predicted_length": (
            None if predicted_length is None else float(predicted_length.item())
        ),
    }


def equalize_3d(axis: Any, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max(float((maximum - minimum).max()) / 2, 1e-4)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def mode_axes_figure(
    pack: dict[str, Any],
    label: str,
    path: Path,
    *,
    length: int,
    degree: int,
) -> None:
    """Plot one mode with one independently scaled panel per predicted dimension."""
    fig = plt.figure(figsize=(13, 17), constrained_layout=True)
    grid = fig.add_gridspec(5, 2, height_ratios=(2.5, 1, 1, 1, 1))
    target = pack["target"]
    prediction = pack["prediction"]
    target_physical = pack["target_physical"]
    prediction_physical = pack["prediction_physical"]
    target_trajectory = spline_decode(target_physical, length, degree)
    prediction_trajectory = spline_decode(prediction_physical, length, degree)
    grid_index = np.arange(target.shape[0])
    gt_color = "#1769aa"
    prediction_color = "#d62828"

    axis_3d = fig.add_subplot(grid[0, :], projection="3d")
    axis_3d.plot(
        *target_trajectory[:, :3].T,
        color=gt_color,
        linewidth=2.7,
        label="GT trajectory",
    )
    axis_3d.plot(
        *prediction_trajectory[:, :3].T,
        color=prediction_color,
        linewidth=2.2,
        linestyle="--",
        label="prediction",
    )
    equalize_3d(
        axis_3d,
        np.concatenate((target_trajectory[:, :3], prediction_trajectory[:, :3])),
    )
    axis_3d.set_title(
        f"{label} · decoded cubic B-spline XYZ (grounded physical coordinates)"
    )
    axis_3d.set_xlabel("x (m)")
    axis_3d.set_ylabel("y (m)")
    axis_3d.set_zlabel("z (m)")
    axis_3d.legend(loc="best", fontsize=9)

    for dim, name in enumerate(DIM_NAMES):
        axis = fig.add_subplot(grid[1 + dim // 2, dim % 2])
        axis.plot(
            grid_index,
            target[:, dim],
            color=gt_color,
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label="GT target",
        )
        axis.plot(
            grid_index,
            prediction[:, dim],
            color=prediction_color,
            linewidth=2.0,
            linestyle="--",
            label="prediction",
        )
        lower = float(min(target[:, dim].min(), prediction[:, dim].min()))
        upper = float(max(target[:, dim].max(), prediction[:, dim].max()))
        padding = max(0.06 * (upper - lower), 0.015)
        axis.set_ylim(lower - padding, upper + padding)
        mse = float(np.mean(np.square(prediction[:, dim] - target[:, dim])))
        axis.set_title(f"{name} · MSE {mse:.5f}")
        axis.set_xlabel("B-spline grid index (0–29)")
        axis.set_ylabel("normalized value")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"{label}: GT vs FSQ reconstruction\n"
        "8 panels use independent y-scales (blue=GT, red dashed=prediction)",
        fontsize=16,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def tensor_figure(processed: dict[str, dict[str, Any]], path: Path) -> None:
    arrays = []
    for mode in ("zero", "start"):
        pack = processed[mode]
        arrays.extend((pack["input"], pack["target"], pack["prediction"]))
    value_limit = max(1.0, max(float(np.abs(array).max()) for array in arrays))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    image_artist = None
    error_artist = None
    for row, mode in enumerate(("zero", "start")):
        pack = processed[mode]
        values = (
            ("Encoder input", pack["input"]),
            ("Reconstruction target", pack["target"]),
            ("Model prediction", pack["prediction"]),
        )
        for column, (title, value) in enumerate(values):
            image_artist = axes[row, column].imshow(
                value.T,
                aspect="auto",
                cmap="coolwarm",
                vmin=-value_limit,
                vmax=value_limit,
                interpolation="nearest",
            )
            axes[row, column].set_title(f"{mode}: {title}")
            axes[row, column].set_xlabel("B-spline grid index")
            axes[row, column].set_yticks(range(len(DIM_NAMES)), DIM_NAMES)
        error = pack["prediction"] - pack["target"]
        error_limit = max(0.05, float(np.abs(error).max()))
        error_artist = axes[row, 3].imshow(
            error.T,
            aspect="auto",
            cmap="PuOr",
            vmin=-error_limit,
            vmax=error_limit,
            interpolation="nearest",
        )
        axes[row, 3].set_title(f"{mode}: prediction - target")
        axes[row, 3].set_xlabel("B-spline grid index")
        axes[row, 3].set_yticks(range(len(DIM_NAMES)), DIM_NAMES)
    fig.colorbar(image_artist, ax=axes[:, :3], shrink=0.78, label="normalized value")
    fig.colorbar(error_artist, ax=axes[:, 3], shrink=0.78, label="normalized error")
    fig.suptitle("Exact 30×8 tensors around the FSQ bottleneck", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def matrix_table(matrix: np.ndarray) -> str:
    header = "".join(f"<th>{html.escape(name)}</th>" for name in DIM_NAMES)
    rows = []
    for index, row in enumerate(matrix):
        cells = "".join(f"<td>{value:+.4f}</td>" for value in row)
        rows.append(f"<tr><th>{index:02d}</th>{cells}</tr>")
    return (
        "<div class='matrix-wrap'><table class='matrix'><thead><tr><th>grid</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def vector(value: np.ndarray, digits: int = 4) -> str:
    return "[" + ", ".join(f"{float(item):.{digits}f}" for item in value) + "]"


def stats_table(start_stats: dict[str, np.ndarray], zero_stats: dict[str, np.ndarray]) -> str:
    rows = []
    for index, name in enumerate(DIM_NAMES):
        rows.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{zero_stats['encoder_min'][index]:+.5f}</td>"
            f"<td>{zero_stats['encoder_max'][index]:+.5f}</td>"
            f"<td>{start_stats['encoder_min'][index]:+.5f}</td>"
            f"<td>{start_stats['encoder_max'][index]:+.5f}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>dimension</th>"
        "<th>zero min</th><th>zero max</th><th>start min</th><th>start max</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build_html(
    samples: list[dict[str, Any]],
    start_stats: dict[str, np.ndarray],
    zero_stats: dict[str, np.ndarray],
    start_run: Path,
    zero_run: Path,
    start_checkpoint: Path,
    zero_checkpoint: Path,
    n_control: int,
    degree: int,
) -> str:
    navigation = "".join(
        f"<a href='#sample-{sample['rank']}'>#{sample['rank']} · task {sample['task_id']} / skill {sample['skill_index']}</a>"
        for sample in samples
    )
    cards = []
    for sample in samples:
        zero = sample["processed"]["zero"]
        start = sample["processed"]["start"]
        mode_cards = []
        for label, pack in (("zero→zero", zero), ("start→start", start)):
            values = pack["model"]
            mode_cards.append(f"""
              <section class="mode-card">
                <div class="eyebrow">{label}</div>
                <div class="token">code {values['token']} · FSQ {html.escape(vector(values['z_q'], 0))}</div>
                <dl>
                  <dt>Transformer output z_e</dt><dd>{html.escape(vector(values['z_e']))}</dd>
                  <dt>FSQ bounded u</dt><dd>{html.escape(vector(values['u_cont']))}</dd>
                  <dt>normalized grid range</dt><dd>{pack['input'].min():+.3f} … {pack['input'].max():+.3f}</dd>
                  <dt>input ↔ target max |Δ|</dt><dd>{pack['input_target_max_abs']:.8f}</dd>
                  <dt>prediction MSE (all)</dt><dd>{pack['prediction_mse']:.6f}</dd>
                  <dt>normalized XYZ MSE</dt><dd>{pack['prediction_position_mse']:.6f}</dd>
                  <dt>physical XYZ RMSE</dt><dd>{html.escape(vector(pack['physical_xyz_rmse']))} m</dd>
                  <dt>normalized rotvec MSE</dt><dd>{pack['prediction_rotation_mse']:.6f}</dd>
                  <dt>normalized gripper MSE</dt><dd>{pack['prediction_gripper_mse']:.6f}</dd>
                </dl>
              </section>
            """)
        cards.append(f"""
        <article class="sample" id="sample-{sample['rank']}">
          <header>
            <div><div class="eyebrow">diverse absolute-start-pose sample #{sample['rank']}</div>
            <h2>task {sample['task_id']} · episode {sample['episode_id']} · skill {sample['skill_index']}</h2></div>
            <div class="chips"><span>{sample['length']} frames</span><span>length norm {sample['length_norm']:+.3f}</span></div>
          </header>
          <p class="path">{html.escape(sample['file'])}</p>
          <p><b>raw start XYZ</b> {html.escape(vector(sample['start_pose'][:3]))} · <b>raw start rotvec</b> {html.escape(vector(sample['start_pose'][3:6]))}</p>
          <div class="mode-grid">{''.join(mode_cards)}</div>
          <h3>zero→zero</h3>
          <figure><img loading="lazy" src="sample_{sample['rank']:02d}_zero_axes.png" alt="zero-grounded GT and prediction by axis"><figcaption>상단은 control-point grid를 cubic B-spline으로 decode한 XYZ 3D 경로다. 하단은 x/y/z/rx/ry/rz/grip0/grip1을 각각 독립 y-scale로 그린 normalized GT와 prediction이다.</figcaption></figure>
          <h3>start→start</h3>
          <figure><img loading="lazy" src="sample_{sample['rank']:02d}_start_axes.png" alt="start-grounded GT and prediction by axis"><figcaption>상단은 control-point grid를 cubic B-spline으로 decode한 XYZ 3D 경로다. 하단은 x/y/z/rx/ry/rz/grip0/grip1을 각각 독립 y-scale로 그린 normalized GT와 prediction이다.</figcaption></figure>
          <details><summary>30×8 heatmap과 정확한 encoder input 값 펼치기</summary>
            <figure><img loading="lazy" src="sample_{sample['rank']:02d}_tensors.png" alt="normalized tensor comparison"><figcaption>FSQ encoder input, reconstruction target, prediction과 error heatmap.</figcaption></figure>
            <h3>zero→zero · 30×8</h3>{matrix_table(zero['input'])}
            <h3>start→start · 30×8</h3>{matrix_table(start['input'])}
          </details>
        </article>
        """)

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FSQ spline input/output · start→start vs zero→zero</title>
<style>
:root{{--bg:#09111d;--panel:#111d2c;--panel2:#17263a;--line:#2b3d55;--text:#ecf2fb;--muted:#9fb0c6;--cyan:#65d6ff;--amber:#ffc76b}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 15% 0,#183353 0,var(--bg) 36%);color:var(--text);font:15px/1.65 Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif}}
main{{max-width:1500px;margin:auto;padding:52px 32px 90px}}h1{{font-size:clamp(31px,4.5vw,55px);line-height:1.08;letter-spacing:-.04em;margin:4px 0 12px}}h2{{margin:0;font-size:25px}}h3{{margin:20px 0 8px}}p{{color:#c8d4e5}}code{{color:#bdeeff}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:#90a6c1}}.lead{{max-width:1080px;font-size:17px}}.callout{{margin:24px 0;padding:18px 21px;border:1px solid #39708a;background:#10283a;border-radius:14px;color:#d8efff}}.pipeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:24px 0}}.pipeline div{{background:var(--panel2);border:1px solid var(--line);border-radius:11px;padding:12px;text-align:center}}.nav{{display:flex;gap:8px;flex-wrap:wrap;margin:25px 0 40px}}a{{color:#86ddff}}.nav a{{text-decoration:none;border:1px solid #38556e;border-radius:999px;padding:7px 12px}}.sample{{margin:36px 0 72px;padding:24px;background:linear-gradient(150deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:17px;box-shadow:0 20px 50px #0005}}.sample header{{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}}.chips{{display:flex;gap:8px;flex-wrap:wrap}}.chips span{{padding:5px 10px;border-radius:999px;background:#21354c;color:#bfe8ff}}.path{{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted);overflow-wrap:anywhere}}.mode-grid{{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin:18px 0}}.mode-card{{padding:16px;border:1px solid var(--line);border-radius:12px;background:#0d1826}}.token{{font-size:20px;font-weight:750;color:var(--amber)}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 13px;margin:11px 0 0}}dt{{color:var(--muted)}}dd{{margin:0;font-family:ui-monospace,monospace;overflow-wrap:anywhere}}figure{{margin:18px 0 28px;padding:11px;background:white;border-radius:12px}}figure img{{display:block;width:100%}}figcaption{{padding:8px 7px 1px;color:#455469}}details{{border:1px solid var(--line);border-radius:10px;padding:12px;background:#0d1724}}summary{{cursor:pointer;color:var(--cyan);font-weight:650}}.table-wrap,.matrix-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;background:#0d1826}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{position:sticky;top:0;background:#18283b;color:#aee8ff}}.matrix{{font:11px/1.35 ui-monospace,monospace}}.matrix th,.matrix td{{padding:4px 7px}}.sources{{font-size:12px;color:var(--muted);overflow-wrap:anywhere}}
@media(max-width:900px){{main{{padding:30px 14px 60px}}.pipeline{{grid-template-columns:1fr}}.mode-grid{{grid-template-columns:1fr}}.sample header{{display:block}}dl{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">actual training tensors · matched reconstruction modes</div>
<h1>FSQ B-spline I/O<br>start→start vs zero→zero</h1>
<p class="lead">같은 원본 skill을 두 preprocessing convention으로 통과시켜, 실제 encoder 입력과 reconstruction target을 만들고 현재 recon-only checkpoint까지 실행했다. 샘플은 결과를 예쁘게 보이도록 고른 것이 아니라 전체 11,221개 중 absolute start pose가 서로 멀도록 deterministic하게 선택했다.</p>
<div class="pipeline"><div>raw 8D state</div><div>SE(3) grounding</div><div>degree-{degree} B-spline<br>{n_control} points</div><div>global min-max<br>normalization</div><div>Transformer → FSQ<br>→ reconstruction</div></div>
<div class="callout"><b>먼저 중요한 사실:</b> 두 run 모두 encoder mode와 reconstructor target mode가 각각 정확히 일치한다. 같은 B-spline 함수와 같은 global min/max 통계를 쓰기 때문에, 각 mode 안에서 <code>encoder input 30×8 == reconstruction target 30×8</code>이다. 차이는 start→start와 zero→zero 사이의 coordinate convention이며, 그림의 model prediction은 quantized FSQ code를 지난 실제 출력이다. 단, prediction MSE는 서로 다른 coordinate/normalization과 checkpoint epoch의 값이므로 두 mode의 성능 순위로 직접 비교하면 안 된다.</div>
<div class="callout"><b>3D 경로가 크게 벌어지는 이유:</b> 그래프는 checkpoint의 공식 <code>sample_control_points()</code> 출력과 동일한 값을 사용하고, 이를 학습/eval과 같은 cubic B-spline decoder로 펼친다. 따라서 큰 간격은 좌표 역변환 오류가 아니라 해당 discrete code가 여러 skill에 공유되면서 decoder가 code-level prototype에 가까운 궤적을 내는 실제 reconstruction error다.</div>
<h2>전체 데이터에서 계산한 normalization 범위</h2>
<p>zero-grounded는 XYZ 평균만 빼고 rotation/gripper는 absolute로 유지한다. start-grounded는 첫 pose를 기준으로 XYZ를 시작 EE frame으로 회전하고 rotation을 <code>R₀⁻¹Rₜ</code>로 만들며 gripper는 그대로 둔다.</p>
{stats_table(start_stats, zero_stats)}
<nav class="nav">{navigation}</nav>
{''.join(cards)}
<p class="sources">start run: {html.escape(str(start_run))}<br>checkpoint: {html.escape(start_checkpoint.name)}<br><br>zero run: {html.escape(str(zero_run))}<br>checkpoint: {html.escape(zero_checkpoint.name)}</p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_run = args.start_run.resolve()
    zero_run = args.zero_run.resolve()
    start_stats = load_stats(start_run)
    zero_stats = load_stats(zero_run)
    bundle = load_bundle(args.skill_bundle.resolve())

    start_checkpoint = resolve_checkpoint(start_run, args.start_checkpoint)
    zero_checkpoint = resolve_checkpoint(zero_run, args.zero_checkpoint)
    start_model, start_cfg = load_fsq_model(start_checkpoint, device="cpu")
    zero_model, zero_cfg = load_fsq_model(zero_checkpoint, device="cpu")
    start_model.eval()
    zero_model.eval()
    n_control = int(cfg_value(start_cfg, "n_control"))
    degree = int(cfg_value(start_cfg, "spline_degree"))
    if n_control != int(cfg_value(zero_cfg, "n_control")) or degree != int(
        cfg_value(zero_cfg, "spline_degree")
    ):
        raise ValueError("Compared checkpoints use different spline grids")
    if scalar_text(start_stats["encoder_input_mode"]) != "start_grounded":
        raise ValueError("start run does not use start_grounded input")
    if scalar_text(zero_stats["encoder_input_mode"]) != "zero_grounded":
        raise ValueError("zero run does not use zero_grounded input")

    selected = select_diverse_samples(bundle, args.samples)
    samples: list[dict[str, Any]] = []
    length_min = float(start_stats["length_min"])
    length_max = float(start_stats["length_max"])
    for rank, index in enumerate(selected, start=1):
        states = skill_states(bundle, index)
        length = len(states)
        processed: dict[str, dict[str, Any]] = {}
        for mode, stats, model in (
            ("zero", zero_stats, zero_model),
            ("start", start_stats, start_model),
        ):
            input_mode = scalar_text(stats["encoder_input_mode"])
            output_mode = scalar_text(stats["reconstructor_output_mode"])
            grounded, normalized_input = normalized_grid(
                states,
                input_mode,
                stats["encoder_min"],
                stats["encoder_max"],
                n_control,
                degree,
            )
            output_grounded, normalized_target = normalized_grid(
                states,
                output_mode,
                stats["reconstructor_min"],
                stats["reconstructor_max"],
                n_control,
                degree,
            )
            values = model_values(model, normalized_input, length)
            target_physical = (
                (normalized_target + 1.0)
                * 0.5
                * (stats["reconstructor_max"] - stats["reconstructor_min"] + 1e-8)
                + stats["reconstructor_min"]
            )
            prediction_physical_manual = (
                (values["prediction"] + 1.0)
                * 0.5
                * (stats["reconstructor_max"] - stats["reconstructor_min"] + 1e-8)
                + stats["reconstructor_min"]
            )
            prediction_physical = values["prediction_physical"]
            decode_consistency = float(
                np.max(np.abs(prediction_physical - prediction_physical_manual))
            )
            if decode_consistency > 1e-6:
                raise RuntimeError(
                    "Manual inverse-normalization differs from sample_control_points: "
                    f"max |delta|={decode_consistency:.8g}"
                )
            processed[mode] = {
                "grounded": grounded,
                "output_grounded": output_grounded,
                "input": normalized_input,
                "target": normalized_target,
                "prediction": values["prediction"],
                "target_physical": target_physical,
                "prediction_physical": prediction_physical,
                "decode_consistency_max_abs": decode_consistency,
                "input_target_max_abs": float(
                    np.max(np.abs(normalized_input - normalized_target))
                ),
                "prediction_mse": float(
                    np.mean(np.square(values["prediction"] - normalized_target))
                ),
                "prediction_position_mse": float(
                    np.mean(np.square(values["prediction"][:, :3] - normalized_target[:, :3]))
                ),
                "physical_xyz_rmse": np.sqrt(
                    np.mean(
                        np.square(prediction_physical[:, :3] - target_physical[:, :3]),
                        axis=0,
                    )
                ),
                "prediction_rotation_mse": float(
                    np.mean(np.square(values["prediction"][:, 3:6] - normalized_target[:, 3:6]))
                ),
                "prediction_gripper_mse": float(
                    np.mean(np.square(values["prediction"][:, 6:] - normalized_target[:, 6:]))
                ),
                "model": values,
            }

        mode_axes_figure(
            processed["zero"],
            "zero→zero",
            output_dir / f"sample_{rank:02d}_zero_axes.png",
            length=length,
            degree=degree,
        )
        mode_axes_figure(
            processed["start"],
            "start→start",
            output_dir / f"sample_{rank:02d}_start_axes.png",
            length=length,
            degree=degree,
        )
        tensor_figure(processed, output_dir / f"sample_{rank:02d}_tensors.png")
        samples.append({
            "rank": rank,
            "index": index,
            "file": str(bundle["files"][index]),
            "task_id": int(bundle["meta_task_id"][index]),
            "episode_id": int(bundle["meta_episode_id"][index]),
            "skill_index": int(bundle["meta_skill_index"][index]),
            "length": length,
            "length_norm": float(
                2.0 * (length - length_min) / (length_max - length_min + 1e-8) - 1.0
            ),
            "start_pose": states[0, :6].copy(),
            "processed": processed,
        })

    html_text = build_html(
        samples,
        start_stats,
        zero_stats,
        start_run,
        zero_run,
        start_checkpoint,
        zero_checkpoint,
        n_control,
        degree,
    )
    (output_dir / "index.html").write_text(html_text)
    serializable = {
        "selection": "greedy farthest-point sampling over standardized absolute start pose",
        "n_control": n_control,
        "spline_degree": degree,
        "start_run": str(start_run),
        "zero_run": str(zero_run),
        "start_checkpoint": str(start_checkpoint),
        "zero_checkpoint": str(zero_checkpoint),
        "samples": [
            {
                "rank": sample["rank"],
                "dataset_index": sample["index"],
                "file": sample["file"],
                "task_id": sample["task_id"],
                "episode_id": sample["episode_id"],
                "skill_index": sample["skill_index"],
                "length": sample["length"],
                "start_pose": sample["start_pose"].tolist(),
                "zero": {
                    "token": sample["processed"]["zero"]["model"]["token"],
                    "z_e": sample["processed"]["zero"]["model"]["z_e"].tolist(),
                    "z_q": sample["processed"]["zero"]["model"]["z_q"].tolist(),
                    "prediction_mse": sample["processed"]["zero"]["prediction_mse"],
                    "prediction_position_mse": sample["processed"]["zero"]["prediction_position_mse"],
                    "prediction_rotation_mse": sample["processed"]["zero"]["prediction_rotation_mse"],
                    "prediction_gripper_mse": sample["processed"]["zero"]["prediction_gripper_mse"],
                },
                "start": {
                    "token": sample["processed"]["start"]["model"]["token"],
                    "z_e": sample["processed"]["start"]["model"]["z_e"].tolist(),
                    "z_q": sample["processed"]["start"]["model"]["z_q"].tolist(),
                    "prediction_mse": sample["processed"]["start"]["prediction_mse"],
                    "prediction_position_mse": sample["processed"]["start"]["prediction_position_mse"],
                    "prediction_rotation_mse": sample["processed"]["start"]["prediction_rotation_mse"],
                    "prediction_gripper_mse": sample["processed"]["start"]["prediction_gripper_mse"],
                },
            }
            for sample in samples
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
