#!/usr/bin/env python3
"""Draw Stage-1 alignment, action-attention, and action-gradient maps."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from eval_config import EvalConfig, load_eval_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.policies.skillVLA.dataset_skillVLA import (
    CAM_WRIST,
    SKILL_CODE,
    SKILL_END_STATE,
    SKILL_END_STATE_VALID,
    SKILL_START_STATE,
    SkillVLADataset,
)
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skill_expert.wrist_patch_target import WristCamera, patch_labels
from lerobot.utils.constants import ACTION


def write_gallery(output_dir: Path, manifest: dict) -> Path:
    """Write a self-contained task/skill browser from ``manifest.json``."""
    samples = manifest.get("samples", [])
    valid = [sample for sample in samples if sample.get("target_valid")]
    exact = sum(sample.get("cell_distance") == 0 for sample in valid)
    near = sum(
        sample.get("cell_distance") is not None and sample["cell_distance"] <= 1
        for sample in valid
    )
    task_rows: list[dict] = []
    seen_tasks: set[str] = set()
    for task in manifest.get("selected_tasks", []):
        key = str(task.get("task_id", "unknown"))
        if key not in seen_tasks:
            task_rows.append(task)
            seen_tasks.add(key)
    for sample in samples:
        key = str(sample.get("task_id", "unknown"))
        if key not in seen_tasks:
            task_rows.append(
                {"task_id": sample.get("task_id"), "task": sample.get("task", "Unknown task")}
            )
            seen_tasks.add(key)

    cards = []
    for sample in samples:
        panel = html.escape(str(sample["panel"]), quote=True)
        action_panel_value = sample.get("action_panel")
        action_panel = (
            html.escape(str(action_panel_value), quote=True)
            if action_panel_value
            else ""
        )
        action_view = (
            f'<a class="panel-view action-view" href="{action_panel}" hidden>'
            f'<img src="{action_panel}" loading="lazy" alt="action attention and gradient maps"></a>'
            if action_panel
            else '<div class="panel-view action-view unavailable" hidden>Action maps were not generated for this sample.</div>'
        )
        query_ids = ", ".join(str(value) for value in sample.get("top_query_indices", []))
        distance = sample.get("cell_distance")
        task_label = html.escape(str(sample.get("task", f"task {sample.get('task_id', '?')}")))
        task_key = html.escape(str(sample.get("task_id", "unknown")), quote=True)
        skill_index = int(sample.get("skill_index", -1))
        skill_code = int(sample.get("skill_code", -1))
        cards.append(
            f"""<article class="card" data-task="{task_key}" data-skill-index="{skill_index}" data-skill-code="{skill_code}" data-episode="{sample['episode']}" data-has-action="{str(bool(action_panel)).lower()}">
<a class="panel-view alignment-view" href="{panel}"><img src="{panel}" loading="lazy" alt="alignment panel"></a>{action_view}
<div class="meta"><b>{task_label}</b><span>episode {sample['episode']} · skill {skill_index} (code {skill_code}) · frame {sample['frame']}</span>
<span>GT {sample['target_cell']} · pred {sample['predicted_cell']} · distance {distance if distance is not None else 'N/A'}</span>
<small>Top queries: {html.escape(query_ids)}</small></div></article>"""
        )
    task_buttons = []
    for task in task_rows:
        key = str(task.get("task_id", "unknown"))
        name = str(task.get("task", "Unknown task"))
        count = sum(str(sample.get("task_id", "unknown")) == key for sample in samples)
        task_buttons.append(
            f"""<button class="task-button" data-task="{html.escape(key, quote=True)}">
<span>Task {html.escape(key)}</span><small>{html.escape(name)}</small><em>{count} images</em></button>"""
        )
    architecture = html.escape(str(manifest.get("architecture", "unknown")))
    checkpoint = html.escape(str(manifest.get("checkpoint", "")))
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{architecture} attention results</title><style>
:root {{ color-scheme: dark; font-family:system-ui,sans-serif; background:#11151b; color:#eef2f7; }}
* {{ box-sizing:border-box; }} body {{ margin:0; }} header {{ padding:24px 28px 18px; border-bottom:1px solid #293341; }}
h1 {{ margin:0 0 8px; }} .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin:14px 0; }}
.pill {{ background:#202833; border:1px solid #344050; border-radius:10px; padding:9px 13px; }}
.path {{ color:#9eabbc; font-size:12px; overflow-wrap:anywhere; }}
.browser {{ display:grid; grid-template-columns:290px minmax(0,1fr); min-height:calc(100vh - 185px); }}
.sidebar {{ padding:20px 14px; border-right:1px solid #293341; background:#141a22; display:flex; flex-direction:column; gap:8px; }}
.sidebar h2 {{ margin:0 8px 10px; font-size:14px; color:#9eabbc; text-transform:uppercase; letter-spacing:.08em; }}
.task-button {{ color:inherit; text-align:left; border:1px solid transparent; border-radius:10px; background:transparent; padding:11px 12px; cursor:pointer; display:grid; gap:3px; }}
.task-button:hover {{ background:#202833; }} .task-button.active {{ background:#263345; border-color:#4e6b91; }}
.task-button span {{ font-weight:700; }} .task-button small {{ color:#b9c4d2; line-height:1.3; }}
.task-button em {{ color:#7f91a7; font-size:11px; font-style:normal; }}
.content {{ padding:22px; min-width:0; }} .content-head h2 {{ margin:0 0 5px; }} .content-head p {{ margin:0; color:#aeb9c8; }}
.view-switch {{ display:flex; gap:8px; margin-top:14px; }}
.view-button {{ border:1px solid #3b4758; background:#1a212b; color:#d9e1eb; border-radius:8px; padding:8px 12px; cursor:pointer; }}
.view-button.active {{ background:#38608c; border-color:#6390c3; color:white; }} .view-button:disabled {{ opacity:.4; cursor:not-allowed; }}
.skill-filters {{ display:flex; gap:8px; flex-wrap:wrap; margin:16px 0 20px; }}
.skill-button {{ border:1px solid #3b4758; background:#1a212b; color:#d9e1eb; border-radius:999px; padding:7px 11px; cursor:pointer; }}
.skill-button.active {{ background:#38608c; border-color:#6390c3; color:white; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:18px; }}
.card {{ background:#1a212b; border:1px solid #303b49; border-radius:12px; overflow:hidden; }}
.card img {{ width:100%; display:block; }} .panel-view[hidden] {{ display:none; }} .unavailable {{ min-height:180px; padding:30px; color:#9eabbc; }} .meta {{ display:grid; gap:5px; padding:12px 14px; }}
.meta span,.meta small {{ color:#aeb9c8; }} .empty {{ color:#9eabbc; padding:40px 0; }}
@media (max-width:800px) {{ .browser {{ grid-template-columns:1fr; }} .sidebar {{ border-right:0; border-bottom:1px solid #293341; display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); }} .sidebar h2 {{ grid-column:1/-1; }} .grid {{ grid-template-columns:1fr; }} }}
</style></head><body><header><h1>{architecture} vision diagnostics</h1>
<div class="summary"><span class="pill">samples <b>{len(samples)}</b></span>
<span class="pill">valid targets <b>{len(valid)}</b></span>
<span class="pill">exact <b>{exact}/{len(valid)}</b></span>
<span class="pill">within 1 cell <b>{near}/{len(valid)}</b></span></div>
<div class="path">{checkpoint}</div></header>
<main class="browser"><nav class="sidebar"><h2>Tasks</h2>{''.join(task_buttons)}</nav>
<section class="content"><div class="content-head"><h2 id="task-title">Select a task</h2><p id="task-name"></p>
<div class="view-switch"><button class="view-button active" data-view="alignment">Alignment head</button><button class="view-button" data-view="action">Action maps</button></div></div>
<div class="skill-filters" id="skill-filters"></div><div class="grid" id="gallery">{''.join(cards)}</div>
<p class="empty" id="empty" hidden>No images match this selection.</p></section></main>
<script>
const taskButtons=[...document.querySelectorAll('.task-button')];
const cards=[...document.querySelectorAll('.card')];
const skillFilters=document.getElementById('skill-filters');
const viewButtons=[...document.querySelectorAll('.view-button')];
let activeTask=null, activeSkill='all', activeView='alignment';
function selectView(value) {{
  activeView=value;
  for(const button of viewButtons) button.classList.toggle('active',button.dataset.view===value);
  for(const card of cards) {{
    card.querySelector('.alignment-view').hidden=value!=='alignment';
    card.querySelector('.action-view').hidden=value!=='action';
  }}
}}
function render() {{
  let visible=0;
  for (const card of cards) {{
    const show=card.dataset.task===activeTask && (activeSkill==='all' || card.dataset.skillIndex===activeSkill);
    card.hidden=!show; if(show) visible++;
  }}
  document.getElementById('empty').hidden=visible!==0;
}}
function selectSkill(value) {{ activeSkill=value; for(const b of skillFilters.children) b.classList.toggle('active',b.dataset.skill===value); render(); }}
function selectTask(button) {{
  activeTask=button.dataset.task; activeSkill='all';
  for(const b of taskButtons) b.classList.toggle('active',b===button);
  document.getElementById('task-title').textContent=button.querySelector('span').textContent;
  document.getElementById('task-name').textContent=button.querySelector('small').textContent;
  const pairs=new Map();
  for(const card of cards) if(card.dataset.task===activeTask) pairs.set(card.dataset.skillIndex,card.dataset.skillCode);
  skillFilters.replaceChildren();
  const all=document.createElement('button'); all.className='skill-button active'; all.dataset.skill='all'; all.textContent=`All skills (${{[...cards].filter(c=>c.dataset.task===activeTask).length}} images)`; all.onclick=()=>selectSkill('all'); skillFilters.append(all);
  for(const [index,code] of [...pairs].sort((a,b)=>Number(a[0])-Number(b[0]))) {{ const b=document.createElement('button'); b.className='skill-button'; b.dataset.skill=index; b.textContent=`Skill ${{index}} · code ${{code}}`; b.onclick=()=>selectSkill(index); skillFilters.append(b); }}
  render();
}}
for(const button of taskButtons) button.onclick=()=>selectTask(button);
for(const button of viewButtons) button.onclick=()=>selectView(button.dataset.view);
if(!cards.some(card=>card.dataset.hasAction==='true')) viewButtons.find(button=>button.dataset.view==='action').disabled=true;
selectView('alignment');
if(taskButtons.length) selectTask(taskButtons[0]); else render();
</script></body></html>"""
    path = output_dir / "index.html"
    path.write_text(body, encoding="utf-8")
    return path


def evenly_spaced(values: list[int], count: int) -> list[int]:
    """Select up to ``count`` ordered values, including both ends when possible."""
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in dict.fromkeys(positions.tolist())]


def configured_episode_ids(config: EvalConfig) -> list[int]:
    """Resolve task-exact episode IDs, optionally intersected with an explicit list."""
    if config.task_episode_ids:
        whitelist = set(config.episode_ids) if config.episode_ids else None
        selected: list[int] = []
        for task_episodes in config.task_episode_ids:
            candidates = [
                episode
                for episode in task_episodes
                if whitelist is None or episode in whitelist
            ]
            selected.extend(candidates[: config.episodes_per_task])
        return selected
    return list(config.episode_ids)


def select_sample_indices(dataset: SkillVLADataset, config: EvalConfig) -> list[int]:
    columns = dataset.hf_dataset.select_columns(
        ["episode_index", "frame_index", "skill_index", "skill_sequence_len"]
    ).with_format("numpy")[:]
    episode = np.asarray(columns["episode_index"]).reshape(-1).astype(int)
    frame = np.asarray(columns["frame_index"]).reshape(-1).astype(int)
    skill = np.asarray(columns["skill_index"]).reshape(-1).astype(int)
    sequence_len = np.asarray(columns["skill_sequence_len"]).reshape(-1).astype(int)
    available_episodes = sorted(set(episode.tolist()))
    wanted_episodes = configured_episode_ids(config) or available_episodes[:1]
    allowed = np.isin(episode, np.asarray(wanted_episodes))
    selected: list[int] = []
    for episode_id in wanted_episodes:
        valid = np.flatnonzero(
            allowed & (episode == episode_id) & (skill < sequence_len - 1)
        )
        skill_ids = evenly_spaced(sorted(set(skill[valid].tolist())), config.skills_per_episode)
        for skill_id in skill_ids:
            rows = valid[skill[valid] == skill_id]
            ordered = rows[np.argsort(frame[rows])].tolist()
            selected.extend(evenly_spaced(ordered, config.frames_per_skill))
    if not selected:
        raise RuntimeError(
            "No valid samples found for "
            f"tasks={list(config.task_ids)}, episodes={list(config.episode_ids)}; "
            f"available episodes={available_episodes[:20]}."
        )
    return selected[: config.max_samples]


def normalize_and_pad_state(raw_state: torch.Tensor, stats: dict, width: int) -> torch.Tensor:
    q01 = torch.as_tensor(stats["q01"], dtype=torch.float32, device=raw_state.device)
    q99 = torch.as_tensor(stats["q99"], dtype=torch.float32, device=raw_state.device)
    state = 2.0 * (raw_state.float() - q01) / (q99 - q01) - 1.0
    return pad_vector(state, width)


def _image_array(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    return image.clamp(0, 1).numpy()


def _probability_map(logits: torch.Tensor, grid: int) -> np.ndarray:
    return logits.float().softmax(dim=-1).reshape(grid, grid).cpu().numpy()


def _contrast_map(values: torch.Tensor, grid: int) -> np.ndarray:
    """Min-max normalize one raw diagnostic map for display only."""
    values = values.detach().float().cpu()
    values = values - values.min()
    values = values / values.max().clamp_min(1e-12)
    return values.reshape(grid, grid).numpy()


def _draw_overlay(
    axis,
    image: np.ndarray,
    heatmap: np.ndarray | None,
    *,
    alpha: float,
    target_cell: int,
    grid: int,
    title: str,
) -> None:
    height, width = image.shape[:2]
    axis.imshow(image)
    if heatmap is not None:
        tensor = torch.from_numpy(heatmap)[None, None]
        resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        axis.imshow(resized[0, 0].numpy(), cmap="magma", alpha=alpha, vmin=0)
    if target_cell >= 0:
        row, column = divmod(target_cell, grid)
        axis.add_patch(
            plt.Rectangle(
                (column * width / grid, row * height / grid),
                width / grid,
                height / grid,
                fill=False,
                edgecolor="#00ff80",
                linewidth=2.0,
            )
        )
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def save_panel(
    path: Path,
    image: np.ndarray,
    pooled_logits: torch.Tensor,
    per_query_logits: torch.Tensor,
    pooling_weights: torch.Tensor,
    contribution_rms: torch.Tensor,
    *,
    top_n: int,
    alpha: float,
    target_cell: int,
) -> list[int]:
    patches = int(pooled_logits.numel())
    grid = int(round(math.sqrt(patches)))
    if grid * grid != patches:
        raise ValueError(f"Alignment head returned a non-square patch count: {patches}")
    top_indices = torch.topk(contribution_rms, k=min(top_n, contribution_rms.numel())).indices.tolist()
    panels = 2 + len(top_indices)
    columns = min(5, panels)
    rows = math.ceil(panels / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.35 * columns, 3.55 * rows), squeeze=False)
    flat = axes.reshape(-1)
    _draw_overlay(
        flat[0], image, None, alpha=alpha, target_cell=target_cell, grid=grid,
        title="wrist image + GT patch",
    )
    predicted = int(pooled_logits.argmax())
    _draw_overlay(
        flat[1], image, _probability_map(pooled_logits, grid), alpha=alpha,
        target_cell=target_cell, grid=grid,
        title=f"pooled alignment (pred={predicted})",
    )
    total_contribution = float(contribution_rms.sum().clamp_min(1e-12))
    for axis, query_index in zip(flat[2:], top_indices, strict=False):
        weight = float(pooling_weights[query_index])
        contribution = float(contribution_rms[query_index])
        _draw_overlay(
            axis,
            image,
            _probability_map(per_query_logits[query_index], grid),
            alpha=alpha,
            target_cell=target_cell,
            grid=grid,
            title=(
                f"query {query_index} | pool {weight:.3f}\n"
                f"spatial contribution {100.0 * contribution / total_contribution:.1f}%"
            ),
        )
    for axis in flat[panels:]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return top_indices


def save_action_panel(
    path: Path,
    image: np.ndarray,
    action_attention: torch.Tensor,
    gradient_saliency: torch.Tensor,
    *,
    alpha: float,
    target_cell: int,
    attention_patch_mass: torch.Tensor | None = None,
) -> None:
    """Draw direct attention rollout and output-gradient maps for every timestep."""
    if action_attention.shape != gradient_saliency.shape or action_attention.ndim != 2:
        raise ValueError(
            "Action attention and gradient saliency must have matching [T,P] shapes, got "
            f"{tuple(action_attention.shape)} and {tuple(gradient_saliency.shape)}."
        )
    timesteps, patches = map(int, action_attention.shape)
    grid = int(round(math.sqrt(patches)))
    if grid * grid != patches:
        raise ValueError(f"Action diagnostics returned a non-square patch count: {patches}")
    columns = min(5, timesteps)
    blocks = math.ceil(timesteps / columns)
    figure, axes = plt.subplots(
        blocks * 2,
        columns,
        figsize=(3.35 * columns, 3.45 * blocks * 2),
        squeeze=False,
    )
    for block in range(blocks):
        for column in range(columns):
            timestep = block * columns + column
            attention_axis = axes[block * 2, column]
            gradient_axis = axes[block * 2 + 1, column]
            if timestep >= timesteps:
                attention_axis.axis("off")
                gradient_axis.axis("off")
                continue
            _draw_overlay(
                attention_axis,
                image,
                _contrast_map(action_attention[timestep], grid),
                alpha=alpha,
                target_cell=target_cell,
                grid=grid,
                title=(
                    f"action t={timestep} · attention rollout\n"
                    f"patch mass={float(attention_patch_mass[timestep]):.3f}"
                    if attention_patch_mass is not None
                    else f"action t={timestep} · attention rollout"
                ),
            )
            _draw_overlay(
                gradient_axis,
                image,
                _contrast_map(gradient_saliency[timestep], grid),
                alpha=alpha,
                target_cell=target_cell,
                grid=grid,
                title=f"action t={timestep} · output gradient",
            )
    figure.suptitle(
        "Top: Action→bottleneck→vision MHA · Bottom: ∂||action output||₂/∂vision patch",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _load_policy(config: EvalConfig, device: torch.device) -> SkillExpertPolicy:
    policy_config = PreTrainedConfig.from_pretrained(config.checkpoint, local_files_only=True)
    policy_config.device = str(device)
    policy_config.dino_model_path = str(config.dino_model_path)
    policy_config.gradient_checkpointing = False
    policy_config.compile_model = False
    policy = SkillExpertPolicy.from_pretrained(
        config.checkpoint, config=policy_config, local_files_only=True
    )
    policy.eval()
    if not hasattr(policy.model, "wrist_patch_alignment_diagnostics"):
        raise TypeError(f"{config.architecture_label} has no wrist alignment diagnostic head.")
    if config.action_maps and not all(
        hasattr(policy.model, name)
        for name in (
            "action_vision_attention_diagnostics",
            "action_vision_gradient_saliency",
        )
    ):
        raise TypeError(
            f"{config.architecture_label} does not expose action-map diagnostics."
        )
    return policy


def _dataset(config: EvalConfig) -> SkillVLADataset:
    episodes = configured_episode_ids(config)
    return SkillVLADataset(
        config.repo_id,
        root=config.dataset_dir,
        episodes=episodes or None,
        video_backend=config.video_backend,
        jitter_pmax=0,
        jitter_early_start_pmax=0,
        jitter_late_start_pmax=0,
        jitter_early_end_pmax=0,
        jitter_late_end_pmax=0,
        include_predictor_start_inputs=True,
        include_skill_end_state_target=True,
    )


def run(config: EvalConfig) -> list[dict]:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("attention_eval_config.yaml requests CUDA, but no GPU is visible.")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    data_dir = output_dir / "data"
    images_dir.mkdir(exist_ok=True)
    if config.save_all_query_arrays:
        data_dir.mkdir(exist_ok=True)
    dataset = _dataset(config)
    indices = select_sample_indices(dataset, config)
    policy = _load_policy(config, device)
    stats = json.loads((config.dataset_dir / "meta" / "stats.json").read_text())["observation.state"]
    exact_tasks = {
        episode: (task_id, name)
        for task_id, name, episodes in zip(
            config.task_ids, config.task_names, config.task_episode_ids, strict=True
        )
        for episode in episodes
    }
    records: list[dict] = []
    real_action_dim = int(policy.config.output_features[ACTION].shape[0])

    for ordinal, dataset_index in enumerate(indices):
        item = dataset[dataset_index]
        raw_state = item["observation.state"].float().unsqueeze(0).to(device)
        state = normalize_and_pad_state(raw_state, stats, policy.config.max_state_dim)
        skill_code = item[SKILL_CODE].reshape(1).long().to(device)
        start_state = item[SKILL_START_STATE].float().unsqueeze(0).to(device)
        end_state = item[SKILL_END_STATE].float().unsqueeze(0).to(device)
        batch = {
            "observation.state": state,
            CAM_WRIST: item[CAM_WRIST].unsqueeze(0).to(device),
            SKILL_CODE: skill_code,
            SKILL_START_STATE: start_state,
            SKILL_END_STATE: end_state,
            SKILL_END_STATE_VALID: item[SKILL_END_STATE_VALID].reshape(1).to(device),
        }
        with torch.no_grad():
            images = policy._collect_images(batch)
            goal = policy._skill_delta_goal(batch)
            condition_tokens = policy.model._condition_tokens(
                images, batch_size=1, skill_code=skill_code
            )
            condition_state = policy.model._project_condition_state(
                state, None, skill_code, goal
            )
            diagnostics = policy.model.wrist_patch_alignment_diagnostics(
                condition_tokens, condition_state
            )
        pooled = diagnostics["pooled_logits"][0].cpu()
        weights = diagnostics["pooling_weights"][0].cpu()
        per_query = diagnostics["per_query_logits"][0].cpu()
        weighted = diagnostics["weighted_logits"][0].cpu()
        contribution = diagnostics["contribution_rms"][0].cpu()
        grid = int(round(math.sqrt(pooled.numel())))
        dino_size = int(policy.config.dino_image_size)
        cell, pixel, valid = patch_labels(
            raw_state[:, :3], raw_state[:, 3:6], end_state[:, :3], WristCamera(),
            grid=grid, height=dino_size, width=dino_size,
        )
        target_cell = int(cell.item())
        episode_id = int(torch.as_tensor(item["episode_index"]).item())
        frame_id = int(torch.as_tensor(item["frame_index"]).item())
        skill_id = int(torch.as_tensor(item["skill_index"]).item())
        task_id = int(torch.as_tensor(item["task_index"]).item())
        benchmark_task_id, task_name = exact_tasks.get(
            episode_id, (None, f"dataset task {task_id}")
        )
        stem = f"{ordinal:03d}_ep{episode_id:04d}_skill{skill_id:02d}_frame{frame_id:04d}"
        panel_path = images_dir / f"{stem}.png"
        top_indices = save_panel(
            panel_path,
            _image_array(item[CAM_WRIST]),
            pooled,
            per_query,
            weights,
            contribution,
            top_n=config.top_n,
            alpha=config.overlay_alpha,
            target_cell=target_cell,
        )
        action_panel_path: Path | None = None
        action_attention: torch.Tensor | None = None
        action_attention_patch_mass: torch.Tensor | None = None
        action_cls_attention: torch.Tensor | None = None
        gradient_saliency: torch.Tensor | None = None
        gradient_cls_saliency: torch.Tensor | None = None
        action_velocity: torch.Tensor | None = None
        action_to_latent: torch.Tensor | None = None
        bridge_layers: torch.Tensor | None = None
        bridge_gate_weights: torch.Tensor | None = None
        if config.action_maps:
            attention_diagnostics = policy.model.action_vision_attention_diagnostics(
                condition_tokens,
                state,
                skill_code,
                goal,
                probe_time=config.action_probe_time,
            )
            with torch.enable_grad():
                gradient_diagnostics = policy.model.action_vision_gradient_saliency(
                    condition_tokens,
                    state,
                    skill_code,
                    goal,
                    action_dims=real_action_dim,
                    probe_time=config.action_probe_time,
                )
            patches = int(pooled.numel())
            full_action_attention = attention_diagnostics["condition_attention"][0].cpu()
            action_cls_attention = full_action_attention[:, 0]
            action_attention = full_action_attention[:, 1 : 1 + patches]
            action_attention_patch_mass = action_attention.sum(dim=-1)
            action_attention = action_attention / action_attention_patch_mass[
                :, None
            ].clamp_min(1e-12)
            full_gradient_saliency = gradient_diagnostics[
                "condition_token_saliency"
            ][0].cpu()
            gradient_cls_saliency = full_gradient_saliency[:, 0]
            gradient_saliency = full_gradient_saliency[:, 1 : 1 + patches]
            action_velocity = gradient_diagnostics["velocity"][0, :, :real_action_dim].cpu()
            action_to_latent = attention_diagnostics["action_to_latent"][0].cpu()
            bridge_layers = attention_diagnostics["bridge_layer_indices"].cpu()
            bridge_gate_weights = attention_diagnostics["bridge_gate_weights"].cpu()
            action_panel_path = images_dir / f"{stem}_action.png"
            save_action_panel(
                action_panel_path,
                _image_array(item[CAM_WRIST]),
                action_attention,
                gradient_saliency,
                alpha=config.overlay_alpha,
                target_cell=target_cell,
                attention_patch_mass=action_attention_patch_mass,
            )
        if config.save_all_query_arrays:
            arrays: dict[str, np.ndarray | np.generic] = {
                "pooled_logits": pooled.numpy(),
                "pooling_weights": weights.numpy(),
                "per_query_logits": per_query.numpy(),
                "weighted_logits": weighted.numpy(),
                "contribution_rms": contribution.numpy(),
                "target_cell": np.int64(target_cell),
                "target_pixel": pixel[0].cpu().numpy(),
                "target_valid": np.bool_(valid.item()),
                "top_query_indices": np.asarray(top_indices, dtype=np.int64),
            }
            if action_attention is not None and gradient_saliency is not None:
                arrays.update({
                    "action_attention_rollout": action_attention.numpy(),
                    "action_attention_patch_mass": action_attention_patch_mass.numpy(),
                    "action_cls_attention": action_cls_attention.numpy(),
                    "action_gradient_saliency": gradient_saliency.numpy(),
                    "action_gradient_cls_saliency": gradient_cls_saliency.numpy(),
                    "action_velocity": action_velocity.numpy(),
                    "action_to_latent": action_to_latent.numpy(),
                    "bridge_layer_indices": bridge_layers.numpy(),
                    "bridge_gate_weights": bridge_gate_weights.numpy(),
                    "action_probe_time": np.float32(config.action_probe_time),
                })
            np.savez_compressed(data_dir / f"{stem}.npz", **arrays)
        predicted = int(pooled.argmax())
        distance = None
        if target_cell >= 0:
            distance = max(
                abs(predicted // grid - target_cell // grid),
                abs(predicted % grid - target_cell % grid),
            )
        record = {
            "panel": panel_path.relative_to(output_dir).as_posix(),
            "action_panel": (
                action_panel_path.relative_to(output_dir).as_posix()
                if action_panel_path is not None else None
            ),
            "data": (
                (data_dir / f"{stem}.npz").relative_to(output_dir).as_posix()
                if config.save_all_query_arrays else None
            ),
            "episode": episode_id,
            "task_id": benchmark_task_id,
            "dataset_task_id": task_id,
            "task": task_name,
            "skill_index": skill_id,
            "skill_code": int(skill_code.item()),
            "frame": frame_id,
            "target_cell": target_cell,
            "target_valid": bool(valid.item()),
            "predicted_cell": predicted,
            "cell_distance": distance,
            "top_query_indices": top_indices,
        }
        records.append(record)
        print(f"[{ordinal + 1}/{len(indices)}] {record['panel']}", flush=True)

    manifest = {
        "architecture": config.architecture_label,
        "checkpoint": str(config.checkpoint),
        "dataset": str(config.dataset_dir),
        "selected_tasks": [
            {
                "task_id": task_id,
                "dataset_task_id": dataset_task_id,
                "task": name,
            }
            for task_id, dataset_task_id, name in zip(
                config.task_ids, config.dataset_task_ids, config.task_names, strict=True
            )
        ],
        "map_semantics": "wrist alignment-head contribution maps, not raw MHA weights",
        "action_map_semantics": (
            "per-timestep direct Action->bottleneck->vision MHA composition plus "
            "gradient norm of action-output L2 magnitude with respect to vision patch tokens; "
            "both use zero noisy action at the configured flow probe time and bypass the alignment head; "
            "displayed attention is patch-renormalized while raw patch/CLS mass is saved"
            if config.action_maps else None
        ),
        "action_probe_time": config.action_probe_time if config.action_maps else None,
        "ranking": "RMS of each pooling-weighted query component after removing spatial mean",
        "samples": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    gallery = write_gallery(output_dir, manifest)
    print(f"Saved {len(records)} samples and gallery to {gallery}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument(
        "--manifest",
        help="Generate index.html from an existing manifest without running inference.",
    )
    args = parser.parse_args()
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        gallery = write_gallery(
            manifest_path.parent,
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
        print(f"Saved gallery to {gallery}")
    else:
        run(load_eval_config(args.config))


if __name__ == "__main__":
    main()
