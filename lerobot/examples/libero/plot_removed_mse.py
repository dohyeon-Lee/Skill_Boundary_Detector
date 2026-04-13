"""
VAE에서 제거된 episode들의 MSE 그래프를 일괄 저장하는 스크립트.

Usage:
    python examples/libero/plot_removed_mse.py \
        --replay_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay_libero90 \
        --latents_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/vae_90/vae90_lat8_hid128/skill_vae_latents_epoch0100.npz \
        --output_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/removed_mse_plots \
        --max_plots 50 \
        --wandb_project DEBUG_skill_90
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    replay_dir: str
    """replay_demo.py 출력 루트 디렉토리 (task* 서브폴더 포함)"""
    latents_path: str
    """skill_vae_latents*.npz — 여기 없는 episode_id가 제거된 것"""
    output_dir: str
    """MSE 그래프 저장 위치"""
    max_plots: int = 0
    """최대 저장 개수. 0 = 전부"""
    wandb_project: str | None = None
    """wandb project 이름. 지정하면 그래프를 wandb에 업로드"""
    wandb_run_name: str = "removed_mse"


def main(args: Args) -> None:
    # 제거된 episode_id 목록
    lat = np.load(args.latents_path)
    latent_ep_ids = set(lat["episode_id"].tolist())

    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모든 replay_results.json 수집
    result_files = sorted(replay_dir.rglob("replay_results.json"))
    print(f"Found {len(result_files)} replay_results.json files")

    removed_records = []
    for rf in result_files:
        results = json.loads(rf.read_text())
        for r in results:
            if r["episode_id"] not in latent_ep_ids and r.get("mse_data") is not None:
                removed_records.append(r)

    print(f"제거된 episode 중 MSE 데이터 있는 것: {len(removed_records)}개")

    if args.max_plots > 0:
        removed_records = removed_records[:args.max_plots]

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={"n_removed": len(removed_records)},
        )

    for r in removed_records:
        ep_id = r["episode_id"]
        lang = r.get("language", "")[:60]
        replan_ts, mse_vals, smoothed_dict_raw, bar_width = r["mse_data"]
        sg_peak_ts = r.get("sg_peak_ts") or []

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(replan_ts, mse_vals, width=bar_width, alpha=0.4, color="mediumpurple", label="raw")

        smoothed_dict = {k: np.array(v) for k, v in smoothed_dict_raw.items()}
        if "savgol" in smoothed_dict:
            sg = smoothed_dict["savgol"]
            sg_mean = float(np.mean(sg))
            ax.plot(replan_ts, sg, color="tab:green", linewidth=1.5, label=f"Savitzky-Golay (mean={sg_mean:.4f})")
            ax.axhline(sg_mean, color="tab:green", linestyle="--", linewidth=1,
                       label=f"SG mean={sg_mean:.4f}")
        if "ma" in smoothed_dict:
            ax.plot(replan_ts, smoothed_dict["ma"], color="tab:orange", linewidth=1.2, label="Moving Avg")

        if sg_peak_ts:
            peak_vals = []
            for pt in sg_peak_ts:
                if pt in replan_ts:
                    idx = replan_ts.index(pt)
                    peak_vals.append(smoothed_dict["savgol"][idx] if "savgol" in smoothed_dict else mse_vals[idx])
                else:
                    peak_vals.append(0)
            ax.scatter(sg_peak_ts, peak_vals, color="red", zorder=5, s=40, label="SG peaks > mean")

        ax.set_title(f"[REMOVED] ep{ep_id}: {lang}", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("MSE xyz")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(replan_ts[::max(1, len(replan_ts)//20)])
        ax.set_xticklabels([str(t) for t in replan_ts[::max(1, len(replan_ts)//20)]], rotation=45, ha="right", fontsize=7)
        fig.tight_layout()

        save_path = output_dir / f"ep{ep_id:05d}_mse.png"
        fig.savefig(str(save_path), dpi=100)

        if wandb_run is not None:
            import wandb
            label = f"ep{ep_id:05d}: {lang[:40]}"
            wandb_run.log({f"removed_mse/{label}": wandb.Image(str(save_path))})

        plt.close(fig)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"저장 완료 → {output_dir}  ({len(removed_records)}개)")


if __name__ == "__main__":
    main(tyro.cli(Args))
