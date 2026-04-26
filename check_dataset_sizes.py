"""
check_dataset_sizes.py — libero_dataset / libero_small_dataset 용량 요약

Usage:
  python check_dataset_sizes.py
  python check_dataset_sizes.py --dirs /data2/dohyeon/SBD/libero_dataset /data2/dohyeon/SBD/libero_small_dataset
  python check_dataset_sizes.py --sort size
"""

import argparse
import json
import subprocess
from pathlib import Path


BASE = Path("/data2/dohyeon/SBD")
DEFAULT_DIRS = [
    BASE / "libero_dataset",
    BASE / "libero_small_dataset",
]


def du(path: Path) -> str:
    result = subprocess.run(
        ["du", "-sh", str(path)], capture_output=True, text=True
    )
    return result.stdout.split("\t")[0] if result.returncode == 0 else "?"


def du_bytes(path: Path) -> int:
    result = subprocess.run(
        ["du", "-sb", str(path)], capture_output=True, text=True
    )
    try:
        return int(result.stdout.split("\t")[0])
    except (ValueError, IndexError):
        return 0


def read_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {}
    try:
        with open(info_path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Dataset size summary")
    parser.add_argument("--dirs", nargs="+", type=Path, default=DEFAULT_DIRS,
                        help="Dataset root directories to scan")
    parser.add_argument("--sort", choices=["name", "size"], default="name",
                        help="Sort order (default: name)")
    args = parser.parse_args()

    rows = []
    for root in args.dirs:
        if not root.exists():
            print(f"[WARN] not found: {root}")
            continue

        datasets = sorted(p for p in root.iterdir() if p.is_dir())
        for ds in datasets:
            info = read_info(ds)
            size_h = du(ds)
            size_b = du_bytes(ds)
            rows.append({
                "root":     root.name,
                "name":     ds.name,
                "size_h":   size_h,
                "size_b":   size_b,
                "episodes": info.get("total_episodes", "-"),
                "frames":   info.get("total_frames", "-"),
                "tasks":    info.get("total_tasks", "-"),
            })

    if args.sort == "size":
        rows.sort(key=lambda r: r["size_b"], reverse=True)

    # ── print ──────────────────────────────────────────────────────────────────
    print(f"\n{'SIZE':>8}  {'EPISODES':>9}  {'FRAMES':>9}  {'TASKS':>5}  DATASET")
    print("-" * 80)

    current_root = None
    for r in rows:
        if r["root"] != current_root:
            print(f"\n[{r['root']}]")
            current_root = r["root"]
        ep  = f"{r['episodes']:>9}" if isinstance(r['episodes'], int) else f"{'?':>9}"
        fr  = f"{r['frames']:>9,}" if isinstance(r['frames'], int) else f"{'?':>9}"
        tsk = f"{r['tasks']:>5}"   if isinstance(r['tasks'], int)    else f"{'?':>5}"
        print(f"{r['size_h']:>8}  {ep}  {fr}  {tsk}  {r['name']}")

    print()
    print(f"Total datasets: {len(rows)}")

    # ── disk usage per root ────────────────────────────────────────────────────
    print()
    for root in args.dirs:
        if root.exists():
            print(f"{du(root):>8}  {root}")


if __name__ == "__main__":
    main()
