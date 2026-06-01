#!/usr/bin/env python3
"""Download original LIBERO HDF5 folders from the Hugging Face mirror.

The upstream LIBERO downloader asks Hugging Face for ``libero_100/*``, but the
mirror stores LIBERO-100 as two folders: ``libero_90/`` and ``libero_10/``.
This helper downloads the actual folder names.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"

DATASET_FOLDERS = {
    "libero_object": ["libero_object"],
    "libero_goal": ["libero_goal"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero_100": ["libero_90", "libero_10"],
    "all": ["libero_object", "libero_goal", "libero_spatial", "libero_90", "libero_10"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--download-dir", required=True, type=Path)
    p.add_argument(
        "--datasets",
        default="libero_100",
        choices=sorted(DATASET_FOLDERS),
        help="`libero_100` downloads the original libero_90 and libero_10 folders.",
    )
    p.add_argument("--force-download", action="store_true")
    p.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel HF download workers. Lower this to 1 or 2 if the Xet/CAS download stalls.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=20,
        help="Retry snapshot_download this many times. HF downloads are resumable.",
    )
    p.add_argument(
        "--retry-sleep",
        type=int,
        default=30,
        help="Seconds to wait between retries.",
    )
    return p.parse_args()


def count_hdf5(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.hdf5")) + sum(1 for _ in root.rglob("*.h5"))


def main() -> None:
    args = parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    folders = DATASET_FOLDERS[args.datasets]
    patterns = [f"{folder}/*" for folder in folders]

    print("Download original LIBERO from Hugging Face")
    print(f"  repo     : {HF_REPO_ID}")
    print(f"  dataset  : {args.datasets}")
    print(f"  folders  : {', '.join(folders)}")
    print(f"  output   : {args.download_dir}")
    print(f"  workers  : {args.max_workers}")
    print(f"  retries  : {args.retries}")

    last_exc: Exception | None = None
    for attempt in range(1, int(args.retries) + 1):
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                local_dir=str(args.download_dir),
                allow_patterns=patterns,
                force_download=bool(args.force_download),
                max_workers=int(args.max_workers),
            )
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"\n[warn] HF download attempt {attempt}/{args.retries} failed: {type(exc).__name__}: {exc}")
            if attempt >= int(args.retries):
                break
            print(f"[warn] Sleeping {args.retry_sleep}s, then resuming download...")
            time.sleep(int(args.retry_sleep))

    if last_exc is not None:
        raise last_exc

    print("\nDownloaded folders:")
    ok = True
    for folder in folders:
        folder_path = args.download_dir / folder
        n = count_hdf5(folder_path) if folder_path.exists() else 0
        print(f"  {folder_path}: {n} hdf5/h5 files")
        ok = ok and n > 0

    if not ok:
        raise SystemExit("Download finished, but at least one expected folder has no HDF5/H5 files.")


if __name__ == "__main__":
    main()
