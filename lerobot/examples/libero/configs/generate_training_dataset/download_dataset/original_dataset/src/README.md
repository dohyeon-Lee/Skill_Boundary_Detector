# Original LIBERO Download

This folder downloads the original LIBERO data into a separate directory under
the `project_root` selected in `configs/global_config.yaml`:

```text
{project_root}/libero_original_dataset
```

It intentionally does not write into `libero_dataset/`. The downloaded files are
the original LIBERO demos and still need to be converted to the exact LeRobot
format used by the current `libero_dataset/libero_90` and `libero_dataset/libero_10`.

## Run

Recommended one-shot path (download on the login node, build + stats in Slurm):

```bash
cd {project_root}/lerobot/examples/libero/configs/generate_training_dataset/download_dataset/original_dataset
./submit_build_original.sh
```

Download only, using the stable Hugging Face path:

```bash
cd {project_root}/lerobot/examples/libero/configs/generate_training_dataset/download_dataset/original_dataset

LIBERO_HF_MAX_WORKERS=2 \
LIBERO_ORIGINAL_DATASETS=libero_100 \
./download_original_libero.sh
```

Short form, if the defaults are already correct:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 LIBERO_HF_MAX_WORKERS=1 LIBERO_HF_RETRIES=20 ./download_original_libero.sh
```

## Output

Default output root:

```text
{project_root}/libero_original_dataset
```

For `LIBERO_ORIGINAL_DATASETS=libero_100`, the Hugging Face mirror stores
LIBERO-100 as the two folders below, so the local downloader maps
`libero_100 -> libero_90 + libero_10`:

```text
{project_root}/libero_original_dataset/libero_90/
{project_root}/libero_original_dataset/libero_10/
```

with `.hdf5` or `.h5` demonstration files inside.

The downloader repository is cloned to:

```text
{project_root}/tools/lerobot-libero
```

## Options

```bash
LIBERO_ORIGINAL_DATASETS=libero_100     # default; contains LIBERO-90 and LIBERO-10
LIBERO_ORIGINAL_DATASET_DIR=...         # default: ${PROJECT_ROOT}/libero_original_dataset
LIBERO_DOWNLOAD_TOOLS_DIR=...           # default: ${PROJECT_ROOT}/tools
LIBERO_UPDATE_REPO=1                    # optional: git pull the downloader repo
LIBERO_INSTALL_REPO=0                   # optional: skip pip install -e
LIBERO_USE_HUGGINGFACE=1                # default: use HF mirror instead of expiring original links
LIBERO_HF_MAX_WORKERS=4                 # lower to 1 or 2 if the HF/Xet download stalls
PYTHON_BIN=...                          # default: ${PROJECT_ROOT}/.venv/bin/python
```

If the progress bar stalls for many minutes at `Fetching 100 files`, stop it
with `Ctrl+C` and retry with the stable command above. If the stable command is
working but too slow, try increasing only the worker count:

```bash
HF_HUB_DISABLE_XET=1 LIBERO_HF_MAX_WORKERS=2 ./download_original_libero.sh
```

Hugging Face downloads are resumable; already completed files are reused.
If the process exits with `httpx.ConnectTimeout` or `read operation timed out`,
the script now waits and retries automatically. If it repeats often, lower
`LIBERO_HF_MAX_WORKERS` to `2` or `1`.

The Hugging Face helper also accepts:

```bash
LIBERO_ORIGINAL_DATASETS=libero_90
LIBERO_ORIGINAL_DATASETS=libero_10
LIBERO_ORIGINAL_DATASETS=libero_object
LIBERO_ORIGINAL_DATASETS=libero_goal
LIBERO_ORIGINAL_DATASETS=libero_spatial
LIBERO_ORIGINAL_DATASETS=all
```

## Convert To LeRobot

After download, convert the original HDF5 files into the LeRobot dataset format
used by the current training pipeline. By default, raw HDF5 files are read from
`{project_root}/libero_original_dataset`, and converted datasets are written to
the `dataset_root` selected in `configs/global_config.yaml`.

Edit:

```text
original_dataset_config.yaml
```

`convert_schema_reference` must point to the canonical existing LeRobot dataset whose
`meta/info.json` feature contract the conversion must reproduce. Only schema metadata is
read from this reference; all frames, states, and actions still come from the original HDF5.
Missing references or non-video image features fail before conversion instead of silently
falling back to another storage format.

To build on the current node after downloading:

```bash
cd {project_root}/lerobot/examples/libero/configs/generate_training_dataset/download_dataset/original_dataset
./build_original_dataset.sh
```

The recommended `./submit_build_original.sh` command above performs the download
first and then submits this same build step to Slurm.

The default conversion uses `convert_vcodec: libsvtav1`, matching the canonical
reference's AV1 video-backed LeRobot format. Image size, FPS, and codec must match
the reference exactly; mismatches fail before any output is written.

Smoke test:

```bash
cd {project_root}/lerobot/examples/libero/configs/generate_training_dataset/download_dataset/original_dataset

{project_root}/.venv/bin/python ./src/convert_original_libero_to_lerobot.py \
  --suite libero_10 \
  --output-name libero_10_smoke \
  --max-tasks 1 \
  --max-episodes-per-task 1
```

Full conversion:

```bash
cd {project_root}/lerobot/examples/libero/configs/generate_training_dataset/download_dataset/original_dataset

{project_root}/.venv/bin/python ./src/convert_original_libero_to_lerobot.py \
  --suite libero_90

{project_root}/.venv/bin/python ./src/convert_original_libero_to_lerobot.py \
  --suite libero_10
```

Outputs:

```text
{project_root}/{dataset_root}/libero_90_full_full
{project_root}/{dataset_root}/libero_10_full_full
```

Validate:

```bash
cd {project_root}/lerobot/examples/libero/configs/generate_training_dataset/visualized_dataset
# Set `dataset` in visualized_dataset_config.yaml to the dataset to inspect.
{project_root}/.venv/bin/python inspect_training_dataset.py
```

The converted data uses the same core column names, shapes, delta-action
convention, and gripper `{-1, 1}` convention as the current training datasets.
Images are resized from the original LIBERO `128x128` observations to `256x256`
to match the current local LeRobot datasets.

## Stats

The current local LeRobot v3 writer creates `meta/stats.json` during conversion.
`submit_build_original.sh` then automatically recomputes exact global
non-video quantiles, including:

```text
q01, q10, q50, q90, q99
```

No extra stats command is needed after the Slurm submit workflow. When running
`src/convert_original_libero_to_lerobot.py` directly, the writer stats are already
usable; to reproduce the submit workflow's exact global quantiles, run:

```bash
cd {project_root}/lerobot/examples/libero/configs/generate_training_dataset/download_dataset/original_dataset

{project_root}/.venv/bin/python ./src/ensure_quantile_stats.py \
  --dataset libero_90_full_full

{project_root}/.venv/bin/python ./src/ensure_quantile_stats.py \
  --dataset libero_10_full_full
```

Force recompute:

```bash
{project_root}/.venv/bin/python ./src/ensure_quantile_stats.py \
  --dataset libero_90_full_full \
  --overwrite
```

Once `*_full_full` is validated, it can be copied or renamed to:

```text
{project_root}/{dataset_root}/libero_90
{project_root}/{dataset_root}/libero_10
```

or used directly as `libero_90_full_full` / `libero_10_full_full` from the
currently selected global `dataset_root`.
