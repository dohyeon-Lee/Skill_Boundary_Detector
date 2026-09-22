#!/usr/bin/env bash
# Shared per-job runtime storage. Source this after PROJECT_ROOT has been
# resolved from the module config, before importing Hugging Face/LeRobot.

: "${PROJECT_ROOT:?PROJECT_ROOT must be set before sourcing runtime_env.sh}"

# `datasets` can create parquet materialization and lock files even when the
# dataset itself is local.  Keep every Hugging Face cache inside the selected
# project root instead of inheriting a machine-specific $HOME cache.
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"

# W&B's transient cache/data can otherwise default to $HOME (and, on some
# clusters, an unrelated or quota-limited scratch mount).
export WANDB_CACHE_DIR="${PROJECT_ROOT}/.cache/wandb/cache"
export WANDB_DATA_DIR="${PROJECT_ROOT}/.cache/wandb/data"
export WANDB_DISABLE_SYMLINKS=true
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"

mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" \
         "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${MPLCONFIGDIR}"
