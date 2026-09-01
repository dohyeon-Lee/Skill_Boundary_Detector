# Oracle init-state matching

This folder builds the dataset-level `eval_init_states.npz` used by exact
LIBERO/Stage-1 evaluation. The map is independent of an FSQ run and is shared
by all SkillVLA datasets derived from the same source dataset.

1. Edit `oracle_matching_config.yaml`.
2. Run `./submit_oracle_matching.sh`.
3. Inspect the Slurm log under `../logs/` and, for LangGap, the generated
   `eval_init_states.diagnostics.json` beside the NPZ.

`dataset_root` is local to this YAML. This is intentional: a LangGap dataset
may live under `dataset` while the global LIBERO pipeline uses
`dataset_filtered`. `auto` is also supported when only one dataset root
contains the requested source.

For split LangGap datasets, `task_ids` always refers to local dataset IDs. The
builder reads `meta/langgap_split.json` (falling back to provenance in
`meta/info.json`) and maps those IDs back to the original 0..55 LangGap task
space before selecting a LIBERO benchmark task.

LangGap does not include the original per-frame MuJoCo state/HDF5 demo. The
generated NPZ therefore gives an exact **episode start**. Evaluating a skill
that begins in the middle of an episode still requires replaying the dataset
actions from that start to the selected skill boundary; that runtime replay is
separate from this matching job.

Python builders, the Slurm worker, and this documentation live under `src/` so
the root exposes only the two canonical shell entry points and YAML
configuration. The same entry points support both LangGap and LIBERO; `mode:
auto` selects the builder from `source_dataset`.
