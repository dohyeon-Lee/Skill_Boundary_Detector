# DrawSVG dataset conversion

This module imports completed episodes directly from the sibling
`drawsvg_pipeline/generated` directory and writes one canonical LeRobot v3
dataset under the `dataset_root` configured in `configs/global_config.yaml`.
It does not copy/download raw data and does not run simulation rollouts.

Edit only `include_groups` and `output_name` in
`drawsvg_dataset_config.yaml`, then submit:

```bash
./submit_build_drawsvg.sh
```

For a direct foreground build on an allocated node:

```bash
./build_drawsvg_dataset.sh
```

The preflight requires every selected `episodeNNN` to contain a successful,
20 Hz schema-v2 `vla_episode.npz` with both top and wrist RGB. It fails before
creating the output if any selected episode is incomplete.

Set `DRAWSVG_SOURCE_ROOT` only when the pipeline is not the normal sibling of
the project. Existing outputs are protected; `FORCE=1` explicitly replaces the
configured output directory.
