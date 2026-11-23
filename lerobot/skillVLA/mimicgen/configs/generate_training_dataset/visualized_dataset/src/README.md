# Coffee dataset gallery

Edit `../visualized_dataset_config.yaml`, then run
`./visualize_training_dataset.sh` in the parent folder. The existing LIBERO
LeRobot-v3 gallery handles both top and wrist views, episode sampling and
browser-compatible H.264/VP9 preview clips. Python files stay under `src/`.

`--dry-run` prints resolved paths; `--list-tasks` lists language tasks and episode
counts. The converted dataset must exist before generating a gallery. No GPU
or environment activation is needed; FFmpeg is resolved automatically.

Default HTML: `outputs/coffee_d0/index.html` under the visualization folder.
