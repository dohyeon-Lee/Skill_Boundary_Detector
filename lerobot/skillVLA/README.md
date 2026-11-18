# SkillVLA workflows

The project-specific pipelines live here, alongside the upstream `examples/` directory:

```text
lerobot/
├── examples/            # Upstream LeRobot examples
└── skillVLA/
    ├── servers/        # Shared machine, storage and scheduler profiles
    ├── abc/configs/    # ABC datasets, simulator, training, and evaluation
    ├── libero/configs/ # LIBERO datasets, training, and evaluation; shared helpers
    └── analysis/       # ABC/LIBERO analysis reports and media
```

Each environment keeps its own `configs/global_config.yaml` for dataset/model
defaults. All workflows share the machine overlays in [`servers/`](servers/):
edit a server profile once to apply it to ABC, LIBERO and other folders here.
Submitted jobs use a private profile copy in their config snapshot; changing the
live profile does not change queued or running jobs.
Run the existing `submit_*.sh` commands from their folders here.
`setup_env.sh` and `hf_sync.sh` at the repository root use this layout too.

Dataset and checkpoint roots (`dataset_*`, `outputs_*`, `models/`) stay at the
repository root or their configured storage volumes. Relative video links keep
the same directory depth. FSQ checkpoint module names are unchanged.

Previously published HF datasets and checkpoints retain their existing remote
paths. Evaluation manifests and report downloads accept the old
`lerobot/examples/{abc,libero,analysis}` prefixes and resolve them locally under
`lerobot/skillVLA/`.
