# SkillVLA workflows

The project-specific pipelines live here, alongside the upstream `examples/` directory:

```text
lerobot/
├── examples/            # Upstream LeRobot examples
└── skillVLA/
    ├── abc/configs/    # ABC datasets, simulator, training, and evaluation
    ├── libero/configs/ # LIBERO datasets, training, and evaluation; shared helpers
    └── analysis/       # ABC/LIBERO analysis reports and media
```

Each environment keeps its own `configs/global_config.yaml` and `configs/servers/`
overlays. Run the existing `submit_*.sh` commands from their folders here.
`setup_env.sh` and `hf_sync.sh` at the repository root use this layout too.

Dataset and checkpoint roots (`dataset_*`, `outputs_*`, `models/`) stay at the
repository root or their configured storage volumes. Relative video links keep
the same directory depth. FSQ checkpoint module names are unchanged.

Previously published HF datasets and checkpoints retain their existing remote
paths. Evaluation manifests and report downloads accept the old
`lerobot/examples/{abc,libero,analysis}` prefixes and resolve them locally under
`lerobot/skillVLA/`.
