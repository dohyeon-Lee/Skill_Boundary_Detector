# pi05 training and evaluation

```text
train_pi05/
├── PT/
│   ├── pi05_PT_config.yaml
│   ├── submit_train.sh
│   ├── src/                    # shared PT/FT config resolver + PT batch script
│   └── eval/                   # PT config, launcher, outputs + shared eval engine
└── FT/
    ├── pi05_FT_config.yaml
    ├── submit_train.sh
    ├── src/                    # FT batch script
    └── eval/                   # FT config, launcher and outputs
```

Run `./submit_train.sh` from `PT/` or `FT/`, and `./submit_eval.sh` from the
corresponding `eval/` folder. FT reuses the config resolver in `PT/src/` and the
evaluation engine in `PT/eval/src/`; evaluation logs and outputs stay in the
calling stage's `eval/` directory.

Training settings, run names, W&B projects and checkpoint directories
(`{outputs_root}/pi05_PT` and `{outputs_root}/pi05_FT`) are unchanged. The former
shared eval outputs have been separated into `PT/eval/outputs/` and
`FT/eval/outputs/`. Historical logs and configuration snapshots are preserved;
unclassified old shared evaluation logs remain under `PT/eval/logs/`.

The shared trainer never uploads checkpoint weights/adapters to W&B, including
when resuming an old config with `wandb.disable_artifact=false`. W&B metrics,
plots/videos, local checkpoint saves and server-to-server archival are unchanged.
This applies at the next trainer launch; it does not change running processes or
delete any previously uploaded artifacts.

## FT normalization statistics probe

Set `training.normalization_stats: ft | pt` in the FT YAML. Default `ft`
uses the grounded FT dataset statistics and preserves existing run names.
Selecting `pt` reads the warm-start checkpoint's saved state/action statistics
and adds `_ptstats` to the run name (before the backend suffix, when present).
Both input normalization and action unnormalization use the selected statistics,
saved in each checkpoint. Grounding is inherited independently from PT. Resume
retains its own FT checkpoint's processors and rejects changing modes; historical
FT checkpoints are treated as `ft`. Existing outputs are not converted.
