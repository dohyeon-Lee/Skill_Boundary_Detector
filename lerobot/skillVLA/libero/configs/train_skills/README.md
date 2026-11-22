# Skill tokenizer training and evaluation

Python implementation files live under their owning component, not directly in
`skillVLA/libero/` or `configs/`.

```text
train_skills/
├── DP/                         # diffusion policy training
│   └── eval/                   # standalone SBD partial builds, cached CPU reports and comparisons
├── FSQ/
│   └── src/                    # FSQ.py, train_FSQ.py, RGB/DINO caches, skills_bundle.py
├── build_data/
│   └── src/                    # segmentation and skill dataset builders
├── skill_eval/
│   ├── src/                    # shared skill galleries and boundary rendering
│   ├── fsq_codebook/src/       # codebook analysis, decoder_eval, DINO patch inspection
│   └── eef_projection/src/
└── src/                        # shared configuration resolver
```

The existing component submit scripts and YAMLs remain the user-facing entry
points. Dataset/checkpoint paths, model settings and the checkpoint format are
unchanged. In particular, `FSQ.py` is still imported as `FSQ` from its new source
directory so saved `FSQ.SplineFSQAEConfig` objects remain loadable; no root-level
compatibility shim is required.

Dataset conversion/inspection tools are under
`generate_training_dataset/tools/`, and the quantizer/spline tests are under
`lerobot/tests/examples/`.

`DP/dp_config.yaml` uses the same history encoder choices as ABC:
`flat`, `GRU_iid`, `GRU_epi`, `transformer`, and `transformer_cls`.
Both GRU variants use two layers and consume current state plus carried hidden
at inference/SBD time, resetting at episode boundaries (`dp_vision: state`).
IID training starts each random history window with zero hidden. Episodic
training carries detached hidden across chronological `dp_n_obs_steps` chunks
and keeps full batches with persistent DataLoader workers. It supports offline
LeRobot datasets, including episode subsets, in single-process training.
Transformer uses two layers/four heads. `dp_unet_size: small|base` applies to
every encoder; run names distinguish `_gru_iid`/`_gru_epi` and `_small` (base
has no size suffix), without a layer-count suffix. Legacy `gru` selects IID.
