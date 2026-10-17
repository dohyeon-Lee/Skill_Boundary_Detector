# Skill tokenizer training and evaluation

Python implementation files live under their owning component, not directly in
`examples/libero/` or `configs/`.

```text
train_skills/
├── DP/                         # diffusion policy training
├── FSQ/
│   └── src/                    # FSQ.py, train_FSQ.py, RGB/DINO caches, skills_bundle.py
├── build_data/
│   └── src/                    # segmentation and skill dataset builders
├── skill_eval/
│   ├── src/                    # shared skill galleries and boundary rendering
│   ├── fsq_codebook/src/       # codebook analysis, decoder_eval, DINO patch inspection
│   ├── skill_boundary_detection/src/
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
