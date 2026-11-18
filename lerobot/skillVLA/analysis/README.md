# SkillVLA analysis results

Cross-environment analysis outputs are collected here while the executable
pipelines remain under `skillVLA/libero` and `skillVLA/abc`.

```text
skill_boundary_detection/
├── libero/<dataset>/<run>/
└── abc/<dataset>/<run>/
```

The generated reports and caches are local artifacts and are ignored by Git.

## Opening reports

Open the `index.html` inside the result folder you want to inspect:

- LIBERO reports: `skill_boundary_detection/libero/<dataset>/<run>/index.html`
- ABC reports: `skill_boundary_detection/abc/<dataset>/<run>/index.html`
- Comparisons: `skill_boundary_detection/together/<comparison>/index.html`

ABC results can have one additional data-source folder (`grab_clutter`,
`put_relative`, or `group_singletask_1k_eps`) below the dataset name.
Media and descriptor caches stay next to their reports, keeping existing
relative links usable after relocation. Dataset and checkpoint roots stay at
the project root; no data download or model rebuild is needed for this move.
