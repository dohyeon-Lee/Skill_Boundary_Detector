# SkillVLA analysis results

Cross-environment analysis outputs are collected here while the executable
pipelines remain under `examples/libero` and `examples/abc`.

```text
skill_boundary_detection/
├── libero/<dataset>/<run>/
└── abc/<dataset>/<run>/
```

The generated reports and caches are local artifacts and are ignored by Git.

## Reports to inspect

- Side-by-side centered 5-GMM comparison: `skill_boundary_detection/together/index.html`
- Side-by-side ABC all-peaks comparison: `skill_boundary_detection/together/all_peaks/index.html`
- Side-by-side ABC power=0.5 comparison: `skill_boundary_detection/together/power0p5_globalmean/index.html`
- Side-by-side within-normalized GMM comparison: `skill_boundary_detection/together/within_normalized_l2/index.html`
- Side-by-side covariance-gated angular-chord comparison: `skill_boundary_detection/together/angular_chord/index.html`
- Side-by-side covariance-gated angular-chord power=1 comparison: `skill_boundary_detection/together/angular_chord_power1/index.html`
- ABC all-peaks SBD: `skill_boundary_detection/abc/abc_sim_all/all_peaks_ck010000/index.html`
- ABC power=0.5 SBD: `skill_boundary_detection/abc/abc_sim_all/power0p5_globalmean_ck010000/index.html`
- ABC power=1 with per-episode motion-start+100 initial mask: `skill_boundary_detection/abc/abc_sim_all/power1_initial_motion_mask100_ck010000/index.html`
- LIBERO standard SBD: `skill_boundary_detection/libero/libero_90_full_full/built_ck010000_tasks0_9_20ep/index.html`
- LIBERO 5-GMM 3-D: `skill_boundary_detection/libero/libero_90_full_full/libero_gmm_3d_tasks0_9_1ep/index.html`
- ABC standard SBD: `skill_boundary_detection/abc/abc_sim_all/built_ck010000/index.html`
- ABC denoising-gain boundaries: `skill_boundary_detection/abc/abc_sim_all/denoising_gain_globalmean_ck010000/index.html`
- ABC 5-GMM 3-D: `skill_boundary_detection/abc/abc_sim_all/abc_gmm_3d_tasks59_82_1ep/index.html`

The two 5-GMM directories also retain their raw per-episode descriptor caches,
so their HTML can be regenerated without another DP inference pass.
