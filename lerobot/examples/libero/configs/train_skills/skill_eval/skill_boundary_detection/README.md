# Skill boundary comparison

Run these commands from this folder, with the updated code on both machines:

```bash
bash submit.sh --prepare-runpod  # Source server: publish missing inputs through HF Desk
bash submit.sh                   # RunPod: download, verify, and evaluate
```

On a Slurm server, `bash submit.sh` keeps the existing submission behavior. Use `--download-only` on RunPod to prepare inputs without starting evaluation. YAML changes require another `--prepare-runpod`; code changes require a separate code update on RunPod.

- Both conditions use the same DP and 7D IID Gaussian probes. PCA is used to describe denoised outputs before GMM; only the second condition excludes gripper from this descriptor.
- The paired probe recipe is FSQ's 96 IID scheduler-Gaussian samples without a mean probe, `pred_original_sample`, step 7, and replan interval 5. Results use the `fsq_sbd_gripper_comparison_fullcal_tasks0_9_20ep` output name.
- This configuration follows the FSQ reference skillset (`score_power=2`, covariance-gated cosine, diagonal GMM, 24-frame terminal mask). The previously generated SkillVLA PT/FT skillsets use a different detector (`score_power=1`, legacy cosine, full GMM, no terminal mask); matching PCA flow alone does not make their boundaries equivalent.
- Each condition independently calibrates its global-mean threshold over all 3,921 skillset episodes. Reports contain only the selected 10 tasks × 20 episodes.
- `threshold_statistics.json` records each threshold, valid point count, and calibration provenance. Partial skillset copies fail rather than calibrating on a smaller subset.
- Resume/report-only runs reuse caches and preserve the full calibration set. The selected report subset never determines the global threshold.
- HF keeps the existing private dataset paths and public DP checkpoint path. Only the report subset's video chunks are required. Completed runs include `index.html`, `summary.json`, `threshold_statistics.json`, and `evaluation_complete.json`; RunPod Desk and the result backup/stop helper recognize these outputs.
