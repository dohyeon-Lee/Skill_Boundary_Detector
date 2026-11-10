# Skill boundary comparison

Run these commands from this folder, with the updated code on both machines:

```bash
bash submit.sh --prepare-runpod  # Source server: publish missing inputs through HF Desk
bash submit.sh                   # RunPod: download, verify, and evaluate
```

On a Slurm server, `bash submit.sh` keeps the existing submission behavior. Use `--download-only` on RunPod to prepare inputs without starting evaluation. YAML changes require another `--prepare-runpod`; code changes require a separate code update on RunPod.

- Both conditions use the same DP and 7D IID Gaussian probes. PCA is used to describe denoised outputs before GMM; only the second condition excludes gripper from this descriptor.
- Each condition independently calibrates its global-mean threshold over all 3,921 skillset episodes. Reports contain only the selected 10 tasks × 20 episodes.
- `threshold_statistics.json` records each threshold, valid point count, and calibration provenance. Partial skillset copies fail rather than calibrating on a smaller subset.
- Resume/report-only runs reuse caches and preserve the full calibration set. The selected report subset never determines the global threshold.
- HF keeps the existing private dataset paths and public DP checkpoint path. Only the report subset's video chunks are required. Completed runs include `index.html`, `summary.json`, `threshold_statistics.json`, and `evaluation_complete.json`; RunPod Desk and the result backup/stop helper recognize these outputs.
