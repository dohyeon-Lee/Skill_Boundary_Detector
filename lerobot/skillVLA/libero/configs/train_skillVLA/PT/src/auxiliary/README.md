# Shared auxiliary training backend

Predictor PT/FT launchers for LIBERO, ABC, MimicGen and DROID use this backend.
VSA FT imports dataset/checkpoint validation helpers from `auxiliary_train_config.py`;
VSA multi-model launchers reuse `run_predictor_group.py`.

- `auxiliary_train_config.py`: config resolution and saved auxiliary contracts.
- `submit_auxiliary.sh`: snapshot, preflight and local/Slurm submission.
- `train.sbatch`: dataset staging and trainer/resume execution.
- `run_predictor_group.py`: independent workers sharing one allocation.

The default config is `../../Predictor/predictor_train_config.yaml`. Component
wrappers set `SKILL_AUX_TRAIN_CONFIG` and `SKILL_AUX_SUBMIT_DIR` for other domains
and FT. Historical Terminator contract helpers remain available for saved-model
compatibility; standalone Terminator and Joint training entries are retired.
