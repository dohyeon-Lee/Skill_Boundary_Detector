"""Fixed DROID build defaults; editable controls stay in the build YAML."""
import json
from pathlib import Path
import re


DEFAULTS = {
    # gaussian_x0hat supplies IID96/seed42, step7/x0-hat, standardized
    # PCA .95/stride3 and weighted diagonal GMM defaults.
    'skillset_mode': 'gaussian_x0hat',
    # Recorded joint velocities are commands, not joint-position deltas.
    'skillset_action_mode': 'dataset',
    'skillset_gripper_mode': 'discrete',
    'skillset_gripper_indices': [7],
    'skillset_gripper_values': [0.0, 1.0],
    'skillset_gripper_threshold': 0.5,
    'skillset_single_gpu_workers': 8,
    'skillset_cpus_per_task': 16,
    'skillset_mem': '64G',
    'skillset_time': '24:00:00',
    'skillset_wandb_project': '',
}


def configure(config):
    """Explicit values, including older frozen recipes, take precedence."""
    return {**DEFAULTS, **config}


def output_suffix(settings):
    """Derive the recipe name from resolved values instead of duplicating YAML."""
    path = Path(settings['dp_policy_path']) / 'config.json'
    if path.is_file():
        policy = json.loads(path.read_text())
        history = int(policy['n_obs_steps'])
        horizon = int(policy['future_action_horizon']
                      if policy.get('action_sequence_mode') == 'history_reconstruction'
                      else policy['horizon'])
    else:
        # Read-only planning may precede checkpoint transfer. Saved metadata
        # takes precedence when available, even if a run label is stale.
        match = re.search(r'_obs(\d+)_future(\d+)(?:_|$)', settings['dp_policy'])
        if not match:
            raise ValueError('DROID build naming requires checkpoint metadata or an obs/future DP run name')
        history, horizon = map(int, match.groups())
    initial = history if settings['skillset_initial_mask'] else 0
    power = f"{settings['skillset_score_power']:g}".replace('.', 'p')
    return (
        f"h{horizon}_r{settings['skillset_replan_interval']}_k{settings['skillset_n_gmm']}"
        f"_nms{settings['skillset_nms_dist']}_sg{settings['skillset_smooth_window']}p{settings['skillset_savgol_polyorder']}"
        f"_pow{power}_imask{initial}_tmask{settings['skillset_terminal_mask_frames']}"
    )
