#!/usr/bin/env python3
"""Keep the common parallel raw-cache -> global calibration -> HTML workflow."""
import argparse
import runpy
import sys
from types import SimpleNamespace
from config import build_settings, workflow_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args, extra = parser.parse_known_args()
    resolved = build_settings(args.config)
    from lerobot.utils import gpu_memory_guard
    # One supervisor covers the complete episode-worker tree. Cached CPU
    # reports must not enroll in the GPU admission/preemption registry.
    device = 'cpu' if '--report-only' in extra or resolved['dp_ablation_cache_ready'] else resolved['dp_ablation_device']
    code = gpu_memory_guard.supervise_if_needed(SimpleNamespace(policy=SimpleNamespace(device=device)))
    if code is not None:
        raise SystemExit(code)
    sys.path.append(str(workflow_dir()))
    runpy.run_path(str(workflow_dir() / 'run_eval.py'), run_name='__main__')

if __name__ == '__main__':
    main()
