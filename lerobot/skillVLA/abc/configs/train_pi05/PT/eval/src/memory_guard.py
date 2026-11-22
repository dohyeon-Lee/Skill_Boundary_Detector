"""Shared-GPU supervision for ABC evaluation (no per-worker allocator caps)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid


def supervise_evaluation(config):
    if config.policy_device == "cpu" or config.policy_type == "hold":
        return None
    if os.environ.get("SBD_GPU_GUARD_CHILD") == "1":
        return None
    if os.environ.get("SBD_GPU_MEMORY_GUARD", "true").lower() not in {"1", "true"}:
        return None
    from lerobot.utils.gpu_memory_guard import GIB, Guard, Nvidia

    env = dict(os.environ)
    defaults = {
        "SBD_GPU_GUARD_RESERVE_GIB": "0.3",
        "SBD_GPU_GUARD_STOP_BELOW_GIB": "0.15",
        "SBD_GPU_GUARD_MINIMUM_GIB": "0.3",
        "SBD_GPU_GUARD_RETRY_MARGIN_GIB": "0.3",
        "SBD_GPU_GUARD_INTERVAL": "0.5",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
        os.environ[key] = env[key]
    env["SBD_GPU_GUARD_TERMINAL_ON_OOM"] = "1"
    probe = Nvidia(env)
    folder = config.project_root / ".cache/gpu_memory_guard" / uuid.uuid4().hex
    guard = Guard([sys.executable, *sys.argv], env, folder, probe)
    stop = threading.Event()

    def protect_this_evaluation():
        # An older trainer's controller may renew the registry cooldown.
        # Enforce this evaluation's threshold without ever stopping its peer.
        consecutive_failures = 0
        while not stop.wait(float(env["SBD_GPU_GUARD_INTERVAL"])):
            try:
                _, _, free = probe.sample()
                with guard.registry.locked() as data:
                    active = [(key, run) for key, run in data["runs"].items()
                              if run["state"] in {"starting", "running", "stop_requested"}]
                    if free is not None and free < float(env["SBD_GPU_GUARD_STOP_BELOW_GIB"]) * GIB and active:
                        newest = max(active, key=lambda pair: (pair[1]["submitted"], pair[1]["sequence"]))[0]
                        if newest == guard.key:
                            data["runs"][guard.key]["state"] = "stop_requested"
                consecutive_failures = 0
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                # NVML can briefly stall under concurrent workloads. A single
                # timed-out query is not evidence of lost GPU supervision; the
                # main Guard also tolerates transient probe failures.
                consecutive_failures += 1
                print(f"ABC eval memory monitor retry {consecutive_failures}/3: {error}", file=sys.stderr, flush=True)
                if consecutive_failures < 3:
                    continue
                guard.cancelled = True
                return
            except Exception as error:
                # Stop this attempt on loss of supervision; preserve all peers.
                print(f"ABC eval memory monitor failed: {error}", file=sys.stderr, flush=True)
                guard.cancelled = True
                return

    thread = threading.Thread(target=protect_this_evaluation, daemon=True)
    thread.start()
    try:
        return guard.run()
    finally:
        stop.set()
        thread.join(timeout=3)
