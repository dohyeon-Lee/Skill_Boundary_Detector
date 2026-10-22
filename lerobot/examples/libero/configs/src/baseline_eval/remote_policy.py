"""CPU simulator facade: the real baseline policy lives in an isolated process."""
from collections import deque
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from rpc import read_message, write_message


class ModelClient:
    def __init__(self, settings, panel):
        command = [settings["model_python"], "-u", str(Path(__file__).with_name("policy_worker.py")),
                   "--family", settings["baseline"], "--source", settings["model_source"],
                   "--project", settings["project_root"], "--checkpoint", panel["policy_path"],
                   "--grounding", panel["grounding"]]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(settings["model_source"]) / "src")
        env["BASELINE_EVAL_SEED"] = str(settings["seed"])
        self.timeout = settings["worker_timeout"]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, bufsize=0, env=env)
        try:
            response = self.receive()
            if response.get("status") != "ready":
                raise RuntimeError(f"Model worker did not initialize: {response}")
        except BaseException:
            self.close()
            raise

    def receive(self):
        response = read_message(self.process.stdout, self.timeout)
        if response.get("status") == "error":
            raise RuntimeError(f"Isolated model failed: {response['error']}; see job stderr")
        return response

    def request(self, operation, **values):
        write_message(self.process.stdin, {"op": operation, **values})
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            try:
                write_message(self.process.stdin, {"op": "close"})
                self.process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class IsolatedPolicy(PreTrainedPolicy):
    # This config is ONLY the CPU evaluator's metadata facade. No pi05 model is instantiated.
    config_class = PI05Config
    name = "isolated_baseline"

    def __init__(self, client, n_action_steps):
        super().__init__(PI05Config(device="cpu", push_to_hub=False))
        self.client, self.n_action_steps = client, n_action_steps
        self.queue = deque()

    def get_optim_params(self):
        raise NotImplementedError("Evaluation-only policy facade")

    def forward(self, batch):
        raise NotImplementedError("Training stays in the model's native environment")

    def reset(self):
        self.queue.clear()
        self.client.request("reset")

    def predict_action_chunk(self, batch, **kwargs):
        payload = {}
        for key in ("observation.images.image", "observation.images.wrist_image", "observation.state", "task"):
            source = "observation.images.image2" if key == "observation.images.wrist_image" and key not in batch else key
            value = batch[source]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if key.startswith("observation.images"):
                    value = (value.clamp(0, 1) * 255).round().to(torch.uint8)
                value = value.numpy()
            payload[key] = value
        actions = np.asarray(self.client.request("predict", batch=payload)["action"])
        if actions.ndim != 3 or actions.shape[-1] != 7 or actions.shape[1] < self.n_action_steps or not np.isfinite(actions).all():
            raise ValueError(f"Invalid decoded action chunk: {actions.shape}")
        return torch.from_numpy(actions.copy())

    def select_action(self, batch, **kwargs):
        if not self.queue:
            chunk = self.predict_action_chunk(batch)
            self.queue.extend(chunk[:, :self.n_action_steps].transpose(0, 1))
        return self.queue.popleft()
