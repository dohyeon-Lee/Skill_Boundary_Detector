"""CPU-owned optimizer states, with unchanged native MolmoAct2 update math.

Only one parameter's moments/compensation visit CUDA at a time. Parameters,
gradients, forward/backward, group clipping and gradient accumulation stay on
their original devices. Copies are blocking so the saved CPU buffers always
contain completed updates, including when the next operation is a checkpoint.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

from lerobot.optim import OptimizerConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2AdamW, MolmoAct2AdamWConfig


class CPUStateMolmoAct2AdamW(MolmoAct2AdamW):
    _offloaded_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "compensation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_groups(self.param_groups)

    @staticmethod
    def _validate_groups(groups):
        for group in groups:
            if any(group.get(key, False) for key in ("capturable", "differentiable", "fused")):
                raise ValueError("MolmoAct2 CPU state offload requires non-capturable, non-fused AdamW.")

    @contextmanager
    def _parameter_state(self, parameter):
        state = self.state[parameter]
        host_buffers = {}
        for key in self._offloaded_keys:
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise RuntimeError(f"Offloaded {key} must be on CPU between updates.")
                host_buffers[key] = value
                state[key] = value.to(parameter.device, non_blocking=False)
        try:
            yield
        finally:
            for key in self._offloaded_keys:
                value = state.get(key)
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    host = host_buffers.get(key)
                    if host is None:
                        host = value.to("cpu", non_blocking=False)
                    else:
                        host.copy_(value, non_blocking=False)
                    state[key] = host

    def _step_parameters(self, *, bfloat16):
        original_groups = self.param_groups
        try:
            for group in original_groups:
                for parameter in group["params"]:
                    if parameter.grad is None or (parameter.dtype == torch.bfloat16) != bfloat16:
                        continue
                    self.param_groups = [{**group, "params": [parameter]}]
                    with self._parameter_state(parameter):
                        if bfloat16:
                            MolmoAct2AdamW._step_compensated_bfloat16(self)
                        else:
                            torch.optim.AdamW.step(self)
        finally:
            self.param_groups = original_groups

    def _step_native_non_bfloat16(self):
        self._step_parameters(bfloat16=False)

    def _step_compensated_bfloat16(self):
        self._step_parameters(bfloat16=True)

    def load_state_dict(self, state_dict):
        """Use PyTorch's loader/dtype rules without staging all states on CUDA.

        Native Optimizer.load_state_dict normally moves moments to each live
        parameter's device. CPU placeholders preserve those loader semantics
        and parameter-group options, while preventing a resume-time CUDA OOM.
        No model weights are copied; placeholders have zero elements.
        """
        self._validate_groups(state_dict["param_groups"])
        original_groups, original_state = self.param_groups, self.state
        real_parameters = [p for group in original_groups for p in group["params"]]
        cpu_to_real = {}
        placeholders = []
        for group in original_groups:
            values = [torch.empty(0, dtype=p.dtype, device="cpu") for p in group["params"]]
            cpu_to_real.update(zip(values, group["params"], strict=True))
            placeholders.append({**group, "params": values})
        self.param_groups = placeholders
        try:
            super().load_state_dict(state_dict)
            self.state = defaultdict(dict, {cpu_to_real.get(key, key): value for key, value in self.state.items()})
            index = 0
            for group in self.param_groups:
                count = len(group["params"])
                group["params"] = real_parameters[index:index + count]
                index += count
        except Exception:
            self.param_groups, self.state = original_groups, original_state
            raise


@OptimizerConfig.register_subclass("sbd_molmoact2_adamw_cpu_state")
@dataclass
class CPUStateMolmoAct2AdamWConfig(MolmoAct2AdamWConfig):
    def build(self, params):
        return CPUStateMolmoAct2AdamW(
            params, lr=self.lr, betas=self.betas, eps=self.eps,
            weight_decay=self.weight_decay, group_grad_clip_norm=self.group_grad_clip_norm,
        )


def offload_optimizer_config(config):
    if isinstance(config, CPUStateMolmoAct2AdamWConfig):
        return config
    if type(config) is not MolmoAct2AdamWConfig:
        raise ValueError("CPU state offload requires the native MolmoAct2 AdamW preset.")
    return CPUStateMolmoAct2AdamWConfig(**asdict(config))
