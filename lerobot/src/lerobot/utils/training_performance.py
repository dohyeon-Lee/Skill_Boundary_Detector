"""Avoid per-metric CUDA waits without changing training or logging reductions."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_DEFER_SCALARS = ContextVar('defer_training_metric_scalars', default=False)


@contextmanager
def defer_scalar_metrics(enabled=True):
    token = _DEFER_SCALARS.set(enabled)
    try:
        yield
    finally:
        _DEFER_SCALARS.reset(token)


def scalar_metric(value):
    """Keep diagnostic scalars detached until the trainer's single collection."""
    return value.detach() if _DEFER_SCALARS.get() else value.item()


def materialize_scalar_metrics(metrics):
    """One host transfer per device/dtype; retain original values and precision."""
    result = dict(metrics)
    groups = {}
    for key, value in metrics.items():
        if torch.is_tensor(value) and value.numel() == 1:
            groups.setdefault((value.device, value.dtype), []).append((key, value.detach().reshape(())))
    for entries in groups.values():
        # Pack without individual casts or rounding double/integer diagnostics.
        values = torch.stack([value for _, value in entries]).cpu().tolist()
        for (key, _), value in zip(entries, values, strict=True):
            result[key] = value
    return result


def prepare_isolated_grad_clip_cache(policy):
    """Call after wrapping/loading/freezing; rebuild if trainable parameters change."""
    groups = policy.isolated_main_optimizer_grad_groups()
    groups = {name: tuple(parameters) for name, parameters in groups.items()}
    excluded = {id(parameter) for parameters in groups.values() for parameter in parameters}
    main = tuple(parameter for parameter in policy.parameters()
                 if parameter.requires_grad and id(parameter) not in excluded)
    policy._sbd_isolated_grad_clip_cache = (main, groups)
    return main, groups
