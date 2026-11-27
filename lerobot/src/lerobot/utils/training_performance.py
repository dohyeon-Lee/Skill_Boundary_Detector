"""Avoid per-metric CUDA waits without changing training or logging reductions."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch


_DEFER_SCALARS = ContextVar('defer_training_metric_scalars', default=False)
_TRAINING_TARGETS = ContextVar('skillvla_forward_targets', default=None)


@contextmanager
def reuse_training_targets():
    """Reuse labels only within one forward, after that forward's random crops.

    Never retain a batch, crop offset, target, or autograd graph between calls.
    Nested forwards get their own scope and exceptions restore the outer scope.
    """
    token = _TRAINING_TARGETS.set({})
    try:
        yield
    finally:
        _TRAINING_TARGETS.reset(token)


def cached_training_target(owner, key, build):
    cache = _TRAINING_TARGETS.get()
    if cache is None:
        return build()
    identity = (id(owner), key)
    if identity not in cache:
        # Keep owner alive so a temporary start-batch dict cannot have its id
        # reused by a different batch in this same forward.
        cache[identity] = (owner, build())
    return cache[identity][1]


@lru_cache(maxsize=128)
def _constant(values, device, dtype):
    return torch.tensor(values, device=device, dtype=dtype)


def constant_tensor(values, *, device, dtype=torch.float32):
    """Private read-only constants; keys include values, actual device and dtype."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return _constant(tuple(values), device, dtype)


@lru_cache(maxsize=64)
def constant_axis(size, device, dtype=torch.float32):
    return torch.arange(size, device=device, dtype=dtype)


@lru_cache(maxsize=32)
def constant_eye(size, device, dtype):
    return torch.eye(size, device=device, dtype=dtype)


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
