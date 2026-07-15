"""Minimal LoRA (low-rank adapters) for pi05 — no `peft` dependency.

Wraps a FROZEN nn.Linear W with a trainable low-rank additive update:
    y = W x + (alpha / r) * B (A x)
with A: (r, in), B: (out, r), B initialised to 0 so the adapter starts as a no-op
(model == base at step 0). Only A, B train; W stays frozen (base preserved).

Used to test whether a frozen backbone + LoRA preserves the backbone's original ability
(e.g. the LLM's language grounding) better than full fine-tuning — measured by zero-shot
transfer to unseen task suites.

``LoRALinear`` is a drop-in for nn.Linear: it delegates ``.weight``/``.bias``/``.in_features``/
``.out_features`` to the base, so custom forward code that reads ``proj.weight.dtype`` keeps working.
"""

import contextlib
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # freeze the base weight
        dtype, device = base.weight.dtype, base.weight.device
        self.lora_A = nn.Linear(base.in_features, r, bias=False).to(device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, base.out_features, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)                # B=0 → ΔW=0 at init (starts == base)
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ── delegate the nn.Linear surface the custom pi05 forward reads directly ──
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_B(self.lora_A(self.drop(x)))
        return out + self.scaling * lora.to(out.dtype)


def inject_lora(root: nn.Module, target_names: set[str], r: int, alpha: float,
                dropout: float = 0.0) -> int:
    """Recursively replace every nn.Linear child whose attribute name is in ``target_names`` with a
    LoRALinear wrapping it. Returns the number of layers wrapped. (Names differ across backbones:
    Gemma self-attn = q_proj/k_proj/v_proj/o_proj; SigLIP attn = q_proj/k_proj/v_proj/out_proj.)"""
    wrapped = 0
    for name, child in list(root.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(root, name, LoRALinear(child, r, alpha, dropout))
            wrapped += 1
        else:
            wrapped += inject_lora(child, target_names, r, alpha, dropout)
    return wrapped


# "q,k,v,o" → the actual Linear attribute names (o covers both Gemma o_proj and SigLIP out_proj).
_TOKEN_TO_NAMES = {
    "q": {"q_proj"}, "k": {"k_proj"}, "v": {"v_proj"}, "o": {"o_proj", "out_proj"},
    "attn": {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"},
    "gate": {"gate_proj"}, "up": {"up_proj"}, "down": {"down_proj"},
    "mlp": {"gate_proj", "up_proj", "down_proj"},
    "fc1": {"fc1"}, "fc2": {"fc2"},
    "action_in": {"action_in_proj"}, "action_out": {"action_out_proj"},
}


def target_names_from_spec(spec: str) -> set[str]:
    """Expand compact target aliases, e.g. ``q,k,v,o,mlp,action_out``."""
    names: set[str] = set()
    for tok in (t.strip() for t in spec.split(",")):
        if tok:
            names |= _TOKEN_TO_NAMES.get(tok, {f"{tok}_proj"})
    return names


# ────────────────────────────────────────────────────────────────────────────────────────────────
# Multi-adapter LoRA (skillVLA): ONE frozen base can carry SEVERAL named low-rank adapters, only a
# selected subset ACTIVE per forward. skillVLA needs this because the SAME VLM LLM is read by the skill
# decoder (adapter "skill") and by the cond stream (adapter "cond") in two separate forwards with
# different adapters live — and one of them (skill) trains at FT while the other (cond) stays frozen.
# The frozen base is stored as ``.base`` (same convention as LoRALinear) so from_pretrained's plain→.base
# routing preserves the pretrained backbone here too. Adapters start B=0 → no-op (model == base at init).
# ────────────────────────────────────────────────────────────────────────────────────────────────

# Which adapter names are live in the CURRENT forward. None = ALL active (single-adapter/back-compat);
# an empty set = base only. Forwards are sequential, so a module-global is safe. Set via active_adapters().
_ACTIVE_ADAPTERS: set[str] | None = None


def set_active_adapters(names) -> None:
    """STICKY adapter selection (no restore) — REQUIRED for any forward that will be backpropped under
    gradient checkpointing: the checkpoint RECOMPUTE runs later, inside loss.backward(), long after a
    `with active_adapters(...)` scope has exited — a restored global would recompute the layers with a
    DIFFERENT adapter set than the original forward (torch aborts on saved-vs-recomputed metadata
    mismatch). A sticky set persists through the backward; the next forward simply overwrites it.
    ``{"skill"}`` → only that adapter; ``set()`` → base only; ``None`` → all adapters."""
    global _ACTIVE_ADAPTERS
    _ACTIVE_ADAPTERS = None if names is None else set(names)


@contextlib.contextmanager
def active_adapters(names):
    """Scoped variant (RESTORES on exit) — safe ONLY for no-grad/inference forwards (no checkpoint
    recompute). For training forwards use set_active_adapters (see its docstring)."""
    global _ACTIVE_ADAPTERS
    prev = _ACTIVE_ADAPTERS
    _ACTIVE_ADAPTERS = None if names is None else set(names)
    try:
        yield
    finally:
        _ACTIVE_ADAPTERS = prev


def _adapter_active(name: str) -> bool:
    return _ACTIVE_ADAPTERS is None or name in _ACTIVE_ADAPTERS


class _LoRAAdapter(nn.Module):
    """One low-rank branch ΔW = (alpha/r)·B·A (B=0 init). No base — it's added onto a shared base."""

    def __init__(self, in_f: int, out_f: int, r: int, alpha: float, dropout: float, dtype, device):
        super().__init__()
        self.lora_A = nn.Linear(in_f, r, bias=False).to(device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, out_f, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.lora_B(self.lora_A(self.drop(x)))


class NamedLoRALinear(nn.Module):
    """A frozen nn.Linear carrying a dict of named adapters; the ACTIVE ones (per active_adapters()) are
    summed onto the base. Drop-in for nn.Linear (delegates .weight/.bias/.in_features/.out_features)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                        # base frozen (pretrained backbone preserved)
        self.adapters = nn.ModuleDict()

    def add_adapter(self, name: str, r: int, alpha: float, dropout: float = 0.0) -> None:
        b = self.base
        self.adapters[name] = _LoRAAdapter(
            b.in_features, b.out_features, r, alpha, dropout, b.weight.dtype, b.weight.device)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for name, ad in self.adapters.items():
            if _adapter_active(name):
                out = out + ad(x).to(out.dtype)
        return out


def route_plain_to_base(state: dict, model_keys: set[str]) -> tuple[dict, int]:
    """Route plain nn.Linear keys into their LoRA-wrapped `.base.*` slots. A wrapped Linear stores its
    pretrained weight under `<name>.base.weight`, but a NON-LoRA checkpoint (pi05_base, a stage-1
    skill_expert, …) has the plain `<name>.weight` — without this remap those land as "missing" and the
    wrapped projections stay at RANDOM init (catastrophic: the whole attention is random; see the pi05
    LoRA incident). Routes only when the model actually exposes the wrapped key; no-op for non-LoRA
    models and for LoRA→LoRA loads. Returns (routed_state, n_routed)."""
    routed, n = {}, 0
    for key, value in state.items():
        if key in model_keys:
            routed[key] = value
            continue
        alt = None
        if key.endswith(".weight"):
            alt = key[: -len(".weight")] + ".base.weight"
        elif key.endswith(".bias"):
            alt = key[: -len(".bias")] + ".base.bias"
        if alt is not None and alt in model_keys and alt not in state:
            routed[alt] = value
            n += 1
        else:
            routed[key] = value
    return routed, n


def inject_named_lora(root: nn.Module, target_names: set[str], adapter_name: str, r: int, alpha: float,
                      dropout: float = 0.0) -> int:
    """Add a NAMED adapter to every targeted nn.Linear under ``root``, wrapping it in a NamedLoRALinear the
    first time (later injections of a different adapter name just append to the existing wrapper). Returns
    the count of Linears carrying this adapter. Calling twice with different names → both adapters coexist,
    switchable via active_adapters()."""
    wrapped = 0
    for name, child in list(root.named_children()):
        if name in target_names and isinstance(child, (nn.Linear, NamedLoRALinear)):
            if isinstance(child, nn.Linear):
                child = NamedLoRALinear(child)
                setattr(root, name, child)
            child.add_adapter(adapter_name, r, alpha, dropout)
            wrapped += 1
        else:
            wrapped += inject_named_lora(child, target_names, adapter_name, r, alpha, dropout)
    return wrapped
