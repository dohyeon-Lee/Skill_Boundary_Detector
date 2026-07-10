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
    "gate": {"gate_proj"}, "up": {"up_proj"}, "down": {"down_proj"}, "fc1": {"fc1"}, "fc2": {"fc2"},
}


def target_names_from_spec(spec: str) -> set[str]:
    """'q,k,v,o' → {'q_proj','k_proj','v_proj','o_proj','out_proj'}."""
    names: set[str] = set()
    for tok in (t.strip() for t in spec.split(",")):
        if tok:
            names |= _TOKEN_TO_NAMES.get(tok, {f"{tok}_proj"})
    return names
