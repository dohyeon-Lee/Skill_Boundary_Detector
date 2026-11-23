"""Private, layer-matched termination reader; never feeds the Action Expert."""
import torch
from torch import nn

from .layerwise_cond_bottleneck import _LayerwiseConditionReader


class DetachedLayerwiseTerminationBottleneck(nn.Module):
    """Read detached VLM layers with eight recurrent, 256-wide latent tokens.

    All projections, normalization and latent updates belong to this branch.
    Detach precedes normalization so even its affine parameters are private.
    """

    def __init__(self, condition_width: int, depth: int):
        super().__init__()
        width = 256
        self.queries = nn.Parameter(torch.randn(8, width) * .02)
        self.memory_norm = nn.LayerNorm(condition_width)
        self.readers = nn.ModuleList([
            _LayerwiseConditionReader(width, condition_width, 8) for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def initial(self, batch_size):
        return self.queries[None].expand(batch_size, -1, -1)

    def step(self, index, latent, hidden, valid):
        memory = self.memory_norm(hidden.detach().to(self.memory_norm.weight.dtype))
        return self.readers[index](latent, memory, key_padding_mask=~valid)

    def readout(self, latent):
        return self.head(latent.mean(dim=1)).float().squeeze(-1)
