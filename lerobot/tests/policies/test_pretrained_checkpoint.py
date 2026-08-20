import pytest
import torch
from safetensors.torch import load_file, save_model
from torch import nn

from lerobot.policies.pretrained import _save_model_as_safetensor


class _StorageViewModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backing = torch.arange(12, dtype=torch.float32)
        self.left = nn.Parameter(backing[:6].view(2, 3))
        self.right = nn.Parameter(backing[6:].view(2, 3))


def test_safetensor_save_clones_nonowning_parameter_views(tmp_path) -> None:
    model = _StorageViewModel()
    broken_path = tmp_path / "broken.safetensors"
    with pytest.raises(RuntimeError, match="None is covering the entire storage"):
        save_model(model, broken_path)

    checkpoint_path = tmp_path / "model.safetensors"
    _save_model_as_safetensor(model, str(checkpoint_path))
    restored = load_file(checkpoint_path)

    torch.testing.assert_close(restored["left"], model.left)
    torch.testing.assert_close(restored["right"], model.right)
