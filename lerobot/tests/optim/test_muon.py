"""Muon probe: hybrid optimizer, param-group splitting, and preset wiring."""

import pytest
import torch
from torch import nn

from lerobot.optim.optimizers import (
    AdamWConfig,
    MuonConfig,
    MuonWithAuxAdamW,
    load_optimizer_state,
    save_optimizer_state,
    split_param_groups_for_muon,
)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 4)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        return self.action_out_proj(self.norm(self.hidden(x)))


def _tagged_groups(model: _Toy) -> list[dict]:
    groups = [
        {"params": list(model.parameters()), "group_name": "vsa", "lr_scale": 1.0}
    ]
    excluded = {id(p) for n, p in model.named_parameters() if "action_out_proj" in n}
    return split_param_groups_for_muon(groups, excluded)


def _build(model: _Toy) -> MuonWithAuxAdamW:
    return MuonConfig(lr=1e-3, weight_decay=0.0).build(_tagged_groups(model))


def test_split_routes_only_eligible_2d_to_muon():
    model = _Toy()
    groups = _tagged_groups(model)
    assert [g["group_name"] for g in groups] == ["vsa_muon", "vsa_adamw"]
    muon, adamw = groups
    # Only hidden.weight is Muon-eligible; the excluded 2D head, its bias, and
    # every 1D tensor stay AdamW.
    assert muon["use_muon"] is True and adamw["use_muon"] is False
    assert [p.shape for p in muon["params"]] == [model.hidden.weight.shape]
    assert all(p.ndim == 2 for p in muon["params"])
    assert id(model.action_out_proj.weight) in {id(p) for p in adamw["params"]}
    # Metadata is copied to both halves and no parameter is lost.
    assert muon["lr_scale"] == adamw["lr_scale"] == 1.0
    total = sum(len(g["params"]) for g in groups)
    assert total == len(list(model.parameters()))


def test_split_rejects_non_dict_groups():
    with pytest.raises(TypeError, match="param groups"):
        split_param_groups_for_muon([torch.nn.Parameter(torch.randn(2, 2))])


def test_hybrid_build_step_and_state_layout():
    model = _Toy()
    optimizer = _build(model)
    assert isinstance(optimizer, MuonWithAuxAdamW)
    assert isinstance(optimizer._muon, torch.optim.Muon)
    assert isinstance(optimizer._adamw, torch.optim.AdamW)
    # Children operate on the parent's own group dicts (shared objects).
    assert optimizer._muon.param_groups[0] is optimizer.param_groups[0]
    assert optimizer._adamw.param_groups[0] is optimizer.param_groups[1]

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name]), name
    # The shared state mapping holds Muon and AdamW entries side by side.
    assert set(optimizer.state[model.hidden.weight]) == {"momentum_buffer"}
    assert "exp_avg" in optimizer.state[model.norm.weight]
    state_dict = optimizer.state_dict()
    assert len(state_dict["state"]) == len(list(model.parameters()))


def test_hybrid_untagged_params_fall_back_to_shape_split():
    model = _Toy()
    optimizer = MuonConfig(lr=1e-3).build(model.parameters())
    muon_params = {
        id(p) for g in optimizer.param_groups if g["use_muon"] for p in g["params"]
    }
    # Without tags every 2D weight (including the head) goes to Muon.
    expected = {id(p) for p in model.parameters() if p.ndim == 2}
    assert muon_params == expected


def test_scheduler_lr_reaches_both_children():
    optimizer = _build(_Toy())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    for child in (optimizer._muon, optimizer._adamw):
        for group in child.param_groups:
            assert group["lr"] == pytest.approx(5e-4)
    scheduler.step()
    assert optimizer._muon.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_hybrid_state_roundtrip_through_lerobot_helpers(tmp_path):
    torch.manual_seed(0)
    model = _Toy()
    optimizer = _build(model)
    for _ in range(2):
        model(torch.randn(4, 8)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    save_optimizer_state(optimizer, tmp_path)

    torch.manual_seed(0)
    fresh_model = _Toy()
    fresh = _build(fresh_model)
    loaded = load_optimizer_state(fresh, tmp_path)
    saved_state = optimizer.state_dict()["state"]
    loaded_state = loaded.state_dict()["state"]
    assert saved_state.keys() == loaded_state.keys()
    for index, entry in saved_state.items():
        for key, value in entry.items():
            torch.testing.assert_close(loaded_state[index][key], value)
    # load_state_dict replaces the parent's group dicts; children must have
    # been rebound onto them and remain steppable.
    assert loaded._muon.param_groups[0] is loaded.param_groups[0]
    fresh_model(torch.randn(4, 8)).sum().backward()
    loaded.step()


def test_skill_expert_preset_default_is_exact_adamw():
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SkillExpertConfig,
    )

    preset = SkillExpertConfig(device="cpu").get_optimizer_preset()
    assert preset == AdamWConfig(
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        grad_clip_norm=1.0,
    )


def test_skill_expert_preset_muon_branch_reuses_adamw_hparams():
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SkillExpertConfig,
    )

    preset = SkillExpertConfig(device="cpu", use_muon=True).get_optimizer_preset()
    assert preset == MuonConfig(
        lr=2.5e-5,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    )
    assert preset.adjust_lr_fn == "match_rms_adamw"


class _DummyConfig:
    def __init__(self, use_muon):
        self.use_muon = use_muon


class _DummyPolicy:
    def __init__(self, model, use_muon):
        self.config = _DummyConfig(use_muon)
        self._model = model

    def named_parameters(self):
        return self._model.named_parameters()


def test_policy_split_hook_is_identity_when_off_and_excludes_io_by_name():
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy

    _DummyPolicy._MUON_ADAMW_ONLY_NAME_PARTS = (
        SkillExpertPolicy._MUON_ADAMW_ONLY_NAME_PARTS
    )
    model = _Toy()
    groups = [{"params": list(model.parameters()), "group_name": "vsa"}]
    unchanged = SkillExpertPolicy._maybe_split_param_groups_for_muon(
        _DummyPolicy(model, use_muon=False), groups
    )
    assert unchanged is groups

    split = SkillExpertPolicy._maybe_split_param_groups_for_muon(
        _DummyPolicy(model, use_muon=True), groups
    )
    muon_params = {
        id(p) for g in split if g["use_muon"] for p in g["params"]
    }
    # action_out_proj is 2D but name-excluded (I/O head convention).
    assert muon_params == {id(model.hidden.weight)}
