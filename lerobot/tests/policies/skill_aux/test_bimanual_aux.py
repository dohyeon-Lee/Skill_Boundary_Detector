"""Exercise ABC's two UV targets, shared crop, three cameras, and termination loss."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux.modeling_skill_aux import SkillAuxPolicy
from lerobot.policies.skill_expert import modeling_skill_predictor as predictor_module
from lerobot.policies.skill_expert.predictor_image_crop import PredictorTopCrop
from test_predictor_mode3_uv import TinyVLM


def test_two_arm_uv_loss_masks_independently_and_inverts_one_crop(monkeypatch):
    monkeypatch.setattr(predictor_module, "build_paligemma_model", lambda *a, **kw: TinyVLM())
    cfg = SkillAuxConfig(
        device="cpu", dtype="float32", train_terminator=False, train_skill_predictor=True,
        bimanual_spatial_auxiliary=True, skill_fsq_levels=[3, 3, 3, 2], skill_vocab_size=54,
        max_state_dim=14, max_action_dim=14,
        skill_image_keys=["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"],
        skill_predictor_mode="mode2", skill_predictor_focus_uv_enabled=True,
        skill_predictor_freeze_vlm=False, skill_predictor_lora=False,
        skill_predictor_reader_heads=2, skill_predictor_reader_depth=1,
    )
    policy = SkillAuxPolicy(cfg)
    predictor = policy.model.skill_predictor
    for parameter in predictor.reader_head_parameters():
        parameter.requires_grad_(True)
    def memory(images, tokens, mask, **kw):
        prefix, valid, ignore = predictor._embed_prefix(images, tokens, mask, **kw)
        return prefix, ignore
    monkeypatch.setattr(predictor, "reader_memory", memory)
    images = [torch.rand(2, 3, 168, 224) for _ in range(3)]
    crop = PredictorTopCrop.sample(images[0], zoom=1.1, training=False, model_size=224)
    monkeypatch.setattr(predictor, "top_crop", lambda *a, **kw: crop)
    targets = torch.tensor([[[.1, .2], [.3, -.2]], [[-.1, .1], [0., 0.]]])
    valid = torch.tensor([[True, False], [True, True]])
    args = (images, torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool), torch.tensor([1, 53]))
    loss, metrics = predictor.loss_with_focus_uv(*args, targets, valid, previous_skill_code=torch.tensor([-1, 2]))
    changed = targets.clone(); changed[0, 1] = 1000
    loss2, _ = predictor.loss_with_focus_uv(*args, changed, valid, previous_skill_code=torch.tensor([-1, 2]))
    torch.testing.assert_close(loss, loss2)
    loss.backward()
    assert torch.isfinite(predictor.focus_uv_head[-2].weight.grad).all()
    assert metrics["focus_uv_valid_fraction"] == .75
    assert len(predictor.vlm.images) == 6  # All three cameras on both calls.
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    batch = {
        "skill_start_image": images[0], "skill_start_wrist_image": images[1],
        "skill_start_right_wrist_image": images[2], "skill_code": args[3],
        "skill_focus_uv": targets, "skill_focus_valid": valid,
        "skill_previous_code": torch.tensor([-1, 2]),
        OBS_LANGUAGE_TOKENS: args[1], OBS_LANGUAGE_ATTENTION_MASK: args[2],
    }
    objective, output = policy(batch)
    assert torch.isfinite(objective)
    assert output["skill_predictor/focus_uv_valid_fraction"] == .75
    transformed = torch.stack([crop.to_model_uv(targets[:, arm])[0] for arm in range(2)], 1).flatten(1)
    restored = predictor.focus_uv_in_original_frame(transformed, images[0], top_crop=crop)
    torch.testing.assert_close(restored, targets.flatten(1), atol=1e-6, rtol=0)


def test_bimanual_fusion_current_and_post_end_backward(monkeypatch):
    from lerobot.policies.skill_aux import bimanual_terminator as module
    import FSQ

    class TinyDINO(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16, num_register_tokens=0)
            self.proj = nn.Linear(3, 16)
        def forward(self, pixel_values):
            return SimpleNamespace(last_hidden_state=self.proj(pixel_values.mean((-1, -2)))[:, None].expand(-1, 5, -1))

    monkeypatch.setattr(FSQ, "_load_dino_model", lambda *a: TinyDINO())
    term = module.BimanualTerminator(
        state_dim=14, fsq_levels=[3, 3, 3, 2], hidden_dim=16, n_layers=1, n_heads=2,
        dropout=0., arch="fusion", vision_backbone="dino", freeze_vision_encoder=True,
        dino_model_path="unused", dino_image_size=32, siglip_image_size=32,
        skill_cond_mode="token", state_min=np.zeros(14), state_max=np.ones(14),
        context_mode="proprio", camera_mode="both", goal_uv=True, start_proprio=True,
        termination_only=True,
    )
    import lerobot.policies.skill_aux.bimanual_terminator as bm
    monkeypatch.setattr(bm, "build_bimanual_terminator", lambda cfg: term)
    keys = ["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"]
    cfg = SkillAuxConfig(
        device="cpu", dtype="float32", bimanual_spatial_auxiliary=True, skill_image_keys=keys,
        train_terminator=True, fsq_path="unused", skill_fsq_levels=[3, 3, 3, 2], skill_vocab_size=54,
        max_state_dim=14, max_action_dim=14, terminator_context="proprio", terminator_goal_uv=True,
        terminator_start_proprio=True, terminator_termination_only=True, terminator_post_end_steps=2,
    )
    policy = SkillAuxPolicy(cfg)
    xyz = torch.tensor([[[100., 80., 1.], [120., 90., 1.]]]).expand(2, -1, -1)
    batch = {key: torch.rand(2, 3, 32, 32) for key in keys}
    batch.update(skill_code=torch.tensor([1, 2]), skill_code_true=torch.tensor([1, 2]), skill_ds=torch.tensor([0, 2]), skill_de=torch.tensor([3, 0]),
        skill_decoder_state=torch.rand(2, 14), skill_start_state=torch.rand(2, 14),
        skill_end_xyz=xyz, skill_focus_valid=torch.ones(2, 2, dtype=torch.bool),
        terminator_top_camera_projection=torch.eye(3, 4)[None].expand(2, -1, -1),
        terminator_post_end_codes=torch.tensor([[0, 0], [1, 0]]),
        terminator_post_end_valid=torch.tensor([[False, False], [True, False]]),
        terminator_post_end_start_state=torch.rand(2, 2, 14),
        terminator_post_end_bimanual_xyz=xyz[:, None].expand(-1, 2, -1, -1),
        terminator_post_end_goal_uv_valid=torch.ones(2, 2, 2, dtype=torch.bool))
    loss, metrics = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["terminator/post_end_conditions_per_frame"] == .5
    assert torch.isfinite(term.goal_proj[0].weight.grad).all()
    assert term.right_wrist_type_embedding.grad is not None
    assert all(p.grad is None for p in term.vision_encoder.parameters())
