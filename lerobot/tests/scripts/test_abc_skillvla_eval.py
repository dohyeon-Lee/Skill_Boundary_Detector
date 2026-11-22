"""Critical deployment contracts: joints/FK, held goals, stable worlds and control RNG."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "lerobot/skillVLA/abc/configs/train_skillVLA/PT/eval/src"
SHARED = ROOT / "lerobot/skillVLA/abc/configs/train_pi05/PT/eval/src"
sys.path.insert(0, str(ROOT / "third_party/abc"))
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(SRC))

import skillvla_adapter as runtime
from joint_overlay import JointPlanFK, camera_projection, denormalize_plan, project_xyz


def test_fk_uses_all_joints_without_changing_live_physics():
    mj = pytest.importorskip("mujoco")
    branches = []
    for arm in ("left", "right"):
        branch = "".join(f'<body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom size=".01" mass=".1"/>' for _ in range(7))
        branches.append(f'<body pos="0 {1 if arm == "left" else -1} 0">' + branch + f'<site name="{arm}_grasp_site" pos=".1 0 0"/>' + "</body>" * 8)
    model = mj.MjModel.from_xml_string('<mujoco><worldbody><camera name="top" pos="0 0 3"/>' + "".join(branches) + '</worldbody></mujoco>')
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data, _qpos_indices=list(range(14)), _gripper_indices=[6, 13])
    fk = JointPlanFK(env)
    joints = np.zeros((3, 14))
    joints[1, 0] = .4
    joints[2, 7] = -.5
    before = (data.qpos.copy(), data.qvel.copy(), data.site_xpos.copy(), data.time)
    xyz = fk.trajectory(data, joints)
    np.testing.assert_allclose(xyz[0], data.site_xpos[fk.sites], atol=1e-7)
    assert not np.allclose(xyz[1, 0], xyz[0, 0])
    assert not np.allclose(xyz[2, 1], xyz[0, 1])
    np.testing.assert_allclose(xyz[1, 1], xyz[0, 1])
    for previous, current in zip(before, (data.qpos, data.qvel, data.site_xpos, data.time)):
        np.testing.assert_array_equal(previous, current)
    uv, valid = project_xyz(np.array([[0., 0., 0.], [0., 0., 4.]]), camera_projection(model, data, "top", 168, 224))
    np.testing.assert_allclose(uv[0], [112, 84])
    assert valid.tolist() == [True, False]


def test_joint_plan_denormalizes_all_fourteen_channels():
    low = np.arange(14) * .1
    high = low + np.arange(14) + 1
    raw = denormalize_plan(np.array([[-1] * 14, [1] * 14]), low, high)
    np.testing.assert_allclose(raw[0], low, atol=1e-6)
    np.testing.assert_allclose(raw[1], high, atol=1e-6)
    with pytest.raises(ValueError):
        denormalize_plan(np.zeros((90, 3)), low, high)


def test_pinned_inference_weights_survive_training_retention(tmp_path):
    from freeze_config import retain_checkpoint
    source = tmp_path / "training/030000/pretrained_model"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"test weights")
    (source / "config.json").write_text("{}")
    retained = tmp_path / "evaluation/src/models/vsa/030000/pretrained_model"
    retain_checkpoint(source, retained)
    assert (source / "model.safetensors").stat().st_ino == (retained / "model.safetensors").stat().st_ino
    retain_checkpoint(source, retained)
    import shutil
    shutil.rmtree(source.parent)
    assert (retained / "model.safetensors").read_bytes() == b"test weights"


def make_adapter(monkeypatch, overlay):
    class VSA:
        config = SimpleNamespace(bspline_tokens=3, chunk_end_state_q01=[0] * 14, chunk_end_state_q99=[1] * 14)
        def reset(self): pass
        def predict_action_chunk(self, batch):
            self.last_codes = batch["skill_code"].clone()
            return torch.rand(len(self.last_codes), 4, 14)
        def predict_independent_plan_for_visualization(self, batch):
            return torch.rand(len(batch["skill_code"]), 3, 14)
    class Predictor:
        def __init__(self): self.previous = []
        def predict_focus_uv(self, images, tokens, mask, *, previous_skill_code):
            self.previous.append(previous_skill_code.clone())
            return (previous_skill_code + 1) % 54, torch.tensor([[.1, .2, -.3, -.4]]).repeat(len(tokens), 1)
    class FK:
        def __init__(self, env): pass
        def trajectory(self, data, joints): return joints[:, :6].reshape(-1, 2, 3)
    monkeypatch.setattr(runtime, "JointPlanFK", FK)
    adapter = runtime.SkillVLAPolicyAdapter.__new__(runtime.SkillVLAPolicyAdapter)
    adapter.torch, adapter.device = torch, torch.device("cpu")
    adapter.config = SimpleNamespace(seed=42, policy_dtype="float32", action_chunk_size=2, max_steps=100)
    adapter.settings = {"fullskill_overlay": overlay, "termination_threshold": .5, "min_skill_steps": 0, "max_skill_steps": 300}
    adapter.vsa, adapter.predictor = VSA(), Predictor()
    def pre(batch):
        n = len(batch["observation.state"])
        return {**batch, "observation.language.tokens": torch.zeros(n, 2).long(),
                "observation.language.attention_mask": torch.ones(n, 2).bool()}
    adapter.preprocessors = {"vsa": pre, "predictor": pre}
    adapter.postprocessors = {"vsa": lambda action: action}
    adapter.models = {"terminator": SimpleNamespace(_code_to_zq=lambda code: code[:, None].float())}
    def terminator(zq, state, top, wrists, **kwargs):
        adapter.term_inputs = (state.clone(), kwargs)
        return torch.zeros(1), torch.full((1,), 10.)
    adapter.terminator = terminator
    adapter.worlds, adapter.fk = {}, {}
    return adapter


def observation(n):
    return {"state": np.arange(n * 14, dtype=np.float32).reshape(n, 14), "prompt": ["grab target"] * n,
            "images": {c: np.full((n, 3, 8, 8), 100, dtype=np.uint8) for c in ("top", "left", "right")}}


def test_stable_world_identity_held_two_arm_goals_and_immediate_switch(monkeypatch, tmp_path):
    adapter = make_adapter(monkeypatch, False)
    env = SimpleNamespace(data=None)
    adapter.infer_for_worlds(observation(2), [4, 9], [env, env])
    np.testing.assert_array_equal(adapter.predictor.previous[0], [-1, -1])
    held = adapter.worlds[9].uv.copy()
    obs = observation(1)
    single = {"state": obs["state"][0] + 9, "prompt": obs["prompt"][0], "images": {c: im[0] for c, im in obs["images"].items()}}
    assert adapter.observe_step(9, single)  # Must stop the old action chunk.
    np.testing.assert_array_equal(adapter.term_inputs[0][0], single["state"])
    np.testing.assert_allclose(adapter.term_inputs[1]["goal_uv"].numpy(), held.reshape(1, 4))
    adapter.infer_for_worlds(obs, [9], [env])  # World 4 disappeared; 9 must keep its own previous code.
    np.testing.assert_array_equal(adapter.predictor.previous[-1], [0])
    assert adapter.worlds[9].code == 1 and adapter.worlds[4].code == 0
    assert adapter.worlds[9].events[0]["end_step"] == 1
    result = adapter.finish_world(9, tmp_path / "trace")
    saved = json.loads(Path(result["skill_trace_path"]).read_text())
    assert [s["skill_code"] for s in saved["skills"]] == [0, 1]


def test_overlay_does_not_consume_action_rng(monkeypatch):
    env = SimpleNamespace(data=None)
    results = []
    for enabled in (False, True):
        adapter = make_adapter(monkeypatch, enabled)
        torch.manual_seed(77)
        first = adapter.infer_for_worlds(observation(1), [0], [env])
        second = adapter.infer_for_worlds(observation(1), [0], [env])
        results.append((first, second, torch.get_rng_state()))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)


def test_gt_conditions_replace_predictor_and_only_terminator_advances(monkeypatch):
    adapter = make_adapter(monkeypatch, False)
    adapter.config.camera_height, adapter.config.camera_width = 8, 8
    adapter.gt_world_indices = {7: 0}
    class Oracle:
        episode_ids = [13125]
        def skill_count(self, world_index): return 2
        def condition(self, world_index, order, env, **kwargs):
            return [12, 43][order], np.array([[.2, -.1], [-.5, .4]]), np.ones(2).astype(bool), {"gt_episode_id": 13125}
    adapter.gt = Oracle()
    def no_predictor(*args, **kwargs): pytest.fail("GT evaluation must not invoke Predictor")
    adapter.predictor.predict_focus_uv = no_predictor
    env = SimpleNamespace(data=None)
    batch = observation(1)
    adapter.infer_for_worlds(batch, [7], [env])
    assert adapter.worlds[7].code == 12
    adapter.infer_for_worlds(batch, [7], [env])
    assert adapter.worlds[7].code == 12  # Extra replanning does not advance GT order.
    single = {"state": batch["state"][0], "prompt": batch["prompt"][0], "images": {k: v[0] for k, v in batch["images"].items()}}
    assert adapter.observe_step(7, single)
    assert not adapter.world_done(7)
    adapter.infer_for_worlds(batch, [7], [env])
    assert adapter.worlds[7].code == 43
    assert adapter.observe_step(7, single)
    assert adapter.world_done(7)  # No clamping/repeating the final GT skill.


def test_actual_vsa_action_and_fullskill_api_with_tiny_backbone(monkeypatch):
    helpers = ROOT / "lerobot/tests/policies/skill_expert"
    monkeypatch.syspath_prepend(str(helpers))
    from test_pi05_vlm_vsa import tiny_vlm_model
    from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_NOPROPRIO_LABEL
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert import dual_bottleneck
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    model = tiny_vlm_model(
        monkeypatch, DUAL_BOTTLENECK_NOPROPRIO_LABEL,
        device="cpu",
        bimanual_spatial_auxiliary=True,
        skill_image_keys=["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"],
        skill_plan_state_dim=14, chunk_end_state_dim=14, bspline_tokens=90, chunk_size=30,
        skill_fsq_levels=[3, 3, 3, 2], skill_vocab_size=54, num_inference_steps=1,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        chunk_end_state_q01=[-1.] * 14, chunk_end_state_q99=[1.] * 14,
        bimanual_goal_xyz_q01=[-1.] * 6, bimanual_goal_xyz_q99=[1.] * 6,
    )
    model.config.dtype = "float32"
    monkeypatch.setattr(dual_bottleneck, "DualBottleneckSkillExpert", lambda config: model)
    policy = SkillExpertPolicy(model.config).eval()
    batch = {key: torch.rand(1, 3, 4, 4) for key in model.config.skill_image_keys}
    batch.update({"observation.state": torch.zeros(1, 14), "skill_code": torch.tensor([53]),
                  "skill_focus_uv": torch.tensor([[[.1, .2], [-.3, -.4]]]),
                  "skill_focus_valid": torch.ones(1, 2).bool()})
    from lerobot.policies.skill_expert.bottleneck_video import capture_bottleneck, decode_auxiliaries
    with torch.inference_mode():
        torch.manual_seed(91)
        baseline = policy.predict_action_chunk(batch)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(91)
        with capture_bottleneck(model) as captured:
            actions = policy.predict_action_chunk(batch)
        torch.testing.assert_close(actions, baseline, rtol=0, atol=0)
        assert torch.equal(rng, torch.get_rng_state())
        plan = policy.predict_independent_plan_for_visualization(batch)
    assert actions.shape == (1, 30, 14) and plan.shape == (1, 90, 14)
    assert torch.isfinite(actions).all() and torch.isfinite(plan).all()
    assert set(captured['outputs']) == {'upper', 'lower'}
    decoded = decode_auxiliaries(captured['outputs'], model.config)
    for tower in ('upper', 'lower'):
        assert decoded[tower]['xyz_world'].shape == (1, 2, 3)
        for key in ('top_left', 'top_right', 'left', 'right'):
            assert decoded[tower][key + '_patch_box_uv'].shape == (1, 4)
    assert not any(key in captured for key in ('top', 'wrist', 'right_wrist'))


def test_fullskill_is_top_only_and_auxiliary_xyz_is_on_all_cameras(monkeypatch):
    import joint_overlay as overlay
    monkeypatch.setattr(overlay, 'camera_projection', lambda *args: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float))
    cameras = ('top', 'left', 'right')
    images = {key: np.zeros((3, 64, 96), np.uint8) for key in cameras}
    env = SimpleNamespace(model=None, data=None)
    xyz = np.array([[[20, 30, 0], [50, 30, 0]], [[25, 30, 0], [55, 30, 0]]])
    frame = overlay.draw_frame(env, images, cameras, xyz, None, '')
    assert frame[30, 22].any()
    assert not frame[17:, 96:].any()  # Neither wrist contains the full-skill path.
    auxiliary = {'upper': {'xyz_world': np.array([[35, 45, 0], [60, 45, 0]]),
                           'top_left_patch_box_uv': np.array([-.8, -.4, -.6, -.2]),
                           'left_patch_box_uv': np.array([-.6, -.4, -.4, -.2]),
                           'right_patch_box_uv': np.array([-.4, -.4, -.2, -.2])}}
    frame = overlay.draw_frame(env, images, cameras, None, None, '', auxiliaries=auxiliary)
    for index in range(3):
        np.testing.assert_array_equal(frame[45, index * 96 + 35], overlay.AUX_COLORS['upper'])
    for index, normalized_x in enumerate([-.8, -.6, -.4]):
        x, y = int((normalized_x + 1) * 95 / 2), int((-.4 + 1) * 63 / 2)
        np.testing.assert_array_equal(frame[y, index * 96 + x], overlay.AUX_COLORS['upper'])


def test_skill_timeout_and_repeated_code_are_distinct_occurrences():
    c = runtime.WorldSkill()
    c.activate(3, np.zeros((2, 2)), np.zeros(14))
    assert not c.advance(.1, threshold=.5, minimum=0, maximum=2)
    assert c.advance(.1, threshold=.5, minimum=0, maximum=2)
    assert c.events[0]["end_reason"] == "timeout"
    c.activate(3, np.zeros((2, 2)), np.zeros(14))
    assert c.order == 1 and c.age == 0 and c.events[-1]["previous_code"] == 3


def test_skillvla_report_keeps_only_html_at_root(tmp_path):
    from report import build_result, render_report
    world = {"world_index": 0, "seed": 1, "steps": 10, "success": True, "prompt": "grab target", "video_path": None,
             "skill_trace_path": str(tmp_path / "src/trace.json")}
    result = build_result({"type": "skillvla"}, "abc_sim_all", [
        {"task": "grab_clutter", "requested_task": "grab_clutter", "num_worlds": 1, "successes": 1, "worlds": [world]}], expected=1)
    render_report(tmp_path, result)
    assert {p.name for p in tmp_path.iterdir()} == {"index.html", "src"}
    text = (tmp_path / "index.html").read_text()
    assert 'href="src/trace.json"' in text and 'src="src/task_success_rates.png"' in text
    assert "FK" in text
