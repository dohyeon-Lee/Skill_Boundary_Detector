#!/usr/bin/env python
"""Open-loop forward-equivalence test — stage2 VSA vs stage1.

Feeds IDENTICAL internal inputs (cond images, state, skill code) and the SAME flow-matching noise
to both action forwards, then compares the predicted action chunk. This isolates the pure forward
math from env / preprocessing / RNG / closed-loop chaos — the only clean way to confirm that the
stage2 "VSA" panel (skill_vla with eval_drop_vlm + adapters OFF) implements the SAME function as the
stage1 "skill_expert" checkpoint. (Their action-side weights were already verified bit-identical.)

Three base comparisons per trial:
  (1) SELF        : s2(VSA) vs s2(VSA), same inputs + SAME noise      → determinism sanity (~0)
  (2) NOISE-SENS  : s2(VSA) vs s2(VSA), same inputs + DIFFERENT noise → how much noise alone moves
                    the action (this is the scale that closed-loop MuJoCo chaos amplifies into
                    visibly different videos)
  (3) CROSS       : s2(VSA) vs s1(skill_expert), same inputs + SAME noise  → THE number of interest

Interpretation: if CROSS ≈ SELF (both ~1e-3 or a bf16 noise floor) and CROSS << NOISE-SENS, then the
two forwards are the same function; the videos differ purely because of independent noise sampling +
closed-loop chaos, NOT weights/inputs/forward code.

When ``--adapter_probes`` is enabled, the same VLM-severed forward additionally runs with
``cond_bridge`` (③) only, ``expert`` (④) only, and both active.  Those probes are measured against
the VSA/base output using the *same* inputs and noise.  They do not test language usefulness; they
isolate whether either adapter can perturb the Stage-1 motor even after the real VLM is cut.

Run on a GPU node, e.g.:
  PROJECT=/scratch/mdorazi/Skill_Boundary_Detector
  PYTHONPATH=$PROJECT/lerobot/src:$PROJECT/lerobot/examples/libero \
    $PROJECT/.venv/bin/python tools/verify_vsa_stage1_equiv.py \
  --stage2 $PROJECT/outputs_filtered/skillVLA_stage2/FSQ555_dino8_both_1000_siglip_015000_state__s2_Ltttr8lr10sv50vf/checkpoints/010000/pretrained_model \
      --stage1 $PROJECT/outputs_filtered/skillVLA_stage1/FSQ555_dino8_both_1000_siglip_batch256_state/checkpoints/015000/pretrained_model
"""
import argparse
from pathlib import Path

import torch


def _cfg(cfg, *names, default=None):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    return default


def load_policies(s2_path, s1_path, device, dtype=None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.skillVLA.modeling_skillVLA import SkillVLAPolicy
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy

    print(f"[load] stage2 (skill_vla) ← {s2_path}")
    s2 = SkillVLAPolicy.from_pretrained(s2_path)
    # VSA panel = drop the real VLM AND turn the LoRA adapters OFF → pure Stage-1 VSA forward.
    s2.config.eval_drop_vlm = True
    s2.config.eval_drop_vlm_keep_adapters = False
    s2 = s2.to(device).eval()

    print(f"[load] stage1 (skill_expert) ← {s1_path}")
    # The stage1 ckpt often has fsq_path hardcoded to the ORIGINAL training machine (e.g. /data2/...),
    # and its terminator build would try to torch.load that stale path. The FSQ terminator is NOT used
    # by sample_actions (action forward = cond_encoder + expert only), so disable it at load time.
    s1cfg = PreTrainedConfig.from_pretrained(s1_path)
    s1cfg.train_terminator = False
    s1 = SkillExpertPolicy.from_pretrained(s1_path, config=s1cfg).to(device).eval()

    if dtype is not None:
        # Force BOTH forwards to the same dtype. fp32 removes bf16 rounding → isolates whether the residual
        # CROSS delta is pure bf16 noise (collapses to ~1e-4) or a structural forward difference (stays).
        s2 = s2.to(dtype)
        s1 = s1.to(dtype)
        print(f"[dtype] both policies cast to {dtype}")
    return s2, s1


def build_inputs(s2, device, source, seed, raw_dataset_dir):
    """Return ``(current_images, skill_start_images, state)`` for the two policy branches.

    The Stage-1 expert and SkillVLA's VSA/cond side both consume the SAME *current* observation.
    The SkillVLA VLM branch instead consumes a distinct frame captured at the (jittered) skill start.
    Keeping those inputs separate matters even for VSA verification: VLM K/V is structurally built to
    preserve the deployed RoPE layout, although eval_drop_vlm prevents it from affecting the motor.
    """
    cfg = s2.config
    B = 1
    n_img = len(list(cfg.image_features))
    res = _cfg(cfg, "image_resolution", default=(224, 224))
    H, W = (res if isinstance(res, (tuple, list)) else (res, res))
    state_dim = int(_cfg(cfg, "max_state_dim", default=32))

    g = torch.Generator(device="cpu").manual_seed(seed)
    if source == "dataset":
        try:
            from lerobot.policies.skillVLA.dataset_skillVLA import (  # noqa: PLC0415
                CAM_3RD, CAM_WRIST, SKILL_START_IMAGE, SKILL_START_WRIST_IMAGE, SkillVLADataset,
            )

            # ``fsq_path`` = .../{run_tag}/FSQ.pt, so its sibling ``skillvla/`` is the augmented
            # Stage-2 dataset that owns skill_start_image/wrist_image.  Do NOT use the raw dataset here:
            # its item has only the current observation and would silently collapse the two branches.
            fsq_raw = str(getattr(s2.config, "fsq_path", "") or "")
            if not fsq_raw:
                raise ValueError("Stage-2 checkpoint has no fsq_path to derive its skillvla dataset")
            fsq_path = Path(fsq_raw)
            skillvla_root = fsq_path.parent / "skillvla"
            if not skillvla_root.is_dir():
                raise FileNotFoundError(f"SkillVLA dataset not found next to fsq_path={fsq_path!s}")
            ds = SkillVLADataset(repo_id="local", root=skillvla_root)
            item = ds[0]
            img_keys = [k for k in cfg.image_features if k in item]
            current, starts = [], []
            start_key = {CAM_3RD: SKILL_START_IMAGE, CAM_WRIST: SKILL_START_WRIST_IMAGE}
            for k in img_keys:
                im = item[k].float()
                if im.ndim == 3:
                    im = im.unsqueeze(0)
                current.append(im.to(device))
                if k not in start_key or start_key[k] not in item:
                    raise KeyError(f"No skill-start image paired with current key {k!r}")
                sim = item[start_key[k]].float()
                if sim.ndim == 3:
                    sim = sim.unsqueeze(0)
                starts.append(sim.to(device))
            st = item["observation.state"].float().reshape(1, -1).to(device)
            if st.shape[1] < state_dim:
                st = torch.cat([st, torch.zeros(1, state_dim - st.shape[1], device=device)], dim=1)
            print("[input] source=dataset  "
                  f"current={[tuple(i.shape) for i in current]}  "
                  f"skill_start={[tuple(i.shape) for i in starts]}  state={tuple(st.shape)}")
            return current, starts, st
        except Exception as e:  # noqa: BLE001
            print(f"[input] dataset load failed ({e!r}) → falling back to synthetic")

    current = [torch.rand(B, 3, H, W, generator=g).to(device) for _ in range(n_img)]
    starts = [torch.rand(B, 3, H, W, generator=g).to(device) for _ in range(n_img)]
    state = torch.randn(B, state_dim, generator=g).to(device)
    print(f"[input] source=synthetic  current/start={n_img}×(1,3,{H},{W})  state=(1,{state_dim})")
    return current, starts, state


@torch.no_grad()
def s2_action(s2, current_images, skill_start_images, state, code, noise, num_steps, adapters=None):
    """skill_vla VSA forward (eval_drop_vlm + adapters OFF already set on the config).

    ``current_images`` are raw [0,1] Stage-1-side camera frames, deliberately shared with ``s1_action``.
    ``skill_start_images`` are distinct raw frames sampled at a skill start. The SkillVLA action-side
    cond encoder consumes the former, while its PaliGemma start-image branch
    (still constructed to preserve the deployed RoPE layout even when VLM reads are severed) expects the
    policy's 224px / [-1,1] preprocessing.  Closed-loop ``_snapshot_vlm_images`` performs this split;
    reproduce it here so a real 256px dataset frame is a valid verifier input.
    """
    from lerobot.policies.pi05.lora import set_active_adapters  # noqa: PLC0415

    B = state.shape[0]
    L = int(_cfg(s2.config, "tokenizer_max_length", "max_lang_tokens", default=48))
    lang_tokens = torch.zeros(B, L, dtype=torch.long, device=state.device)   # VLM is severed → discarded
    lang_masks = torch.ones(B, L, dtype=torch.bool, device=state.device)
    start_images = [s2._preprocess_vlm_tensor(img) for img in skill_start_images]
    if adapters is not None:
        # sample_actions() normally selects ∅ for this VSA config.  Probe a precise subset by
        # calling its sampling core after setting the sticky adapter scope ourselves.  The core still
        # honors eval_drop_vlm=True, so no VLM K/V crosses into cond/action.
        set_active_adapters(adapters)
        return s2.model._sample_actions_A(current_images, start_images, lang_tokens, lang_masks, state,
                                           code, noise, num_steps)
    return s2.model.sample_actions(current_images, start_images, lang_tokens, lang_masks, state,
                                   skill_code=code, noise=noise, num_steps=num_steps)


@torch.no_grad()
def s1_action(s1, imgs, state, code, noise, num_steps):
    """skill_expert forward."""
    img_masks = [torch.ones(state.shape[0], dtype=torch.bool, device=state.device) for _ in imgs]
    return s1.model.sample_actions(imgs, img_masks, state, code, noise=noise, num_steps=num_steps)


def stats(a, b):
    d = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    return d.max().item(), d.mean().item(), cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", required=True, help="stage2 skill_vla checkpoint (…/pretrained_model)")
    ap.add_argument("--stage1", required=True, help="stage1 skill_expert checkpoint (…/pretrained_model)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--source", choices=["synthetic", "dataset"], default="synthetic",
                    help="synthetic (default, robust) or a real frame from --raw_dataset_dir")
    ap.add_argument("--raw_dataset_dir", default="", help="LeRobotDataset root for --source dataset")
    ap.add_argument("--codes", default="0,31,62,93,124", help="comma-separated skill codes to sweep")
    ap.add_argument("--num_steps", type=int, default=10, help="flow-matching integration steps (both)")
    ap.add_argument("--input_seed", type=int, default=0)
    ap.add_argument("--out", default="", help="also write the report (txt + json) to this path")
    ap.add_argument("--dtype", choices=["asis", "fp32"], default="asis",
                    help="fp32 forces both forwards to float32 (removes bf16 rounding → isolates whether "
                         "the CROSS delta is pure rounding or a structural difference)")
    ap.add_argument("--adapter_probes", action=argparse.BooleanOptionalAction, default=True,
                    help="also measure ③ bridge-only, ④ expert-only, and ③+④ VLM-severed deltas "
                         "relative to the adapter-off VSA base")
    args = ap.parse_args()

    force_dtype = torch.float32 if args.dtype == "fp32" else None

    lines = []
    def P(s=""):
        print(s)
        lines.append(str(s))

    device = torch.device(args.device)
    s2, s1 = load_policies(args.stage2, args.stage1, device, dtype=force_dtype)
    current_images, skill_start_images, state = build_inputs(
        s2, device, args.source, args.input_seed, args.raw_dataset_dir)

    chunk = int(_cfg(s2.config, "chunk_size", default=10))
    adim = int(_cfg(s2.config, "max_action_dim", default=32))
    # REAL action dims only — sample_actions returns max_action_dim (padded); eval slices to real_dim.
    # Comparing the padding dims (unused, held at ~0) would overstate the delta.
    try:
        from lerobot.utils.constants import ACTION
        real_dim = int(s2.config.output_features[ACTION].shape[0])
    except Exception:  # noqa: BLE001
        real_dim = adim
    P(f"[dims] max_action_dim={adim}  real_action_dim={real_dim}  (comparing first {real_dim})")
    codes = [int(c) for c in args.codes.split(",") if c.strip() != ""]

    # In VLM-severed inference, adapter "cond" lives only in the VLM LLM and cannot affect the motor.
    # These are exactly the two adapter paths that can alter a drop_vlm rollout.
    adapter_probes = (("bridge", frozenset({"cond_bridge"})),
                      ("expert", frozenset({"expert"})),
                      ("bridge_expert", frozenset({"cond_bridge", "expert"})))
    present_adapters = set()
    for mod in s2.modules():
        if hasattr(mod, "adapters"):
            present_adapters.update(mod.adapters.keys())
    P(f"[adapters] checkpoint contains {sorted(present_adapters) or 'none'}; "
      f"probes={'on' if args.adapter_probes else 'off'}")

    def rd(a):
        return a[:, :, :real_dim]

    def make_noise(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(state.shape[0], chunk, adim, generator=g).to(device)

    noiseA, noiseB = make_noise(1234), make_noise(5678)   # two independent noise draws

    P("\n" + "=" * 78)
    P(f"{'code':>5} | {'SELF max|Δ|':>14} | {'NOISE-SENS max|Δ|':>18} | {'CROSS max|Δ|':>14} | {'CROSS cos':>9}")
    P("-" * 78)
    agg = {"self": 0.0, "sens": 0.0, "cross": 0.0, "cross_cos": 1.0}
    per_code, probe_rows = [], []
    probe_worst = {name: 0.0 for name, _ in adapter_probes}
    for code in codes:
        c = torch.tensor([code], dtype=torch.long, device=device)
        a_s2 = s2_action(s2, current_images, skill_start_images, state, c, noiseA, args.num_steps)
        a_s2_same = s2_action(s2, current_images, skill_start_images, state, c, noiseA, args.num_steps)  # (1)
        a_s2_diff = s2_action(s2, current_images, skill_start_images, state, c, noiseB, args.num_steps)  # (2)
        a_s1 = s1_action(s1, current_images, state, c, noiseA, args.num_steps)  # (3) cross, SAME noise

        self_mx, _, _ = stats(rd(a_s2), rd(a_s2_same))
        sens_mx, _, _ = stats(rd(a_s2), rd(a_s2_diff))
        cross_mx, cross_mean, cross_cos = stats(rd(a_s2), rd(a_s1))
        P(f"{code:>5} | {self_mx:>14.3e} | {sens_mx:>18.3e} | {cross_mx:>14.3e} | {cross_cos:>9.5f}")
        per_code.append({"code": code, "self_max": self_mx, "noise_sens_max": sens_mx,
                         "cross_max": cross_mx, "cross_mean": cross_mean, "cross_cos": cross_cos})
        agg["self"] = max(agg["self"], self_mx)
        agg["sens"] = max(agg["sens"], sens_mx)
        agg["cross"] = max(agg["cross"], cross_mx)
        agg["cross_cos"] = min(agg["cross_cos"], cross_cos)

        if args.adapter_probes:
            row = {"code": code}
            for name, adapters in adapter_probes:
                a_probe = s2_action(s2, current_images, skill_start_images, state, c, noiseA,
                                    args.num_steps, adapters=adapters)
                mx, mean, cos = stats(rd(a_s2), rd(a_probe))
                row[name] = {"max": mx, "mean": mean, "cos": cos,
                             "active": sorted(adapters),
                             "in_checkpoint": sorted(set(adapters) & present_adapters)}
                probe_worst[name] = max(probe_worst[name], mx)
            probe_rows.append(row)

    P("=" * 78)
    P(f"worst-case over codes: SELF={agg['self']:.3e}  NOISE-SENS={agg['sens']:.3e}  "
      f"CROSS={agg['cross']:.3e}  (min cos={agg['cross_cos']:.5f})")
    ratio = (agg["cross"] / agg["sens"]) if agg["sens"] > 0 else float("inf")
    P("\nVerdict (criterion: does severing the VLM + adapters RECOVER the stage1 function?):")
    if agg["cross"] <= max(1e-3, 5 * agg["self"]):
        verdict = "SAME_FUNCTION"
        P("  ✅ CROSS ≈ SELF (≈0) → VSA reproduces stage1 to fp precision → intent implemented EXACTLY.")
    elif agg["cross_cos"] >= 0.98 and ratio <= 0.25:
        verdict = "IMPLEMENTED_AS_INTENDED"
        P("  ✅ cos≥0.98 and CROSS ≪ NOISE-SENS → VSA recovers the stage1 function; the residual is")
        P("     numerical (bf16 RoPE-offset + HF-forward vs custom-KV impl), NOT a design/logic error.")
    else:
        verdict = "CHECK"
        P("  ⚠️  cos dropped OR CROSS ~ NOISE-SENS → a possible LOGIC difference (mask/positions/skill")
        P("     injection) — not just numerical. Worth tracing.")
    P(f"  metrics: cos={agg['cross_cos']:.4f}  CROSS/NOISE-SENS={ratio:.3f}  "
      f"(CROSS={agg['cross']:.2e}, NOISE-SENS={agg['sens']:.2e})")
    P("  note: closed-loop video divergence is driven by noise-sensitivity + chaos, not by CROSS.")

    if args.adapter_probes:
        P("\nAdapter drift probes (VLM severed; SAME input/noise; Δ vs adapter-off VSA):")
        P(f"{'code':>5} | {'③ bridge max|Δ|':>18} | {'④ expert max|Δ|':>18} | {'③+④ max|Δ|':>18}")
        P("-" * 78)
        for row in probe_rows:
            P(f"{row['code']:>5} | {row['bridge']['max']:>18.3e} | "
              f"{row['expert']['max']:>18.3e} | {row['bridge_expert']['max']:>18.3e}")
        P("worst-case: " + ", ".join(f"{name}={mx:.3e}" for name, mx in probe_worst.items()))

    if args.out:
        import json
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump({"stage2": args.stage2, "stage1": args.stage1, "source": args.source,
                       "num_steps": args.num_steps, "verdict": verdict, "worst": agg,
                       "checkpoint_adapters": sorted(present_adapters), "per_code": per_code,
                       "adapter_probes": probe_rows, "adapter_probe_worst": probe_worst}, f, indent=2)
        P(f"\n[saved] {args.out}\n[saved] {os.path.splitext(args.out)[0] + '.json'}")


if __name__ == "__main__":
    main()
