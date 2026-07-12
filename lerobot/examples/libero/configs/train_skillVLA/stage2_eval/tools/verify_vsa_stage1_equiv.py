#!/usr/bin/env python
"""Open-loop forward-equivalence test — stage2 VSA vs stage1.

Feeds IDENTICAL internal inputs (cond images, state, skill code) and the SAME flow-matching noise
to both action forwards, then compares the predicted action chunk. This isolates the pure forward
math from env / preprocessing / RNG / closed-loop chaos — the only clean way to confirm that the
stage2 "VSA" panel (skill_vla with eval_drop_vlm + adapters OFF) implements the SAME function as the
stage1 "skill_expert" checkpoint. (Their action-side weights were already verified bit-identical.)

Three comparisons per trial:
  (1) SELF        : s2(VSA) vs s2(VSA), same inputs + SAME noise      → determinism sanity (~0)
  (2) NOISE-SENS  : s2(VSA) vs s2(VSA), same inputs + DIFFERENT noise → how much noise alone moves
                    the action (this is the scale that closed-loop MuJoCo chaos amplifies into
                    visibly different videos)
  (3) CROSS       : s2(VSA) vs s1(skill_expert), same inputs + SAME noise  → THE number of interest

Interpretation: if CROSS ≈ SELF (both ~1e-3 or a bf16 noise floor) and CROSS << NOISE-SENS, then the
two forwards are the same function; the videos differ purely because of independent noise sampling +
closed-loop chaos, NOT weights/inputs/forward code.

Run on a GPU node, e.g.:
  PROJECT=/scratch/mdorazi/Skill_Boundary_Detector
  PYTHONPATH=$PROJECT/lerobot/src:$PROJECT/lerobot/examples/libero \
    $PROJECT/.venv/bin/python tools/verify_vsa_stage1_equiv.py \
      --stage2 $PROJECT/outputs_filtered/skillVLA_stage2/FSQ555_dino8_both_1000_siglip_015000_state__s2_Lttr8lr10/checkpoints/020000/pretrained_model \
      --stage1 $PROJECT/outputs_filtered/skillVLA_stage1/FSQ555_dino8_both_1000_siglip_batch256_state/checkpoints/015000/pretrained_model
"""
import argparse

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
    """Return (cond_images: list[Tensor], state: Tensor[B, max_state_dim]). Identical tensors are fed
    to BOTH forwards, so the only thing that must match is the SHAPE each expects (same image set,
    state padded to max_state_dim)."""
    cfg = s2.config
    B = 1
    n_img = len(list(cfg.image_features))
    res = _cfg(cfg, "image_resolution", default=(224, 224))
    H, W = (res if isinstance(res, (tuple, list)) else (res, res))
    state_dim = int(_cfg(cfg, "max_state_dim", default=32))

    g = torch.Generator(device="cpu").manual_seed(seed)
    if source == "dataset":
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
            ds = LeRobotDataset(repo_id="local", root=raw_dataset_dir)
            item = ds[0]
            img_keys = [k for k in cfg.image_features if k in item]
            imgs = []
            for k in img_keys:
                im = item[k].float()
                if im.ndim == 3:
                    im = im.unsqueeze(0)
                imgs.append(im.to(device))
            st = item["observation.state"].float().reshape(1, -1).to(device)
            if st.shape[1] < state_dim:
                st = torch.cat([st, torch.zeros(1, state_dim - st.shape[1], device=device)], dim=1)
            print(f"[input] source=dataset  images={[tuple(i.shape) for i in imgs]}  state={tuple(st.shape)}")
            return imgs, st
        except Exception as e:  # noqa: BLE001
            print(f"[input] dataset load failed ({e!r}) → falling back to synthetic")

    imgs = [torch.rand(B, 3, H, W, generator=g).to(device) for _ in range(n_img)]
    state = torch.randn(B, state_dim, generator=g).to(device)
    print(f"[input] source=synthetic  images={n_img}×(1,3,{H},{W})  state=(1,{state_dim})")
    return imgs, state


@torch.no_grad()
def s2_action(s2, imgs, state, code, noise, num_steps):
    """skill_vla VSA forward (eval_drop_vlm + adapters OFF already set on the config)."""
    B = state.shape[0]
    L = int(_cfg(s2.config, "tokenizer_max_length", "max_lang_tokens", default=48))
    lang_tokens = torch.zeros(B, L, dtype=torch.long, device=state.device)   # VLM is severed → discarded
    lang_masks = torch.ones(B, L, dtype=torch.bool, device=state.device)
    return s2.model.sample_actions(imgs, imgs, lang_tokens, lang_masks, state,
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
    args = ap.parse_args()

    force_dtype = torch.float32 if args.dtype == "fp32" else None

    lines = []
    def P(s=""):
        print(s)
        lines.append(str(s))

    device = torch.device(args.device)
    s2, s1 = load_policies(args.stage2, args.stage1, device, dtype=force_dtype)
    imgs, state = build_inputs(s2, device, args.source, args.input_seed, args.raw_dataset_dir)

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
    per_code = []
    for code in codes:
        c = torch.tensor([code], dtype=torch.long, device=device)
        a_s2 = s2_action(s2, imgs, state, c, noiseA, args.num_steps)
        a_s2_same = s2_action(s2, imgs, state, c, noiseA, args.num_steps)   # (1) determinism
        a_s2_diff = s2_action(s2, imgs, state, c, noiseB, args.num_steps)   # (2) noise sensitivity
        a_s1 = s1_action(s1, imgs, state, c, noiseA, args.num_steps)        # (3) cross, SAME noise

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

    if args.out:
        import json
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump({"stage2": args.stage2, "stage1": args.stage1, "source": args.source,
                       "num_steps": args.num_steps, "verdict": verdict, "worst": agg,
                       "per_code": per_code}, f, indent=2)
        P(f"\n[saved] {args.out}\n[saved] {os.path.splitext(args.out)[0] + '.json'}")


if __name__ == "__main__":
    main()
