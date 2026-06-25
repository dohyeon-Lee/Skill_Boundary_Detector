#!/usr/bin/env python3
"""Stage-2 VLM attention heatmap — PURE attention mask only.

For a few skill-start frames, run ONLY the Stage-2 VLM prefix [3rd image patches · wrist image
patches · language · skill-query] and read out its self-attention (the VLM is eager, so
`output_attentions=True` returns the softmax weights). The single output is the LANGUAGE → 3rd-person
IMAGE attention: for each language word, the attention distribution over the image patches, reshaped to
the patch grid and overlaid on the frame. Nothing else (no recon / skill-pred / extra panels).

  attention A[head, query, key] over the prefix → head-mean → slice [language rows, 3rd-image cols]
  → (n_words, n_patch) → per word reshape to (g, g) → upsample to the frame → alpha overlay.

Offline, no simulator. Loads the policy exactly like input_probe.py (PreTrainedConfig + make_policy +
make_pre_post_processors).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("attn_viz")

CAM_3RD = "observation.images.image"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy_path", required=True, help="Stage-2 checkpoint pretrained_model dir")
    p.add_argument("--dataset_dir", required=True, help="skillvla dataset (frames + task language)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--layers", default="last",
                   help='"rollout" (end-to-end flow, most faithful) | "last" | "all" | "mid" | comma ints')
    p.add_argument("--weighting", default="value", choices=["value", "attn"],
                   help='"value" = value-weighted A·‖V‖ (default, sharper); "attn" = raw softmax weights')
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _find_tokenizer(pre):
    """Best-effort: pull the language tokenizer out of the preprocessor pipeline for word labels."""
    for step in getattr(pre, "steps", []):
        for attr in ("input_tokenizer", "tokenizer", "language_tokenizer", "_tokenizer"):
            tok = getattr(step, attr, None)
            if tok is not None and hasattr(tok, "convert_ids_to_tokens"):
                return tok
    return None


_STRUCT_WORDS = {"task", "state", "action"}


def _task_word_indices(raw_words, valid):
    """Keep only the TASK-instruction language tokens: alphabetic word-pieces, dropping the PI05
    prompt scaffolding ("Task:/State:/Action:" + the discretized-state DIGITS + punctuation + pad)."""
    import re
    out = []
    for i, rw in enumerate(raw_words):
        if not bool(valid[i]) or rw in ("<pad>", "<eos>", "<bos>", "<unk>"):
            continue
        w = str(rw).replace("▁", "").replace("Ġ", "").strip()
        if not re.search(r"[A-Za-z]", w) or w.lower() in _STRUCT_WORDS:   # drops state digits / scaffold
            continue
        out.append(i)
    return out


def _layer_indices(spec: str, n_layers: int) -> list[int]:
    spec = str(spec).strip().lower()
    if spec == "last":
        return [n_layers - 1]
    if spec == "all":
        return list(range(n_layers))
    if spec == "mid":
        return [n_layers // 2]
    return [int(x) for x in spec.split(",") if x.strip()]


@torch.no_grad()
def vlm_attention(policy, batch, layer_ids, weighting="value", rollout=False):
    """Run the VLM prefix capturing attention → (A, n_third, n_img_total, n_lang, lang_masks).

    A is the head+layer-averaged attention (B, S, S), A[query, key]. weighting:
      "attn"  → raw softmax weights A[q,k] (diffuse; attention != contribution).
      "value" → VALUE-weighted A[q,k]·‖V[k]‖ : down-weights keys the query attends to but whose value
                vector injects little signal, so only keys that actually DRIVE the output stay bright.
    Both directions are sliced by the caller; returns the token-block sizes for slicing."""
    m = policy.model
    start_images = policy._snapshot_vlm_images(batch)          # [3rd, wrist] preprocessed for the VLM
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    n_imgs = [m.paligemma_with_expert.embed_image(img).shape[1] for img in start_images]
    n_third = int(n_imgs[0])
    n_img_total = int(sum(n_imgs))
    n_lang = int(lang_tokens.shape[1])

    vlm_embeds, vlm_pad, _ = m._vlm_tokens(start_images, lang_tokens, lang_masks)
    att_2d = vlm_pad[:, None, :] & vlm_pad[:, :, None]
    att_4d = torch.where(att_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE).to(vlm_embeds.dtype)
    position_ids = torch.cumsum(vlm_pad, dim=1) - 1

    # The custom PiGemma decoder layer drops attention weights and its output_attentions plumbing is
    # broken, so FORCE eager (sdpa returns no weights) and HOOK each self_attn for its 2nd return value
    # (eager_attention_forward → (output, attn_weights)) + v_proj for the value vectors (value weighting).
    m._vlm.config._attn_implementation = "eager"
    cap_a: dict[int, torch.Tensor] = {}   # attn weights  (B, Hq, S, S)
    cap_v: dict[int, torch.Tensor] = {}   # value-proj out (B, S, Hkv*head_dim)
    hooks = []
    for li, layer in enumerate(m._vlm.layers):
        def _ah(_m, _i, out, li=li):
            w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
            if w is not None:
                cap_a[li] = w.detach()
        hooks.append(layer.self_attn.register_forward_hook(_ah))
        if weighting == "value":
            def _vh(_m, _i, out, li=li):
                cap_v[li] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
            hooks.append(layer.self_attn.v_proj.register_forward_hook(_vh))
    try:
        m._vlm.forward(
            inputs_embeds=vlm_embeds, attention_mask=att_4d, position_ids=position_ids,
            past_key_values=None, use_cache=False, adarms_cond=None)
    finally:
        for h in hooks:
            h.remove()
    if not cap_a:
        raise RuntimeError("No attention weights captured — self_attn did not return them under eager.")

    head_dim = int(m._vlm.layers[0].self_attn.head_dim)
    if rollout:
        # Attention rollout (Abnar & Zuidema 2020): head-mean RAW (row-stochastic) attention per layer,
        # add the residual (0.5·A + 0.5·I) and renormalize, then matrix-product over ALL layers → the
        # end-to-end token-to-token attention flow (accounts for what the skill-query accumulated across
        # the whole stack). Uses raw attention — value weighting doesn't compose with the stochastic
        # product, so it is ignored in this mode.
        idxs = sorted(cap_a)
        B, _, S, _ = cap_a[idxs[0]].shape
        eye = torch.eye(S, device=cap_a[idxs[0]].device).unsqueeze(0)          # (1, S, S)
        R = eye.expand(B, S, S).clone()
        for i in idxs:
            Ah = cap_a[i].float().mean(1)                                      # (B, S, S) row-stochastic
            Ah = 0.5 * Ah + 0.5 * eye
            Ah = Ah / Ah.sum(-1, keepdim=True).clamp(min=1e-9)
            R = Ah @ R                                                         # accumulate Ã_L·…·Ã_1
        return R, n_third, n_img_total, n_lang, lang_masks

    per_layer = []
    for i in layer_ids:
        if i not in cap_a:
            continue
        Ah = cap_a[i].float()                                       # (B, Hq, S, S)
        if weighting == "value" and i in cap_v:
            B, Hq, S, _ = Ah.shape
            V = cap_v[i].float()                                    # (B, S, Hkv*head_dim)
            Hkv = V.shape[-1] // head_dim
            vnorm = V.view(B, S, Hkv, head_dim).norm(dim=-1).permute(0, 2, 1)   # (B, Hkv, S)
            if Hq != Hkv:                                            # GQA: kv heads → q heads
                vnorm = vnorm.repeat_interleave(Hq // Hkv, dim=1)
            Ah = Ah * vnorm[:, :, None, :]                          # weight each KEY by its value norm
        per_layer.append(Ah.mean(1))                                # mean over heads → (B, S, S)
    A = torch.stack(per_layer, 0).mean(0)                           # (B, S, S)
    return A, n_third, n_img_total, n_lang, lang_masks


def overlay_panel(ax, frame_hw3, heat_gg, title):
    ax.imshow(frame_hw3)
    H, W = frame_hw3.shape[:2]
    heat = torch.tensor(heat_gg)[None, None]
    heat = torch.nn.functional.interpolate(heat, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    ax.imshow(heat, cmap="jet", alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render_grid(frame, items, suptitle, png):
    """items = [(label, heat_gg), ...] → one figure: skill-start frame + per-item attention overlays."""
    ncol = min(6, max(1, len(items) + 1))
    nrow = max(1, (len(items) + 1 + ncol - 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.1 * nrow), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    axes[0, 0].imshow(frame); axes[0, 0].set_title("skill-start frame", fontsize=8); axes[0, 0].axis("off")
    for j, (label, heat) in enumerate(items, start=1):
        overlay_panel(axes[j // ncol, j % ncol], frame, heat, label)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_index(path, title, policy_path, cards):
    rows = "\n".join(
        f'<div class="card"><h3>sample {si} · frame {fid}</h3>'
        f'<img src="images/{html.escape(name)}"></div>' for si, fid, name in cards)
    path.write_text(
        f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#111;color:#eee}"
        ".card{margin:18px 0;border-bottom:1px solid #333;padding-bottom:12px}"
        "img{max-width:100%;border:1px solid #333}h3{font-weight:600}</style>"
        f"<h1>{html.escape(title)}</h1><p>{html.escape(policy_path)}</p>{rows}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    ds = LeRobotDataset(repo_id="local/attn_viz", root=args.dataset_dir)
    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    policy.eval().to(device)
    pre, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    tok = _find_tokenizer(pre)
    n_layers = len(policy.model._vlm.layers)
    rollout = str(args.layers).strip().lower() == "rollout"        # end-to-end flow over ALL layers
    layer_ids = _layer_indices("all" if rollout else args.layers, n_layers)
    log.info("loaded %s | VLM layers=%d → %s", args.policy_path, n_layers,
             "rollout" if rollout else f"using {layer_ids}")

    out = Path(args.output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    frame_ids = rng.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)

    cards_l2i, cards_i2l, cards_sq = [], [], []
    for si, fid in enumerate(frame_ids.tolist()):
        raw = ds[fid]
        batch = pre({k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in raw.items()})
        A, n_third, n_img_total, n_lang, lang_masks = vlm_attention(
            policy, batch, layer_ids, args.weighting, rollout)
        A = A[0].cpu().numpy()                                      # (S, S) = A[query, key]
        valid = lang_masks[0].bool().cpu().numpy()

        g = int(round(math.sqrt(n_third)))
        if g * g != n_third:
            log.warning("3rd-image patches=%d not a perfect square; cropping to %dx%d", n_third, g, g)
        frame = raw[CAM_3RD]
        frame = frame.permute(1, 2, 0).float().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
        frame = np.clip(frame, 0, 1) if frame.max() <= 1.001 else np.clip(frame / 255.0, 0, 1)

        ids = batch[OBS_LANGUAGE_TOKENS][0].cpu().tolist()
        raw_words = tok.convert_ids_to_tokens(ids) if tok is not None else [f"t{i}" for i in range(len(ids))]
        widx = (_task_word_indices(raw_words, valid) if tok is not None
                else [i for i in range(len(raw_words)) if valid[i]])[:24]
        lang0 = n_img_total                                        # language block start (absolute)

        def lbl(ti):
            return str(raw_words[ti]).replace("▁", "").replace("Ġ", "").strip() or str(raw_words[ti])
        # dir 1: language Q → 3rd-image K  (each word's attention over the patches; row → grid)
        items_l2i = [(lbl(ti), A[lang0 + ti, : g * g].reshape(g, g)) for ti in widx]
        # dir 2: 3rd-image Q → language K  (each patch's attention to the word; column → grid)
        items_i2l = [(lbl(ti), A[: g * g, lang0 + ti].reshape(g, g)) for ti in widx]

        # dir 3: skill-query token (the LAST prefix token = skill read-out) → image / language.
        # This is what the model attends to WHEN PREDICTING THE SKILL (skill_head reads this token's
        # hidden), so it's the decision-relevant grounding. → image = one map; → language = top words.
        sq = A[-1]                                                 # skill-query row, (S,)
        heat_sq = sq[: g * g].reshape(g, g)
        topw = sorted(((lbl(ti), float(sq[lang0 + ti])) for ti in widx), key=lambda x: -x[1])[:6]
        top_str = ", ".join(word for word, _ in topw)

        wmode = "rollout" if rollout else args.weighting
        p1 = out / "images" / f"lang2img_{si:02d}.png"
        p2 = out / "images" / f"img2lang_{si:02d}.png"
        p3 = out / "images" / f"skillq_{si:02d}.png"
        render_grid(frame, items_l2i, f"sample {si} (frame {fid})  language Q → 3rd-image K  [{wmode}]", p1)
        render_grid(frame, items_i2l, f"sample {si} (frame {fid})  3rd-image Q → language K  [{wmode}]", p2)
        render_grid(frame, [("skill-query → image", heat_sq)],
                    f"sample {si} (frame {fid})  skill-query → image  [{wmode}]\ntop words: {top_str}", p3)
        cards_l2i.append((si, fid, p1.name))
        cards_i2l.append((si, fid, p2.name))
        cards_sq.append((si, fid, p3.name))
        log.info("sample %d/%d (frame %d) → %s, %s, %s", si + 1, len(frame_ids), fid, p1.name, p2.name, p3.name)

    write_index(out / "index_lang2img.html", "Stage-2 VLM — language Q → image K (word grounding)",
                args.policy_path, cards_l2i)
    write_index(out / "index_img2lang.html", "Stage-2 VLM — image Q → language K (patch → word)",
                args.policy_path, cards_i2l)
    write_index(out / "index_skillq.html", "Stage-2 VLM — skill-query → image (what drives the skill)",
                args.policy_path, cards_sq)
    log.info("Saved → %s , %s , %s",
             out / "index_lang2img.html", out / "index_img2lang.html", out / "index_skillq.html")


if __name__ == "__main__":
    main()
