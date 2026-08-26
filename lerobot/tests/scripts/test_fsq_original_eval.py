from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/libero"))
sys.path.insert(0, str(ROOT / "src"))

from FSQ import encoder_grounding_position, prepare_encoder_trajectory  # noqa: E402
from FSQ_original import FSQOriginalConfig, SplineFSQOriginalAE  # noqa: E402
from fsq_original_eval import Args, main  # noqa: E402


def _write_skills(root: Path, count: int = 8) -> tuple[Path, list[np.ndarray], list[np.ndarray]]:
    skills = root / "skillset" / "skills"
    skills.mkdir(parents=True)
    rng = np.random.default_rng(0)
    segments, actions = [], []
    for i in range(count):
        length = int(rng.integers(12, 40))
        t = np.linspace(0.0, 1.0, length)[:, None]
        harmonics = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        pose = np.sin(2 * np.pi * (t + i / 5.0) * harmonics[None]).astype(np.float32)
        gripper = np.tile((t > 0.5).astype(np.float32), (1, 2)) * 0.04
        states = np.concatenate([pose, gripper], axis=1).astype(np.float32)
        acts = rng.uniform(-1.0, 1.0, size=(length, 7)).astype(np.float32)
        segments.append(states)
        actions.append(acts)
        np.savez(
            str(skills / f"ep{i:03d}_skill0.npz"),
            actions=acts, states=states,
            episode_id=np.int64(i), task_id=np.int64(0), skill_index=np.int64(0),
            frame_start=np.int64(0), frame_end=np.int64(length),
        )
    return skills, segments, actions


def _checkpoint(
    tmp: Path, segments, actions, arch: str, quantizer: str = "fsq",
    encoder_arch: str = "spline",
) -> Path:
    mode = "optimal"
    prep = np.concatenate([prepare_encoder_trajectory(s, mode) for s in segments])
    starts = np.stack([encoder_grounding_position(s) for s in segments])
    lengths = [len(s) for s in segments]
    cfg = FSQOriginalConfig(
        enc_dim=8, n_control=12, hidden_dim=32, fsq_levels=[3, 3, 3],
        quantizer=quantizer, bsq_code_dim=4, encoder_arch=encoder_arch,
        num_layers=1, num_heads=4, dropout=0.0, decoder_layers=2,
        decoder_arch=arch, encoder_input_mode=mode, action_dim=7,
        length_min=float(min(lengths)), length_max=float(max(lengths)),
        encoder_min=prep.min(0), encoder_max=prep.max(0),
        encoder_start_min=starts.min(0), encoder_start_max=starts.max(0),
        device="cpu",
    )
    model = SplineFSQOriginalAE(cfg)
    path = tmp / f"FSQ_epoch0001_{arch}_{quantizer}_{encoder_arch}.pt"
    torch.save(
        {"format_version": cfg.format_version, "cfg": cfg, "model_state": model.state_dict()},
        str(path),
    )
    return path


def _run_eval(tmp_path: Path, arch: str, quantizer: str = "fsq") -> dict:
    skills, segments, actions = _write_skills(tmp_path)
    ckpt = _checkpoint(tmp_path, segments, actions, arch, quantizer)
    out = tmp_path / f"out_{arch}_{quantizer}"
    main(Args(model_path=str(ckpt), skills_dir=str(skills), output_dir=str(out), device="cpu"))
    payload = json.loads((out / "metrics.json").read_text())
    assert (out / "fsq_original_eval.html").is_file()
    return payload


def test_eval_oneshot_reports_state_recon(tmp_path: Path) -> None:
    payload = _run_eval(tmp_path, "oneshot")
    assert payload["decoder_arch"] == "oneshot"
    assert set(payload["splits"]) == {"train", "val", "all"}
    for split in payload["splits"].values():
        assert "recon_mse_xyz" in split and "recon_mse_gripper" in split
        assert "top1_share_pct" in split and "near_boundary_pct" in split
        assert "termination_abs_err_mean" not in split
    counts = {name: payload["splits"][name]["n_skills"] for name in payload["splits"]}
    assert counts["train"] + counts["val"] == counts["all"]


def test_eval_bsq_reports_bit_metrics_and_figures(tmp_path: Path) -> None:
    skills, segments, actions = _write_skills(tmp_path)
    ckpt = _checkpoint(tmp_path, segments, actions, "rnn", "bsq")
    evals = tmp_path / "evals"
    for tag in ("epoch0001", "epoch0002"):
        main(Args(model_path=str(ckpt), skills_dir=str(skills),
                  output_dir=str(evals / tag), device="cpu"))
    payload = json.loads((evals / "epoch0002" / "metrics.json").read_text())
    assert payload["quantizer"] == "bsq"
    assert payload["codebook_size"] == 16
    for split in payload["splits"].values():
        assert split["codebook_size"] == 16
        assert 0.0 <= split["bit_plus_ratio_min_pct"] <= split["bit_plus_ratio_max_pct"] <= 100.0
        assert "termination_abs_err_mean" in split
    # Codes are persisted per epoch for the migration graph, and the html
    # embeds both figures (bit histograms + Hamming usage/flow).
    for tag in ("epoch0001", "epoch0002"):
        assert (evals / tag / "codes.npy").is_file()
    html = (evals / "epoch0002" / "fsq_original_eval.html").read_text()
    assert "bit_histograms" in html and "hamming_usage_flow" in html
    assert "data:image/png;base64," in html
    from fsq_original_eval import _previous_codes
    prev = _previous_codes(evals / "epoch0002")
    assert prev is not None and len(prev) == payload["splits"]["all"]["n_skills"]


def test_eval_action_seq_encoder(tmp_path: Path) -> None:
    skills, segments, actions = _write_skills(tmp_path)
    ckpt = _checkpoint(tmp_path, segments, actions, "rnn", encoder_arch="action_seq")
    out = tmp_path / "out_action_seq"
    main(Args(model_path=str(ckpt), skills_dir=str(skills), output_dir=str(out), device="cpu"))
    payload = json.loads((out / "metrics.json").read_text())
    for split in payload["splits"].values():
        assert "recon_mse_xyz" in split and "termination_abs_err_mean" in split
        assert "top1_share_pct" in split


def test_eval_rnn_reports_action_recon_and_termination(tmp_path: Path) -> None:
    payload = _run_eval(tmp_path, "rnn")
    assert payload["decoder_arch"] == "rnn"
    for split in payload["splits"].values():
        assert "recon_mse_xyz" in split
        assert "termination_abs_err_mean" in split
        assert 0.0 <= split["early_rate"] <= 1.0
        assert 0.0 <= split["no_fire_rate"] <= 1.0
