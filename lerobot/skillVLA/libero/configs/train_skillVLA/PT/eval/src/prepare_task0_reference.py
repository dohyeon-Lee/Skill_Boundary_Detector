"""Build a reviewable object-local EEF reference from the FT task0 demo (CPU).

No policy loading, GPU, environment stepping, or job submission. Original states
are restored in a separate headless reference simulator, never a rollout env.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build_data/src"))

from lerobot.utils.task0_intervention import PROTOCOL, file_sha256, goal_value, scene_context


def prepare(run, original, output, *, episode=5, stable_steps=3):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from skill_episode_data import SkillEvaluationDataset, localize_model_xml_assets

    dataset = SkillEvaluationDataset(skill_dataset_dir=run / "skillvla", skill_latents_path=run / "skill_latents.npz",
                                 eval_init_states_path=run.parent / "eval_init_states.npz",
                                 original_dataset_dir=original, suite_name="libero_10")
    aligned = dataset.load_aligned_episode(episode)
    if "alphabet_soup_and_the_tomato_sauce" not in aligned.source.scene_file:
        raise ValueError("Selected reference episode is not LIBERO-10 task0.")
    task = benchmark.get_benchmark_dict()["libero_10"]().get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    import libero
    if not bddl.is_file():
        bddl = Path(libero.__file__).resolve().parent / "libero/bddl_files" / task.problem_folder / task.bddl_file
    assets = next((p for p in (Path(get_libero_path("assets")),
                              Path(libero.__file__).resolve().parent / "libero/assets",
                              Path.home() / ".cache/libero/assets") if p.is_dir()), None)
    if assets is None:
        raise FileNotFoundError("No local LIBERO assets; prepare the server runtime first.")
    env = ControlEnv(bddl_file_name=str(bddl), has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset_from_xml_string(localize_model_xml_assets(aligned.model_xml, libero_assets_dir=assets))
        snapshots = []
        for index in aligned.original_action_indices:
            env.set_init_state(aligned.original_states[int(index)])
            snapshots.append(scene_context(env.env, task_id=0, height=256, width=256))
    finally:
        env.close()
    goals = [p["state"] for p in snapshots[0]["predicates"]]
    first = next(g for g in goals if g[:2] == ["in", "alphabet_soup_1"])
    second = next(g for g in goals if g[:2] == ["in", "tomato_sauce_1"])
    containers = [name for name in snapshots[0]["objects"] if second[2] == name or second[2].startswith(name + "_")]
    if len(containers) != 1:
        raise ValueError(f"Could not identify the basket object for {second[2]}.")
    container = containers[0]
    streak, first_done = 0, None
    for frame, snapshot in enumerate(snapshots):
        streak = streak + 1 if goal_value(snapshot, first) and not snapshot["grasps"][first[1]] else 0
        if streak >= stable_steps:
            first_done = frame
            break
    if first_done is None or not goal_value(snapshots[-1], second):
        raise ValueError("Reference must successfully place alphabet soup first, then tomato sauce.")
    with np.load(run / "skill_latents.npz", allow_pickle=False) as data:
        rows = sorted([(int(i), int(s), int(e), int(c)) for ep, i, s, e, c in zip(
            *(data[k] for k in ("episode_id", "skill_index", "frame_start", "frame_end", "tokens")), strict=True
        ) if int(ep) == episode])
    phases = []
    with np.load(run / "skill_focus_uv.npz", allow_pickle=False) as data:
        focus = {(int(ep), int(i)): np.asarray(uv) for ep, i, uv in zip(
            data["episode_id"], data["skill_index"], data["focus_uv"], strict=True)}
    initial_target = np.asarray(snapshots[0]["objects"][second[1]]["position"])
    for index, start, end, code in rows:
        endpoint = min(end, len(snapshots) - 1)
        # Runtime takeover is at the NEXT learned boundary, not in the middle
        # of the release skill. Replaying that same release occurrence would
        # be an off-by-one and can deadlock same-skill rejection.
        if start <= first_done:
            continue
        xyz = np.asarray(aligned.filtered_states[endpoint, :3], dtype=float).copy()
        if dataset.proprio_grounding == "episode_start_xyz":
            xyz += aligned.episode_start_xyz
        # The processed dataset endpoint is authoritative for the trained UV
        # head. Recorded simulator states and processed EEF observations need
        # not be sub-mm identical; retain their discrepancy for audit instead.
        state_delta = float(np.linalg.norm(xyz - np.asarray(snapshots[endpoint]["eef_world"])))
        if state_delta > .05:
            raise ValueError(f"Restored EEF and action-aligned dataset endpoint disagree at {endpoint}: "
                             f"dataset={xyz.tolist()}, simulator={snapshots[endpoint]['eef_world']}.")
        from lerobot.policies.skillVLA.focus_projection import project_eef
        point = project_eef(xyz, np.asarray(snapshots[endpoint]["camera_transform"]), height=256, width=256)
        if not np.allclose(point.normalized_xy, focus[(episode, index)], atol=1e-4, rtol=0):
            raise ValueError("Restored original-frame UV disagrees with the training target archive.")
        basket = np.asarray(snapshots[endpoint]["objects"][container]["position"])
        # Explicit diagnostic heuristic: nearest endpoint XY to the remaining
        # can's ORIGINAL position or the basket. Saved phases can be reviewed.
        anchor = second[1] if np.linalg.norm(xyz[:2] - initial_target[:2]) < np.linalg.norm(xyz[:2] - basket[:2]) else container
        pose = snapshots[start]["objects"][anchor]
        local = np.asarray(pose["rotation"]).T @ (xyz - np.asarray(pose["position"]))
        phases.append({"skill_index": index, "skill_code": code, "frame_start": start, "frame_end": end,
                       "anchor": anchor, "eef_end_local_xyz": local.tolist(), "reference_eef_world": xyz.tolist(),
                       "reference_anchor_start_pose": pose, "reference_focus_uv": list(point.normalized_xy),
                       "recorded_state_eef_delta_m": state_delta})
    if not phases or phases[-1]["anchor"] != container:
        raise ValueError("Could not construct second-can grasp/placement reference phases.")
    info = json.loads((run / "skillvla/meta/info.json").read_text())
    reference = {"protocol": PROTOCOL, "task_id": 0, "skill_code_space_id": info.get("skill_code_space_id"),
                 "fsq_sha256": file_sha256(run / "FSQ.pt"),
                 "first_goal": first, "second_goal": second, "container_object": container, "phases": phases,
                 "source": {"run": str(run.resolve()), "episode": episode, "demo": aligned.source.demo,
                            "scene_file": aligned.source.scene_file, "first_placement_frame": first_done,
                            "first_intervention_boundary_frame": phases[0]["frame_start"],
                            "alignment_max_error": aligned.alignment_max_error},
                 "interpretation": "Object-relative demonstration EEF endpoints; reference phase/anchor heuristic, NOT exact rollout GT."}
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=5)
    args = parser.parse_args()
    prepare(args.run, args.original, args.output, episode=args.episode)
