# ABC Skill Boundary Detector evaluation

After `train_skills/DP` reaches the initially selected checkpoint 010000 and
`train_skills/build_data` finishes, generate a report from the already-built
curves and boundaries with:

```bash
./submit.sh
```

This default report is CPU-only: it performs no DP inference and does not alter
the global threshold. It shows five episodes for each of the 24 `grab_clutter`
language prompts using a top-camera episode player with a live skill badge and
clickable skill timeline, followed by the final boundary score and denoising
gain. Delta-BIC, gripper diagnostics, and static start/end frame pairs are
omitted. The `영상 크게 보기` button enters a video-only fullscreen view while
retaining the live skill badge. Set `ABC_SBD_REPORT_EPISODES=1` for a smaller
report.

To make a separate report whose boundaries are local maxima of the saved
smoothed denoising gain above its pooled global mean, while retaining the same
NMS and minimum-skill-length settings as the built SBD dataset, run:

```bash
ABC_SBD_BOUNDARY_SOURCE=denoising_gain ./submit.sh
```

This writes to
`lerobot/examples/analysis/skill_boundary_detection/abc/<dataset>/denoising_gain_globalmean_ck<checkpoint>/`
and leaves the default built-boundary report unchanged. Its graph shows only
the denoising gain, pooled mean threshold, accepted peaks, and resulting skill
cuts. All ABC and LIBERO SBD reports share the
`lerobot/examples/analysis/skill_boundary_detection/` result root.

The expensive full-vs-no-gripper DP ablation remains available separately:

```bash
./submit_ablation.sh
```

To preview every saved SBD local maximum without a height threshold, while
retaining the standard NMS, start margin, and minimum-skill-length guards, use:

```bash
ABC_SBD_BOUNDARY_SOURCE=all_peaks ./submit.sh
```

This writes a separate `all_peaks_ck<checkpoint>` report. The built skillset
and its original threshold remain unchanged.

To compress score heights with `score_power=0.5` and restore the pooled global
mean threshold in that score space, run:

```bash
ABC_SBD_BOUNDARY_SOURCE=power_half_mean ./submit.sh
```

This writes `power0p5_globalmean_ck<checkpoint>` without modifying the built
skillset.

To recut at `score_power=1` while excluding each episode's reset/lift motion
from both global-mean calibration and peak candidates, run:

```bash
ABC_SBD_BOUNDARY_SOURCE=power_one_initial_motion_mask ./submit.sh
```

The ABC-only initial guard detects the first five consecutive arm-state changes
above `0.005 rad/frame` (grippers excluded), then masks through 100 frames after
that motion start. Episodes with an extreme reset jump above `0.5 rad/frame`
within the first 10 frames extend the mask until the power=1 SBD curve remains
below the provisional global mean for three consecutive anchors, capped at 100
additional frames. The report shades the base mask blue and any reset-recovery
extension orange, and writes to
`power1_initial_motion_mask100_ck<checkpoint>`; the built skillset is unchanged.
