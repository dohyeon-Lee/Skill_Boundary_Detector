"""Lightweight shared identities for the Top-UV Bspline ablations.

Persist lowercase labels like other VSA models; YAML/CLI names are case
insensitive. Display/run names retain the user-facing ``UV_`` prefix.
"""

UV_BSPLINE_BASE_LABELS = {
    "uv_" + base: base
    for base in (
        "fullskill_bspline_3_lastlayer_lastvis",
        "fullskill_bspline_3_lastlayer_lastvis_current",
        "fullskill_bspline_3_lastlayer_lastvis_current_none",
        "fullskill_bspline_3_lastlayer_lastvis_current_preskill",
    )
}
UV_BSPLINE_REVISIONS = {
    label: label + "_top_uv_start_xyz_v1" for label in UV_BSPLINE_BASE_LABELS
}
UV_BSPLINE_DIFFAUX_LABELS = tuple(label + "_diffaux" for label in UV_BSPLINE_BASE_LABELS)
UV_BSPLINE_BASE_LABELS.update({
    label + "_diffaux": base for label, base in tuple(UV_BSPLINE_BASE_LABELS.items())
})
UV_BSPLINE_REVISIONS.update({
    label: label + "_top_uv_start_xyz_end_state8_v1" for label in UV_BSPLINE_DIFFAUX_LABELS
})

# Keep semantic base labels unchanged: `_last` only selects the suffix bridge,
# independently of held/current vision, preskill memory and auxiliary heads.
UV_BSPLINE_LAST_LAYER_BASE_LABELS = {
    label + "_last": label
    for label in (
        "uv_fullskill_bspline_3_lastlayer_lastvis",
        "uv_fullskill_bspline_3_lastlayer_lastvis_diffaux",
        "uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill_diffaux",
        "uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill",
    )
}
for label, baseline in UV_BSPLINE_LAST_LAYER_BASE_LABELS.items():
    UV_BSPLINE_BASE_LABELS[label] = UV_BSPLINE_BASE_LABELS[baseline]
    UV_BSPLINE_REVISIONS[label] = UV_BSPLINE_REVISIONS[baseline].removesuffix("_v1") + "_last_visual_layer_v1"
    if baseline in UV_BSPLINE_DIFFAUX_LABELS:
        UV_BSPLINE_DIFFAUX_LABELS += (label,)

UV_BSPLINE_JOINT_SELF_ATTENTION_NAME = "UV_Bspline_lastlayer_current_preskill_diffaux"
UV_BSPLINE_JOINT_SELF_ATTENTION_LABEL = UV_BSPLINE_JOINT_SELF_ATTENTION_NAME.lower()
UV_BSPLINE_BASE_LABELS[UV_BSPLINE_JOINT_SELF_ATTENTION_LABEL] = (
    "fullskill_bspline_3_lastlayer_lastvis_current_preskill"
)
UV_BSPLINE_DIFFAUX_LABELS += (UV_BSPLINE_JOINT_SELF_ATTENTION_LABEL,)
UV_BSPLINE_REVISIONS[UV_BSPLINE_JOINT_SELF_ATTENTION_LABEL] = (
    UV_BSPLINE_JOINT_SELF_ATTENTION_LABEL + "_top_uv_start_xyz_end_state8_joint_self_attention_v1"
)


UV_BSPLINE_PREDICTED_XYZ_NAMES = {
    name.lower(): name
    for mode in ("detach", "attach")
    for suffix in ("", "_topw1")
    for name in (f"UV_Bspline_lastlayer_current_preskill_diffaux2_{mode}{suffix}",)
}
UV_BSPLINE_PREDICTED_XYZ_LABELS = tuple(UV_BSPLINE_PREDICTED_XYZ_NAMES)
for label in UV_BSPLINE_PREDICTED_XYZ_LABELS:
    UV_BSPLINE_BASE_LABELS[label] = "fullskill_bspline_3_lastlayer_lastvis_current_preskill"
    UV_BSPLINE_REVISIONS[label] = label + "_top_uv_predicted_xyz3_joint_self_attention_v1"
UV_BSPLINE_DIFFAUX_LABELS += UV_BSPLINE_PREDICTED_XYZ_LABELS

UV_BSPLINE_DIFFAUX3_NAMES = {
    name.lower(): name
    for prefix in ("UV_Bspline_lastlayer_current_preskill", "UV_Bspline_lastlayer")
    for name in (f"{prefix}_diffaux3",)
}
UV_BSPLINE_DIFFAUX3_LAST_LAYER_BASE_LABELS = {
    label + "_last": label for label in UV_BSPLINE_DIFFAUX3_NAMES
}
UV_BSPLINE_DIFFAUX3_NAMES.update({
    label + "_last": name + "_last"
    for label, name in tuple(UV_BSPLINE_DIFFAUX3_NAMES.items())
})
UV_BSPLINE_DIFFAUX3_LABELS = tuple(UV_BSPLINE_DIFFAUX3_NAMES)
for label in UV_BSPLINE_DIFFAUX3_LABELS:
    # The legacy base describes Expert/proprio/plan routing only. Every public
    # diffaux3 name selects joint 100-token attention, including held-start ones.
    UV_BSPLINE_BASE_LABELS[label] = "fullskill_bspline_3_lastlayer_lastvis" + (
        "_current_preskill" if "_current_preskill_" in label else ""
    )
    UV_BSPLINE_REVISIONS[label] = label + "_gt_xyz_shared_goal_noise_v2"

UV_BSPLINE_DIFFAUX4_NAMES = {
    name.lower(): name
    for prefix in ("UV_Bspline_lastlayer", "UV_Bspline_lastlayer_current_preskill")
    for suffix in ("", "_uvtoken")
    for name in (f"{prefix}_diffaux4{suffix}",)
}
UV_BSPLINE_DIFFAUX4_START_XYZ_BASE_LABELS = {
    f"{prefix}_diffaux4_start".lower(): f"{prefix}_diffaux4".lower()
    for prefix in ("UV_Bspline_lastlayer", "UV_Bspline_lastlayer_current_preskill")
}
UV_BSPLINE_DIFFAUX4_NAMES.update({
    label: UV_BSPLINE_DIFFAUX4_NAMES[baseline] + "_start"
    for label, baseline in UV_BSPLINE_DIFFAUX4_START_XYZ_BASE_LABELS.items()
})
UV_BSPLINE_DIFFAUX4_LAST_LAYER_BASE_LABELS = {
    label + "_last": label
    for label in UV_BSPLINE_DIFFAUX4_NAMES
    if not label.endswith("_uvtoken")
}
UV_BSPLINE_DIFFAUX4_NAMES.update({
    label: UV_BSPLINE_DIFFAUX4_NAMES[baseline] + "_last"
    for label, baseline in UV_BSPLINE_DIFFAUX4_LAST_LAYER_BASE_LABELS.items()
})
UV_BSPLINE_DIFFAUX4_START_XYZ_BASE_LABELS.update({
    label + "_last": baseline + "_last"
    for label, baseline in tuple(UV_BSPLINE_DIFFAUX4_START_XYZ_BASE_LABELS.items())
})
UV_BSPLINE_DIFFAUX4_LABELS = tuple(UV_BSPLINE_DIFFAUX4_NAMES)
UV_BSPLINE_GT_XYZ_LABELS = (*UV_BSPLINE_DIFFAUX3_LABELS, *UV_BSPLINE_DIFFAUX4_LABELS)
for label in UV_BSPLINE_DIFFAUX4_LABELS:
    UV_BSPLINE_BASE_LABELS[label] = "fullskill_bspline_3_lastlayer_lastvis" + (
        "_current_preskill" if "_current_preskill_" in label else ""
    )
    UV_BSPLINE_REVISIONS[label] = label + "_gt_xyz_shared_goal_noise_gemma_readout_v1"
for label, baseline in UV_BSPLINE_DIFFAUX4_START_XYZ_BASE_LABELS.items():
    UV_BSPLINE_REVISIONS[label] = (
        UV_BSPLINE_REVISIONS[baseline].removesuffix("_v1") + "_start_xyz_only_v1"
    )
for label, baseline in UV_BSPLINE_DIFFAUX4_LAST_LAYER_BASE_LABELS.items():
    UV_BSPLINE_REVISIONS[label] = (
        UV_BSPLINE_REVISIONS[baseline].removesuffix("_v1") + "_last_visual_layer_v1"
    )

# Diffaux5/6 retain all three bottleneck targets and add a separate,
# start-observation-only Gemma XYZ head. The suffix reads vision at all its
# layers by default; `_last` restricts this to the final suffix layer.
UV_BSPLINE_DUAL_XYZ_NAMES = {
    name.lower(): name
    for series in (5, 6)
    for prefix in ("UV_Bspline_lastlayer", "UV_Bspline_lastlayer_current_preskill")
    for suffix in ("", "_last")
    for name in (f"{prefix}_diffaux{series}{suffix}",)
}
UV_BSPLINE_DUAL_XYZ_LABELS = tuple(UV_BSPLINE_DUAL_XYZ_NAMES)
UV_BSPLINE_DUAL_XYZ_LAST_LAYER_LABELS = tuple(
    label for label in UV_BSPLINE_DUAL_XYZ_LABELS if label.endswith("_last")
)
UV_BSPLINE_EXPERT_UV_XYZ_LABELS = tuple(
    label for label in UV_BSPLINE_DUAL_XYZ_LABELS if "_diffaux6" in label
)
UV_BSPLINE_GT_XYZ_LABELS += UV_BSPLINE_DUAL_XYZ_LABELS
for label in UV_BSPLINE_DUAL_XYZ_LABELS:
    UV_BSPLINE_BASE_LABELS[label] = "fullskill_bspline_3_lastlayer_lastvis" + (
        "_current_preskill" if "_current_preskill_" in label else ""
    )
    UV_BSPLINE_REVISIONS[label] = label + "_gt_xyz_shared_goal_noise_dual_xyz_start_w1_v1"


# Independent, start-only planners on the ordinary (non-current/preskill) aux3 base.
UV_BSPLINE_SKILL_NAMES = {
    name.lower(): name for name in (
        "UV_Bspline_skill_learnable", "UV_Bspline_skill_denoising", "UV_Bspline_skill_learnable_query",
        "UV_Bspline_skill_denoising_none",
        "UV_Bspline_skill_learnable_wovis", "UV_Bspline_skill_denoising_none_wovis",
    )
}
UV_BSPLINE_SKILL_UV_LABELS = tuple(
    label for label in UV_BSPLINE_SKILL_NAMES
    if label.removesuffix("_wovis") in ("uv_bspline_skill_learnable", "uv_bspline_skill_denoising_none")
)
UV_BSPLINE_SKILL_NOUV_BASE_LABELS = {label + "_nouv": label for label in UV_BSPLINE_SKILL_UV_LABELS}
UV_BSPLINE_SKILL_NAMES.update({
    label: UV_BSPLINE_SKILL_NAMES[baseline] + "_nouv"
    for label, baseline in UV_BSPLINE_SKILL_NOUV_BASE_LABELS.items()
})
UV_BSPLINE_SKILL_LABELS = tuple(UV_BSPLINE_SKILL_NAMES)
UV_BSPLINE_SKILL_COND_UV_LABELS = (*UV_BSPLINE_SKILL_UV_LABELS, *UV_BSPLINE_SKILL_NOUV_BASE_LABELS)
UV_BSPLINE_SKILL_WOVIS_LABELS = tuple(
    label for label in UV_BSPLINE_SKILL_LABELS if label.removesuffix("_nouv").endswith("_wovis")
)
UV_BSPLINE_GT_XYZ_LABELS += tuple(label for label in UV_BSPLINE_SKILL_LABELS if label not in UV_BSPLINE_SKILL_COND_UV_LABELS)
for label in UV_BSPLINE_SKILL_LABELS:
    UV_BSPLINE_BASE_LABELS[label] = "fullskill_bspline_3_lastlayer_lastvis"
    UV_BSPLINE_REVISIONS[label] = label + "_aux3_independent_start_plan_v1"
    if label in UV_BSPLINE_SKILL_UV_LABELS:
        UV_BSPLINE_REVISIONS[label] = label + "_aux3_independent_start_plan_expert_uv_v2"
    elif label in UV_BSPLINE_SKILL_NOUV_BASE_LABELS:
        UV_BSPLINE_REVISIONS[label] = label + "_aux3_independent_start_plan_expert_start_xyz_only_v1"


def bspline_base_label(label: str) -> str:
    return UV_BSPLINE_BASE_LABELS.get(label, label)
