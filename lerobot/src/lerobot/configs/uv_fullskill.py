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


def bspline_base_label(label: str) -> str:
    return UV_BSPLINE_BASE_LABELS.get(label, label)
