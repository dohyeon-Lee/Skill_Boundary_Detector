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


def bspline_base_label(label: str) -> str:
    return UV_BSPLINE_BASE_LABELS.get(label, label)
