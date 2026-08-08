"""Small dependency-free contract shared by report generation and review serving."""

from __future__ import annotations

import hashlib
import json

REVIEW_SCHEMA = "stage1_skill_eval_human_review_v1"


def review_id_for_signature(signature: dict) -> str:
    """Return the stable identifier shared by an HTML report and its reviews."""
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
