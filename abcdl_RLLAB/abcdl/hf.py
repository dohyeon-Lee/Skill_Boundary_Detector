"""Publish/fetch abcdl & mcap datasets on the HuggingFace Hub (format=branch, version=tag)."""

from __future__ import annotations

from typing import Optional


def _hub():
    try:
        import huggingface_hub as h
    except ImportError as e:
        raise ImportError("HF integration needs `pip install huggingface_hub`") from e
    return h


def _version_tag(fmt: str, version: str) -> str:
    """Tag name for a (format, version) pair, e.g. ``abcdl_224-v1``.

    Git tags are repo-global, so a bare ``v1`` would collide across format
    branches. Scoping the tag with the format keeps each format's versions
    independent within one dataset repo.
    """
    return f"{fmt}-{version}"


def push(
    repo_id: str,
    local_dir: str,
    fmt: str,
    version: str,
    message: Optional[str] = None,
    token: Optional[str] = None,
    card: Optional[dict] = None,
) -> str:
    """Upload *local_dir* to *repo_id* on the Hub.

    The dataset format (e.g. ``"abcdl_224"``) is stored as a branch; the version
    (e.g. ``"v1"``) is stored as a format-scoped tag ``<fmt>-<version>`` pointing
    to that branch's HEAD (a bare ``v1`` would collide across format branches).

    When *card* (a metadata dict) is given, a dataset card (``README.md`` with HF
    YAML frontmatter) is auto-generated from it and uploaded to the ``main``
    branch — so ``push``/``push_to_hub`` publishes the card like LeRobot, with no
    manual post-processing. Returns the branch name (``fmt``).
    """
    h = _hub()
    h.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    h.create_branch(
        repo_id=repo_id, repo_type="dataset", branch=fmt, exist_ok=True, token=token
    )
    h.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=local_dir,
        revision=fmt,
        commit_message=message or f"upload {fmt} {version}",
        token=token,
    )
    h.create_tag(
        repo_id=repo_id, repo_type="dataset", tag=_version_tag(fmt, version),
        revision=fmt, token=token, exist_ok=True,
    )
    if card is not None:
        refs = h.list_repo_refs(repo_id, repo_type="dataset", token=token)
        branches = [b.name for b in refs.branches]
        readme = _build_card(repo_id, {**card, "fmt": fmt, "version": version}, branches)
        h.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            commit_message=f"update dataset card ({fmt} {version})",
            token=token,
        )
    return fmt


def _branch_desc(b: str) -> str:
    if b == "mcap":
        return "full-resolution source recordings (MCAP; the only re-derivable original)"
    if b == "lerobot":
        return "LeRobot v3.0 dataset (parquet + videos)"
    if b.startswith("abcdl_"):
        return f"abcdl training cache (MP4+binary) @ {b.split('_', 1)[1]}px"
    return "dataset branch"


def _build_card(repo_id: str, card: dict, branches: list) -> str:
    """Render a HuggingFace dataset card (README.md) from a metadata dict."""
    name = repo_id.split("/")[-1]
    fmt = card.get("fmt", "abcdl")
    version = card.get("version", "v1")
    cams = card.get("camera_keys") or []
    tasks = card.get("tasks") or []
    res = card.get("resolution")
    res_s = f"{res[0]}x{res[1]}" if res else "n/a"
    rows = "\n".join(
        f"| `{b}` | {_branch_desc(b)} |" for b in branches if b != "main"
    ) or "| (none yet) | |"
    task_list = "\n".join(f"- {t}" for t in tasks[:25]) or "- (unspecified)"
    default_desc = (
        f"`{name}` is a bimanual robot-manipulation imitation-learning dataset "
        f"({card.get('num_episodes', '?')} episodes, {card.get('num_frames', '?')} frames "
        f"at {card.get('fps', '?')} FPS) collected on a two-arm station with "
        f"{len(cams)} cameras. Each frame carries a "
        f"{card.get('state_dim', '?')}-D proprioceptive state and a "
        f"{card.get('action_dim', '?')}-D action, plus synchronized camera images."
    )
    description = card.get("description") or default_desc
    frontmatter = (
        "---\n"
        "task_categories:\n- robotics\n"
        "tags:\n- abcdl\n- robotics\n- imitation-learning\n- bimanual-manipulation\n"
        "---\n\n"
    )
    body = f"""# {name}

{description}

Published with [**abcdl**](https://github.com/jellyho/abcdl_RLLAB) — the ABC
*A Behavior Cloning Dataloader* format. The same data is offered in several
**formats**, each stored as its own git **branch**; every **version** is a tag
`<format>-<version>` so resolutions and revisions coexist in one repo.

- **`mcap`** — full-resolution source recordings (the only branch you can re-process
  into other resolutions/alignments; everything else is derived from it).
- **`abcdl_<size>`** — the training cache: cameras stacked into one MP4 at `<size>`px
  plus a raw state/action binary, encoded for near-free random frame access.
- **`lerobot`** — a LeRobot v3.0 dataset (parquet + per-camera videos).

## Available formats (branches)

| branch | contents |
|---|---|
{rows}

## `{fmt}` @ `{version}`

| field | value |
|---|---|
| Episodes | {card.get('num_episodes', '?')} |
| Frames | {card.get('num_frames', '?')} |
| Cameras | {', '.join(cams) or '?'} |
| Resolution | {res_s} |
| FPS | {card.get('fps', '?')} |
| State dim | {card.get('state_dim', '?')} |
| Action dim | {card.get('action_dim', '?')} |
| Robot | {card.get('robot_type', '?')} |

## Tasks

{task_list}

## Install

```bash
pip install git+https://github.com/jellyho/abcdl_RLLAB    # provides the `abcdl` package
```

## Usage (LeRobot-compatible loader)

```python
from abcdl.dataset import AbcdlDataset
from torch.utils.data import DataLoader

# auto-downloads the `{fmt}` branch from the Hub, then loads locally
ds = AbcdlDataset("{repo_id}", fmt="{fmt}", version="{version}")
print(len(ds), "frames |", ds.meta.camera_keys)

item = ds[0]
#   item["observation.state"]            -> FloatTensor ({card.get('state_dim', '?')},)
#   item["action"]                       -> FloatTensor ({card.get('action_dim', '?')},)
#   item["observation.images.<cam>"]     -> FloatTensor (3, H, W) in [0, 1]
#   item["task"]                         -> str

# action chunks (e.g. 16 steps) for chunked policies:
ds = AbcdlDataset("{repo_id}", fmt="{fmt}", delta_timestamps={{"action": [i/ {card.get('fps', 30)} for i in range(16)]}})

# standard multi-worker training loop:
loader = DataLoader(ds, batch_size=64, num_workers=8, shuffle=True)
for batch in loader:
    ...
```

Pull the raw full-resolution source instead:

```python
from abcdl import hf
mcap_dir = hf.pull("{repo_id}", fmt="mcap", version="latest")
```

<sub>Dataset card auto-generated by <code>abcdl.hf.push</code>.</sub>
"""
    return frontmatter + body


def pull(
    repo_id: str,
    fmt: str,
    version: str = "latest",
    dest: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Download *repo_id* from the Hub and return the local directory path.

    When *version* is ``"latest"``, checks out the branch head (``fmt``);
    otherwise checks out the format-scoped tag ``<fmt>-<version>``.
    """
    h = _hub()
    revision = fmt if version == "latest" else _version_tag(fmt, version)
    return h.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        token=token,
    )


def list_versions(
    repo_id: str, fmt: Optional[str] = None, token: Optional[str] = None
) -> dict:
    """Return ``{"branches": [...], "tags": [...]}`` for *repo_id*.

    If *fmt* is given, ``branches`` is filtered to that branch name and ``tags``
    to that format's versions, with the ``<fmt>-`` prefix stripped (so a tag
    ``abcdl_224-v1`` is returned as ``v1``).
    """
    h = _hub()
    refs = h.list_repo_refs(repo_id, repo_type="dataset", token=token)
    branches = [b.name for b in refs.branches]
    tags = [t.name for t in refs.tags]
    if fmt is not None:
        branches = [b for b in branches if b == fmt]
        prefix = f"{fmt}-"
        tags = [t[len(prefix):] for t in tags if t.startswith(prefix)]
    return {"branches": branches, "tags": tags}
