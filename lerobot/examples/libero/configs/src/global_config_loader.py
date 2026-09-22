#!/usr/bin/env python3
"""Load ``global_config.yaml`` together with its per-server overlay.

``global_config.yaml`` holds the settings every machine shares. Machine-specific keys (Slurm
partition/qos/nodes, storage locations) live in ``servers/<name>.yaml`` next to it, so no server
ever edits the shared file. The server is chosen by, in order:

1. the ``SBD_SERVER`` environment variable,
2. ``server:`` in ``global_config.yaml`` (``auto`` = detect),
3. auto-detection: the one server whose ``detect:`` path prefixes contain this repository.

Detection never guesses: zero or several matches is an error naming ``SBD_SERVER``.
``project_root`` defaults to the repository root, so no server file has to spell it out.
Merge order: global < server (< the module YAML, applied by each caller).

A ``global_config.yaml`` without a ``servers/`` directory beside it is returned unchanged: that
is the self-contained layout of config snapshots taken before the overlay existed.

Every config module loads this file by path (``load_global_config``); the CLI prints the merged
result:  ``python src/global_config_loader.py [--server NAME] [--key project_root]``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

SERVERS_DIR = "servers"
SERVER_ENV = "SBD_SERVER"
# <repo>/lerobot/examples/libero/configs — used to find the repository from any config path.
_CONFIGS_RELATIVE = Path("lerobot") / "examples" / "libero" / "configs"


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if " #" in text:
        text = text.split(" #", 1)[0].rstrip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.strip("\"'")


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    """pyyaml-free fallback for flat ``key: value`` / ``- item`` files (same as train_skills_config)."""
    out: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_list_key is not None:
            out[current_list_key].append(_parse_scalar(line[2:]))
            continue
        if ":" not in line:
            current_list_key = None
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if value == "":
            out[key] = []
            current_list_key = key
        else:
            out[key] = _parse_scalar(value)
            current_list_key = None
    return out


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _load_flat_yaml(path)
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping.")
    return data


def repo_root(path: Path) -> Path:
    """The repository containing ``path`` (a live config or a snapshot bundle inside the tree)."""
    for directory in (path.resolve().parent, *path.resolve().parents):
        if (directory / _CONFIGS_RELATIVE).is_dir():
            return directory
    raise FileNotFoundError(f"{path} is not inside a Skill_Boundary_Detector checkout ({_CONFIGS_RELATIVE}).")


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _under(path: Path, prefix: str) -> bool:
    prefix = prefix.rstrip("/") or "/"
    text = str(path)
    return text == prefix or text.startswith(prefix + "/")


def select_server(servers_dir: Path, repo: Path, requested: str = "auto") -> tuple[str, dict[str, Any]]:
    servers = {path.stem: read_yaml(path) for path in sorted(servers_dir.glob("*.yaml"))}
    if not servers:
        raise FileNotFoundError(f"No server configs in {servers_dir}.")
    name = (os.environ.get(SERVER_ENV) or requested or "auto").strip()
    if name != "auto":
        if name not in servers:
            raise ValueError(f"Unknown server {name!r}; choose one of {sorted(servers)} ({servers_dir}/*.yaml).")
        return name, servers[name]
    # The unresolved and resolved checkout paths both count, so symlinked mounts still match.
    locations = {repo.absolute(), repo.resolve()}
    matches = [
        server for server, config in servers.items()
        if any(_under(location, prefix) for prefix in _as_list(config.get("detect")) for location in locations)
    ]
    if len(matches) != 1:
        found = "no server matches" if not matches else f"several servers match ({', '.join(matches)})"
        raise ValueError(
            f"Cannot pick a server for {repo}: {found} the detect: prefixes in {servers_dir}. "
            f"Set {SERVER_ENV}=<name> or server: <name> in global_config.yaml."
        )
    return matches[0], servers[matches[0]]


def load_global_config(path: Path | str, server: str | None = None) -> dict[str, Any]:
    """``global_config.yaml`` merged with the selected ``servers/<name>.yaml``."""
    path = Path(path)
    config = read_yaml(path)
    servers_dir = path.parent / SERVERS_DIR
    if not servers_dir.is_dir():
        return config
    repo = repo_root(path)
    requested = server or str(config.pop("server", "auto") or "auto")
    config.pop("server", None)
    name, overlay = select_server(servers_dir, repo, requested)
    overlay = {key: value for key, value in overlay.items() if key != "detect"}
    merged = {**config, **overlay}
    if not str(merged.get("project_root", "") or "").strip():
        merged["project_root"] = str(repo)
    merged["server"] = name
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "global_config.yaml")
    parser.add_argument("--server", default=None, help=f"override detection (like {SERVER_ENV})")
    parser.add_argument("--key", default=None, help="print only this key")
    args = parser.parse_args()
    merged = load_global_config(args.config, args.server)
    if args.key:
        value = merged.get(args.key, "")
        print(json.dumps(value) if isinstance(value, (list, dict)) else value)
    else:
        print(json.dumps(merged, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
