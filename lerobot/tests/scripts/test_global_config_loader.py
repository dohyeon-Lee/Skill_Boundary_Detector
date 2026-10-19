"""Per-server global config (configs/servers/*.yaml) and RunPod storage links."""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
LOADER = runpy.run_path(str(CONFIGS / "src/global_config_loader.py"))
LINKER = runpy.run_path(str(CONFIGS / "src/link_storage.py"))
load_global_config = LOADER["load_global_config"]


def _checkout(tmp_path: Path, servers: dict[str, str], shared: str = "server: auto\ndataset_root: dataset_filtered\n") -> Path:
    configs = tmp_path / "repo/lerobot/examples/libero/configs"
    (configs / "servers").mkdir(parents=True)
    (configs / "global_config.yaml").write_text(shared)
    for name, text in servers.items():
        (configs / "servers" / f"{name}.yaml").write_text(text)
    return configs / "global_config.yaml"


def _servers(tmp_path: Path) -> dict[str, str]:
    return {
        "here": f"detect:\n  - {tmp_path}\ntrain_partition:\n  - a6000\ntrain_qos: base_qos\ndataset_root: dataset_other\n",
        "elsewhere": "detect:\n  - /definitely/not/here\ntrain_partition: debug\n",
    }


def test_auto_detects_the_server_from_the_checkout_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    config = load_global_config(_checkout(tmp_path, _servers(tmp_path)))
    assert config["server"] == "here"
    assert config["project_root"] == str((tmp_path / "repo").resolve())
    assert config["train_partition"] == ["a6000"] and config["train_qos"] == "base_qos"
    assert config["dataset_root"] == "dataset_other"          # server wins over the shared file
    assert "detect" not in config


def test_env_and_global_key_select_a_server_explicitly(tmp_path: Path, monkeypatch) -> None:
    path = _checkout(tmp_path, _servers(tmp_path), shared="server: elsewhere\n")
    monkeypatch.delenv("SBD_SERVER", raising=False)
    assert load_global_config(path)["server"] == "elsewhere"
    monkeypatch.setenv("SBD_SERVER", "here")
    assert load_global_config(path)["server"] == "here"
    monkeypatch.setenv("SBD_SERVER", "nope")
    with pytest.raises(ValueError, match="Unknown server 'nope'"):
        load_global_config(path)


@pytest.mark.parametrize("detects", [("/definitely/not/here", "/nor/here"), ("{tmp}", "{tmp}/repo")])
def test_detection_refuses_to_guess(tmp_path: Path, monkeypatch, detects) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    servers = {f"s{index}": f"detect:\n  - {prefix.format(tmp=tmp_path)}\n" for index, prefix in enumerate(detects)}
    with pytest.raises(ValueError, match="SBD_SERVER"):
        load_global_config(_checkout(tmp_path, servers))


def test_an_environment_variable_can_identify_the_server(tmp_path: Path, monkeypatch) -> None:
    """RunPod checkouts can live anywhere (/workspace, /root/workspace): $RUNPOD_POD_ID says it is a pod."""
    monkeypatch.delenv("SBD_SERVER", raising=False)
    servers = {"pod": "detect:\n  - /workspace\ndetect_env:\n  - SBD_TEST_POD_ID\n",
               "elsewhere": "detect:\n  - /definitely/not/here\n"}
    path = _checkout(tmp_path, servers)
    monkeypatch.setenv("SBD_TEST_POD_ID", "abc123")
    config = load_global_config(path)
    assert config["server"] == "pod" and "detect_env" not in config
    monkeypatch.delenv("SBD_TEST_POD_ID")
    with pytest.raises(ValueError, match="SBD_SERVER"):
        load_global_config(path)


def test_relative_storage_paths_follow_the_checkout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    servers = {"here": f"detect:\n  - {tmp_path}\nstorage_outputs: ..\nstorage_volume: /abs/volume\n"}
    config = load_global_config(_checkout(tmp_path, servers))
    assert config["storage_outputs"] == str(Path(config["project_root"]).parent)
    assert config["storage_volume"] == "/abs/volume"
    shipped = load_global_config(CONFIGS / "global_config.yaml", "runpod")
    assert shipped["storage_outputs"] == str(CONFIGS.parents[3].resolve().parent)


def test_single_file_layout_is_returned_unchanged(tmp_path: Path) -> None:
    """Config snapshots taken before servers/ existed carry one self-contained global file."""
    path = tmp_path / "global_config.yaml"
    path.write_text("project_root: /old/root\ntrain_partition: debug\n")
    assert load_global_config(path) == {"project_root": "/old/root", "train_partition": "debug"}


def test_flat_parser_matches_pyyaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    path = _checkout(tmp_path, _servers(tmp_path))
    with_yaml = load_global_config(path)
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert load_global_config(path) == with_yaml


BASE_PARTITIONS = [
    "suma_rtx4090", "base_suma_rtx3090", "dell_rtx3090", "asus_a5000", "gigabyte_a5000",
    "gigabyte_a6000", "suma_a6000", "asus_6000ada", "tyan_a6000",
]
SLURM_PROFILES = {
    "base": (BASE_PARTITIONS, "base_qos"),
    "big": (BASE_PARTITIONS[:3] + ["big_suma_rtx3090"] + BASE_PARTITIONS[3:], "big_qos"),
    "pro6000": (["asus_pro6000", "gigabyte_pro6000"], "pro6000_qos"),
    "48gb": (["gigabyte_a6000", "suma_a6000", "asus_6000ada", "tyan_a6000"], "base_qos"),
}


def _yonsei_profile_yaml(profile: str) -> str:
    lines = (CONFIGS / "servers/yonsei.yaml").read_text().splitlines()
    assert lines[0].startswith("slurm_profile:")  # the only selector stays at the very top
    lines[0] = f"slurm_profile: {profile}"
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("without_yaml", [False, True])
@pytest.mark.parametrize("profile", SLURM_PROFILES)
def test_each_shipped_slurm_profile_resolves(tmp_path: Path, monkeypatch, profile: str, without_yaml: bool) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    path = _checkout(tmp_path, {"yonsei": _yonsei_profile_yaml(profile)})
    if without_yaml:
        monkeypatch.setitem(sys.modules, "yaml", None)
    config = load_global_config(path, "yonsei")
    partitions, qos = SLURM_PROFILES[profile]
    assert config["slurm_profile"] == profile
    assert config["train_partition"] == partitions
    assert config["train_qos"] == qos
    assert config["scheduler"] == "slurm"
    assert config["dataset_root"] == "dataset_filtered"
    assert config["train_nodelist"] == ""
    assert config["train_exclude_nodes"] == [
        "node18", "node40", "node24", "node54", "node52", "node30", "node33", "node29", "node23",
    ]
    assert not any(key.startswith("slurm_profile_") for key in config)


def test_selected_profile_overrides_stale_direct_slurm_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    server = _yonsei_profile_yaml("48gb") + "train_partition: stale_pool\ntrain_qos: stale_qos\n"
    path = _checkout(tmp_path, {"yonsei": server}, shared="train_partition: old_pool\ntrain_qos: old_qos\n")
    config = load_global_config(path, "yonsei")
    assert (config["train_partition"], config["train_qos"]) == SLURM_PROFILES["48gb"]


@pytest.mark.parametrize("selector", ["typo", '""', "[]", "null", "false", "48"])
def test_invalid_slurm_profile_fails_clearly(tmp_path: Path, monkeypatch, selector: str) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    path = _checkout(tmp_path, {"yonsei": _yonsei_profile_yaml(selector)})
    with pytest.raises(ValueError, match="yonsei.yaml: .*slurm_profile.*base.*big.*pro6000"):
        load_global_config(path, "yonsei")


@pytest.mark.parametrize("definition, error", [
    ("slurm_profile_fast_partitions: []\nslurm_profile_fast_qos: base_qos\n", "partitions must be"),
    ("slurm_profile_fast_partitions: a6000\nslurm_profile_fast_qos: base_qos\n", "partitions must be"),
    ("slurm_profile_fast_partitions: [null]\nslurm_profile_fast_qos: base_qos\n", "partitions must be"),
    ('slurm_profile_fast_partitions: [" "]\nslurm_profile_fast_qos: base_qos\n', "partitions must be"),
    ("slurm_profile_fast_partitions: [a6000]\n", "qos must be"),
    ('slurm_profile_fast_partitions: [a6000]\nslurm_profile_fast_qos: ""\n', "qos must be"),
    ("slurm_profile_fast_partitions: [a6000]\nslurm_profile_fast_qos: []\n", "qos must be"),
])
def test_incomplete_slurm_profile_never_falls_back(tmp_path: Path, monkeypatch, definition: str, error: str) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    server = "slurm_profile: fast\ntrain_partition: stale\ntrain_qos: stale_qos\n" + definition
    path = _checkout(tmp_path, {"here": server})
    with pytest.raises(ValueError, match=error):
        load_global_config(path, "here")


def test_unselected_server_profile_is_not_resolved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    servers = _servers(tmp_path)
    servers["elsewhere"] += "slurm_profile: nonexistent\n"
    config = load_global_config(_checkout(tmp_path, servers))
    assert config["server"] == "here"
    assert config["train_partition"] == ["a6000"]
    assert "slurm_profile" not in config


def test_slurm_profile_is_frozen_by_submit_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SBD_SERVER", raising=False)
    path = _checkout(tmp_path, {"yonsei": _yonsei_profile_yaml("48gb")}, shared="server: yonsei\n")
    helper = path.parent / "src/snapshot_config.sh"
    helper.parent.mkdir()
    shutil.copy2(CONFIGS / "src/snapshot_config.sh", helper)
    component = path.parent / "train_example/train_config.yaml"
    component.parent.mkdir()
    component.write_text("schedule: {steps: 100000}\n")
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; snapshot_config "$2"', "_", str(helper), str(component)],
        check=True, capture_output=True, text=True,
    )
    snapshot = Path(result.stdout.strip())
    assert snapshot.read_text() == component.read_text()
    (path.parent / "servers/yonsei.yaml").write_text(_yonsei_profile_yaml("big"))
    assert load_global_config(path)["train_qos"] == "big_qos"
    frozen = load_global_config(snapshot.parent / "global_config.yaml")
    assert frozen["slurm_profile"] == "48gb"
    assert (frozen["train_partition"], frozen["train_qos"]) == SLURM_PROFILES["48gb"]


def test_flattened_snapshot_with_profile_metadata_stays_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "global_config.yaml"
    path.write_text("slurm_profile: base\ntrain_partition: old_pool\ntrain_qos: old_qos\n")
    assert load_global_config(path) == {
        "slurm_profile": "base", "train_partition": "old_pool", "train_qos": "old_qos",
    }


@pytest.mark.parametrize("server", ["yonsei", "rllab", "runpod"])
def test_shipped_servers_resolve(server: str) -> None:
    config = load_global_config(CONFIGS / "global_config.yaml", server)
    assert config["server"] == server
    assert Path(config["project_root"]) == CONFIGS.parents[3].resolve()
    for key in ("train_partition", "train_qos", "train_nodelist", "train_exclude_nodes", "dataset_root", "outputs_root"):
        assert key in config
    assert ("storage_volume" in config) is (server == "runpod")


def test_runpod_storage_links_keep_the_usual_tree(tmp_path: Path) -> None:
    repo, volume, disk = tmp_path / "repo", tmp_path / "global", tmp_path / "ckpt"
    (repo / "models/dino").mkdir(parents=True)
    (repo / "models/dino/config.json").write_text("{}")                  # tracked by git
    (volume / "dataset_filtered/libero").mkdir(parents=True)
    (volume / "models/dino").mkdir(parents=True)
    (volume / "models/dino/config.json").write_text("{}")                # identical copy -> ok
    (volume / "models/dino/model.safetensors").write_text("w")
    (volume / "models/pi05_base").mkdir()
    (volume / "outputs_filtered/pi05_PT/run_a/checkpoints/030000").mkdir(parents=True)
    (volume / "outputs_filtered/skillVLA_stage1/VSA/run_b/checkpoints/010000").mkdir(parents=True)   # 3 levels
    (volume / "outputs_filtered/FSQ/run_f").mkdir(parents=True)
    (volume / "outputs_filtered/FSQ/run_f/FSQ.pt").write_text("w")                                  # no checkpoints/
    (volume / "hf_cache/blobs").mkdir(parents=True)                      # unrelated -> ignored
    (disk / "Skill_Boundary_Detector").mkdir(parents=True)               # e.g. the checkout itself -> ignored
    config = {
        "server": "runpod", "project_root": str(repo), "dataset_root": "dataset_filtered",
        "outputs_root": "outputs_filtered", "storage_volume": str(volume), "storage_outputs": str(disk),
    }
    counts = LINKER["link_storage"](config)
    assert counts == {"linked": 7, "ok": 1, "conflict": 0}
    assert (repo / "dataset_filtered/libero").is_dir()
    assert (repo / "outputs_filtered").resolve() == (disk / "outputs_filtered").resolve()
    assert (repo / "outputs_filtered/pi05_PT/run_a").is_dir()           # warm starts find it
    assert (disk / "outputs_filtered/skillVLA_stage1/VSA/run_b").is_symlink()
    assert not (disk / "outputs_filtered/skillVLA_stage1/VSA").is_symlink()  # new runs there stay local
    assert (disk / "outputs_filtered/FSQ/run_f").is_symlink()
    assert (repo / "models/dino/model.safetensors").is_symlink()
    assert not (repo / "models/dino/config.json").is_symlink()
    assert (repo / "models/pi05_base").is_symlink()
    assert not (repo / "hf_cache").exists()
    assert not (repo / "Skill_Boundary_Detector").exists()
    assert LINKER["link_storage"](config) == {"linked": 0, "ok": 8, "conflict": 0}   # idempotent

    (volume / "models/dino/README.md").write_text("volume")
    (repo / "models/dino/README.md").write_text("repo")
    assert LINKER["link_storage"](config)["conflict"] == 1               # never overwrites


def test_old_group_links_are_replaced_by_run_links(tmp_path: Path) -> None:
    repo, volume, disk = tmp_path / "repo", tmp_path / "volume", tmp_path / "disk"
    repo.mkdir()
    (volume / "outputs_filtered/skillVLA_stage1/VSA/run_b/checkpoints/010000").mkdir(parents=True)
    (disk / "outputs_filtered/skillVLA_stage1").mkdir(parents=True)
    os.symlink(volume / "outputs_filtered/skillVLA_stage1/VSA", disk / "outputs_filtered/skillVLA_stage1/VSA")
    config = {"server": "runpod", "project_root": str(repo), "outputs_root": "outputs_filtered",
              "storage_volume": str(volume), "storage_outputs": str(disk)}
    assert LINKER["link_storage"](config)["conflict"] == 0
    assert not (disk / "outputs_filtered/skillVLA_stage1/VSA").is_symlink()
    assert (disk / "outputs_filtered/skillVLA_stage1/VSA/run_b").is_symlink()
    assert (volume / "outputs_filtered/skillVLA_stage1/VSA/run_b/checkpoints/010000").is_dir()   # data untouched


def test_container_disk_only_keeps_every_checkpoint_in_one_outputs_tree(tmp_path: Path) -> None:
    """Without the volume there is nothing to link: a second outputs tree next to the checkout would
    only split new runs (written inside it) from pulled checkpoints."""
    repo, disk = tmp_path / "workspace/repo", tmp_path / "workspace"
    repo.mkdir(parents=True)
    config = {
        "server": "runpod", "project_root": str(repo), "dataset_root": "dataset_filtered",
        "outputs_root": "outputs_filtered", "storage_volume": str(tmp_path / "workspace-global"),   # not mounted
        "storage_outputs": str(disk),
    }
    assert LINKER["link_storage"](config) == {"linked": 0, "ok": 0, "conflict": 0}
    assert not (disk / "outputs_filtered").exists() and not (repo / "outputs_filtered").exists()
    assert not (tmp_path / "workspace-global").exists()


def test_servers_without_storage_link_nothing(tmp_path: Path) -> None:
    assert LINKER["link_storage"]({"server": "yonsei", "project_root": str(tmp_path)}) == {
        "linked": 0, "ok": 0, "conflict": 0,
    }
