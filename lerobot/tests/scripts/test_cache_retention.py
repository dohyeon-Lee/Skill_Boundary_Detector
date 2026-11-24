import fcntl
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys


SOURCE = Path(__file__).resolve().parents[2] / "skillVLA/libero/configs/src/cache_retention.py"
spec = importlib.util.spec_from_file_location("cache_retention", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pruning_keeps_current_previous_and_in_use_archives(tmp_path):
    archives = []
    for i in range(5):
        path = tmp_path / f"venv-{i:016x}.tar.zst"
        path.write_bytes(b"archive")
        Path(str(path) + ".unpacked_size").write_text("7")
        os.utime(path, (i + 1, i + 1))
        archives.append(path)
    untouched = tmp_path / "results.tar.zst"
    untouched.write_bytes(b"unique result")
    with Path(str(archives[0]) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        removed = module.prune_venv_archives(tmp_path, archives[-1])
        assert set(removed) == {archives[1], archives[2]}
        assert archives[0].is_file()
    module.prune_venv_archives(tmp_path, archives[-1])
    assert not archives[0].exists()
    assert all(p.is_file() for p in archives[-2:])
    assert not Path(str(archives[1]) + ".unpacked_size").exists()
    assert untouched.read_bytes() == b"unique result"


def test_fsq_and_eval_share_one_archive_pool_and_limit_generations(tmp_path):
    source = tmp_path / ".venv"
    (source / "bin").mkdir(parents=True)
    (source / "bin/python").symlink_to(sys.executable)
    config = source / "pyvenv.cfg"
    config.write_text("generation 0")
    common = SOURCE.with_name("node_local_venv.sh")
    fsq = SOURCE.parents[1] / "train_skills/FSQ/src/fsq_node_local_venv.sh"

    def prepare(helper, function):
        result = subprocess.run(
            ["bash", "-c", f"source {shlex.quote(str(helper))}; "
             f"{function} {shlex.quote(str(tmp_path))}"],
            check=True, capture_output=True, text=True,
        )
        return Path(result.stdout.strip())

    first = prepare(common, "prepare_node_local_venv_archive")
    assert prepare(fsq, "fsq_prepare_venv_archive") == first
    for i in range(1, 4):
        config.write_text(f"generation {i}")
        latest = prepare(fsq, "fsq_prepare_venv_archive")
    assert latest.exists()
    assert len(list((tmp_path / ".cache/node_local_venv").glob("*.tar.zst"))) == 2
    assert not (tmp_path / ".cache/fsq_venv").exists()
