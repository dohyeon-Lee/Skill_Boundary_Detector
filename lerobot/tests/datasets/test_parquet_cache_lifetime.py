"""Local parquet loading must not accumulate shared Arrow generations."""
import json
import os
from pathlib import Path
import subprocess
import sys

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.io_utils import load_nested_dataset


def test_filtered_parquet_uses_temporary_cache_and_preserves_source(tmp_path, monkeypatch):
    parquet = tmp_path / "data/chunk-000/file-000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 0, 1], "value": [1., 2., 3.]}), parquet)
    original = parquet.read_bytes()
    shared = tmp_path / "shared-hf-cache"
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", shared)
    ds = load_nested_dataset(parquet.parent.parent, episodes=[1])
    assert ds[0] == {"episode_index": 1, "value": 3.}
    assert len(ds) == 1
    assert ds.cache_files and all(Path(f["filename"]).is_file() for f in ds.cache_files)
    assert not shared.exists()
    assert parquet.read_bytes() == original


def test_temporary_cache_survives_spawn_then_is_removed_on_parent_exit(tmp_path):
    parquet = tmp_path / "data/chunk-000/file-000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 1], "value": [10, 20]}), parquet)
    # An actual spawned process checks that cleanup ownership isn't pickled
    # into a DataLoader worker and that the parent cache stays alive for it.
    script = tmp_path / "check_spawn.py"
    script.write_text('''
import json, multiprocessing as mp, sys
from pathlib import Path
from lerobot.datasets.io_utils import load_nested_dataset
def worker(ds, queue):
    queue.put(ds[1]["value"])
if __name__ == "__main__":
    ds = load_nested_dataset(Path(sys.argv[1]))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=worker, args=(ds, queue))
    process.start()
    assert queue.get(timeout=30) == 20
    process.join(timeout=30)
    assert process.exitcode == 0
    paths = [f["filename"] for f in ds.cache_files]
    assert all(Path(p).is_file() for p in paths)
    print(json.dumps(paths))
''')
    proc = subprocess.run([sys.executable, str(script), str(parquet.parent.parent)],
                          capture_output=True, text=True, timeout=90, env=os.environ.copy())
    assert proc.returncode == 0, proc.stderr
    paths = json.loads(proc.stdout.strip().splitlines()[-1])
    assert paths and all(not Path(p).exists() for p in paths)
    assert parquet.is_file()
