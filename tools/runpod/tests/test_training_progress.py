"""Read existing training logs, including resumed and parallel trainers."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.runpod.desk_probe import snapshot, training_progress


class TrainingProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "lerobot/examples/libero/configs").mkdir(parents=True)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.job = {"name": "skill_aux", "output": str(self.logs / "train.out"),
                    "error": str(self.logs / "train.err")}

    def run_folder(self, name="predictor_mode2", saved=70000, target=100000):
        run = self.root / "outputs_filtered/Predictor" / name
        checkpoint = run / "checkpoints" / f"{saved:06d}"
        (checkpoint / "training_state").mkdir(parents=True)
        (checkpoint / "training_state/training_step.json").write_text(json.dumps({"step": saved}))
        (checkpoint / "pretrained_model").mkdir()
        (checkpoint / "pretrained_model/train_config.json").write_text(json.dumps({"steps": target}))
        (checkpoint.parent / "last").symlink_to(checkpoint.name)
        return run

    def progress(self):
        return training_progress(self.root, self.job, [16 * 1024 * 1024])

    def test_current_live_log_and_saved_checkpoint_are_separate_and_no_raw_logs_escape(self):
        run = self.run_folder()
        (run / "checkpoints/080000").mkdir()  # An unfinished save is not a checkpoint.
        Path(self.job["output"]).write_text(f"schedule  : steps=100000 mode=warmup_constant\noutput : {run}\n")
        Path(self.job["error"]).write_text(f"INFO Output dir: {run}\nINFO cfg.steps=100000 (100K)\n"
            "HF_TOKEN=never-export-this\n\rTraining: 79%|████| 79123/100000 [24:48:11<6:36:13, 1.14s/step]\r")
        value = self.progress()
        self.assertEqual((value["completed_steps"], value["total_steps"]), (79123, 100000))
        self.assertAlmostEqual(value["percent"], 79.123)
        self.assertEqual(value["runs"][0]["latest_checkpoint"], 70000)
        self.assertNotIn("never-export-this", json.dumps(value))
        self.assertNotIn(str(self.root), json.dumps(value))

    def test_resumed_tqdm_counter_is_relative_to_remaining_steps(self):
        run = self.run_folder(saved=70000)
        Path(self.job["error"]).write_text(f"Output dir: {run}\ncfg.steps=100000 (100K)\n"
            "Training: 50%|████| 25000/50000 [1:00<1:00, 1.1s/step]\r")
        value = self.progress()
        self.assertEqual(value["completed_steps"], 75000)
        self.assertEqual(value["percent"], 75)

    def test_new_checkpoint_target_supersedes_old_header_in_appended_log(self):
        run = self.run_folder(saved=100000, target=150000)
        Path(self.job["error"]).write_text(f"Output dir: {run}\ncfg.steps=100000\n" + " " * 270000
            + "\rTraining: 2%| | 1000/50000 [1:00<1:00]\r")
        value = self.progress()
        self.assertEqual((value["completed_steps"], value["total_steps"]), (101000, 150000))

    def test_unknown_target_or_inconsistent_resume_never_fabricates_progress(self):
        Path(self.job["error"]).write_text("Training: 20%| | 10000/50000 [1:00<1:00]\r")
        value = self.progress()
        self.assertIsNone(value["completed_steps"])
        self.assertIsNone(value["percent"])
        run = self.run_folder(saved=90000)
        Path(self.job["error"]).write_text(f"Output dir: {run}\ncfg.steps=100000\n"
            "Training: 20%| | 10000/50000 [1:00<1:00]\r")
        self.assertIsNone(self.progress()["completed_steps"])

    def test_parallel_trainers_keep_distinct_progress_and_weighted_total(self):
        a, b = self.run_folder("mode_a", 50000), self.run_folder("mode_b", 20000)
        Path(self.job["output"]).write_text("\n".join(
            f"START mode_{key} pid=123 CUDA_VISIBLE_DEVICES=0 output={run} logs={self.logs / key}.{{out,err}}"
            for key, run in (("a", a), ("b", b))))
        for key, run, step in (("a", a, 50000), ("b", b, 25000)):
            (self.logs / f"{key}.err").write_text(f"Output dir: {run}\ncfg.steps=100000\nTraining: 50%| | {step}/100000\r")
        value = self.progress()
        self.assertEqual((value["completed_steps"], value["total_steps"], value["percent"]), (75000, 200000, 37.5))
        self.assertEqual([run["completed_steps"] for run in value["runs"]], [50000, 25000])

    def test_failed_job_keeps_real_step_and_is_not_marked_one_hundred_percent(self):
        Path(self.job["error"]).write_text("cfg.steps=100000\nTraining: 10%| | 10000/100000\rTraceback: OOM\n")
        jobs = self.root / ".cache/local_jobs"
        jobs.mkdir(parents=True)
        (jobs / "123.job").write_text(f"name=skill_aux\nscript={self.root}/train.sbatch\n"
            f"output={self.job['output']}\nerror={self.job['error']}\n")
        (jobs / "123.exit").write_text("1")
        with patch("tools.runpod.desk_probe.subprocess.check_output", side_effect=OSError):
            job = snapshot(self.root)["jobs"][0]
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["training_progress"]["percent"], 10)

    def test_large_logs_read_only_bounded_head_and_tail(self):
        path = Path(self.job["error"])
        with path.open("wb") as stream:
            stream.write(b"cfg.steps=100000\n")
            stream.seek(80 * 1024 * 1024)
            stream.write(b"\rTraining: 79%| | 79123/100000\r")
        budget = [512 * 1024]
        value = training_progress(self.root, self.job, budget)
        self.assertEqual(value["completed_steps"], 79123)
        self.assertEqual(budget[0], 256 * 1024)

    def test_log_and_checkpoint_symlinks_cannot_read_outside_project(self):
        with tempfile.TemporaryDirectory() as temp:
            secret = Path(temp) / "secret.log"
            secret.write_text("cfg.steps=100000\nTraining: 50%| | 50000/100000\r")
            Path(self.job["error"]).symlink_to(secret)
            with self.assertRaises(ValueError):
                self.progress()
            Path(self.job["error"]).unlink()
            run = self.root / "outputs_filtered/run"
            (run / "checkpoints").mkdir(parents=True)
            (run / "checkpoints/last").symlink_to(Path(temp))
            Path(self.job["error"]).write_text(f"Output dir: {run}\ncfg.steps=100000\n")
            with self.assertRaises(ValueError):
                self.progress()


if __name__ == "__main__":
    unittest.main()
