"""Live progress stays correct across tqdm redraws, task resets and batches."""

import json
from pathlib import Path
import tempfile
import unittest

from lerobot.utils.eval_progress import plan, publish
from tools.runpod.desk_probe import batch_progress, evaluation_progress


def bar(n, total=10, rate=None):
    return f"\rStepping through eval batches: 0%| | {n}/{total}" + (
        f" [running_success_rate={rate}%]" if rate is not None else "")


class ProgressTests(unittest.TestCase):
    def test_redraw_postfix_and_rollout_success_never_count_extra_episodes(self):
        log = bar(0) + bar(0, rate=100) + bar(1, rate=100)
        log += bar(1, rate=50) + bar(2, rate=50)
        log += "\rRunning rollout with at most 520 steps: 20%| | 104/520 [running_success_rate=100.0%]"
        self.assertEqual(batch_progress(log, 10, 1), [{"completed": 2, "successes": 1}])

    def test_completed_task_redraw_and_next_task_reset(self):
        log = bar(0) + bar(10, rate=60) + bar(10, rate=60)
        log += bar(0) + bar(3, rate=66.7)
        self.assertEqual(batch_progress(log, 10, 1), [
            {"completed": 10, "successes": 6}, {"completed": 3, "successes": 2}])

    def test_partial_vector_batch_and_missing_rates(self):
        self.assertEqual(batch_progress(bar(0, 3) + bar(3, 3, 60), 5, 2),
                         [{"completed": 5, "successes": 3}])
        self.assertEqual(batch_progress(bar(0) + bar(1), 10, 1)[0]["successes"], None)
        with self.assertRaises(ValueError):
            batch_progress(bar(5, rate=40), 10, 1)

    def environment(self, root):
        return {"MODELS_JSON": json.dumps([{"label": "30k", "policy_path": "private"}, {"label": "40k"}]),
            "EVAL_GPU_GROUPS_JSON": json.dumps([[{"tag": "w0", "items": [0, 1], "panels": [0]},
                                                {"tag": "w1", "items": [0, 1], "panels": [1]}]]),
            "STAGE1_EVAL_WORK_DIR": str(root / "eval"), "SBD_LOCAL_JOB_ID": "123",
            "SBD_LOCAL_JOBS_DIR": str(root / ".cache/local_jobs"),
            "EVAL_OUT_DIR": str(root / "eval/outputs/run"), "TARGET_TASK": "libero_10",
            "N_EPISODES": "10", "EVAL_BATCH_SIZE": "1", "HF_TOKEN": "never-export"}

    def test_plan_and_two_checkpoint_aggregation_without_secret_or_path_leak(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            env = self.environment(root)
            self.assertNotIn("never-export", json.dumps(plan(env, "123")))
            self.assertNotIn("private", json.dumps(plan(env, "123")))
            publish(env, "123")
            logs = root / "eval/logs"
            logs.mkdir(parents=True)
            (logs / "123_0_w0.err").write_text(bar(0) + bar(10, rate=60) + bar(0) + bar(3, rate=66.7))
            (logs / "123_0_w1.err").write_text(bar(0) + bar(5, rate=40))
            base = root / ".cache/local_jobs/123"
            result = evaluation_progress(root, base, env["EVAL_OUT_DIR"])
            self.assertEqual((result["completed_episodes"], result["total_episodes"], result["percent"]), (18, 40, 45))
            self.assertEqual((result["successes"], result["observed_episodes"]), (10, 18))
            self.assertEqual(result["models"][0]["completed_tasks"], 1)
            self.assertIsNone(result["models"][1]["tasks"][1]["success_pct"])
            self.assertNotIn(str(logs), json.dumps(result))

    def test_log_paths_outside_project_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            env = self.environment(root)
            value = plan(env, "123")
            value["workers"][0]["log"] = "/etc/passwd"
            folder = Path(env["SBD_LOCAL_JOBS_DIR"])
            folder.mkdir(parents=True)
            (folder / "123.eval.json").write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                evaluation_progress(root, folder / "123", env["EVAL_OUT_DIR"])


if __name__ == "__main__":
    unittest.main()
