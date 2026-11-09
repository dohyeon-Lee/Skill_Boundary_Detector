import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/lerobot/utils/eval_gpu_memory.py"
spec = importlib.util.spec_from_file_location("eval_gpu_memory", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_budget_keeps_headroom_and_caps_each_worker():
    result = module.budget(88 * module.GIB, 179 * module.GIB, 2)
    assert result["budget_bytes"] == 30 * module.GIB
    assert result["budget_bytes"] * 2 + result["reserve_bytes"] <= result["free_before_bytes"]
    assert module.budget(40 * module.GIB, 179 * module.GIB, 2)["budget_bytes"] == 12 * module.GIB
    with pytest.raises(ValueError):
        module.budget(25 * module.GIB, 179 * module.GIB, 2)


def test_selects_single_visible_gpu_without_ssh_or_cuda_initialization():
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return "0, GPU-mine, 183359, 91000\n"
    uuid, total, free = module.query({"CUDA_VISIBLE_DEVICES": "0"}, runner)
    assert uuid == "GPU-mine" and total > free
    assert calls[0][0] == "nvidia-smi"
    with pytest.raises(ValueError):
        module.query({"CUDA_VISIBLE_DEVICES": "0,1"}, runner)


def test_existing_stricter_process_limit_is_preserved():
    limits = []
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(total_memory=100 * module.GIB),
        memory=SimpleNamespace(get_per_process_memory_fraction=lambda _: .2),
        set_per_process_memory_fraction=lambda fraction, device: limits.append((fraction, device)))
    torch = SimpleNamespace(cuda=cuda)
    result = module.apply_limit(torch, {"SBD_EVAL_GPU_BUDGET_BYTES": str(30 * module.GIB)})
    assert limits == [(.2, 0)] and result["allocator_limit_gib"] == 20
    assert module.apply_limit(torch, {}) is None


def test_eval_limit_applied_before_ready_and_model_loading():
    script = SCRIPT.parents[3] / "examples/libero/configs/train_skillVLA/PT/eval/src/run_eval.py"
    tree = ast.parse(script.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "eval_main")
    names = [node.value.func.id for node in function.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)]
    assert names.index("apply_eval_gpu_memory_limit") < names.index("_mark_startup_ready")
