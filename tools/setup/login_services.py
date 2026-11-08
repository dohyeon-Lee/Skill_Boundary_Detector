"""Non-interactive login, called only by setup_env.sh --runpod.

RunPod injects HF_TOKEN and WANDB_API_KEY from template Secrets. SDK login
persists credentials for later shells; credentials never appear in argv/logs.
"""
from __future__ import annotations

import contextlib
import io
import os


def login_services() -> int:
    failed = False
    for service, variable, command in (
        ("Hugging Face", "HF_TOKEN", "hf auth login"),
        ("W&B", "WANDB_API_KEY", "wandb login"),
    ):
        token = os.environ.get(variable, "").strip()
        if not token:
            print(f"{service}: {variable} 없음 — 자동 로그인 생략. 필요 시 {command}")
            continue
        try:
            # SDK diagnostics and exception messages may contain credentials.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if variable == "HF_TOKEN":
                    from huggingface_hub import login

                    login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
                else:
                    import wandb

                    if not wandb.login(key=token, relogin=True, verify=True, timeout=30):
                        raise RuntimeError("Login was not verified")
        except Exception:
            failed = True
            print(f"{service}: 자동 로그인 실패. {variable}의 Secret 값·권한 및 네트워크를 확인하세요.")
        else:
            print(f"{service}: 로그인 완료.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(login_services())
