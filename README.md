# Skill Boundary Detector (SkillVLA)

## 1. RunPod 계정·SSH 준비

- HF **Write 토큰**과 [W&B API 키](https://wandb.ai/authorize)를 준비한다.
- HF에서 [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224), [DINOv3-S](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), [DINOv3-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 약관에 동의한다.
- RunPod Secrets에 키를 저장하고 개인 템플릿에 연결한다. 실제 키는 Git에 저장하지 않는다.

| 템플릿 환경변수 | Secret 참조 |
| --- | --- |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` |
| `WANDB_API_KEY` | `{{ RUNPOD_SECRET_wandb_key }}` |

템플릿에서 **22/tcp**, public IP, SSH 공개키를 설정한다.
**SSH·VS Code를 사용할 PC**에서 실행한다. 첫 실행에 RunPod API 키를 입력하면 이후 재사용한다.
Python 3.9 이상이 필요하며 Windows에서는 Git Bash를 사용한다.

```bash
bash runpod_ssh.sh   # 새 Pod의 IP·포트로 Host runpod 갱신
ssh runpod
```

## 2. 설치

RunPod 터미널에서 실행한다.

```bash
cd /workspace
git clone --branch skillVLA_VSA https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git
cd Skill_Boundary_Detector
bash setup_env.sh --runpod
```

기본 환경(VSA·Predictor·Terminator)을 설치하고 Secrets로 HF·W&B에 로그인한다.
다른 서버·로컬은 `bash setup_env.sh`를 사용하며 자동 로그인하지 않는다.
MolmoAct2·GR00T는 필요할 때만 추가한다.

```bash
bash setup_env.sh --molmoact2-only
bash setup_env.sh --groot-only
```

## 3. SkillVLA PT·FT 평가

서버와 RunPod에서 사용할 **eval 폴더**로 이동한다. PT는 경로의 `FT`를 `PT`로 바꾼다.

```bash
cd lerobot/examples/libero/configs/train_skillVLA/FT/eval
```

| 평가 | 읽는 YAML |
| --- | --- |
| FT | `ft_eval_config.yaml` |
| PT | `pt_eval_config.yaml` |

```bash
# 연구실 서버: HF 준비만 수행 (Slurm 제출 안 함)
bash submit_eval.sh --prepare-runpod

# RunPod: 필요한 입력 다운로드·해시 검증 후 평가
bash submit_eval.sh
```

- HF에 있는 파일은 재사용한다. 없는 데이터는 **기존 비공개 데이터 경로**, 체크포인트는 **공개 저장소의 기존 `outputs_*` 경로**로 Desk에서 업로드한다.
- 체크포인트는 평가용 `pretrained_model`만 전송한다. 작은 설정·다운로드 목록은 `dataset_filtered/eval_configs/pt.json` 또는 `ft.json`에 저장한다.
- RunPod는 서버가 준비한 YAML 스냅샷을 사용한다. **YAML·체크포인트 변경 후에는 서버에서 다시 준비한다.**
- 다운로드만 하려면 `--download-only`를 붙인다. 데이터 전체를 미리 받을 필요는 없다.
- Slurm 서버에서 옵션 없이 실행하면 기존처럼 평가 작업을 제출한다.
- GPU별 worker의 모델 로딩은 순차적으로, 평가는 병렬로 실행해 로딩 중 RAM 피크를 줄인다.

스킬 경계 비교 평가도 같은 흐름으로 실행한다.

```bash
cd lerobot/examples/libero/configs/train_skills/skill_eval/skill_boundary_detection
bash submit.sh --prepare-runpod   # 서버: HF 입력 준비
bash submit.sh                    # RunPod: 다운로드·검증 후 평가
```

조건별 threshold는 전체 3,921개 에피소드에서 따로 계산하고, 보고서는 선택한 200개를 담는다.
실행 설정은 HF에 보관하며, 코드 변경은 별도로 RunPod에 반영해야 한다.

## 4. 학습·기타 작업

프로젝트 루트에서 필요한 데이터와 모델을 받는다. 일부만 선택하려면 `bash hf_sync.sh pull`을 사용한다.

```bash
bash hf_sync.sh pull dataset_filtered --models --yes
```

이어학습용 체크포인트는 `<run>`을 실제 폴더명으로 바꿔 받는다.

```bash
bash hf_sync.sh pull \
  --checkpoints 'outputs_filtered/skillVLA_PT/VSA/<run>/checkpoints/010000' --yes
```

각 작업 폴더의 기존 `submit_*.sh`를 실행한다. Slurm 제출 / RunPod 백그라운드 실행은 자동 선택한다.
RunPod 감지가 안 되면 `export SBD_SERVER=runpod`로 지정한다.

```bash
# 예: VSA PT 학습 (프로젝트 루트에서)
bash lerobot/examples/libero/configs/train_skillVLA/PT/VSA/submit_train.sh
```

## 5. 작업 관리·체크포인트 보관

연구실 서버에서 `bash sync_server_ui.sh`를 실행하고 **8765** 포트를 전달하면
**RunPod Desk**에서 GPU·잡·누적 비용·전체·태스크별 성공률을 확인하고 Pod를 중단할 수 있다.
실행 중인 평가의 진행률과 체크포인트·태스크별 중간 성공률은 잡 이름을 펼쳐 확인한다.
학습 잡은 현재/목표 step, 진행률, 최근 저장 체크포인트를 표시한다.
수집된 결과는 Pod 삭제 후에도 서버에 남는다. [설정·비용 안내](tools/runpod/DESK_README.md).
결과 폴더 전체를 공개 HF에 보관한 뒤 Pod를 자동 중단하는 방법도 같은 안내에 있다.

RunPod 작업 로그는 각 실행 폴더의 `logs/`에 남는다. 프로젝트 루트에서 조회·중단한다.

```bash
bash lerobot/examples/libero/configs/src/submit_job.sh list
bash lerobot/examples/libero/configs/src/submit_job.sh stop <id>
```

체크포인트 자동 업로드는 tmux에서 실행한다.

```bash
tmux new -s hfwatch
bash hf_sync.sh watch --keep 3 --protect 050000,100000
```

**RunPod에서는 최근 3개와 보호 step 외의 오래된 체크포인트를 HF·로컬에서 삭제한다.**
`Ctrl-b`, `d`로 tmux에서 나온다. 컨테이너 디스크는 Pod 정지·삭제 시 사라질 수 있으므로 종료 전에 업로드를 확인한다.

[Hugging Face Desk 사용법](tools/server_sync/HF_README.md) · [전송 속도 측정 기록](tools/runpod/TRANSFER_BENCHMARK_2026-11-09.md)
