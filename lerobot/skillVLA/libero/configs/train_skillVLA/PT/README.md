# SkillVLA PT training

PT는 VSA와 Predictor를 각각 학습합니다. 데이터와 공통 모델 경로는
[`stage1_common_config.yaml`](stage1_common_config.yaml)에서 선택합니다.

| Component | Config | Launcher | Output |
| --- | --- | --- | --- |
| VSA | `VSA/vsa_train_config.yaml` | `VSA/submit_train.sh` | `outputs_root/skillVLA_PT/VSA/<run>` |
| Predictor | `Predictor/predictor_train_config.yaml` | `Predictor/submit_train.sh` | `outputs_root/skillVLA_PT/Predictor/<run>` |

Predictor PT·FT의 공통 실행기와 VSA FT가 쓰는 설정 함수는
[`src/auxiliary`](src/auxiliary)에 있습니다. VSA의 병렬 실행기도 이 위치를 사용합니다.
별도 Terminator 학습 진입점은 제거했고, 현재 VSA는 내장 termination reader를 학습합니다.
기존 외부 Terminator 체크포인트를 사용하는 평가는 계속 가능합니다.
제출 시 YAML과 실행 셸을 snapshot으로 저장하고, Python 모듈은 공통 소스에서 읽습니다.

## VSA: 세 가지 Term bottleneck 모델

설정 파일은 `VSA/vsa_train_config.yaml` 하나를 사용합니다. 공통 architecture는
`UV_Bspline_denoising_dualbottleneck_pi05vlm_9_9_noproprio_termbn_detach`이며,
아래 두 옵션으로 모델과 결과 폴더명이 결정됩니다.

| 모델 구분 | `vision.top_crop.enabled` | `bspline.gripper_action` | full-skill target |
| --- | --- | --- | --- |
| `termbn_detach` | false | false | pose 6D + finger proprio 2D |
| `termbn_detach_topcrop1p1_gripaction` | true | true | pose 6D + gripper action 1D |
| `termbn_detach_gripaction` | false | true | pose 6D + gripper action 1D |

`top_crop.zoom: 1.1`일 때 `_topcrop1p1`, action gripper를 쓸 때 `_gripaction`이
자동으로 붙습니다. 현재 YAML은 마지막 모델을 선택합니다. `run.suffix: base`와
기존 이름 생성 규칙을 유지하므로 이어학습 폴더도 그대로 결정됩니다.

세 모델 모두 현재 Top/Wrist 이미지를 π0.5 VLM으로 인코딩하고, 두 독립적인
100-token bottleneck이 Action Expert의 L1–L9 / L10–L18에 연결됩니다.
Expert에 현재 proprio를 직접 넣지 않으며, full-skill readout은 L9입니다.
별도의 termination reader는 각 VLM 층을 detach해서 읽는 18층 bottleneck으로,
Action Expert에 연결되지 않고 gradient clipping도 본류와 분리됩니다.

`vlm_termination`은 loss weight 1, sigma 1, positive weight 2, 종료 후 10스텝을
사용합니다. Skill transition randomization에 따른 현재/이전 스킬과 UV 처리는
그대로 유지합니다. Termination loss는 action loss와 별도로 기록합니다.

Top crop을 켜면 학습 중 random crop, 평가 중 center crop을 사용하며,
VLM·Expert의 Top UV 조건과 visual auxiliary UV 정답을 함께 변환합니다.
XYZ와 Wrist 목표는 원래 좌표를 유지합니다. Full-skill은 30개 B-spline
샘플로 학습하며 `bspline.gripper_action`으로 마지막 gripper 표현만 바꿉니다.

데이터셋·transition randomization·π0.5/tokenizer 경로는
`stage1_common_config.yaml`에서 설정합니다. VSA YAML에는 모델 선택과 loss,
학습·저장·실행 설정만 둡니다. DINO 선택, 비활성 Best-of-N, 과거 VSA ablation
설정은 현재 설정 목록에서 제거했습니다.

```bash
cd lerobot/skillVLA/libero/configs/train_skillVLA/PT/VSA
./submit_train.sh
# 기존 할당 사용: ./submit_train.sh --JOB_ID
# 연세 서버 사용: ./submit_train.sh --yonsei
```

`training.gradient_checkpointing`에서 LLM(`vlm`), Vision(`vision`),
Action Expert(`action_expert`), Skill planner(`skill_planner`)를 각각 지정합니다.
현재는 모두 false입니다. 체크포인트는 최신 1개와 50k/100k를 보관하며,
동일 모델·suffix로 재실행하면 `checkpoints/last`에서 이어학습합니다.

## Predictor modes and spatial targets

Only `mode1` and `mode2` are selectable for new training. Both use Top/Wrist/proprio,
1.1x Top zoom and random crop during training. `mode2` additionally reads the previous
skill (GT during training, the executed skill at evaluation, learned BOS at reset).
Both condition the spatial reader on the selected current skill.

`skill_predictor.spatial_target: uv` predicts absolute skill-end UV in the cropped
Top image. Finite out-of-frame targets are clipped to the border and supervised.
Evaluation uses a center crop and restores UV to the original-image coordinate system.
`xyz` predicts skill-end XYZ minus the measured jittered-start XYZ, which is added
back outside the network at inference. Transition randomization is a separate setting.

Historical `mode3` and `mode3_preskill` checkpoints remain readable without moving
or rewriting their files. They map in memory to `mode1` and `mode2`, respectively.
New checkpoints carry a mode schema so old, different `mode1`/`mode2` models cannot
be mistaken for the renamed models. Other retired variants are not training choices.

## Predictor modes sharing one GPU

`Predictor/predictor_train_config.yaml` keeps the original single-model behavior
with a scalar `skill_predictor.mode: mode2`. To train independent models
simultaneously on one large GPU, explicitly select a list instead:

```yaml
skill_predictor:
  mode: [mode1, mode2]
training:
  dataloader: {batch_size: 16, workers: 6, gpus: 1}
slurm: {gres: "gpu:1", cpus: 16, memory: 192G, time: "48:00:00"}
```

Run the same `Predictor/submit_train.sh`. This submits **one allocation**, not a
job array: both trainers inherit the same `CUDA_VISIBLE_DEVICES`, have their own
weights/optimizer/W&B run, and use the usual mode-specific output folders. The
dataset is staged only once. Batch size and workers apply **per model**; Slurm
CPU and host-memory limits are **totals**. The example resources are not a VRAM
guarantee: model/optimizer/activation memory adds up on the shared GPU, and GPU
compute is shared too. No MPS or fractional GPU reservation is enabled.

The allocation log reports each worker's status; separate stdout/stderr and
exit codes are under `Predictor/logs/predictor_group_<job id>/`. A failed worker
does not cancel healthy ones, but the allocation exits nonzero if any failed.
Cancellation stops the whole group, including dataloader workers. Resubmitting
uses each model's own `checkpoints/last`, just like a scalar run. Duplicate modes
and multi-GPU list configurations are rejected before submission.

Use `SKILL_AUX_DRY_RUN=1 bash Predictor/submit_train.sh` to inspect the snapshot,
resolved output paths, and one-GPU request without submitting. Lists are PT-only;
FT continues to inherit its architecture from its warm-start checkpoint.

## GPU allocation in Stage-1 eval

The existing `eval_num_gpus` and `eval_max_workers_per_gpu` settings also apply
to multi-model comparisons and checkpoint sweeps. Work is split over **task x
panel** pairs, where one panel is one model/checkpoint combination. For example,
ten tasks and nine panels with `eval_num_gpus: 10` and
`eval_max_workers_per_gpu: 5` request ten one-GPU array jobs, each running five
independent evaluation processes and covering nine task/panel pairs in total.
GPU count is an upper bound; a smaller evaluation uses fewer GPUs. The per-GPU
process cap is 1–10. Inside an existing allocation, evaluation uses one GPU
and processes its assigned task/panel groups there.

Every worker keeps the full panel order for comparison videos and evaluates
only its assigned panels. Existing seeds, resume checks, panel directories,
merged success-rate tables/charts and side-by-side video layouts are preserved.
This distributes more independent work; throughput still depends on available
GPUs and the CPU/VRAM resources shared by each GPU's processes.

## Checkpoint sweeps in Stage-1 eval

In `eval/pt_eval_config.yaml`, use
`model_defaults.checkpoint: [10000, 20000, 50000]` to evaluate every model at
those checkpoints, or set a model's `checkpoint` list. Numeric steps are
normalized to six-digit folder names. Multiple models use the same ordered
checkpoint list for side-by-side evaluation.

The result folder retains the existing bar charts and also writes these sweep
line charts automatically: `checkpoint_success_rates.png` overlays the models'
overall mean SR; `checkpoint_task_success_rates/<index>_<model>.png` contains
one chart per model with a separate line for each task. Thus one model produces
two line charts, and N models produce N+1. Curves sort checkpoints numerically,
and unavailable task results are gaps. Every finished array worker refreshes
the charts using the metrics available so far.
