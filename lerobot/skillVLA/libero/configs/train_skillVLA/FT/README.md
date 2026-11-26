# SkillVLA FT

VSA와 Predictor는 [`newtask_ft_common_config.yaml`](newtask_ft_common_config.yaml)의
데이터 선택과 각 PT 체크포인트를 사용해 독립적으로 FT합니다.

| Component | Config | Launcher | Output |
| --- | --- | --- | --- |
| VSA | `VSA/vsa_ft_config.yaml` | `VSA/submit_train.sh` | `outputs_root/skillVLA_FT/VSA/<run>` |
| Predictor | `Predictor/predictor_ft_config.yaml` | `Predictor/submit_train.sh` | `outputs_root/skillVLA_FT/Predictor/<run>` |

현재 VSA는 내장 termination reader를 사용합니다. 별도 Terminator 학습과 Joint FT
진입점은 제거했습니다. 기존 체크포인트와 평가 결과는 유지하며, 과거 모델의 평가도
저장된 계약대로 지원합니다. Predictor 공통 코드는 `../PT/src/auxiliary/`에 있습니다.
`NEWTASK_FT_DRY_RUN=1 VSA/submit_train.sh` 또는
`SKILL_AUX_DRY_RUN=1 Predictor/submit_train.sh`로 제출 전 설정을 확인합니다.

VSA는 GT skill·spatial target으로 학습하므로 Predictor와 동시에 학습할 수 있습니다.
Transition randomization은 각 모델의 PT 체크포인트에서 상속하며, VSA는 일반 프레임 샘플링으로
학습합니다. 공통 trainer는 W&B에 checkpoint weights를 올리지 않습니다.

## Dataset requirements

Run `./visualize_sbd.sh` from this directory to compare the saved boundaries
and FSQ codes configured under `sbd_preview.builds` in the common YAML. The
defaults compare the original `full_1`, `full_full ft`, and `full_full ft_own`
builds on the episodes selected by `dataset`. Each episode has one common
video and three time-aligned plots/tracks; clicking a track/plot seeks the
video. Subset-local episode IDs are mapped back through `source_episode_indices`
so the columns always refer to the same original recording. An unavailable
episode in the old `full_1` build is marked missing, never replaced by a
different episode. `--all-episodes` browses the entire current build without
changing the training selection; `--dry-run` validates the mapping without
rendering. This uses the project venv and CPU only, and neither runs DP nor
resegments the data. HTML goes under `FT/outputs/sbd_comparison_*`, with videos
and SVGs stored separately under `src/` to keep the main report small.

LIBERO can build `libero_10_full_full` once and select training episodes in
`newtask_ft_common_config.yaml`. With `dataset.episode_selection: per_task`,
`n_episodes: 1` selects the first episode in each task, ordered by the dataset's
episode IDs. `n_episodes: all` uses the full build. A nonempty nested
`episode_ids` overrides the count: one inner list per task in ascending
`task_index` order, containing **zero-based positions within that task**.
For example, `[[0, 2], [1]]` selects the first and third episodes of the first
task and the second episode of the next task; `[]` skips that task. Missing
task lists and unavailable positions fail before submission. VSA and Predictor receive the same resolved global episode IDs. Output names add
`_ep<count>_pertask` and an ID hash for explicit selections. MimicGen's default
`global` selection continues to use a flat list of global episode IDs.

The configured source/run must already have a complete SkillVLA build; episode
selection does not build missing data. Use the same segmentation and PT
normalization statistics when comparing this selection with `libero_10_full_1`.

* Source checkpoints are named as `{run, checkpoint}` in the shared YAML and located automatically.
* Labeled by the same FSQ model as the source checkpoints. This is verified
  from `fsq_source.json` on both sides; the checkpoints' `skill_code_space_id`
  is then kept even though the new run tag differs.
* Same FSQ levels, state/action dimensions, and `proprio_grounding`.
* 세 VSA 모델 모두 `skill_focus_uv.npz`가 필요합니다
  (`focus_uv.enabled=true`로 빌드).

## VSA: 세 가지 Term bottleneck 모델의 FT

`VSA/vsa_ft_config.yaml` 하나로 PT의 세 모델을 모두 FT합니다.
`newtask_ft_common_config.yaml`의 `warm_start.vsa_checkpoint`에서
`termbn_detach`, `termbn_detach_topcrop1p1_gripaction`, `termbn_detach_gripaction`
중 원하는 PT run과 checkpoint를 선택합니다.

Architecture, 9+9 split, detach termination reader, Top crop 설정,
full-skill gripper 표현(8D proprio / 7D action), normalization 통계와
loss coefficient는 선택한 PT 체크포인트에서 상속합니다.
FT YAML에서 이를 다시 정의하지 않습니다. 결과 폴더에는 원본 PT run의
식별자가 유지되며, 긴 이름은 기존 규칙에 따라 일부를 hash로 줄입니다.

현재 adaptation 설정:

- `freeze_expert: true`: Expert 전 층·Action I/O·full-skill head를 동결합니다.
- `skill_flow_loss: true`: full-skill loss의 gradient는 동결된 Expert를 통과해
  학습 가능한 visual bridge·bottleneck·VLM으로 전달됩니다.
- `visual_aux_loss: true`: 기존 Top/Wrist/XYZ auxiliary supervision을 유지합니다.
- `termination_loss: true`: 독립된 detach termination reader를 함께 FT합니다.
  false이면 termination supervision을 끄며 출력 이름에 `_notermloss`를 붙입니다.

VLM과 vision encoder는 학습하며, 별도의 DINO 선택이나 LR-scale 옵션은
현재 VSA YAML에 노출하지 않습니다. Gradient checkpointing은
`vlm`, `vision`, `action_expert`로 각각 설정합니다.

데이터와 task별 episode 선택은 공통 YAML을 따릅니다.
Transition randomization의 범위·분포·gripper 보존은 각 warm-start 체크포인트에서
상속하고, FT 데이터셋의 사전 계산 범위 안인지 제출 전에 검증합니다.
VSA는 표준 프레임 샘플링을 사용합니다.
저장 경로와 서버 선택은 global config를 따릅니다.

```bash
cd lerobot/skillVLA/libero/configs/train_skillVLA/FT/VSA
./submit_train.sh
# 기존 할당 사용: ./submit_train.sh --JOB_ID
# 연세 서버 사용: ./submit_train.sh --yonsei
```

총 20k steps, 최신 체크포인트 1개와 `[3000, 5000, 10000, 20000]`을 보관합니다.
20k 종료까지 해당 보호 step에서도 저장하며, 완료된 저장을 확인한 뒤 이전
일반 체크포인트를 정리합니다. 재실행 시 기존 `last`의 optimizer·RNG도 복원합니다.

## Predictor

`Predictor/submit_train.sh`는 `../PT/src/auxiliary/submit_auxiliary.sh`를 사용합니다.
PT 구조·정규화 통계를 선택한 Predictor 체크포인트에서 상속하고, 데이터와 episode
선택은 공통 YAML을 따릅니다. `warm_start.predictor_checkpoint`에 여러 체크포인트를
지정하면 한 GPU에서 독립적인 FT를 병렬 실행합니다. VSA는 이 Predictor 선택을 무시합니다.
각 모델의 batch·worker 수는 component YAML에서 정하며, GPU·CPU·메모리는 합산해 확보합니다.

## Evaluation

`eval/ft_eval_config.yaml` 하나에서 VSA/Predictor checkpoint와 평가 조건을 선택하고
`eval/submit_eval.sh`를 실행합니다. 현재 세 VSA 모델은 `advance_mode: own`으로
내장 termination 확률을 사용합니다. 기존 외부 Terminator 또는 Joint checkpoint를
읽는 평가 코드는 과거 결과 재현을 위해 유지합니다.
