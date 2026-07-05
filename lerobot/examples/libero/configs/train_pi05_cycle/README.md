# train_pi05_cycle — block-cyclic PT (continual-learning curriculum) 미니실험

Replay-free continual learning 가설 검증용: PT를 iid 셔플 대신 **task 그룹 단위 block-cyclic
커리큘럼**으로 돌려서, 이후 FT 시 forgetting 저항성이 생기는지 본다 (PT 전용; FT/eval은 기존
`configs/train_pi05` 파이프라인 재사용).

> **연구 배경·가설·이론·설계 근거 전체는 [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)** —
> 새 환경/새 Claude 세션은 그 문서부터 읽을 것.

## 구성

```
src/lerobot_train_cycle.py       # 학습 스크립트 (lerobot 본체 무수정, 단독 루프)
src/train_pi05_cycle_config.py   # yaml → shell export 리졸버 (PT + eval 겸용)
src/aggregate_group_success.py   # eval 결과를 그룹별 success로 집계 (groups.json × LIBERO language)
cycle/cycle_config.yaml          # 실험 설정 (조건 토글)
cycle/cycle_PT.sbatch, submit_cycle_PT.sh
cycle_eval/cycle_eval_config.yaml, eval.sbatch, submit_eval.sh   # closed-loop 평가 + 그룹 집계
cycle_ft/ft_config.yaml, ft.sbatch, submit_ft.sh   # replay-free FT + 실시간 forgetting probe
src/lerobot_train_ft_probe.py                      # FT 학습 스크립트 (PT probe 계측 내장)
```

**FT (가설 판정)**: `cd cycle_ft` → `ft_config.yaml`의 `ft_source`를 조건별로 바꿔가며
`./submit_ft.sh`. libero_10을 replay 없이 FT하면서 `probe_every`마다 PT 그룹 probe를 측정
(PT 때와 같은 배치·노이즈 → 직접 비교 가능). wandb `VLA_cycle_FT`의 `probe/g{j}_forget` =
**FT 중 잊혀지는 실시간 곡선** — 조건 간 이 곡선 비교가 핵심 figure. FT 후 closed-loop 유지율은
cycle_eval을 FT 출력(groups.json 자동 복사됨)에 돌리면 그룹 집계까지 나옴.
주의: FT 레시피는 전 조건 동일 고정, normalizer는 PT ckpt 것 유지(의도된 설계, yaml 주석 참조).

**Eval**: `cd cycle_eval && ./submit_eval.sh` — `eval_model`(비우면 yaml 토글로 자동 구성)의
checkpoint를 libero_90 전체에서 rollout → `outputs/<run>/group_success.{json,png}` (그룹별
성공률 + not_in_pt 분리). env 오버라이드: `EVAL_MODEL=... CHECKPOINT=015000 ./submit_eval.sh`

**Compare eval**: `cd cycle_eval && ./submit_compare.sh` — yaml의 `compare_models`
({model_dir, label} 리스트, 2개 이상)를 **동일 init state에서** task별로 순차 rollout하고,
task 하나 끝날 때마다 그 자리에서 side-by-side 영상 합성(라벨 배너, 초록=성공/빨강=실패) →
`outputs_compare/<A_vs_B_ckpt>/videos/task{ID}_ep{E}.mp4` + `compare_summary.json` +
`compare_chart.png` (task별 모델 비교 바 + 전체 성공률 — 전부 task마다 갱신, 돌아가는 중에
열어봐도 됨). 모델은 한 번만 로드하고 task 루프를 돎.
멀티 GPU: `N_SHARDS=4 ./submit_compare.sh` → SLURM array 4개(각 1 GPU)가 task를 round-robin
분담, 전부 끝나면 merge 잡이 자동으로 shard summary들을 `compare_summary.json`+최종 차트로 합침.

## 조건 매트릭스 (yaml 토글 → run name 자동 인코딩)

| 조건 | 설정 | run name 예 |
|---|---|---|
| iid baseline | `cycle_iid_baseline=true` (권장: 동일 probe 계측) | `PTiid_..._g8p500` |
| pure cyclic | `delta_lambda=0, reptile_beta=1` | `PTcyc_..._g8p500` |
| cyclic + Δ | `delta_lambda=0.5` | `PTcyc_..._g8p500_lam05` |
| cyclic + Reptile | `reptile_beta=0.5` | `PTcyc_..._g8p500_b05` |
| 전부 | 둘 다 | `PTcyc_..._g8p500_lam05_b05` |

핵심 다이얼: `cycle_phase_steps` (k) — 섭동 크기/cross-term 이득. sweep 대상.
`cycle_n_cycles > 0`이면 phase_steps 대신 사이클 수를 지정 (`phase = steps//(groups×cycles)`
자동 계산, run name → `g{G}c{N}`). n_groups sweep 시 사이클 수 보존용 — 예:
`CYCLE_N_GROUPS=73 CYCLE_N_CYCLES=5 ./submit_cycle_PT.sh` (task 단위 그룹). 단 73그룹은
phase 경계가 촘촘해져 probe 오버헤드가 커짐(경계마다 전 그룹 forward) — 주의.

## 실행

```bash
cd cycle && ./submit_cycle_PT.sh                      # yaml 기본값으로 제출
CYCLE_PHASE_STEPS=1000 PT_EXP=p1k ./submit_cycle_PT.sh  # env 오버라이드 sweep
# 스모크 테스트 (1사이클 = 24스텝, Δ+Reptile 경로까지 전부 통과 확인):
PT_STEPS=24 CYCLE_PHASE_STEPS=3 CYCLE_DELTA_LAMBDA=0.5 CYCLE_REPTILE_BETA=0.5 \
  PT_SAVE_FREQ=24 PT_EXP=smoke ./submit_cycle_PT.sh
```

## wandb 지표 (phase 경계마다)

- `probe/g{j}_loss` — 그룹별 고정 probe loss (고정 배치 + 고정 flow-matching 노이즈 →
  파라미터 변화만 측정). `cycle/active_group`과 조합하면 간섭 행렬 Δ_ij·복구 곡선 복원 가능
- `probe/g{j}_forget` — 자기 phase 마지막 대비 상대 악화량 (Δ-feedback의 입력)
- `cycle/w_active` — 현재 phase의 Δ-가중치
- `probe/grad_cos_g{X}` — (옵션 `cycle_probe_grad_group≥0`) probe gradient 회전각,
  k-상한(테일러 유효성) 진단. boundary당 backward 1회 + CPU ~6GB

## 주의

- **단일 GPU 전용, resume 없음** (기존 output dir은 삭제 후 fresh 시작)
- Reptile 켠 채 `steps`가 사이클 배수가 아니면 마지막 부분 사이클에도 보간이 적용됨
- Adam 모멘트는 Reptile 보간 시 리셋하지 않음 (v1 설계; ablation 후보)
