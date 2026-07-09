# RESEARCH_CONTEXT — block-cyclic PT 커리큘럼 연구 (continual learning)

> 이 문서는 다른 서버/새 Claude Code 세션으로의 핸드오프용. 이 폴더의 코드가 왜 존재하고,
> 어떤 가설을 검증하며, 설계 결정들이 왜 그렇게 내려졌는지를 담는다. (2026-07-03 작성)

## 0. 한 줄 요약

**pi05(VLA)의 pretraining을 iid 셔플 대신 "task 그룹 block-cyclic 커리큘럼(+Δ-feedback,
+Reptile)"으로 돌리면, 이후 새 task를 replay 없이 FT할 때 기존 task를 덜 잊는 파라미터
구조가 미리 형성된다** — 는 가설의 검증 실험 (PT 전용 미니실험, pi05에서 먼저; 최종 목표는
skillVLA 적용).

## 1. 문제 설정

- 최종 목표: skillVLA의 continual learning — 새 task FT 시 기존 task를 **replay 데이터 없이**
  최대한 보존.
- 제약의 정확한 의미: **FT(적응) 시점**에 과거 데이터/버퍼 금지. PT 시점엔 전체 데이터가
  어차피 있으므로 뭐든 허용 (PT 안에서의 revisit은 replay 위반이 아님).
- 이 구도에 원형이 맞는 기존 방법 = OML (meta-CL): 메타학습 때 다 쓰고 배포 후엔 맨몸.
  단 OML은 2차 미분(inner loop 관통 backprop)이라 3B VLA에서 계산 불가 →
  **이 연구 = OML의 목적을 1차 연산만으로 근사하는 시도**.

## 2. 가설의 구조 (중요: 두 층위 구분)

- **층위 1 (행동 보존)**: 옛 입력에 옛 출력. distillation/GT supervision이 다루는 것.
- **층위 2 (구조 강건성)**: 그 매핑이 내부적으로 어떻게 구현되어 있어서 **미래의 gradient
  업데이트가 안 건드리는가**. gradient 기하 문제 — 우리가 원하는 건 이것.
- 어떤 loss 항도 revisit 시점에 층위 2를 직접 못 만든다 (저장된 스칼라엔 방향 정보가 없음 —
  "두 시나리오 사고실험"). 층위 2는 **사이클 동역학의 창발적 산물**로만 유도 가능하다는 게
  이 방법의 베팅.
- 스트레스 테스트 논리: forgetting의 원인은 task-응집적 거시 변위인데, iid는 그런 섭동을
  한 번도 안 가해서 강건해질 선택압이 없다. block phase가 그 섭동의 리허설(=FT의 리허설)
  역할. adversarial training과 같은 논리.

## 3. 이론 배경 (대화에서 유도한 것들의 요약)

### 3.1 실질적 목적함수

순차(블록) SGD + 매 사이클 순서 셔플의 평균 업데이트는 다음의 gradient descent와 일치
(α² 차수까지):

```
F = Σᵢ Lᵢ − c·Σᵢ≠ⱼ ⟨gᵢ, gⱼ⟩,   c ∝ α × phase길이(k)
```

- 어디에도 내적을 계산하는 코드는 없음 — 뒤 스텝의 gradient가 "앞 스텝이 옮겨놓은 지점"에서
  평가된다는 사실만으로 cross term `α²·H_B·g_A`가 변위에 자동으로 담김.
- 셔플이 필수인 이유: A→B 판은 H_B·g_A만, B→A 판은 H_A·g_B만 줌. 둘의 평균이어야
  `∇⟨g_A,g_B⟩ = H_A·g_B + H_B·g_A` (항등식)로 완성됨.
- cross term은 스텝 **쌍**마다 생겨 그룹 간 ~k²로 누적, 1차 항은 ~k → **k가 정렬신호 증폭기**.
- 이 항은 α²라 loss를 못 이김 — "loss 동률인 해들 중 정렬된 해를 고르는 동률 심판".

### 3.2 순수 cyclic만의 독립 메커니즘 (2차 효과와 무관)

revisit 반복 = alternating projection: 각 phase가 그 그룹의 해집합으로의 사영 → 사이클이
**교집합(모든 그룹을 동시에 푸는 영역)으로 기하 수렴** (Evron et al. 증명, 선형 세팅).
단 이건 "본 task retention"의 안전망일 뿐 — iid도 교집합엔 도달함. 차별점은 **교집합
manifold 위의 어느 점에 앉느냐**: iid는 배치 노이즈 implicit bias가 고른 점, cyclic은
"거시 블록 섭동을 수백 번 맞고도 안 밀리게 된" 스트레스-선별된 점.

### 3.3 Reptile (β 보간)의 정확한 역할

- Reptile = "phase/사이클 변위 Δ를 gradient처럼 쓰는 outer SGD". `θ ← anchor + β(θ−anchor)`.
- **β는 증폭기가 아님** (Δ의 두 성분에 똑같이 곱해짐). 증폭기는 k.
- β<1의 실제 역할: ① 큰 k의 정찰 정보를 수집하되 이사는 일부만 커밋(진동/recency 제거),
  ② anchor가 거의 제자리 → 같은 지점 근방에서 셔플 샘플 N개 확보 → 체계적 정렬신호(∝N)만
  남고 노이즈(∝√N)는 상쇄. **k=증폭기, β=안정기** 세트.
- 작은 LR과의 차이: LR은 감지와 커밋을 같이 줄여 정렬신호가 **제곱**으로 죽음(α²). Reptile은
  감지는 α 그대로, 커밋만 β배 → 같은 진행속도 기준 정렬신호 1/β배.
- **anchor는 사이클 단위** (옵션 B): 그룹 간 cross term이 Δ 안에 담기려면 여러 그룹이 같은
  inner 궤적 안에 있어야 함. phase 단위 anchor(옵션 A)는 그룹 내 정렬만 직접 담김 —
  ablation 변형으로 보류. Sequential Reptile(ICLR'22)도 B 구조.

### 3.4 Δ-feedback의 정확한 형태와 이유

- 원안 `loss + (loss − loss_last)`는 **무효**: 저장된 스칼라는 상수라 미분에서 소멸,
  gradient는 잊은 양과 무관하게 2·∇loss. 
- 올바른 형태: **detach된 곱셈 계수** `w = 1 + λ·max(0, Δ_rel)`. 양수 상수 곱은 최솟값
  위치/방향 불변, effective LR만 조절 → "같은 목표, 잊은 만큼 더 급하게". detach 안 하면
  loss² 항이 생겨 목적함수가 변형됨(잊은 양이 아니라 loss 절대값이 증폭됨).
- 역할: alternating projection 수렴의 그룹별 gain 균형 (feedback controller). 층위 2를
  직접 만드는 항이 아님.
- 측정은 **고정 probe 배치 + 고정 flow-matching 노이즈**(fork_rng)로: 같은 자로 재기.
  pi05는 forward마다 노이즈/t를 샘플링하므로 노이즈 고정 없이는 Δ가 샘플링 잡음에 오염됨.

### 3.5 k(phase 길이)의 자체 상한

테일러 유효성: k가 크면 정찰 자체가 anchor 근방을 벗어나 Δ 안의 정보가 왜곡됨 — β로 구제
불가(β는 Δ가 만들어진 후에 곱해짐). 선험 계산은 불가(3B 비볼록), 학습 중 진단으로 조작적
정의: ① probe gradient 회전각(cos, ~0.7 안전/0.5 위험), ② 1차 예측 `ΔL≈⟨g,Δ⟩` 대비 실측
편차, ③ 복구 속도(revisit 초반 몇 %에 이전 수준 복귀). 전부 probe에서 나옴.

## 4. 선행연구 지도

| 논문 | 관계 |
|---|---|
| **Sequential Reptile** (Lee et al., ICLR'22, 2110.02600) | 메커니즘 원형(순차 inner loop + Reptile = task 정렬, multilingual NLP). **novelty 겹침 주의, 필독** |
| **Evron et al.** (COLT'22 2205.09588, ICML'23) | cyclic ordering 이론: forgetting ≤ T²·min{1/√k, d/k}, alternating projection/Kaczmarz 동치 |
| **Lesort et al.** (CoLLAs'23, 2207.04543, SCoLe) | 재등장만 있으면 SGD만으로 지식 누적 (딥넷 실증) |
| **"Pretrained VLA Surprisingly Resistant to Forgetting"** (2026, 2603.03818) | VLA는 이미 덜 잊음 + PT가 CL 강건성 좌우. **headroom 확인 필독** |
| MER (ICLR'19) | Reptile+replay, online CL. 버퍼가 하던 "과거 공급"을 우리는 커리큘럼 revisit으로 대체 |
| OML (NeurIPS'19) | 같은 목적의 2차 구현 = 이상적 upper bound (스케일 불가) |
| GPM (ICLR'21) | 직교 투영(하드) 경쟁자. plasticity 단조감소가 약점. FT 병용 가능 부품이기도 |
| Mirzadeh et al. 2020 | training regime/flat minima가 forgetting 결정 → joint+SAM baseline 근거 |
| DWA (2019) | Δ-계수의 MTL 선행 (loss 변화율 기반 task weight) |

**Novelty 위치** (2026-07 웹서치 기준): "VLA에서 PT 커리큘럼으로 이후 replay-free FT
강건성을 심는다"는 조합은 빈칸. Δ-feedback 부품도 이 계열에 없음. 차별화 축: 그룹-phase
단위 굵은 궤적 + PT→FT 전이 목적 + Δ-controller.

## 5. 실험 계획

### 조건 매트릭스 (PT 6조건 → 동일 FT → 기존 task 유지율)

1. joint iid — **`cycle_iid_baseline=true`로 이 폴더 안에서 실행 (권장)**: 학습은 순수 iid
   글로벌 셔플, probe/로깅 계측은 cyclic 런과 완전 동일 → 같은 자로 잰 비교가 됨 (run name
   `PTiid_...`). Δ/Reptile 자동 무시.
2. joint + SAM — flat-minima 경쟁자 (미구현)
3. cyclic만 (`lam=0, b=1`)
4. cyclic + Δ
5. cyclic + Reptile
6. cyclic + Δ + Reptile

판정: 3≈4면 Δ 기여 없음, 2≥6이면 커리큘럼 불필요, 5/6>3이면 정렬 증폭이 유효.

### 핵심 figure: phase 길이(k) sweep → 역U 커브

k=1(=iid)부터 극단까지 sweep → 동일 FT → 유지율. 중간 봉우리가 뜨면 "섭동-복구가 강건함을
만든다"는 메커니즘의 존재 증명. 평평하면 가설 기각.

### 측정 3층

| 층 | 시점 | 내용 |
|---|---|---|
| 진단 | PT 중 | 회전각·예측편차·복구속도 (k 유효범위) |
| 구조 | PT 중 | 간섭 행렬 Δ_ij, 그룹 간 grad cosine — 사이클 지날수록 개선되는지 (메커니즘 증거) |
| 최종 | FT 후 | held-out task FT 후 기존 task 유지율 (가설 판정) |

프록시에서 3층을 다 찍어 대응관계 확보 → 본 스케일에선 싼 지표만으로 k·β 세팅.
FT 프로토콜 주의: 모든 조건에 FT 레시피 완전 동일, FT task는 PT에서 완전 제외(held-out).

## 6. 구현 맵 (이 폴더)

- `src/lerobot_train_cycle.py` — 단독 학습 스크립트. lerobot 본체 무수정.
  `CycleTrainPipelineConfig(TrainPipelineConfig)` 서브클래스로 draccus CLI 통합.
  핵심 함수: `build_groups`(frame-balanced greedy bin-packing, task 통째 배정, groups.json 저장 —
  고정 k와 데이터 공정성을 동시에 만족) / `GroupCursor`(그룹별 epoch 커서 — revisit이 미완
  epoch을 이어받아 frame 소비가 정확히 균등, iid의 epoch 순열과 동일한 hygiene) /
  `build_probe_batches`+`measure_probe`(fork_rng 고정 노이즈) / `update_policy_scaled`
  (Δ-계수는 detach scalar 곱, **로깅은 unscaled loss**) / `reptile_interpolate`
  (사이클 anchor, CPU 스트리밍) / `measure_probe_grad`+`grad_cosine`(회전각 진단, 옵션).
- `cycle/cycle_config.yaml` — 토글: `cycle_phase_steps`(k) / `cycle_n_cycles`(>0이면
  phase_steps = steps//(groups×cycles) 자동, n_groups sweep 시 사이클 보존용) /
  `cycle_delta_lambda` / `cycle_reptile_beta` / `cycle_iid_baseline`. 전부 env 오버라이드 가능
  (CYCLE_*). run name에 조건 자동 인코딩 (`PTcyc_..._g8p500_lam05_b05`, `PTiid_...`, `g8c5`).
- **파라미터화 원칙**: `pt_steps`(예산)는 고정 앵커 — 조건 간 동일 연산량 비교가 실험의 전제.
  자유 변수는 groups + {phase XOR cycles}. steps를 바꾸는 실험(같은 k에 사이클 보충 등)은
  k↕cycles 교란을 분리하는 의도적 대조군으로만, 수동으로. **bs도 조건 간 고정 필수.**
- `cycle_eval/` compare 모드 — `compare_models`(2+ 모델)로 동일 init state에서 task별 순차
  rollout + **즉시 side-by-side 영상 합성**(`src/eval_compare.py`, 모델 1회 로드, task마다
  summary 갱신 → 진행 중 열람 가능). 다른 서버의 stage1_eval/video_compare의 온라인 버전.
- `cycle_eval/` — closed-loop LIBERO 평가 + **그룹별 success 집계**: lerobot-eval rollout 후
  `src/aggregate_group_success.py`가 eval_info.json × groups.json × LIBERO task language를
  조인해 `group_success.{json,png}` 생성 → probe/forget 곡선과 그룹 단위로 대응되는 실제
  성공률. ⚠️ 데이터 사실: libero_90의 90 scene-task는 유일 지시문 74개(12개 중복)이고 PT
  데이터셋 "73 tasks"는 지시문 단위(scene-task 89/90 커버, 결측=task51 butter/basket).
  그룹핑·조인 모두 지시문 키라 일관 (매칭 89/90 검증됨).
- wandb (`VLA_cycle`) — 섹션 분리됨 (WandBLogger의 mode 화이트리스트를 우회해 raw run으로
  로깅하는 `wandb_log_section` 헬퍼):
  - `train/`: loss·grad_norm·lr·epochs(전역) — log_freq마다
  - `probe_loss/` g{j}, `probe_forget/` g{j}, `grad_cos/` g{X} — 각각 별도 섹션, phase 경계마다.
    active group의 forget은 "자기 직전 phase 끝 대비 순 drift"(복구 완성도) — own_last 갱신
    전에 로깅하는 순서 때문. FT 스크립트도 동일 섹션명(probe_loss/probe_forget, 기준=FT step 0)
  - `cycle/`: position(소수점 사이클 위치), index, active_group, w_active
  - `epoch/`: g{j}(그룹별 데이터 소비 바퀴수, GroupCursor 기반 정확값), active_group
  → 간섭 행렬/복구 곡선 오프라인 복원 가능. Reptile 보간 직후에도 probe 1회 (pull-back 가시화).

### 알려진 v1 한계 (의도된 설계)

- 단일 GPU 전용(assert), **resume 없음**(output dir 삭제 후 fresh — walltime 계산 필수:
  스텝시간 × steps < pt_time 확인하고 제출할 것)
- Adam 모멘트를 Reptile 보간 때 리셋 안 함 — ablation 후보
- steps가 사이클 배수가 아니면 마지막 부분 사이클에도 보간 적용
- 회전각 진단은 단일 지정 그룹만(`probe_grad_group`), CPU ~6GB
- probe는 매 경계 × 전 그룹이라, 그룹 수를 크게(예: 73 task 단위) 하면 오버헤드 ~3× —
  그 실험 전에 probe thinning(m경계마다 측정) 추가 필요

## 7. 현재 상태 (2026-07-04, 서버 이전 시점)

**돌고 있는 런 3개** (bs16/20k, wandb `VLA_cycle`): PTcyc_g8p500, PTcyc_g8p250, PTiid_g8p250.
중간 분석 결과 (wandb API, cyc500이 18.5k 시점):

- ✅ 계측 검증: PTiid forget ≈ ±1% (노이즈), PTcyc forget 평균 +5~12%/최대 +50% →
  cyclic의 forget 신호는 진짜 커리큘럼 유발 간섭
- ✅ 섭동 발생: 경계 측정의 60~80%에서 >2% forget — "잊을 만큼 잊는" 상태
- ✅ 복구 작동: active group net drift ≈ 0 이하 (p250은 가끔 +0.02~0.04 = 복구 약간 부족)
- ✅ 안전망: train loss cyc500 0.091 / cyc250 0.095 / iid 0.096 (iid는 9.4k 시점) — 커리큘럼 비용 없음
- ⚠️ **forget 진폭이 사이클마다 감소** (cyc500: c1 +11.8% → c4 +4.2%) — 원하는 "무간섭 평형
  이동" 신호처럼 보이지만 **같은 구간 LR이 2.5e-5→3e-6으로 cosine decay 중이라 교란됨.**
  단위 LR당 간섭으로 거칠게 정규화하면 감소가 사라짐 → 현재 데이터로는 구조 형성 주장 불가.

**분리 실험 (배선 완료)**: `pt_constant_lr: true` (yaml) 또는 `PT_CONSTANT_LR=true` (env) —
decay_lr을 peak과 같게 만들어 scheduler alpha=1 → warmup 후 **상수 LR**. run name에
`_constlr` 자동 태그:
```bash
PT_CONSTANT_LR=true CYCLE_PHASE_STEPS=250 ./submit_cycle_PT.sh
```
판정: constant-LR cyclic에서도 진폭이 줄면 구조 형성 증거 확보, 안 줄면 기존 감소는 LR
아티팩트 (가설 자체는 FT 유지율로 최종 판정되므로 죽지 않음).

**constant-LR 최종 결과 (2026-07-05, 20k 완주): 구조 형성 신호가 두 교란 모두 생존 — 확보됨.**
- cyc250-constlr 사이클별 forget mean: c1 +12.6% → c3 10.8% → c4-c8 ~5-7% → c9 +2.9%
  (**~4× 감소, LR은 끝까지 2.5e-5 상수**)
- grad_norm 정규화 후에도 감소 유지 (0.131 → 0.031). 후반 grad_norm은 오히려 상승(0.86→0.95)
  하는데 forget은 감소 → "gradient가 작아져서"도 아님. **파라미터는 계속 같은 크기로 움직이되
  다른 그룹을 점점 덜 해치는 방향으로** = 정렬/무간섭화의 직접 증거 (probe-loss 공간 한정).
- decay 런은 같은 구간 c1 11.2% → c9 0.8% (더 가파름 = LR 효과가 추가로 얹힘). iid는 전 구간
  ±0.4% 노이즈 바닥 (계측 대조 유지).
- grad_cos(g0): 초기 사이클 0.08~0.4 (초기 급강하 구간은 전개 붕괴 — 초반 cross-term 정보는
  노이즈였음), 이후 const 런은 0.6~0.8에서 유지 = **k=250@2.5e-5는 런 내내 유효 경계 근처**.
  decay 런이 0.95+로 오르는 건 LR 축소 효과.
- 최종 train loss: const 0.0907 / decay 0.0865 / iid 0.0856 — annealing 없는 만큼만 소폭 높음.

**이전 중간 분석 기록 (10.25k 시점):**
- 완주 3런 최종 train loss: cyc500 0.0849 / cyc250 0.0865 / iid 0.0856 — 동률.
  **커리큘럼 성능 비용 없음 확정** (20k 기준).
- const 런 사이클별 forget: c1 +12.6% → c2 10.1% → c3 10.8% → c4 5.4% —
  **LR 고정에서도 진폭 감소 존재** → 구조 형성 신호가 순수 LR 아티팩트는 아님.
  단 decay 런(c4 +2.7%)보다 완만 = 이전 감소의 일부는 LR 몫이었음. 둘 다 사실.
- 남은 유보: ① 4사이클뿐(완주 후 c6~c9 확인이 결정타), ② loss 수렴→gradient 축소 교란
  잔존 (train/grad_norm 로깅됨 → 완주 후 진폭/grad_norm 정규화로 사후 분리 가능).
- 회전각 실측: const 런 grad_cos_g0 ≈ 0.75~0.83 (1회 0.52) → **k=250@2.5e-5는 유효 경계
  근처.** k=500 constant는 경계 초과 가능성 높음. decay 런들이 0.95+인 건 LR 축소 효과일 뿐.

기타: 이전에 bs64로 낸 런들은 스텝당 5.3s → 20k=30h > 24h walltime라 폐기했음 (resume 없음
주의 — 스텝시간 × steps < pt_time 확인 후 제출). 첫 train/ wandb 포인트는 log_freq=200 ×
스텝시간 후에야 나타남 — 초반에 probe/cycle/epoch 창만 보이는 건 정상.

**⚠️ 치명 버그 발견·수정 (2026-07-05):** 초기 train 스크립트가 `make_pre_post_processors`에
`postprocessor_overrides`(unnormalizer에 데이터셋 stats 주입)를 빠뜨림 → 체크포인트가
**pi05_base의 unnormalizer stats를 그대로 상속** → eval에서 액션 역정규화 스케일이 틀려
모션 붕괴/성공률 0 근처. **학습 자체(가중치)는 정상** (입력 normalizer는 제대로 오버라이드됨,
loss 곡선 정상). 진단 근거: 3-way compare에서 ref(기존 pi05_PT@15k)만 정상 + iid 체크포인트의
policy_postprocessor.json이 pi05_base와 바이트 동일. 조치: ① train 스크립트에
postprocessor_overrides 추가(수정됨), ② `src/repair_postprocessor.py`로 기존 체크포인트
전부 수리 완료(재학습 불필요). **주의: 수정 전에 시작돼 아직 돌고 있는 런(constlr 등)은 이후
저장되는 체크포인트가 다시 오염되므로, 완주 후 repair 스크립트 재실행 필요.**
이전에 관찰된 낮은 eval 성공률/이상 모션은 전부 이 버그 소산 — **수리 후 재평가 필수.**

**FT 판정 파이프라인 구축 (2026-07-05):** `cycle_ft/` + `src/lerobot_train_ft_probe.py` —
cycle-PT ckpt를 libero_10으로 replay-free FT하면서 PT 그룹 probe(같은 groups.json·probe_seed
→ PT 때와 값 직접 비교 가능)를 probe_every=250마다 측정. `probe/g{j}_forget`(기준 = FT step 0)
= FT 중 forgetting 실시간 곡선. 판정식: forgetting(PTcyc) < forgetting(PTiid) ⇔ 가설 성립.
설계 결정: FT 레시피 전 조건 동일 고정(15k/2.5e-5/freeze 없음), **normalizer는 PT ckpt 유지**
(FT dataset stats로 재정규화하면 그 자체가 old-task 파괴 → 파라미터 forgetting과 교란).
FT 출력에 groups.json 자동 복사 → cycle_eval 그룹 집계 그대로 동작. ref(일반 PT) 소스도
ft_source에 절대경로 + ft_groups_json(아무 cycle 런 것)으로 지원.

**β-스케줄 구현 (2026-07-05):** `cycle_reptile_beta_end ≥ 0`이면 β를 사이클에 걸쳐
`cycle_reptile_beta → beta_end`로 cosine anneal (per-cycle, run name `_b{s}to{e}`, wandb
cycle/reptile_beta). 이론적 근거: LR decay는 감지·커밋을 같이 죽이지만 β decay는 **커밋만
anneal** — 정찰·정렬·스트레스 선별은 풀 스케일 유지, anchor는 말기에 합의 지점에 안착
(constant-LR endpoint의 orbit/recency 문제 해결). 의도된 조합 = `pt_constant_lr: true` +
`β: 0.5→0.1`. 다음 3파전: {const+β스케줄, const+β고정, decay} → FT 판정.

**메커니즘 확정 (2026-07-07): cyclic의 forgetting 이점 = 더 flat한 minima, 언어모델에 집중.**
- 진단 도구: `src/measure_term2_blocks.py`(블록별 gradient 정렬) + `src/measure_flatness_blocks.py`
  (블록별 filter-normalized 섭동 → 옛-task probe loss 상승 ΔL/L; forgetting-relevant sharpness).
  둘 다 fork_rng 고정 노이즈, 사이클 probe와 동일 자. sbatch: cycle_eval/term2.sbatch, flatness.sbatch
  (MODEL/CKPT env). 결과 outputs_term2/, outputs_flatness/.
- **term2(블록 정렬): cyclic ≈ iid, 둘 다 랜덤(0.125) 근처 → 기각.** 블록별 "공유 몰아주기"
  전략은 타겟 없음. (놀랍게도 action_expert가 vision보다 정렬 높음 — 매니퓰레이션 공유구조가
  모터에 있을 수 있음, 근데 iid와 차이 없어 무의미.)
- **flatness(basin 곡률): cyclic이 모든 블록에서 더 flat (CYC<IID, 두 eps 모두, 일관).** →
  Mirzadeh 2020 "flat minima→덜 잊음" 실측. cyclic이 바꾼 건 gradient 방향관계가 아니라
  앉은 자리의 곡률.
- **핵심: 옛-task 지식은 압도적으로 language_model(VLM)에 있다.** language_model fragility
  (ΔL/L@0.05 = cyc 0.173 / iid 0.262)가 나머지 블록(전부 <0.005)의 ~100배. 그리고 cyclic의
  이득이 정확히 거기 집중 (34~50% 덜 fragile). vision/action/flow는 흔들어도 옛 loss 거의 불변.
- 스토리: cyclic PT → flatter minima → 그 강건화가 옛 지식이 사는 언어모델에 집중 → FT 섭동에서
  언어모델이 덜 밀림 → 덜 잊음. 기존 skillVLA의 VLM-freeze/언어 modulation 관찰과 정합.
- ⚠️ 노이즈: 8그룹×2배치라 term2 블록차는 노이즈 수준. flatness는 3seed×2eps로 신호 명확.
  더 굳히려면 probe 배치↑ 또는 seed↑. 측정은 cyc250_constlr@20k vs iid250(decay)@20k.
- **FT 이동량 교차검증 (`src/measure_ft_delta_blocks.py`, forward-free CPU, FT010k vs PT020k):
  cyclic과 iid가 VLM을 똑같은 양 민다 (rel_move 0.00279 vs 0.00285) → "cyclic이 VLM을 덜
  건드려서"(가설 A) 기각, "같이 밀려도 flat해서 안 무너짐"(가설 B) 확정.** 게다가 VLM은 FT 때
  전 블록 중 이동 최소(0.0028)인데 forgetting은 거기서 남 — 새 task는 모터쪽(action/flow,
  rel 0.025~0.048)에 학습되고 VLM은 거의 안 건드리는데, VLM이 100배 취약해서 그 작은 이동이
  유일한 forgetting 원천. **forgetting = 취약성 × 이동량, cyclic은 순수하게 취약성(flatness)
  으로 이김.** flatness 그림 + 이동량 그림이 상호 검증 = 논문 핵심 2-figure.

## 8. 다음 단계

1. (필요시) 스모크: `PT_STEPS=24 CYCLE_PHASE_STEPS=3 CYCLE_DELTA_LAMBDA=0.5
   CYCLE_REPTILE_BETA=0.5 PT_SAVE_FREQ=24 PT_EXP=smoke ./submit_cycle_PT.sh`
2. **첫 비교쌍**: 동일 예산으로 `./submit_cycle_PT.sh`(PTcyc) + `CYCLE_IID_BASELINE=true
   ./submit_cycle_PT.sh`(PTiid). FT 없이도 판단할 것: ① probe 톱니(phase 중 다른 그룹 loss
   상승 = forgetting 발생 확인, 안 오르면 섭동 부족), ② **g{j}_forget 진폭이 사이클이 갈수록
   감소하는지 = 무간섭 평형 이동의 첫 메커니즘 증거**, ③ 최종 loss가 iid에 크게 안 밀리는지.
3. k-sweep: `CYCLE_PHASE_STEPS ∈ {50, 200, 500, 2000}` (+ Δ/Reptile 토글 조건들)
4. FT 프로토콜 연결 (기존 train_pi05 FT 재사용, libero_10 held-out) → U커브
5. 결과 좋으면: joint+SAM baseline 추가, 본 스케일(100k), skillVLA 포팅
6. 열린 질문: Adam 모멘트 처리, 옵션 A(phase-anchor) ablation, 그룹 수 sweep(n_cycles 고정
   모드 활용), 상대 Δ vs 절대 Δ, 73그룹(task 단위) 실험 시 probe thinning 구현 필요(현재는
   경계마다 전 그룹 probe라 오버헤드 ~3×), "본 task 무간섭 평형이 unseen task 섭동에도
   강건한가"(핵심 도약)
