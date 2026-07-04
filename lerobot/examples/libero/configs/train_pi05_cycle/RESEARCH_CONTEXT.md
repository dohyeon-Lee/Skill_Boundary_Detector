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

1. joint iid (기존 `train_pi05`) — baseline
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
- `cycle/cycle_config.yaml` — 토글: `cycle_phase_steps`(k) / `cycle_delta_lambda` /
  `cycle_reptile_beta`. run name에 조건 자동 인코딩 (`PTcyc_..._g8p500_lam05_b05`).
- wandb (`VLA_cycle`): phase 경계마다 `probe/g{j}_loss`, `probe/g{j}_forget`,
  `cycle/active_group`, `cycle/w_active` → 간섭 행렬/복구 곡선 오프라인 복원 가능.
  Reptile 보간 직후에도 probe 1회 (pull-back 효과 가시화).

### 알려진 v1 한계 (의도된 설계)

- 단일 GPU 전용(assert), resume 없음(output dir 삭제 후 fresh)
- Adam 모멘트를 Reptile 보간 때 리셋 안 함 — ablation 후보
- steps가 사이클 배수가 아니면 마지막 부분 사이클에도 보간 적용
- 회전각 진단은 단일 지정 그룹만(`probe_grad_group`), CPU ~6GB
- GPU 스모크 테스트 미실행 상태로 인계될 수 있음 — README의 smoke 커맨드 먼저 실행할 것

## 7. 다음 단계

1. 스모크 테스트 (README 커맨드) → probe/Δ/Reptile 경로 확인
2. 미니 sweep: k ∈ {1(=iid 대조), 200, 500, 2000} × {순수, +Δ, +Reptile, +둘다}, 20k스텝
3. FT 프로토콜 연결 (기존 train_pi05 FT 재사용, libero_10 held-out) → U커브
4. 결과 좋으면: joint+SAM baseline 추가, 본 스케일(100k), skillVLA 포팅
5. 열린 질문: Adam 모멘트 처리, 옵션 A(phase-anchor) ablation, 그룹 수 sweep,
   상대 Δ vs 절대 Δ, "본 task 무간섭 평형이 unseen task 섭동에도 강건한가"(핵심 도약)
