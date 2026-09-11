# VSA Cond–Expert Bridge 설계 대화 기록

> 상태: 아이디어 논의 단계. 아직 코드로 구현하지 않음.
>
> 배경 논문: *Semantically Structured Mixture-of-Experts for Compositional Robotic Manipulation* (SMoDP, 2026). 아래 설계는 논문 구조를 그대로 복제하는 것이 아니라, skill 관련 motion 지식의 보존과 1–5 episode few-shot adaptation이라는 현재 프로젝트 목적에 맞춰 확장한 아이디어다.

## 1. 프로젝트 목적

- Stage 1의 대규모 데이터에서 학습한 skill/motion 정보를 최대한 보존한다.
- Stage 2 및 FT, 특히 1–5 episode만 있는 FT 데이터에서도 새로운 scene/object 배치에 적응한다.
- 현재처럼 frozen VSA의 입구에서 skill/noise만 steering하는 것보다 높은 적응 자유도를 제공한다.
- Language는 VSA의 Cond–Expert 연결에 직접 넣지 않는다.
- Language는 현재 방식처럼 Stage 2의 noise predictor를 통한 noise steering에만 사용한다.

## 2. 현재 VSA 구조에 대한 확인

현재 `gemma_300m` Action Expert는 다음 구성이다.

- hidden width 1024
- 18 Transformer layers
- 각 layer에 joint self-attention과 Expert MLP가 존재
- action token은 Cond token과 모든 action token을 읽을 수 있음
- Cond token은 action token을 읽지 못함
- Expert MLP는 Cond token을 직접 읽지는 않지만, joint attention을 거쳐 Cond가 섞인 action hidden을 입력받음

현재 한 layer의 개념적 순서는 다음과 같다.

```text
action hidden + Cond hidden
           ↓
Cond–Action joint self-attention
           ↓
Expert MLP
           ↓
next layer
```

따라서 현재 구조에도 Cond token을 직접 읽지 않는 큰 파라미터 덩어리(특히 Expert MLP)는 있지만, action-action attention과 Cond→Action attention이 동일한 joint attention 안에 얽혀 있어 `motion core`와 `Cond connector`가 명확하게 분리된 것은 아니다.

## 3. Delta Bridge의 기본 개념

Delta Bridge는 frozen Cond hidden을 읽어 frozen Expert hidden에 task-specific residual을 추가하는 작은 학습 모듈이다.

```text
Q = Wq_bridge × expert_hidden
K = Wk_bridge × cond_hidden
V = Wv_bridge × cond_hidden

delta = Wo_bridge × Attention(Q, K, V)
expert_hidden_new = expert_hidden + gate × delta
```

- `Q`의 입력은 Expert hidden이지만 `Wq_bridge`는 Expert Core가 아니라 Bridge 소속이다.
- Bridge의 `Wq/Wk/Wv/Wo`, norm, gate만 독립적으로 학습할 수 있다.
- Language token은 Bridge의 K/V에 포함하지 않는다.
- Delta는 실제 xyz action에 직접 더해지는 값이 아니라 action token의 hidden feature에 더해진다.
- gate 또는 output projection을 zero-init하면 학습 시작 시 기존 VSA와 동일한 출력을 보장할 수 있다.

## 4. 기존 VSA에 즉시 붙이는 Retrofit 방식

기존 Stage 1/2 checkpoint를 다시 학습하지 않고 바로 사용할 수 있다.

```text
Frozen current VSA layer
  - current Cond
  - current joint attention
  - current Expert MLP
           ↓
Trainable FT Delta Bridge
           ↓
다음 frozen VSA layer
```

FT에서 가장 보존적인 구성:

```text
Frozen
- 기존 Cond
- 기존 joint attention
- 기존 Expert 전체
- action input/output projections
- skill/time/latent conditioning

Trainable
- 새 FT Delta Bridge
- noise predictor
- 필요한 skill/latent predictor
```

장점:

- Stage 1 재학습 불필요
- 현재 checkpoint로 즉시 검증 가능
- zero-init으로 FT 시작 시 기존 출력 완전 보존 가능
- noise-only steering보다 observation-dependent한 고차원 보정 가능

한계:

- 기존 VSA 자체는 이미 Cond와 Expert가 joint attention으로 섞여 학습됨
- frozen Expert를 순수한 skill-only motion core라고 해석할 수 없음
- 새 Bridge가 1–5 episode에서 처음 학습되므로 너무 크면 과적합 가능
- 기존 joint attention과 새 Bridge의 역할이 일부 겹칠 수 있음

## 5. 처음부터 다시 학습하는 Clean Separation 방식

Action trajectory 내부 상호작용과 Cond 연결을 처음부터 분리한다.

```text
Skill + timestep + noisy action (+ latent)
                 ↓
        Expert action self-attention
        Expert MLP
                 ↓
        Cond→Expert Base Bridge
                 ↑
        Cond(image, proprio)
                 ↓
        다시 Expert Core
                 ↓
        다시 Base Bridge
                 ↓
                ...
```

구조적 역할:

- Expert Core: noisy action, timestep, skill, latent만 받으며 action dynamics와 skill motion prior를 담당
- Cond Encoder: image/proprio를 해석
- Base Bridge: Cond 정보를 Expert action hidden으로 전달
- Language: 이 경로에 들어가지 않고 noise predictor에서만 사용

Stage 1에서는 Expert, Cond, Base Bridge, action head를 모두 같은 action/flow objective로 end-to-end 학습할 수 있다.

처음부터 분리해 학습할 때 기대되는 이점:

- Expert와 Cond grounding의 경계가 구조적으로 명확함
- Base Bridge가 대규모 Stage 1 데이터에서 일반적인 vision–motion 연결을 미리 학습
- FT에서는 작은 task-specific delta만 배워도 될 가능성이 높음
- Expert를 freeze하는 것이 skill motion library 보존이라는 해석과 더 잘 맞음

비용 및 위험:

- Stage 1부터 재학습 필요
- 분리가 지나치게 강하면 full policy 성능이 떨어질 수 있음
- 구조 분리만으로 Expert Core가 motion을 담당하게 된다는 보장은 없음

## 6. 가장 중요한 미해결점: Core에 motion이 담긴다는 보장은 없음

Core와 Bridge를 나눠도 동일한 데이터와 최종 action loss로 동시에 학습하면 다음과 같은 퇴화해가 가능하다.

```text
Case A
Core: 거의 의미 없는 기본 hidden
Bridge: 실제 action 대부분 담당

Case B
Core: 평균적인 motion 대부분 담당
Bridge: 거의 사용되지 않음
```

최종 action loss는 어느 모듈에 정보가 저장되는지 구분하지 않으므로, architecture separation만으로는 역할 분리가 보장되지 않는다.

Core에 skill motion을 넣기 위한 후보:

### A. Core-only auxiliary objective

```text
Full path:
skill + noise + Cond → Core + Bridge → action → L_main

Core-only path:
skill + noise → Core → action → L_core

L = L_main + lambda_core × L_core
```

현재 프로젝트의 `arch0_skill` / `arch0_skill_chunk` 보조 경로와 유사한 원리다.

### B. Bridge dropout

일부 batch에서 Bridge를 0으로 만들어 Core만으로 flow loss를 맞추게 한다.

```text
h_new = h_core + mask × Bridge(h_core, cond)

mask=1: full path
mask=0: core-only path
```

한 batch에서 두 번 forward하지 않아도 되어 계산량 증가가 작다. Cond 없이 동일 skill의 정확한 개별 action을 구분할 수는 없으므로 Core는 skill 공통 motion 또는 noise/latent에 따른 multimodal skill distribution을 배우는 것이 목표다.

### C. 순차 학습

역할 분리를 가장 강하게 유도하는 방법이다.

```text
Stage 1-A
- Bridge 없이 Expert Core를 skill-conditioned flow policy로 선학습

Stage 1-B
- Core를 freeze하거나 매우 작은 LR 사용
- Cond + Base Bridge를 full action loss로 학습

Stage 1-C (선택)
- Core에 작은 LR을 주고 전체 joint fine-tuning
- Core-only objective 유지
```

보장 강도는 대략 다음 순서다.

```text
구조만 분리
< Core-only loss / Bridge dropout
< Core 선학습 후 freeze하고 Bridge 학습
```

## 7. 1–5 episode FT에 대한 잠정 방향

현재 대화에서 가장 보존적인 후보는 다음과 같다.

```text
Stage 1 base model
- Expert Core 학습
- Cond 학습
- Base Bridge 학습

Few-shot FT
- Expert Core freeze
- Base Cond freeze
- Base Bridge freeze
- zero-init FT Delta Bridge만 학습
- noise predictor 및 필요한 predictor 학습
```

이는 기존 지식을 덮어쓰지 않고 다음처럼 동작한다.

```text
FT output
= Frozen Stage 1 output
 + task-specific Delta Bridge correction
```

Cond 전체 fine-tuning이나 Cond LoRA는 필수로 결정되지 않았다. 동일 LIBERO domain에서 frozen Cond feature가 충분하다면 FT Delta Bridge만으로 먼저 검증하는 것이 가장 강한 보존 baseline이다.

## 8. Retrofit과 Clean Separation 비교

| 구분 | 기존 VSA + FT Delta Bridge | 분리형 VSA Stage 1 재학습 |
|---|---|---|
| Stage 1 재학습 | 불필요 | 필요 |
| 기존 checkpoint 활용 | 직접 가능 | 일부 weight 초기화만 가능 |
| Core/Cond 역할 분리 | 불명확 | 구조적으로 명확 |
| Base Bridge 사전학습 | 없음 | Stage 1 전체 데이터로 가능 |
| 빠른 검증 | 적합 | 부적합 |
| 장기적인 연구 구조 | 제한적 | 더 적합 |

잠정적인 실험 순서:

1. 기존 VSA에 작은 zero-init FT Delta Bridge를 붙여 noise-only FT보다 성능이 개선되는지 확인
2. Bridge가 실제로 유효하면 clean separation 구조를 Stage 1부터 학습
3. clean 구조에서는 Core-only loss, Bridge dropout 또는 순차 학습으로 Core 역할을 강제

## 9. 이후 결정할 사항

- Retrofit을 먼저 구현할지, clean separation으로 바로 갈지
- Bridge 삽입 layer 수와 위치
- layer별 Bridge를 둘지 weight를 공유할지
- Bridge bottleneck width 또는 low-rank 크기
- residual gate 초기화 방식
- FT에서 frozen Cond만 사용할지 Cond LoRA/full update도 옵션으로 둘지
- Base Bridge를 FT에서 freeze하고 별도 Delta Bridge를 둘지, Base Bridge 자체를 update할지
- Core-only target을 main action chunk로 할지 canonical skill trajectory로 할지
- Bridge dropout 비율 또는 Stage 1 순차 학습 일정
- skill을 Bridge query/gate에 명시적으로 한 번 더 넣을지
- 기존 `arch0_skill` / `arch0_skill_chunk` 보조 경로와 어떻게 통합할지
- 보존/적응 평가 지표: base-task retention, FT success, skill sensitivity, bridge-to-expert RMS, bridge gate 등

