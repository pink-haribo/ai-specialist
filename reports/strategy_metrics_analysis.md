# GAIN-MTL Strategy 성능 평가 분석

## 1. 분석 개요

본 보고서는 GAIN-MTL 프레임워크의 **6개 학습 전략(S1, S2, S3, S7, S8, S6)**에 대한 성능 평가 결과를 분석한다.

평가는 두 가지 관점에서 수행한다:
1. **분류 성능 평가**: Accuracy, Recall
2. **해석 성능 평가**: CAM-IoU, Point Game, Energy Inside, Loc-IoU

| Strategy | 설명 | 핵심 Loss 구성 |
|----------|------|---------------|
| S1 | Classification Only (Baseline) | `L_cls` |
| S2 | + CAM Guidance | `L_cls + L_cam_guide` |
| S3 | + Attention Mining | `L_cls + L_am + L_guide` |
| S7 | Attention Mining + Counterfactual (Lightweight) | S3 + `L_cf` |
| S8 | S7 + GT Mask Curriculum | S7 + multiplicative fusion curriculum |
| S6 | Full + GT Mask Curriculum | All losses + multiplicative fusion curriculum |

---

## 2. 분류 성능 평가 (Classification Performance)

### 2.1 결과 요약

| Metric | S1 | S2 | S3 | S7 | S8 | S6 |
|--------|------|------|------|------|------|------|
| **Accuracy** | 0.888 | 0.893 | 0.902 | 0.896 | 0.887 | **0.907** |
| **Recall** | 0.880 | 0.899 | 0.938 | 0.934 | 0.930 | **0.949** |

### 2.2 Metric별 순위

**Accuracy 순위:**

| 순위 | Strategy | Accuracy | Baseline 대비 |
|------|----------|----------|--------------|
| 1 | **S6** | **0.907** | +0.019 |
| 2 | S3 | 0.902 | +0.014 |
| 3 | S7 | 0.896 | +0.008 |
| 4 | S2 | 0.893 | +0.005 |
| 5 | S1 | 0.888 | — (Baseline) |
| 6 | S8 | 0.887 | -0.001 |

**Recall 순위:**

| 순위 | Strategy | Recall | Baseline 대비 |
|------|----------|--------|--------------|
| 1 | **S6** | **0.949** | +0.069 |
| 2 | S3 | 0.938 | +0.058 |
| 3 | S7 | 0.934 | +0.054 |
| 4 | S8 | 0.930 | +0.050 |
| 5 | S2 | 0.899 | +0.019 |
| 6 | S1 | 0.880 | — (Baseline) |

### 2.3 Strategy별 상세 분석

#### S1 — Classification Only (Baseline)

| Metric | Value |
|--------|-------|
| Accuracy | 0.888 |
| Recall | 0.880 |

- 분류만 수행하는 **Baseline 모델**로, Accuracy 0.888 / Recall 0.880의 기본 성능을 보인다.
- Recall 0.880은 전체 Strategy 중 **최저치**로, 결함 샘플의 약 **12%를 미탐지**한다.
- 해석 가능성 관련 loss가 없는 순수 분류 모델이므로, 이후 Strategy의 분류 성능 변화를 측정하는 기준점이 된다.

#### S2 — + CAM Guidance

| Metric | Value | vs S1 |
|--------|-------|-------|
| Accuracy | 0.893 | +0.005 |
| Recall | 0.899 | +0.019 |

- CAM Guidance(`L_cam_guide`)를 추가하여 Accuracy와 Recall이 모두 소폭 상승했다.
- Recall이 +0.019 향상되어, **CAM의 공간적 지도(spatial supervision)가 분류 성능에도 긍정적으로 작용**함을 보여준다.
- CAM을 GT mask 방향으로 유도하는 것이 feature 학습의 질을 높여 분류에도 간접적인 이득을 제공한 것으로 해석된다.

#### S3 — + Attention Mining

| Metric | Value | vs S2 | vs S1 |
|--------|-------|-------|-------|
| Accuracy | 0.902 | +0.009 | +0.014 |
| Recall | 0.938 | +0.039 | +0.058 |

- **분류 성능이 크게 향상되는 핵심 전환점**이다.
- Recall이 S2 대비 **+0.039의 큰 도약**을 보여, Attention Mining(`L_am`)과 Guided Attention(`L_guide`)이 결함 탐지 민감도를 크게 강화했다.
- Accuracy도 0.902로 S1~S3 구간에서 **가장 높은 수치**를 기록했다.
- 학습 가능한 attention이 실제 결함 feature에 집중하도록 유도함으로써, 분류 성능과 해석 가능성을 동시에 향상시키는 효과가 확인된다.
- **비용 대비 성능이 가장 우수한 Strategy**로, 3개의 loss만으로 높은 분류 성능을 달성했다.

#### S7 — Attention Mining + Counterfactual (Lightweight)

| Metric | Value | vs S3 |
|--------|-------|-------|
| Accuracy | 0.896 | -0.006 |
| Recall | 0.934 | -0.004 |

- S3에 Counterfactual(`L_cf`)을 추가한 **경량화 Strategy**이다.
- Accuracy와 Recall이 S3 대비 각각 -0.006, -0.004 소폭 하락했다.
- Counterfactual loss가 attention을 더 보수적(conservative)으로 만들어 **분류 성능에 약간의 trade-off**가 발생한 것으로 보인다.
- 다만 하락 폭이 미미하여, Counterfactual이 분류에 미치는 부정적 영향은 제한적이다.

#### S8 — S7 + GT Mask Curriculum

| Metric | Value | vs S7 | vs S3 |
|--------|-------|-------|-------|
| Accuracy | 0.887 | -0.009 | -0.015 |
| Recall | 0.930 | -0.004 | -0.008 |

- S7에 GT Mask Curriculum을 추가했으나, 분류 성능이 **추가 하락**했다.
- Accuracy 0.887은 **전체 Strategy 중 최저**로, Baseline(S1) 수준까지 떨어졌다.
- Localization loss 없이 GT mask curriculum만 적용한 것이 효과적이지 않았으며, 오히려 분류 학습에 간섭을 일으킨 것으로 분석된다.
- **경량 모델(S7/S8)에서는 curriculum이 분류 성능 향상에 기여하지 못한다**는 점이 확인된다.

#### S6 — Full + GT Mask Curriculum (Best)

| Metric | Value | vs S1 | vs S3 |
|--------|-------|-------|-------|
| Accuracy | **0.907** | **+0.019** | +0.005 |
| Recall | **0.949** | **+0.069** | +0.011 |

- **분류 성능 전체 1위** — Accuracy 0.907, Recall 0.949로 두 metric 모두 최고치를 달성했다.
- Recall 0.949는 결함 미탐지율이 약 **5.1%**에 불과하며, Baseline(S1) 대비 미탐지율을 **12.0% → 5.1%로 약 58% 감소**시켰다.
- 모든 loss 구성요소 + GT Mask Curriculum의 조합이 시너지 효과를 발휘하여, **분류와 해석 가능성 목표가 상호 보완적으로 작용**했다.
- S8과 달리 full loss stack(localization 포함) 위에 curriculum을 적용한 것이 핵심적인 차이로, **curriculum의 효과는 충분한 loss 구조가 뒷받침될 때 극대화**됨을 시사한다.

### 2.4 핵심 분석

#### (1) Accuracy vs Recall 분포 특성

- **Accuracy 범위**: 0.887 ~ 0.907 (2.0%p) — Strategy 간 차이가 상대적으로 작다.
- **Recall 범위**: 0.880 ~ 0.949 (6.9%p) — Strategy 간 차이가 크다.
- Recall이 Accuracy보다 **약 3.5배 넓은 범위**를 보여, 학습 전략이 결함 탐지 민감도(Recall)에 더 큰 영향을 미침을 확인할 수 있다.
- 이는 attention 기반 loss 구성이 주로 **결함 클래스의 feature 표현을 강화**하는 방향으로 작용하기 때문이다.

#### (2) Strategy 진행 경로별 성능 변화

**Core progression (S1 → S2 → S3):**
- 단계적 loss 추가가 분류 성능을 **일관되게 향상**시킨다.
- 특히 S2→S3 전환에서 Recall이 +0.039 급증하며, Attention Mining이 결함 탐지의 핵심 기여 요소임을 보여준다.

**Lightweight path (S3 → S7 → S8):**
- Counterfactual과 curriculum을 추가할수록 분류 성능이 **점진적으로 하락**한다.
- Localization loss 없이 추가적인 regularization(counterfactual, curriculum)을 적용하면, attention 학습이 과도하게 제약되어 분류에 부정적 영향을 미친다.

**Full model (S6):**
- 모든 loss 구성요소를 갖춘 상태에서 curriculum을 적용하면 **최고 분류 성능**을 달성한다.
- S8 대비 S6의 우위(Accuracy +0.020, Recall +0.019)는 **localization loss가 curriculum의 효과를 뒷받침하는 필수 요소**임을 시사한다.

#### (3) 결함 미탐지율(Miss Rate) 비교

| Strategy | Recall | Miss Rate | S1 대비 개선율 |
|----------|--------|-----------|--------------|
| S1 | 0.880 | 12.0% | — |
| S2 | 0.899 | 10.1% | 15.8% |
| S3 | 0.938 | 6.2% | 48.3% |
| S7 | 0.934 | 6.6% | 45.0% |
| S8 | 0.930 | 7.0% | 41.7% |
| S6 | 0.949 | 5.1% | **57.5%** |

- S6는 Baseline 대비 결함 미탐지율을 **57.5% 감소**시켜, 제조 현장에서의 실질적 안전성 향상에 가장 크게 기여한다.
- S3도 48.3% 감소로 우수하며, loss 복잡도 대비 효율이 높다.

---

## 3. 해석 성능 평가 (Interpretation Performance)

> **데이터 입력 대기 중** — CAM-IoU, Point Game, Energy Inside, Loc-IoU 수치 제공 시 분석을 추가한다.

---
