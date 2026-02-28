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

#### S7 — S3 + Counterfactual (Localization 미포함)

| Metric | Value | vs S3 |
|--------|-------|-------|
| Accuracy | 0.896 | -0.006 |
| Recall | 0.934 | -0.004 |

- S3에 Counterfactual(`L_cf`)을 추가한 Strategy이다. 단, S6과 달리 **Localization 관련 loss(`L_loc`, `L_consist`)와 Localization head가 포함되지 않는다.**
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
- **Localization이 없는 S7/S8에서는 curriculum이 분류 성능 향상에 기여하지 못한다**는 점이 확인된다.

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

**기본 loss 누적 경로 (S1 → S2 → S3):**
- S1(분류만) → S2(+CAM Guidance) → S3(+Attention Mining) 순서로 **해석 가능성 관련 loss를 단계적으로 추가**하는 경로이다.
- 각 단계에서 분류 성능이 **일관되게 향상**되며, 특히 S2→S3 전환에서 Recall이 +0.039 급증한다.
- Attention Mining이 결함 탐지의 핵심 기여 요소임을 보여준다.

**Localization 미포함 경로 (S3 → S7 → S8):**
- S3에 Counterfactual(S7), GT Mask Curriculum(S8)을 추가하되, **Localization 관련 loss(`L_loc`, `L_consist`)는 포함하지 않는** 경로이다.
- 구성요소가 추가됨에도 분류 성능이 **점진적으로 하락**한다.
- Localization loss 없이 추가적인 regularization(counterfactual, curriculum)을 적용하면, attention 학습이 과도하게 제약되어 분류에 부정적 영향을 미친다.

**전체 loss + Curriculum (S6):**
- 모든 loss 구성요소(Localization 포함)를 갖춘 상태에서 curriculum을 적용하면 **최고 분류 성능**을 달성한다.
- S8 대비 S6의 우위(Accuracy +0.020, Recall +0.019)는 **Localization loss가 curriculum의 효과를 뒷받침하는 필수 요소**임을 시사한다.

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

### 3.1 결과 요약

| Metric | S1 | S2 | S3 | S7 | S8 | S6 |
|--------|------|------|------|------|------|------|
| **CAM-IoU** | 0.181 | 0.478 | 0.500 | **0.518** | 0.504 | 0.479 |
| **Point Game** | 0.085 | 0.896 | 0.892 | 0.879 | **0.920** | 0.919 |
| **Energy Inside** | 0.166 | 0.689 | 0.694 | 0.696 | 0.757 | **0.759** |
| **Loc-IoU** | 0.028 | 0.029 | 0.036 | 0.035 | 0.035 | **0.253** |

### 3.2 Metric별 순위

**CAM-IoU 순위** (attention map과 GT mask의 공간적 겹침):

| 순위 | Strategy | CAM-IoU | Baseline 대비 |
|------|----------|---------|--------------|
| 1 | **S7** | **0.518** | +0.337 |
| 2 | S8 | 0.504 | +0.323 |
| 3 | S3 | 0.500 | +0.319 |
| 4 | S6 | 0.479 | +0.298 |
| 5 | S2 | 0.478 | +0.297 |
| 6 | S1 | 0.181 | — (Baseline) |

**Point Game 순위** (최대 activation이 결함 내부에 위치하는 비율):

| 순위 | Strategy | Point Game | Baseline 대비 |
|------|----------|------------|--------------|
| 1 | **S8** | **0.920** | +0.835 |
| 2 | S6 | 0.919 | +0.834 |
| 3 | S2 | 0.896 | +0.811 |
| 4 | S3 | 0.892 | +0.807 |
| 5 | S7 | 0.879 | +0.794 |
| 6 | S1 | 0.085 | — (Baseline) |

**Energy Inside 순위** (attention 에너지 중 결함 내부에 집중된 비율):

| 순위 | Strategy | Energy Inside | Baseline 대비 |
|------|----------|---------------|--------------|
| 1 | **S6** | **0.759** | +0.593 |
| 2 | S8 | 0.757 | +0.591 |
| 3 | S7 | 0.696 | +0.530 |
| 4 | S3 | 0.694 | +0.528 |
| 5 | S2 | 0.689 | +0.523 |
| 6 | S1 | 0.166 | — (Baseline) |

**Loc-IoU 순위** (Localization head의 예측 정확도):

| 순위 | Strategy | Loc-IoU | Baseline 대비 |
|------|----------|---------|--------------|
| 1 | **S6** | **0.253** | +0.225 |
| 2 | S3 | 0.036 | +0.008 |
| 3 | S7 / S8 | 0.035 | +0.007 |
| 4 | S2 | 0.029 | +0.001 |
| 5 | S1 | 0.028 | — (Baseline) |

### 3.3 Strategy별 상세 분석

#### S1 — Classification Only (Baseline)

| Metric | Value |
|--------|-------|
| CAM-IoU | 0.181 |
| Point Game | 0.085 |
| Energy Inside | 0.166 |
| Loc-IoU | 0.028 |

- 모든 해석 metric에서 **압도적으로 낮은 수치**를 기록하며, 전형적인 **"맞는 답, 틀린 근거(right answer, wrong reason)"** 문제를 보인다.
- Point Game 0.085는 최대 activation이 결함 내부에 위치할 확률이 **8.5%에 불과**함을 의미한다. 즉 91.5%의 경우 모델이 결함과 무관한 영역을 가장 강하게 활성화한다.
- Energy Inside 0.166은 attention 에너지의 **83.4%가 결함 외부에 분산**되어 있음을 보여준다.
- 분류 성능(Accuracy 0.888)이 양호함에도 불구하고, 모델의 판단 근거가 결함과 무관하다는 점에서 **제조 현장 배포 시 신뢰할 수 없는 모델**이다.

#### S2 — + CAM Guidance

| Metric | Value | vs S1 |
|--------|-------|-------|
| CAM-IoU | 0.478 | +0.297 (2.64x) |
| Point Game | 0.896 | +0.811 (10.5x) |
| Energy Inside | 0.689 | +0.523 (4.15x) |
| Loc-IoU | 0.029 | +0.001 |

- **해석 가능성이 극적으로 향상되는 첫 번째 전환점**이다.
- CAM Guidance(`L_cam_guide`)만으로 Point Game이 0.085 → 0.896으로 **10.5배 상승**했다. 최대 activation이 이제 **89.6% 확률로 결함 내부**에 위치한다.
- Energy Inside도 0.166 → 0.689로, attention 에너지의 약 **69%가 결함 영역에 집중**되도록 변화했다.
- CAM을 GT mask 방향으로 직접 지도(supervision)하는 것이 해석 가능성 확보의 **가장 효과적인 단일 요소**임을 보여준다.
- Loc-IoU는 0.029로 거의 변화 없음 — Localization head가 없으므로 구조적 한계이다.

#### S3 — + Attention Mining

| Metric | Value | vs S2 |
|--------|-------|-------|
| CAM-IoU | 0.500 | +0.022 |
| Point Game | 0.892 | -0.004 |
| Energy Inside | 0.694 | +0.005 |
| Loc-IoU | 0.036 | +0.007 |

- Attention Mining 추가 후 **CAM-IoU가 0.500으로 상승**하며 해석 가능성을 소폭 개선했다.
- Point Game은 -0.004로 거의 동일하게 유지되고, Energy Inside는 +0.005 소폭 상승했다.
- S2에서의 극적인 향상에 비하면 증분이 작지만, **분류 성능(Recall +0.039)과 해석 가능성을 동시에 개선**했다는 점에서 의미가 크다.
- 학습 가능한 attention이 분류와 spatial alignment 두 목표를 균형 있게 달성하고 있음을 보여준다.

#### S7 — S3 + Counterfactual (Localization 미포함)

| Metric | Value | vs S3 |
|--------|-------|-------|
| CAM-IoU | **0.518** | +0.018 |
| Point Game | 0.879 | -0.013 |
| Energy Inside | 0.696 | +0.002 |
| Loc-IoU | 0.035 | -0.001 |

- **CAM-IoU 0.518로 전체 Strategy 중 최고치**를 달성했다.
- Counterfactual loss(`L_cf`)가 "결함이 없었다면?" 시뮬레이션을 통해 attention이 실제 결함 영역과 더 정밀하게 겹치도록 유도한 결과이다.
- 또한 Localization loss의 multi-task interference가 없어 attention이 **spatial alignment에 더 순수하게 집중**할 수 있는 구조적 이점도 기여한다.
- 반면 **Point Game은 0.879로 S2~S8 중 최저**이다. CAM-IoU(전체 겹침)는 높지만 **최대 activation 지점의 정확도는 다소 낮은** 특성을 보인다. 즉, attention map이 결함 영역을 넓게 커버하지만 peak 위치는 약간 분산되는 경향이 있다.

#### S8 — S7 + GT Mask Curriculum

| Metric | Value | vs S7 | vs S3 |
|--------|-------|-------|-------|
| CAM-IoU | 0.504 | -0.014 | +0.004 |
| Point Game | **0.920** | **+0.041** | +0.028 |
| Energy Inside | 0.757 | **+0.061** | +0.063 |
| Loc-IoU | 0.035 | ±0.000 | -0.001 |

- GT Mask Curriculum이 S7의 약점을 정확히 보완하여 **Point Game과 Energy Inside를 크게 향상**시켰다.
- Point Game이 0.879 → 0.920으로 **+0.041 상승**하며 **전체 Strategy 중 1위**를 달성했다. Curriculum을 통한 GT mask 주입이 최대 activation 위치를 결함 중심부로 끌어당기는 효과를 발휘했다.
- Energy Inside도 0.696 → 0.757로 **+0.061 상승**하며, attention 에너지가 결함 내부에 더 밀집하게 되었다.
- CAM-IoU는 0.518 → 0.504로 소폭 하락했으나, 이는 attention이 결함 영역 전체를 넓게 커버하기보다 **결함 핵심 영역에 더 집중(sharp)**해진 것으로 해석된다.
- **Localization 없이도 높은 해석 가능성을 달성**할 수 있음을 보여주는 Strategy이다.

#### S6 — Full + GT Mask Curriculum (Best Overall)

| Metric | Value | vs S1 | vs S8 |
|--------|-------|-------|-------|
| CAM-IoU | 0.479 | +0.298 | -0.025 |
| Point Game | 0.919 | +0.834 | -0.001 |
| Energy Inside | **0.759** | **+0.593** | +0.002 |
| Loc-IoU | **0.253** | **+0.225** | **+0.218** |

- **Energy Inside 0.759로 전체 1위**, Point Game 0.919로 S8과 거의 동률(0.001 차이)이다.
- S8과의 핵심 차별점은 **Loc-IoU 0.253**으로, Localization head를 포함한 유일한 Strategy이다. 결함 위치를 pixel 수준에서 예측할 수 있는 능력을 갖춘다.
- CAM-IoU 0.479는 S7(0.518) 대비 낮지만, 이는 Localization loss가 attention 학습에 간섭하여 attention map의 공간적 겹침이 다소 줄어든 결과이다.
- 종합적으로 **Point Game, Energy Inside, Loc-IoU 3개 metric에서 최상위권**을 유지하며, 분류 성능(Accuracy 0.907, Recall 0.949)까지 포함하면 **가장 균형 잡힌 Strategy**이다.

### 3.4 핵심 분석

#### (1) Metric 간 1위 Strategy가 분산됨

| Metric | 1위 Strategy | 값 |
|--------|-------------|-----|
| CAM-IoU | S7 | 0.518 |
| Point Game | S8 | 0.920 |
| Energy Inside | S6 | 0.759 |
| Loc-IoU | S6 | 0.253 |

- 단일 Strategy가 모든 해석 metric을 지배하지 않으며, **각 metric의 특성에 따라 최적 Strategy가 다르다.**
- CAM-IoU(전체 겹침)는 Localization 미포함 + Counterfactual(S7)에서, Point Game(peak 위치)과 Energy Inside(에너지 집중)는 GT Mask Curriculum(S8, S6)에서 최적화된다.
- Loc-IoU는 Localization head가 있는 S6만이 유의미한 수치(0.253)를 보인다.

#### (2) S1 → S2 전환: 해석 가능성의 결정적 전환점

| Metric | S1 | S2 | 향상 배율 |
|--------|------|------|----------|
| CAM-IoU | 0.181 | 0.478 | 2.64x |
| Point Game | 0.085 | 0.896 | 10.5x |
| Energy Inside | 0.166 | 0.689 | 4.15x |

- S1→S2 전환에서 **모든 해석 metric이 수 배~10배 이상 향상**된다.
- 이후 S2→S3→...→S6 구간의 향상 폭(소수점 단위)과 비교하면, **CAM Guidance(`L_cam_guide`)가 해석 가능성 확보의 가장 핵심적인 단일 요소**임이 명확하다.
- S2 이후의 추가 구성요소들은 이 기반 위에서 **점진적인 미세 조정(fine-tuning)** 역할을 수행한다.

#### (3) GT Mask Curriculum의 효과: Point Game과 Energy Inside에 집중

- S7→S8, S5(추정)→S6 모두에서 curriculum 적용 시 **Point Game과 Energy Inside가 뚜렷하게 향상**된다.
- S7→S8: Point Game +0.041, Energy Inside +0.061
- 훈련 초기에 GT mask를 attention에 직접 주입하여, 최대 activation 위치와 에너지 분포를 결함 중심으로 유도하는 효과가 확인된다.
- 반면 CAM-IoU는 curriculum 적용 시 소폭 하락(S7→S8: -0.014)하는 경향이 있어, curriculum이 attention을 **더 좁고 집중된 형태(sharper)**로 만드는 특성을 보인다.

#### (4) Localization 포함 여부에 따른 해석 metric 차이

| Metric | Localization 미포함 최고 (S7 or S8) | S6 (Localization 포함) | 차이 |
|--------|--------------------------------------|------------------------|------|
| CAM-IoU | 0.518 (S7) | 0.479 | -0.039 |
| Point Game | 0.920 (S8) | 0.919 | -0.001 |
| Energy Inside | 0.757 (S8) | 0.759 | +0.002 |
| Loc-IoU | 0.035 (S7/S8) | 0.253 | **+0.218** |

- Point Game과 Energy Inside는 Localization 포함 여부에 **거의 영향을 받지 않는다** (차이 0.002 이내).
- CAM-IoU는 Localization이 없는 S7이 +0.039 높다. Localization loss가 attention의 spatial 학습에 다소 간섭하여 전체 겹침 면적이 줄어드는 것으로 보인다.
- **Loc-IoU는 Localization head 포함 여부에 따라 결정적 차이**(0.035 vs 0.253)가 발생하며, pixel 수준의 결함 위치 예측이 필요한 경우 S6가 유일한 선택이다.

---
