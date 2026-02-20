# 배경 지식

## 1. 제조 결함 검출과 딥러닝

### 1.1 전통적 접근: 이미지 분류 기반 검사

제조 환경에서 양품/불량 판정은 대표적인 이진 분류(Binary Classification) 문제이다. CNN(Convolutional Neural Network) 기반 분류 모델은 높은 정확도를 달성할 수 있으나, 다음과 같은 한계가 존재한다:

- **Black-box 문제**: 모델이 "왜" 불량이라고 판단했는지 알 수 없음
- **Spurious Correlation**: 결함 자체가 아닌 배경 패턴, 조명 조건 등에 의존하여 판정
- **일반화 취약성**: 학습 데이터의 편향에 의존하면 환경 변화 시 성능 급감

### 1.2 해석 가능성(Interpretability)의 필요성

제조 현장에서 AI 모델을 배포하려면 단순 정확도 이상의 **신뢰성**이 요구된다:

| 요구사항 | 설명 |
|---------|------|
| **판단 근거 확인** | 운영자가 모델의 분류 근거를 시각적으로 확인 가능해야 함 |
| **결함 위치 파악** | 단순 양/불 판정을 넘어 결함의 위치를 알아야 후속 공정 개선 가능 |
| **품질 감사(Audit)** | 제조 품질 기준 충족 여부를 외부에 설명할 수 있어야 함 |
| **실패 분석** | 오판 시 원인을 분석하고 모델을 개선할 수 있어야 함 |

---

## 2. Class Activation Map (CAM)

### 2.1 CAM의 개념

CAM(Class Activation Map)은 CNN이 특정 클래스를 예측할 때 **입력 이미지의 어느 영역에 주목**했는지를 시각화하는 기법이다.

```
기본 원리:
  특성 맵 (Feature Map)  ×  분류기 가중치 (Classifier Weights)
       (C, H, W)                    (C,)
                    ↓
            가중합 (Weighted Sum)
                    ↓
          Class Activation Map (H, W)
```

### 2.2 Weight-based CAM

본 연구에서 사용하는 Weight-based CAM은 분류기의 가중치를 직접 활용하여 생성한다:

```
CAM(x, y) = Σ_c  w_c · F_c(x, y)

  w_c : 클래스 c에 대한 분류기 가중치
  F_c : c번째 채널의 특성 맵
```

- 별도의 후처리 없이 **미분 가능(differentiable)**하므로 학습 중 직접 supervision 가능
- 이것이 기존 Grad-CAM 등과의 핵심적 차이점

### 2.3 CAM 품질 평가 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| **CAM-IoU** | \|CAM ∩ GT\| / \|CAM ∪ GT\| | Attention과 실제 결함 영역의 겹침 비율 |
| **PointGame** | 1(argmax(CAM) ∈ GT) | 최대 활성화 지점이 결함 내부에 있는지 |
| **Energy-Inside** | Σ(CAM · GT) / Σ(CAM) | 전체 attention 에너지 중 결함 내부 비율 |

---

## 3. Attention Mechanism

### 3.1 Channel Attention

**"어떤 특성(what)이 중요한가?"**를 학습하는 메커니즘이다.

```
입력: F ∈ R^(C×H×W)

  F → AvgPool → MLP → σ → Channel Weight (C, 1, 1)
  F → MaxPool → MLP ↗

출력: F' = F × Channel Weight
```

- Global Average Pooling과 Max Pooling으로 채널별 통계를 추출
- 공유 MLP를 통해 채널 간 상호작용을 모델링
- Sigmoid로 각 채널의 중요도 가중치 생성

### 3.2 Spatial Attention

**"어디(where)가 중요한가?"**를 학습하는 메커니즘이다.

```
입력: F ∈ R^(C×H×W)

  AvgPool(F, dim=C) → Concat → Conv(7×7) → σ → Spatial Weight (1, H, W)
  MaxPool(F, dim=C) ↗

출력: F' = F × Spatial Weight
```

- 채널 축을 따라 평균/최대 풀링하여 공간적 정보를 압축
- 7×7 컨볼루션으로 공간적 주의 가중치 생성

### 3.3 CBAM (Convolutional Block Attention Module)

Channel Attention과 Spatial Attention을 **순차적으로 결합**한 모듈이다:

```
F → Channel Attention → F' → Spatial Attention → F''
    (what에 주목)            (where에 주목)
```

본 연구에서는 CBAM을 GAIN Attention Module의 전처리 단계로 사용하여 특성을 정제한다.

---

## 4. Multi-Task Learning (MTL)

### 4.1 개념

하나의 모델이 **여러 관련된 작업을 동시에 학습**하는 패러다임이다.

```
              ┌─── Task 1: 분류 (양품/불량)
공유 표현 ────┤
(Backbone)    ├─── Task 2: 위치 추정 (결함 영역 segmentation)
              └─── Task 3: 반사실적 추론 (결함 제거 시 양품 판정)
```

### 4.2 MTL의 장점

| 장점 | 설명 |
|------|------|
| **Inductive Bias** | 보조 작업이 주 작업의 학습을 정규화하여 과적합 방지 |
| **Feature Sharing** | 위치 추정 작업이 backbone에 더 풍부한 공간 정보 학습을 유도 |
| **Data Efficiency** | 여러 supervision 신호로 같은 데이터에서 더 많은 정보 추출 |

### 4.3 Multi-Task Interference

여러 작업의 loss가 서로 상충하여 **성능이 오히려 하락**하는 문제가 발생할 수 있다.

```
예시 (본 연구 S4):
  S3 (분류 + attention)  →  Accuracy 88.3%
  S4 (+ localization)    →  Accuracy 87.2%  ← 오히려 하락!
```

이를 해결하기 위한 전략:
- **단계적 학습 (Staged Training)**: loss를 점진적으로 추가
- **Loss Weight 조절**: 각 loss의 기여도를 균형 있게 설정
- **Warmup**: 새로운 loss를 서서히 도입

---

## 5. Feature Pyramid Network (FPN)

### 5.1 개념

CNN backbone의 **다양한 해상도의 특성 맵을 융합**하여 멀티스케일 표현을 구축하는 구조이다.

```
Backbone 출력 (상위 → 하위 해상도):
  Stage 3 ────→ ┌──────┐
  Stage 4 ────→ │      │
  Stage 5 ────→ │  FPN │ ──→ 융합된 멀티스케일 특성
  Stage 6 ────→ │      │
                └──────┘
```

### 5.2 본 연구에서의 역할

- Localization Head에 멀티스케일 특성을 제공하여 **다양한 크기의 결함**을 검출
- 큰 결함과 작은 결함 모두를 효과적으로 처리

---

## 6. Counterfactual Reasoning (반사실적 추론)

### 6.1 개념

**"만약 결함이 없었다면 어떤 결과가 나왔을까?"**를 시뮬레이션하는 학습 방법이다.

```
원본 이미지 (불량)  →  분류기  →  "불량" (정상 동작)
       ↓
결함 영역 제거      →  분류기  →  "양품" (기대되는 결과)
```

### 6.2 학습 메커니즘

1. **Feature Suppression**: 결함 마스크 영역의 특성을 억제/대체
2. **재분류**: 수정된 특성으로 다시 분류 수행
3. **Loss 적용**: 수정된 특성은 "양품"으로 분류되어야 한다는 제약 조건

### 6.3 왜 중요한가

- 모델이 **실제 결함 특성에 의존**하고 있는지를 검증
- Spurious correlation 의존을 효과적으로 방지
- 결함 영역을 제거했을 때 판정이 바뀌지 않으면 → 모델이 결함이 아닌 다른 것을 보고 있다는 증거

---

## 7. EfficientNetV2

### 7.1 개요

Google이 제안한 고효율 CNN 아키텍처로, **학습 속도와 파라미터 효율성**이 뛰어나다.

| 특성 | 설명 |
|------|------|
| **Fused-MBConv** | 초기 레이어에서 depthwise conv 대신 일반 conv 사용으로 학습 가속 |
| **Progressive Learning** | 이미지 크기, 정규화 강도를 점진적으로 증가 |
| **ImageNet 사전학습** | 풍부한 시각적 특성을 이미 학습한 상태에서 미세 조정 |

### 7.2 본 연구에서의 구성

```
EfficientNetV2-S:
  Stem → Block1 → Block2 → ... → Block6 → Conv_head(1280ch)
                                    ↓              ↓
                              FPN 입력         분류/CAM 입력
                          (멀티스케일 특성)    (고수준 의미 특성)
```

- Conv_head의 1280채널은 ImageNet에서 사전학습된 의미적으로 풍부한 특성
- 이를 분류기와 CAM 생성에 활용하여 높은 품질의 Weight-based CAM 생성 가능

---

## 8. 주요 Loss Function

### 8.1 Focal Loss (분류)

클래스 불균형 문제를 해결하기 위한 loss 함수이다.

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

  α : 클래스별 가중치 (양품/불량 비율 보정)
  γ : 초점 파라미터 (어려운 샘플에 더 집중, 기본값 2.0)
```

- 쉬운 샘플의 loss 기여를 줄이고, **분류가 어려운 샘플에 집중**
- 제조 데이터에서 흔한 양품/불량 비율 불균형 처리에 효과적

### 8.2 Dice Loss (위치 추정)

영역 겹침을 최적화하는 segmentation용 loss이다.

```
Dice = 2 × |A ∩ B| / (|A| + |B|)
Loss = 1 - Dice
```

- 픽셀 수 불균형(배경 >> 결함)에 강건
- BCE Loss와 결합하여 사용

### 8.3 IoU Loss (Attention 정렬)

Attention map과 GT mask의 겹침을 직접 최적화한다.

```
IoU = |A ∩ B| / |A ∪ B|
Loss = 1 - IoU
```
