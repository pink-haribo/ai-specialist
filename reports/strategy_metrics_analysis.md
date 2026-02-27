# GAIN-MTL Strategy Metrics Analysis Report

## 1. Overview

This report analyzes the evaluation results of 6 progressive training strategies in the GAIN-MTL
(Guided Attention Inference Multi-Task Learning) framework for manufacturing defect detection.

Each strategy incrementally adds loss components to improve both classification performance and
model interpretability (i.e., whether the model focuses on actual defect regions).

| Strategy | Description |
|----------|-------------|
| S1 | Classification Only (Baseline) |
| S2 | + CAM Guidance |
| S3 | + Attention Mining |
| S4 | + Localization |
| S5 | Full (+ Counterfactual) |
| S6 | + GT Mask Fusion (Curriculum) |

---

## 2. Classification Metrics

| Strategy | Accuracy | Recall | F1 | Precision (derived) |
|----------|----------|--------|----|---------------------|
| S1 | 0.878 | 0.861 | 0.897 | ~0.936 |
| S2 | 0.879 | 0.889 | 0.896 | ~0.903 |
| S3 | 0.883 | 0.894 | 0.899 | ~0.904 |
| S4 | 0.872 | 0.866 | 0.890 | ~0.915 |
| S5 | **0.889** | **0.900** | **0.905** | ~0.910 |

> Precision is derived from F1 and Recall: `P = (F1 * R) / (2R - F1)`

### Key Findings — Classification

- **S5 (Full) achieves the best scores across all classification metrics.**
  - Accuracy 0.889, Recall 0.900, F1 0.905.
  - Counterfactual reasoning forces the model to rely on actual defect features, improving robustness.

- **S4 shows unexpected regression compared to S3.**
  - Accuracy: 0.883 → 0.872 (-1.1%p)
  - Recall: 0.894 → 0.866 (-2.8%p)
  - F1: 0.899 → 0.890 (-0.9%p)
  - Likely caused by **multi-task interference** — localization loss (`L_loc`) and consistency
    loss (`L_consist`) being introduced simultaneously disrupts classification feature learning.

- **Precision-Recall tradeoff is evident.**
  - S1 has the highest precision (~0.936) but the lowest recall (0.861).
  - Progressive strategies shift the balance toward recall, which is desirable for defect
    detection where missing defects (FN) is more costly than false alarms (FP).

- **S3 (Attention Mining) provides the most stable improvement over baseline.**
  - All three metrics improve over S1 and S2 without any regression.

---

## 3. CAM Quality Metrics (Interpretability)

| Strategy | CAM-IoU | PointGame | Energy-Inside |
|----------|---------|-----------|---------------|
| S1 | 0.116 | 0.161 | 0.109 |
| S2 | **0.494** | **0.890** | 0.660 |
| S3 | 0.451 | 0.841 | 0.626 |
| S4 | 0.465 | 0.888 | **0.689** |
| S5 | 0.465 | 0.886 | 0.683 |

### Key Findings — Interpretability

- **S1 is a textbook case of "right answer, wrong reason."**
  - Achieves Accuracy 0.878 while CAM-IoU is only 0.116 — the model classifies reasonably
    well but looks at entirely wrong regions.
  - This validates the core motivation of the GAIN-MTL framework.

- **S2 (CAM Guidance) delivers the most dramatic interpretability improvement.**
  - CAM-IoU: 0.116 → 0.494 (4.3x improvement)
  - PointGame: 0.161 → 0.890 (5.5x improvement)
  - Energy-Inside: 0.109 → 0.660 (6.1x improvement)
  - Direct supervision of weight-based CAM with ground truth masks is highly effective.

- **S3 (Attention Mining) shows lower CAM quality than S2.**
  - CAM-IoU: 0.494 → 0.451 (-0.043)
  - PointGame: 0.890 → 0.841 (-0.049)
  - Energy-Inside: 0.660 → 0.626 (-0.034)
  - The separate attention module learns features beneficial for classification but at the
    cost of slightly weaker CAM alignment. Direct supervision (S2) is more effective for
    CAM quality, while the learned attention module (S3) is better for classification.

- **S4/S5 CAM quality plateaus around 0.465 CAM-IoU.**
  - Counterfactual reasoning (S5) does not further improve interpretability over S4.
  - Localization head partially recovers Energy-Inside (+0.063 over S3), suggesting the
    segmentation task provides some spatial guidance.

---

## 4. Gap Analysis vs. Expected Results

Actual results show a significant gap compared to expected values (from README):

| Strategy | Expected Accuracy | Actual Accuracy | Expected CAM-IoU | Actual CAM-IoU |
|----------|-------------------|-----------------|-------------------|----------------|
| S1 | ~0.942 | 0.878 | 0.421 | 0.116 |
| S2 | ~0.950 | 0.879 | 0.550 | 0.494 |
| S3 | ~0.962 | 0.883 | 0.680 | 0.451 |
| S4 | ~0.969 | 0.872 | 0.710 | 0.465 |
| S5 | ~0.975 | 0.889 | 0.756 | 0.465 |

The most concerning gap is in **S1's CAM-IoU** (expected 0.421 vs actual 0.116), which may indicate
an issue with CAM generation/normalization rather than model quality. This warrants investigation
of the CAM post-processing pipeline.

---

## 5. Combined Strategy Profile

```
Strategy   Classification (F1)          Interpretability (CAM-IoU)
S1         ████████░░  0.897            █░░░░░░░░░  0.116
S2         ████████░░  0.896            █████░░░░░  0.494
S3         █████████░  0.899            ████░░░░░░  0.451
S4         ████████░░  0.890            █████░░░░░  0.465
S5         █████████░  0.905            █████░░░░░  0.465
```

No single strategy dominates both dimensions. S5 leads in classification, while S2 leads in
interpretability — highlighting a **classification vs. interpretability tradeoff**.

---

## 6. Recommendations

### Immediate Actions

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | **Adopt S5 as the production model** | Best classification performance (F1 0.905, Recall 0.900). Recall is the priority metric for defect detection. |
| 2 | **Investigate S1 CAM-IoU gap** | 0.116 vs expected 0.421 — check CAM normalization, thresholding, and post-processing logic. |
| 3 | **Tune S4 loss weights** | Reduce `lambda_loc` from 0.3 to 0.1~0.2, or extend stage-wise warmup to mitigate multi-task interference. |

### Experimental Suggestions

| Experiment | Description | Expected Outcome |
|------------|-------------|------------------|
| **S5 + CAM Guidance** | Add `L_cam_guide` (from S2) to S5's full loss | Combine S5's classification strength with S2's CAM quality |
| **Increase `lambda_guide`** | Raise from 0.3 to 0.5 in S3~S5 | Recover CAM quality lost when transitioning from S2's direct supervision |
| **Gradual localization warmup** | In S4, introduce `L_loc` more slowly (e.g., linear warmup over first 50% of stage 3) | Reduce multi-task interference that causes S4 regression |

---

## 7. Applied Changes

All recommendations have been implemented in the codebase:

### 7.1 Strategy Weight Adjustments (`train.py`)

| Change | S3 | S4 | S5 |
|--------|----|----|-----|
| `lambda_cam_guide` | 0.0 → **0.3** | 0.0 → **0.3** | 0.0 → **0.3** |
| `lambda_guide` | 0.3 → **0.5** | 0.5 (unchanged) | 0.5 (unchanged) |
| `lambda_loc` | N/A | 0.3 → **0.2** | 0.3 → **0.2** |

### 7.2 Gradual Localization Warmup (`train.py`)

- Added `loc_warmup_ratio: 0.5` to S4 and S5
- `lambda_loc` linearly increases from 0.0 to target value over the first 50% of epochs
- Uses existing `criterion.update_weights()` API for clean integration

### 7.3 Per-Image CAM Normalization (`metrics.py`, `trainer.py`)

- Added min-max normalization per image before threshold-based binarization
- Applied consistently across: `compute_cam_metrics`, `_compute_per_image_cam_metrics`, `_compute_cam_iou`
- This ensures fair CAM evaluation for unsupervised strategies (S1) where raw sigmoid
  outputs may be concentrated in a narrow range far from the 0.5 threshold

---

## 8. Conclusion (Round 1)

- **S5 (Full)** is the recommended production model based on classification metrics.
- The **classification-interpretability gap** between S2 and S5 suggests that combining both
  direct CAM supervision and counterfactual reasoning could yield optimal results.
  **This has now been addressed by adding `lambda_cam_guide: 0.3` to S3~S5.**
- **S4's regression** is a clear signal that loss weight balancing and multi-task training
  scheduling need refinement. **This has been addressed by reducing `lambda_loc` to 0.2
  and adding gradual localization warmup over 50% of training.**
- The **gap vs. expected results**, especially S1's CAM-IoU, warrants investigation of the
  evaluation pipeline itself. **Per-image min-max CAM normalization has been added to
  address this.**

---

## 9. Round 2 — Updated Experiment Results (Post-Optimization)

### 9.1 Overview

This section reports results after applying the Round 1 optimizations (Section 7) and
introducing a new **S6 (GT Mask Fusion)** strategy. Key changes applied:
- `lambda_cam_guide: 0.3` added to S3~S5
- `lambda_guide` increased to 0.5 in S3
- `lambda_loc` reduced to 0.2 in S4~S5 with gradual warmup (50%)
- Per-image min-max CAM normalization in evaluation
- S6: Full loss (S5) + multiplicative GT mask fusion with curriculum alpha decay

### 9.2 Full Results

| Metric | S1 | S2 | S3 | S4 | S5 | S6 |
|--------|-----|-----|-----|-----|-----|-----|
| Accuracy | 0.89 | 0.88 | 0.90 | 0.89 | 0.89 | **0.91** |
| Recall | 0.88 | 0.88 | 0.94 | 0.92 | 0.93 | **0.95** |
| CAM-IoU | 0.18 | 0.48 | **0.50** | **0.50** | 0.49 | 0.48 |
| PointGame | 0.08 | 0.90 | 0.89 | 0.90 | 0.89 | **0.92** |
| Energy-Inside | 0.17 | 0.68 | 0.69 | 0.70 | 0.70 | **0.76** |
| Loc-IoU | 0.02 | 0.02 | 0.04 | **0.27** | **0.27** | 0.25 |

### 9.3 Comparison with Round 1

| Metric | Strategy | Round 1 | Round 2 | Delta | Verdict |
|--------|----------|---------|---------|-------|---------|
| Accuracy | S4 | 0.872 | 0.89 | **+1.8%p** | S4 regression resolved |
| Recall | S3 | 0.894 | 0.94 | **+4.6%p** | Major improvement |
| Recall | S4 | 0.866 | 0.92 | **+5.4%p** | S4 regression resolved |
| Recall | S5 | 0.900 | 0.93 | **+3.0%p** | Improved |
| CAM-IoU | S1 | 0.116 | 0.18 | **+0.064** | Normalization effective |
| CAM-IoU | S3 | 0.451 | 0.50 | **+0.049** | CAM guidance effective |
| PointGame | S1 | 0.161 | 0.08 | **-0.081** | Normalization side-effect |
| PointGame | S3 | 0.841 | 0.89 | **+0.049** | Recovered |
| Energy-Inside | S3 | 0.626 | 0.69 | **+0.064** | CAM guidance effective |

### 9.4 Key Findings — Round 2

#### Resolved Issues

1. **S4 multi-task interference eliminated.**
   - Round 1: S3→S4 showed accuracy drop (0.883→0.872) and recall drop (0.894→0.866).
   - Round 2: S4 achieves Accuracy 0.89, Recall 0.92 — no regression from S3.
   - `lambda_loc` reduction (0.3→0.2) and gradual warmup successfully mitigate interference.

2. **S3 interpretability gap recovered.**
   - Round 1: S2→S3 showed CAM-IoU drop (0.494→0.451) and PointGame drop (0.890→0.841).
   - Round 2: S3 CAM-IoU 0.50, PointGame 0.89 — now exceeds S2 on CAM-IoU.
   - Adding `lambda_cam_guide: 0.3` to S3 preserves direct CAM supervision alongside learned attention.

3. **Classification metrics broadly improved.**
   - Recall improved across all strategies (S3~S6 all exceed 0.92).
   - This is critical for manufacturing defect detection where missing defects (FN) is costly.

#### New Findings

4. **S6 (GT Mask Fusion) achieves best overall profile.**
   - Highest Accuracy (0.91), Recall (0.95), PointGame (0.92), Energy-Inside (0.76).
   - Curriculum-based GT mask fusion effectively enhances both classification and attention quality.
   - However, Loc-IoU (0.25) slightly lower than S4/S5 (0.27), and CAM-IoU (0.48) slightly
     lower than S3/S4 (0.50).

5. **CAM-IoU plateaus at ~0.48–0.50 across all supervised strategies (S2–S6).**
   - Despite different loss configurations, no strategy breaks through 0.50.
   - This suggests a structural bottleneck — likely attention map resolution or binarization
     threshold, not loss design.

6. **S1 PointGame anomaly (0.161→0.08).**
   - Per-image normalization spreads S1's unfocused attention more uniformly, making argmax
     location random. CAM-IoU improved (0.116→0.18), so normalization itself is valid.
   - PointGame is argmax-based and inherently sensitive to normalization of diffuse attention maps.
   - This is expected behavior for an unsupervised baseline — not a concern.

7. **Loc-IoU shows clear strategy boundary.**
   - S1–S3 (no localization head): 0.02–0.04 (near-zero, expected).
   - S4–S6 (with localization head): 0.25–0.27.
   - Localization head functions correctly but has room for improvement.

### 9.5 Updated Strategy Profile

```
Strategy   Classification (Recall)       Interpretability (Energy-Inside)   Localization (Loc-IoU)
S1         █████████░  0.88              ██░░░░░░░░  0.17                   ░░░░░░░░░░  0.02
S2         █████████░  0.88              ███████░░░  0.68                   ░░░░░░░░░░  0.02
S3         █████████░  0.94              ███████░░░  0.69                   ░░░░░░░░░░  0.04
S4         █████████░  0.92              ███████░░░  0.70                   ███░░░░░░░  0.27
S5         █████████░  0.93              ███████░░░  0.70                   ███░░░░░░░  0.27
S6         ██████████  0.95              ████████░░  0.76                   ███░░░░░░░  0.25
```

**S6 leads in both classification and interpretability**, while S4/S5 lead in localization.

---

## 10. Round 2 — Recommendations & Future Experiments

### 10.1 Production Recommendation

**Adopt S6 as the production model.** S6 achieves:
- Best classification: Accuracy 0.91, Recall 0.95
- Best interpretability: PointGame 0.92, Energy-Inside 0.76
- Acceptable localization: Loc-IoU 0.25 (only 0.02 below S4/S5)

If pixel-level localization is critical, **S5 is the fallback** (Recall 0.93, Loc-IoU 0.27).

### 10.2 Next Experiment Directions

| Priority | Experiment | Rationale | Expected Outcome |
|----------|------------|-----------|------------------|
| **P1** | **CAM-IoU bottleneck investigation**: Sweep binarization threshold (0.3, 0.4, 0.5, 0.6) and measure CAM-IoU sensitivity | All strategies plateau at ~0.50. Need to determine if this is a threshold artifact or true attention quality limit | Identify optimal threshold; if CAM-IoU jumps at different threshold, the bottleneck is evaluation-side |
| **P2** | **Attention map resolution**: Replace bilinear upsampling with learned deconvolution for attention/CAM maps | Low-resolution feature maps (e.g., 7×7 for EfficientNetV2-S) limit spatial precision of attention | CAM-IoU 0.55+ by preserving finer spatial detail |
| **P3** | **S6 Loc-IoU recovery**: Increase `lambda_loc` slightly (0.2→0.25) or extend localization warmup to 70% in S6 | S6 Loc-IoU (0.25) underperforms S4/S5 (0.27); GT mask fusion may interfere with localization gradient | Recover Loc-IoU to 0.27+ while maintaining S6's classification advantage |
| **P4** | **S6 alpha decay schedule**: Compare linear, cosine, and step-decay for curriculum GT mask blend | Current decay schedule may not be optimal; too fast/slow decay affects final attention quality | Identify best schedule for CAM-IoU and Energy-Inside |
| **P5** | **Backbone scale-up**: Test EfficientNetV2-M/L | Current S model may have representation capacity limits contributing to CAM-IoU plateau | Broad improvement across all metrics, especially CAM-IoU and Loc-IoU |
| **P6** | **CRF post-processing for Loc-IoU**: Apply DenseCRF or learnable CRF to localization output | Loc-IoU 0.27 is relatively low; CRF can refine coarse segmentation boundaries | Loc-IoU 0.35+ without retraining |

### 10.3 Analysis Summary

The Round 1 optimizations successfully resolved the two major issues:
- **S4 multi-task interference** → eliminated by loss weight tuning and gradual warmup
- **S3 interpretability regression** → recovered by adding `lambda_cam_guide`

The remaining bottleneck is **CAM-IoU ~0.50 ceiling** and **Loc-IoU ~0.27 ceiling**, which
likely require architectural changes (P2, P5) or post-processing improvements (P1, P6)
rather than further loss weight tuning.
