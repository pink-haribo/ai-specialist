# GAIN-MTL Strategy Metrics Analysis Report

## 1. Overview

This report analyzes the evaluation results of 5 progressive training strategies in the GAIN-MTL
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

## 7. Conclusion

- **S5 (Full)** is the recommended production model based on classification metrics.
- The **classification-interpretability gap** between S2 and S5 suggests that combining both
  direct CAM supervision and counterfactual reasoning could yield optimal results.
- **S4's regression** is a clear signal that loss weight balancing and multi-task training
  scheduling need refinement.
- The **gap vs. expected results**, especially S1's CAM-IoU, warrants investigation of the
  evaluation pipeline itself before drawing final conclusions.
