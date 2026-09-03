# VCPS Trajectory Calibration vs. Baselines: MQT-LLaVA

> **Document Purpose**: Comprehensive unified benchmark comparison comparing standard calibration baselines (TS, Platt, Spline, ATS, Residual Calibrator), proposed trajectory calibration methods (**VCPS-5D**, **VCPS-17D**), and UMPIRE multi-pass uncertainty quantification baselines across all 14 datasets.

## Evaluation Setup & Temperature Protocol Note

> [!IMPORTANT]
> **CRITICAL TEMPERATURE & COMPUTATION PROTOCOL**:
> 1. **Greedy Deterministic Single-Pass ($T = 0.0, K = 1$)**:
>    - Evaluated for: **Naive Confidence (NC)**, **Temperature Scaling (TS)**, **Platt Scaling (1D)**, **Trajectory LR (No Bias)**, **Quadratic Platt (Logit-Only)**, **Spline Calibration (PCHIP)**, **Adaptive TS (ATS)**, **Residual Calibrator**, **VCPS-5D**, and **VCPS-17D**.
>    - Evaluated on exact autoregressive logit trajectories from standard single-pass greedy decoding. Computational overhead: **$1\times$ forward pass** (real-time zero rollout overhead).
> 2. **Stochastic Multi-Pass Sampling ($T = 0.5, K = 10$)**:
>    - Evaluated for: **`ln_entropy`**, **`semantic_entropy`**, **`eigen_score`**, and **`umpire`**.
>    - Requires generating $K=10$ stochastic rollout paths per prompt, followed by bidirectional DeBERTa NLI cross-encoder semantic clustering or sentence-embedding covariance SVD decomposition. Computational overhead: **$10\times$ autoregressive generation + NLI inference**.

- **Architecture**: MQT-LLaVA (7B parameters)
- **Total Benchmarks Evaluated**: 14 diverse multimodal datasets
- **Benchmark Results Source**: `results/experiments/benchmark/benchmark_mqt_summary.csv`
- **UMPIRE Results Source**: `results/umpire_eval/mqt_llava_cumulative_summary.csv`

---

## Section 1: Per-Dataset Comprehensive Comparison Tables (14 Datasets)

In each table:
- **ECE (%)**: Standard Expected Calibration Error (lower is better $\downarrow$). For UMPIRE methods, Calibrated ECE (cece) is reported.
- **Adaptive ECE (%)**: Bin-balanced Adaptive Expected Calibration Error (lower is better $\downarrow$).
- **AUROC**: Area Under ROC Curve for error detection / selective prediction (higher is better $\uparrow$).
- **Highlighting**: Lowest ECE in **bold**, second-lowest ECE in *italics*. Highest AUROC in **bold**, second-highest AUROC in *italics*.

### Benchmark: `ai2d`
**Dataset**: `ai2d` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 24.34% | 24.31% | 0.705 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 11.04% | 13.47% | 0.705 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 7.89% | 9.52% | 0.705 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 8.62% | 11.37% | 0.689 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 11.25% | 13.56% | 0.681 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **6.24%** | 9.68% | 0.705 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 8.72% | 8.98% | 0.699 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 11.61% | 13.92% | 0.705 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | *7.14%* | 11.34% | *0.709* | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.33% | 13.46% | **0.709** | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.83% | 11.90% | 0.701 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.40% | - | 0.611 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 35.70% | - | 0.594 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 31.50% | - | 0.639 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 31.80% | - | 0.612 | `VALID` |

### Benchmark: `chartqa`
**Dataset**: `chartqa` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 62.99% | 62.99% | 0.467 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 41.54% | 41.54% | 0.467 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 2.83% | 7.04% | 0.533 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 3.84% | 6.74% | 0.739 | `SCRAMBLED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 8.91% | 11.40% | 0.402 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 2.86% | 7.05% | 0.533 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 6.58% | 7.07% | 0.513 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 41.53% | 41.53% | 0.467 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 1.76% | 6.69% | 0.587 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **0.61%** | 6.18% | 0.506 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *1.30%* | 6.49% | 0.498 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.80% | - | 0.710 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 37.70% | - | 0.830 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 24.90% | - | *0.855* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.00% | - | **0.861** | `VALID` |

### Benchmark: `docvqa`
**Dataset**: `docvqa` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 51.18% | 51.18% | 0.644 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 46.63% | 46.63% | 0.644 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 0.47% | 5.04% | 0.644 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 0.82% | 2.76% | 0.775 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 1.08% | 3.25% | 0.722 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 1.49% | 4.27% | 0.699 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 1.07% | 2.48% | 0.662 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 46.62% | 46.62% | 0.644 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.45% | 4.82% | 0.669 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **0.33%** | 4.84% | 0.671 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *0.45%* | 4.17% | 0.658 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.70% | - | 0.805 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.30% | - | 0.816 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 33.50% | - | *0.839* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.00% | - | **0.853** | `VALID` |

### Benchmark: `gqa`
**Dataset**: `gqa` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 13.64% | 9.87% | 0.757 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.21% | 10.14% | 0.757 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.56% | 10.27% | 0.757 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 9.80% | 10.75% | 0.756 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | *8.24%* | 9.78% | 0.757 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 12.09% | 9.84% | 0.757 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 10.81% | 9.78% | 0.748 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 9.13% | 9.10% | *0.757* | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 9.91% | 9.26% | 0.757 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 9.88% | 9.28% | 0.757 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **8.18%** | 8.02% | **0.757** | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.80% | - | 0.698 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.00% | - | 0.624 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.70% | - | 0.696 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.70% | - | 0.715 | `VALID` |

### Benchmark: `infographicvqa`
**Dataset**: `infographicvqa` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 75.51% | 75.51% | *0.769* | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 52.63% | 52.63% | *0.769* | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *0.28%* | 1.09% | *0.769* | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 0.30% | 1.07% | **0.910** | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 1.21% | 2.19% | 0.286 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **0.28%** | 1.08% | *0.769* | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 0.60% | 0.60% | 0.746 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 52.61% | 52.61% | *0.769* | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.29% | 1.09% | *0.769* | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.29% | 1.09% | *0.769* | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.29% | 1.09% | *0.769* | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 34.10% | - | 0.615 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 47.50% | - | 0.648 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 47.80% | - | 0.648 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 26.10% | - | 0.646 | `VALID` |

### Benchmark: `lego-puzzles`
**Dataset**: `lego-puzzles` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 34.60% | 34.14% | 0.510 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 23.68% | 24.08% | 0.510 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *0.60%* | 11.70% | 0.490 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 2.28% | 10.13% | **0.620** | `SCRAMBLED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 3.47% | 8.47% | *0.618* | `SCRAMBLED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 5.99% | 9.18% | 0.582 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 4.15% | 5.27% | 0.522 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 23.68% | 24.08% | 0.510 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 2.16% | 9.21% | 0.588 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.98% | 12.18% | 0.579 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **0.42%** | 10.44% | 0.551 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.30% | - | 0.531 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 45.20% | - | 0.533 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.40% | - | 0.502 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.60% | - | 0.537 | `VALID` |

### Benchmark: `mmbench`
**Dataset**: `mmbench` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 15.38% | 14.87% | 0.828 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.81% | 10.07% | 0.828 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.11% | 8.24% | 0.828 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | *5.95%* | 7.08% | **0.849** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.70% | 5.43% | *0.835* | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.87%** | 6.80% | 0.828 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 7.18% | 10.04% | 0.828 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 9.38% | 9.89% | 0.831 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 8.33% | 8.53% | 0.831 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.38% | 7.95% | 0.832 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.05% | 8.15% | 0.828 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.20% | - | 0.613 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 23.30% | - | 0.596 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 20.90% | - | 0.675 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.10% | - | 0.618 | `VALID` |

### Benchmark: `mmmu`
**Dataset**: `mmmu` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 52.24% | 52.24% | 0.315 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 49.32% | 49.32% | 0.315 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 0.16% | 2.15% | 0.315 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **0.09%** | 1.87% | *0.779* | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 0.24% | 1.85% | **0.832** | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *0.12%* | 2.15% | 0.432 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 0.29% | 2.27% | 0.315 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 49.32% | 49.32% | 0.315 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.15% | 2.14% | 0.356 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.15% | 2.15% | 0.335 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.15% | 2.15% | 0.321 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.10% | - | 0.584 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 44.40% | - | 0.588 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 45.50% | - | 0.627 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 37.70% | - | 0.598 | `VALID` |

### Benchmark: `pope`
**Dataset**: `pope` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 7.04% | 6.45% | 0.818 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.86% | 5.90% | 0.818 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 4.37% | 5.29% | 0.818 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.12% | 6.54% | **0.819** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.85% | 6.54% | 0.818 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.54% | 5.37% | 0.818 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 7.30% | 7.97% | 0.818 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 5.52% | 6.02% | 0.817 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | **3.75%** | 6.58% | *0.818* | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.92% | 6.61% | 0.818 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *4.37%* | 5.36% | 0.816 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.90% | - | 0.683 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.70% | - | 0.368 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.10% | - | 0.616 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.70% | - | 0.693 | `VALID` |

### Benchmark: `scienceqa`
**Dataset**: `scienceqa` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 23.29% | 23.04% | 0.775 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 10.47% | 10.44% | 0.775 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 7.75% | 6.51% | 0.775 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | *5.55%* | 6.17% | **0.782** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 7.43% | 9.87% | 0.776 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **5.00%** | 5.90% | 0.771 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 7.63% | 7.97% | 0.773 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 10.10% | 10.18% | 0.777 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.17% | 7.23% | 0.780 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.71% | 7.05% | *0.781* | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.61% | 6.28% | 0.776 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.80% | - | 0.581 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.50% | - | 0.568 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.00% | - | 0.588 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.20% | - | 0.581 | `VALID` |

### Benchmark: `seedbench`
**Dataset**: `seedbench` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 18.56% | 17.74% | 0.733 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *6.68%* | 7.99% | 0.733 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.51% | 6.88% | 0.733 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 10.28% | 8.54% | **0.740** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 9.07% | 7.14% | 0.721 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.38% | 6.84% | 0.733 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 8.73% | 7.65% | 0.736 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | **6.28%** | 8.00% | 0.734 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 8.35% | 9.17% | 0.734 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.10% | 9.04% | 0.735 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.58% | 7.55% | *0.737* | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.10% | - | 0.610 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.00% | - | 0.599 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.00% | - | 0.639 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.70% | - | 0.605 | `VALID` |

### Benchmark: `textvqa`
**Dataset**: `textvqa` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 31.24% | 30.96% | 0.681 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 17.62% | 18.27% | 0.681 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.78% | 11.59% | 0.681 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.95% | 5.51% | 0.763 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 7.15% | 8.73% | 0.749 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.55% | 8.69% | 0.681 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **5.49%** | 7.32% | 0.684 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 18.15% | 18.15% | 0.692 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | *6.96%* | 6.71% | 0.712 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.91% | 7.01% | 0.723 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.83% | 6.74% | 0.709 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.80% | - | 0.793 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 31.50% | - | 0.789 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.30% | - | *0.802* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.10% | - | **0.818** | `VALID` |

### Benchmark: `vizwiz-vqa`
**Dataset**: `vizwiz-vqa` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 30.26% | 30.26% | 0.802 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 28.77% | 28.77% | 0.802 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.70% | 6.81% | 0.802 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.22% | 5.66% | **0.804** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.29% | 8.55% | 0.778 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **4.92%** | 5.14% | 0.803 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 6.06% | 5.64% | 0.798 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 29.05% | 28.45% | 0.801 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.26% | 6.61% | *0.803* | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.30% | 5.75% | 0.802 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *5.61%* | 6.22% | 0.802 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.50% | - | 0.718 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.70% | - | 0.689 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.20% | - | 0.744 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.30% | - | 0.754 | `VALID` |

### Benchmark: `vqav2`
**Dataset**: `vqav2` | **Model**: MQT-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 12.14% | 11.52% | 0.746 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 15.10% | 15.96% | 0.746 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 14.26% | 15.96% | 0.746 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | *7.80%* | 8.46% | **0.803** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 9.60% | 9.00% | 0.788 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **6.25%** | 9.08% | 0.765 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 8.84% | 15.17% | 0.749 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 11.79% | 13.10% | 0.765 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 9.30% | 13.12% | 0.751 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 10.63% | 11.16% | 0.757 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 11.02% | 11.68% | 0.751 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.30% | - | 0.767 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.70% | - | 0.699 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.20% | - | 0.773 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 7.80% | - | *0.790* | `VALID` |

---

## Section 2: Macro-Average Summary Table (Across All 14 Benchmarks)

Macro-averaged evaluation metrics across all 14 datasets for **MQT-LLaVA**.

| Calibration Method | Category | Regime / Sampling | Macro ECE (%) $\downarrow$ | Macro Adaptive ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 32.32% | 31.79% | 0.682 |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 23.45% | 23.94% | 0.682 |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.88% | 7.72% | 0.686 |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.40% | 6.62% | **0.774** |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.11% | 7.56% | 0.698 |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **4.90%** | 6.50% | *0.706* |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 5.96% | 7.02% | 0.685 |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 23.20% | 23.64% | 0.685 |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.07% | 7.32% | 0.705 |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *4.97%* | 7.41% | 0.698 |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.05% | 6.87% | 0.691 |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.77% | - | 0.666 |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 33.37% | - | 0.639 |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.57% | - | 0.689 |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 21.27% | - | 0.691 |

---

## Section 3: Win-Count & Comparative Analysis

### A. Win-Count Summary
- **Overall Best ECE (#1 across ALL evaluated methods)**: Our VCPS methods (**VCPS-5D** / **VCPS-17D**) achieve the absolute lowest ECE on **4 out of 14 datasets** (28.6% win rate).
  - **Specific Datasets Won in ECE**: `chartqa` (VCPS-5D (Our Method): **0.61%**), `docvqa` (VCPS-5D (Our Method): **0.33%**), `gqa` (VCPS-17D (Our Method): **8.18%**), `lego-puzzles` (VCPS-17D (Our Method): **0.42%**).
- **Win Rate vs. UMPIRE Multi-Pass Baselines in ECE**: VCPS trajectory calibration achieves lower ECE than all four UMPIRE multi-pass baselines on **13 out of 14 datasets**.
- **Overall Best AUROC (#1 across ALL evaluated methods)**: Our VCPS methods achieve the highest selective prediction AUROC on **2 out of 14 datasets**.
  - **Specific Datasets Won in AUROC**: `ai2d` (VCPS-5D (Our Method): **0.709**), `gqa` (VCPS-17D (Our Method): **0.757**).

### B. Detailed Win Breakdown Table
| Dataset | Lowest ECE Method | Best ECE (%) | Highest AUROC Method | Best AUROC | VCPS Win Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `ai2d` | Quadratic Platt (Logit-Only) | **6.24%** | VCPS-5D (Our Method) | **0.709** | `Top AUROC` |
| `chartqa` | VCPS-5D (Our Method) | **0.61%** | `umpire` | **0.861** | `Top ECE` |
| `docvqa` | VCPS-5D (Our Method) | **0.33%** | `umpire` | **0.853** | `Top ECE` |
| `gqa` | VCPS-17D (Our Method) | **8.18%** | VCPS-17D (Our Method) | **0.757** | `Top ECE, Top AUROC` |
| `infographicvqa` | Quadratic Platt (Logit-Only) | **0.28%** | Trajectory LR | **0.910** | `Competitive` |
| `lego-puzzles` | VCPS-17D (Our Method) | **0.42%** | Trajectory LR | **0.620** | `Top ECE` |
| `mmbench` | Quadratic Platt (Logit-Only) | **3.87%** | Trajectory LR | **0.849** | `Competitive` |
| `mmmu` | Trajectory LR | **0.09%** | Trajectory LR (No Bias) | **0.832** | `Competitive` |
| `pope` | Residual Calibrator | **3.75%** | Trajectory LR | **0.819** | `Competitive` |
| `scienceqa` | Quadratic Platt (Logit-Only) | **5.00%** | Trajectory LR | **0.782** | `Competitive` |
| `seedbench` | Adaptive TS (ATS) | **6.28%** | Trajectory LR | **0.740** | `Competitive` |
| `textvqa` | Spline Calibration | **5.49%** | `umpire` | **0.818** | `Competitive` |
| `vizwiz-vqa` | Quadratic Platt (Logit-Only) | **4.92%** | Trajectory LR | **0.804** | `Competitive` |
| `vqav2` | Quadratic Platt (Logit-Only) | **6.25%** | Trajectory LR | **0.803** | `Competitive` |
