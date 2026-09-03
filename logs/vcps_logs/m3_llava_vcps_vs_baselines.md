# VCPS Trajectory Calibration vs. Baselines: M3-LLaVA

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

- **Architecture**: M3-LLaVA (7B parameters)
- **Total Benchmarks Evaluated**: 14 diverse multimodal datasets
- **Benchmark Results Source**: `results/experiments/benchmark/benchmark_m3_summary.csv`
- **UMPIRE Results Source**: `results/umpire_eval/m3_llava_cumulative_summary.csv`

---

## Section 1: Per-Dataset Comprehensive Comparison Tables (14 Datasets)

In each table:
- **ECE (%)**: Standard Expected Calibration Error (lower is better $\downarrow$). For UMPIRE methods, Calibrated ECE (cece) is reported.
- **Adaptive ECE (%)**: Bin-balanced Adaptive Expected Calibration Error (lower is better $\downarrow$).
- **AUROC**: Area Under ROC Curve for error detection / selective prediction (higher is better $\uparrow$).
- **Highlighting**: Lowest ECE in **bold**, second-lowest ECE in *italics*. Highest AUROC in **bold**, second-highest AUROC in *italics*.

### Benchmark: `ai2d`
**Dataset**: `ai2d` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 41.80% | 41.80% | 0.598 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.56% | 10.73% | 0.598 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *3.49%* | 7.49% | 0.598 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.27% | 8.57% | *0.645* | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 7.27% | 8.57% | **0.645** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.14%** | 7.06% | 0.598 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 6.04% | 6.98% | 0.594 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.50% | 10.69% | 0.598 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.35% | 8.27% | 0.613 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.68% | 8.38% | 0.638 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.39% | 8.92% | 0.638 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.40% | - | 0.617 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 31.60% | - | 0.604 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.20% | - | 0.607 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.60% | - | 0.608 | `VALID` |

### Benchmark: `chartqa`
**Dataset**: `chartqa` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 70.70% | 70.70% | 0.835 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 45.45% | 45.07% | 0.835 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 2.96% | 2.34% | 0.835 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **2.30%** | 3.55% | 0.833 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | *2.53%* | 3.13% | **0.842** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 3.13% | 2.12% | 0.835 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 3.92% | 4.23% | 0.807 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 45.43% | 45.05% | 0.835 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 3.38% | 2.03% | 0.836 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.35% | 2.88% | *0.838* | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 2.96% | 2.33% | 0.837 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 8.90% | - | 0.734 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.80% | - | 0.790 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.30% | - | 0.818 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 8.80% | - | 0.828 | `VALID` |

### Benchmark: `docvqa`
**Dataset**: `docvqa` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 73.52% | 73.52% | 0.732 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 48.86% | 48.86% | 0.732 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *1.80%* | 2.42% | 0.732 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **1.53%** | 2.75% | 0.725 | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.20% | 6.00% | 0.567 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 2.11% | 2.33% | 0.732 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 1.98% | 2.75% | 0.715 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 48.84% | 48.84% | 0.732 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 1.81% | 2.36% | 0.732 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 1.99% | 2.51% | 0.737 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 1.91% | 2.75% | 0.734 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 15.80% | - | 0.786 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 37.10% | - | 0.809 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.70% | - | *0.830* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.40% | - | **0.834** | `VALID` |

### Benchmark: `gqa`
**Dataset**: `gqa` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 32.54% | 32.43% | 0.685 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.24% | 10.84% | 0.685 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | **4.08%** | 7.71% | 0.685 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.41% | 5.72% | 0.700 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.33% | 7.41% | 0.686 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *4.23%* | 7.71% | 0.685 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 5.93% | 6.57% | 0.697 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.86% | 9.84% | 0.702 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.52% | 6.58% | 0.698 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.15% | 7.08% | *0.705* | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.44% | 6.56% | 0.705 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.80% | - | 0.696 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.30% | - | 0.623 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 17.60% | - | 0.690 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 17.40% | - | **0.712** | `VALID` |

### Benchmark: `infographicvqa`
**Dataset**: `infographicvqa` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 84.55% | 84.55% | 0.425 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 53.91% | 53.91% | 0.425 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *0.04%* | 1.12% | 0.575 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **0.02%** | 0.99% | *0.644* | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 0.27% | 1.42% | 0.504 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 0.05% | 1.24% | **0.696** | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 0.60% | 0.60% | 0.500 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 53.89% | 53.89% | 0.425 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.04% | 1.29% | 0.579 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.07% | 1.23% | 0.425 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.07% | 1.31% | 0.424 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.50% | - | 0.547 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.20% | - | 0.534 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 34.80% | - | 0.561 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.60% | - | 0.562 | `VALID` |

### Benchmark: `lego-puzzles`
**Dataset**: `lego-puzzles` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 73.16% | 73.16% | 0.477 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 34.71% | 34.71% | 0.477 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | **0.71%** | 10.07% | 0.477 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.15% | 10.52% | 0.601 | `SCRAMBLED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 8.75% | 10.41% | *0.651* | `SCRAMBLED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.34% | 10.04% | **0.668** | `SCRAMBLED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 6.36% | 6.36% | 0.500 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 34.66% | 34.66% | 0.477 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 2.77% | 6.80% | 0.548 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *0.88%* | 11.97% | 0.515 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 2.74% | 11.86% | 0.506 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 21.20% | - | 0.551 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.00% | - | 0.582 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 15.00% | - | 0.571 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 11.00% | - | 0.588 | `VALID` |

### Benchmark: `mmbench`
**Dataset**: `mmbench` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 28.62% | 28.58% | 0.629 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.68% | 7.77% | 0.629 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *3.07%* | 4.88% | 0.629 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.03% | 6.60% | *0.690* | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.99% | 5.77% | 0.679 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **1.61%** | 4.36% | 0.634 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 4.66% | 4.45% | 0.623 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.48% | 9.99% | 0.685 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.73% | 7.81% | 0.676 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.10% | 6.97% | 0.685 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.39% | 6.18% | **0.695** | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 20.50% | - | 0.660 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.00% | - | 0.646 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.20% | - | 0.681 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.30% | - | 0.658 | `VALID` |

### Benchmark: `mmmu`
**Dataset**: `mmmu` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 78.76% | 78.76% | **0.639** | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 52.21% | 52.21% | **0.639** | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *1.81%* | 3.45% | **0.639** | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 1.96% | 3.59% | 0.607 | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 2.60% | 3.92% | *0.621* | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 2.26% | 4.10% | 0.274 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **1.45%** | 2.64% | 0.555 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 52.19% | 52.19% | **0.639** | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 1.85% | 3.47% | **0.639** | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 1.89% | 3.50% | **0.639** | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 1.97% | 3.49% | **0.639** | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 36.40% | - | 0.566 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 47.60% | - | 0.568 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 50.10% | - | 0.602 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.70% | - | 0.592 | `VALID` |

### Benchmark: `pope`
**Dataset**: `pope` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 4.03% | 4.12% | 0.752 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.67% | 3.96% | 0.752 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.69% | 4.05% | 0.752 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.51% | 6.32% | 0.752 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.36% | 5.56% | 0.750 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 3.64% | 4.06% | 0.752 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 4.92% | 5.10% | 0.754 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 4.63% | 4.16% | *0.757* | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | *3.52%* | 4.31% | 0.751 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.14% | 4.02% | 0.754 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **3.15%** | 3.47% | **0.761** | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.80% | - | 0.668 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.90% | - | 0.387 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.50% | - | 0.678 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.50% | - | 0.666 | `VALID` |

### Benchmark: `scienceqa`
**Dataset**: `scienceqa` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 11.08% | 10.66% | 0.701 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.69% | 9.76% | 0.701 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.63% | 7.46% | 0.701 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **3.09%** | 5.03% | **0.707** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.12% | 6.85% | 0.701 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *3.18%* | 4.13% | 0.680 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 4.44% | 5.16% | 0.695 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 7.70% | 9.43% | *0.705* | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 4.26% | 5.91% | 0.701 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.87% | 5.43% | 0.703 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.71% | 6.00% | 0.703 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 26.80% | - | 0.659 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.00% | - | 0.633 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.10% | - | 0.636 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.10% | - | 0.662 | `VALID` |

### Benchmark: `seedbench`
**Dataset**: `seedbench` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 32.40% | 32.40% | 0.500 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 1.31% | 2.64% | 0.500 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *1.17%* | 2.25% | 0.500 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 3.48% | 4.60% | 0.641 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 3.44% | 4.23% | 0.641 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 1.63% | 2.89% | 0.500 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.72%** | 1.17% | 0.513 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 2.85% | 9.98% | 0.628 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 8.58% | 9.08% | 0.641 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.09% | 8.21% | 0.643 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.55% | 7.83% | 0.626 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 23.30% | - | *0.644* | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.00% | - | 0.629 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.70% | - | **0.655** | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.40% | - | *0.644* | `VALID` |

### Benchmark: `textvqa`
**Dataset**: `textvqa` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 8.32% | 8.72% | 0.832 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.70% | 8.35% | 0.832 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.01% | 5.70% | 0.832 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.78% | 4.41% | *0.841* | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | **3.54%** | 2.93% | **0.841** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.77% | 4.73% | 0.832 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 10.11% | 10.25% | 0.832 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.22% | 8.80% | 0.832 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.01% | 5.70% | 0.834 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.66% | 5.46% | 0.834 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *4.66%* | 5.16% | 0.834 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 11.80% | - | 0.811 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.20% | - | 0.774 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 26.60% | - | 0.814 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.40% | - | 0.829 | `VALID` |

### Benchmark: `vizwiz-vqa`
**Dataset**: `vizwiz-vqa` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 16.71% | 16.71% | 0.835 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 15.56% | 15.56% | 0.835 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.76% | 7.25% | 0.835 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | *5.11%* | 4.18% | 0.827 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | **4.93%** | 6.50% | 0.821 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.26% | 7.41% | 0.835 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 8.13% | 8.45% | 0.826 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 15.45% | 15.45% | 0.835 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.47% | 7.08% | *0.836* | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.46% | 7.02% | **0.836** | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.25% | 7.18% | 0.835 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.80% | - | 0.723 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.40% | - | 0.701 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.30% | - | 0.757 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.70% | - | 0.768 | `VALID` |

### Benchmark: `vqav2`
**Dataset**: `vqav2` | **Model**: M3-LLaVA

| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 8.51% | 7.74% | 0.756 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 10.48% | 10.63% | 0.756 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 7.13% | 9.48% | 0.756 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 6.15% | 6.64% | 0.775 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.91% | 6.01% | 0.776 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.69%** | 5.09% | 0.769 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 7.25% | 10.14% | 0.761 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 9.10% | 9.49% | 0.765 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.84% | 9.09% | 0.761 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.22% | 9.36% | 0.767 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *5.06%* | 8.14% | 0.767 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.40% | - | 0.787 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 11.40% | - | 0.690 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 6.20% | - | *0.816* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 5.80% | - | **0.819** | `VALID` |

---

## Section 2: Macro-Average Summary Table (Across All 14 Benchmarks)

Macro-averaged evaluation metrics across all 14 datasets for **M3-LLaVA**.

| Calibration Method | Category | Regime / Sampling | Macro ECE (%) $\downarrow$ | Macro Adaptive ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 40.33% | 40.27% | 0.671 |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 21.93% | 22.50% | 0.671 |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *3.24%* | 5.41% | 0.682 |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 3.91% | 5.25% | **0.713** |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.59% | 5.62% | 0.695 |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.22%** | 4.81% | 0.678 |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 4.75% | 5.35% | 0.669 |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 22.06% | 23.03% | 0.687 |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 4.37% | 5.70% | *0.703* |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.11% | 6.00% | 0.694 |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.23% | 5.80% | 0.693 |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.24% | - | 0.675 |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.25% | - | 0.641 |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 24.74% | - | 0.694 |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 17.34% | - | 0.698 |

---

## Section 3: Win-Count & Comparative Analysis

### A. Win-Count Summary
- **Overall Best ECE (#1 across ALL evaluated methods)**: Our VCPS methods (**VCPS-5D** / **VCPS-17D**) achieve the absolute lowest ECE on **1 out of 14 datasets** (7.1% win rate).
  - **Specific Datasets Won in ECE**: `pope` (VCPS-17D (Our Method): **3.15%**).
- **Win Rate vs. UMPIRE Multi-Pass Baselines in ECE**: VCPS trajectory calibration achieves lower ECE than all four UMPIRE multi-pass baselines on **14 out of 14 datasets**.
- **Overall Best AUROC (#1 across ALL evaluated methods)**: Our VCPS methods achieve the highest selective prediction AUROC on **3 out of 14 datasets**.
  - **Specific Datasets Won in AUROC**: `mmbench` (VCPS-17D (Our Method): **0.695**), `pope` (VCPS-17D (Our Method): **0.761**), `vizwiz-vqa` (VCPS-5D (Our Method): **0.836**).

### B. Detailed Win Breakdown Table
| Dataset | Lowest ECE Method | Best ECE (%) | Highest AUROC Method | Best AUROC | VCPS Win Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `ai2d` | Quadratic Platt (Logit-Only) | **3.14%** | Trajectory LR (No Bias) | **0.645** | `Competitive` |
| `chartqa` | Trajectory LR | **2.30%** | Trajectory LR (No Bias) | **0.842** | `Competitive` |
| `docvqa` | Trajectory LR | **1.53%** | `umpire` | **0.834** | `Competitive` |
| `gqa` | Platt Scaling (1D) | **4.08%** | `umpire` | **0.712** | `Competitive` |
| `infographicvqa` | Trajectory LR | **0.02%** | Quadratic Platt (Logit-Only) | **0.696** | `Competitive` |
| `lego-puzzles` | Platt Scaling (1D) | **0.71%** | Quadratic Platt (Logit-Only) | **0.668** | `Competitive` |
| `mmbench` | Quadratic Platt (Logit-Only) | **1.61%** | VCPS-17D (Our Method) | **0.695** | `Top AUROC` |
| `mmmu` | Spline Calibration | **1.45%** | Naive Confidence (NC) | **0.639** | `Competitive` |
| `pope` | VCPS-17D (Our Method) | **3.15%** | VCPS-17D (Our Method) | **0.761** | `Top ECE, Top AUROC` |
| `scienceqa` | Trajectory LR | **3.09%** | Trajectory LR | **0.707** | `Competitive` |
| `seedbench` | Spline Calibration | **0.72%** | `eigen_score` | **0.655** | `Competitive` |
| `textvqa` | Trajectory LR (No Bias) | **3.54%** | Trajectory LR (No Bias) | **0.841** | `Competitive` |
| `vizwiz-vqa` | Trajectory LR (No Bias) | **4.93%** | VCPS-5D (Our Method) | **0.836** | `Top AUROC` |
| `vqav2` | Quadratic Platt (Logit-Only) | **3.69%** | `umpire` | **0.819** | `Competitive` |
