# VCPS Trajectory Calibration vs. Baselines: M3-LLaVA

> **Document Purpose**: Comprehensive unified benchmark comparison comparing standard calibration baselines (TS, Platt, Spline, ATS, Residual Calibrator), proposed trajectory calibration methods (**VCPS-5D**, **VCPS-17D**), and UMPIRE multi-pass uncertainty quantification baselines across all 14 datasets.

## Evaluation Setup & Temperature Protocol Note

> [!IMPORTANT]
> **CRITICAL TEMPERATURE & COMPUTATION PROTOCOL**:
> 1. **Greedy Deterministic Single-Pass ($T = 0.0, K = 1$)**:
>    - Evaluated for: **Naive Confidence (NC)**, **Temperature Scaling (TS)**, **Platt Scaling (1D)**, **Trajectory LR**, **Trajectory LR (No Bias)**, **Trajectory Platt (5D)**, **Trajectory Platt (17D)**, **Quadratic Platt (Logit-Only)**, **Spline Calibration (PCHIP)**, **Adaptive TS (ATS)**, **Residual Calibrator**, **VCPS-5D**, and **VCPS-17D**.
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
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.56% | 9.38% | 0.598 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.49% | 5.19% | 0.598 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.42% | 9.58% | **0.645** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.39% | 8.58% | 0.632 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *3.14%* | 5.10% | 0.598 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.74%** | 7.68% | 0.598 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.54% | 10.25% | 0.605 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.35% | 8.13% | 0.613 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.96% | 9.85% | *0.638* | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.72% | 9.57% | 0.637 | `VALID` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 2.96% | 2.35% | 0.835 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 3.89% | 4.01% | 0.827 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.96% | 4.27% | 0.831 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 3.13% | 2.14% | 0.835 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *2.68%* | 3.80% | 0.828 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 45.43% | 45.05% | 0.835 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 3.32% | 2.27% | 0.836 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.08% | 2.19% | *0.837* | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **2.67%** | 2.22% | **0.838** | `VALID` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *1.80%* | 2.39% | 0.732 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 1.91% | 2.36% | 0.737 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 2.71% | 3.99% | 0.691 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 2.11% | 2.30% | 0.732 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **1.28%** | 1.93% | 0.732 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 48.84% | 48.84% | 0.732 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 1.84% | 2.40% | 0.733 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 2.05% | 2.65% | 0.738 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 2.17% | 3.11% | 0.739 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 15.80% | - | 0.786 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 37.10% | - | 0.809 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.70% | - | *0.830* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.40% | - | **0.834** | `VALID` |

### Benchmark: `gqa`
**Dataset**: `gqa` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 32.54% | 32.43% | 0.685 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.24% | 9.90% | 0.685 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | **4.08%** | 6.70% | 0.685 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 6.68% | 5.79% | 0.690 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.88% | 8.51% | 0.682 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *4.23%* | 6.69% | 0.685 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 6.74% | 6.44% | 0.685 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.86% | 9.41% | 0.701 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.42% | 5.74% | 0.697 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.38% | 5.41% | 0.709 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.02% | 5.25% | *0.710* | `VALID` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 0.04% | 1.13% | 0.575 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **0.01%** | 1.09% | *0.606* | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | *0.03%* | 1.18% | 0.569 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 0.05% | 1.24% | **0.696** | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 0.12% | 1.23% | 0.500 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 53.89% | 53.89% | 0.425 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.04% | 1.13% | 0.577 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.06% | 1.31% | 0.423 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.06% | 1.22% | 0.407 | `COLLAPSED` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | **0.71%** | 11.89% | 0.477 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.19% | 8.81% | 0.634 | `SCRAMBLED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 11.28% | 11.61% | *0.641* | `SCRAMBLED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.34% | 9.74% | **0.668** | `SCRAMBLED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 6.28% | 12.33% | 0.571 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 34.66% | 34.66% | 0.477 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 2.45% | 8.94% | 0.528 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *0.90%* | 10.78% | 0.513 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.65% | 9.19% | 0.546 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 21.20% | - | 0.551 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.00% | - | 0.582 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 15.00% | - | 0.571 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 11.00% | - | 0.588 | `VALID` |

### Benchmark: `mmbench`
**Dataset**: `mmbench` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 28.62% | 28.58% | 0.629 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.68% | 7.88% | 0.629 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.07% | 5.50% | 0.629 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.82% | 6.37% | *0.692* | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.77% | 5.60% | 0.680 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *1.61%* | 4.69% | 0.634 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **1.46%** | 4.81% | 0.500 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.78% | 9.89% | 0.689 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.53% | 7.56% | 0.679 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.92% | 4.54% | 0.691 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.49% | 4.99% | **0.693** | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 20.50% | - | 0.660 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.00% | - | 0.646 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.20% | - | 0.681 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.30% | - | 0.658 | `VALID` |

### Benchmark: `mmmu`
**Dataset**: `mmmu` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 78.76% | 78.76% | 0.639 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 52.21% | 52.21% | 0.639 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *1.81%* | 3.44% | 0.639 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 2.06% | 3.64% | **0.645** | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 2.85% | 5.05% | 0.550 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 2.26% | 4.09% | 0.274 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.24%** | 3.03% | 0.573 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 52.19% | 52.19% | 0.639 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 1.85% | 3.46% | 0.639 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 1.89% | 3.48% | 0.639 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 1.98% | 3.47% | *0.643* | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 36.40% | - | 0.566 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 47.60% | - | 0.568 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 50.10% | - | 0.602 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.70% | - | 0.592 | `VALID` |

### Benchmark: `pope`
**Dataset**: `pope` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 4.03% | 4.30% | 0.752 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.67% | 4.23% | 0.752 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.69% | 4.33% | 0.752 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.38% | 5.64% | 0.757 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.38% | 5.65% | 0.757 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 3.64% | 4.35% | 0.752 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *3.55%* | 3.82% | 0.752 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 3.79% | 3.87% | *0.761* | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 3.69% | 3.81% | 0.752 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.65% | 4.03% | 0.760 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **3.38%** | 3.86% | **0.765** | `VALID` |
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
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **2.60%** | 6.43% | 0.703 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 3.35% | 6.33% | 0.698 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 3.18% | 4.13% | 0.680 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 3.36% | 7.51% | 0.701 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 7.70% | 8.47% | **0.705** | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 4.06% | 6.05% | 0.702 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.44% | 5.88% | 0.703 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *2.60%* | 6.62% | *0.704* | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 26.80% | - | 0.659 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.00% | - | 0.633 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.10% | - | 0.636 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.10% | - | 0.662 | `VALID` |

### Benchmark: `seedbench`
**Dataset**: `seedbench` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 32.40% | 32.40% | 0.500 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 1.31% | 6.16% | 0.500 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *1.17%* | 5.98% | 0.500 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.06% | 5.74% | 0.643 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 3.69% | 6.32% | *0.645* | `SCRAMBLED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 1.63% | 6.89% | 0.500 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 5.55% | 7.60% | 0.500 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | **1.09%** | 10.36% | 0.581 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 8.58% | 10.21% | 0.641 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.92% | 7.58% | 0.639 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.66% | 7.82% | 0.639 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 23.30% | - | 0.644 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.00% | - | 0.629 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.70% | - | **0.655** | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.40% | - | 0.644 | `VALID` |

### Benchmark: `textvqa`
**Dataset**: `textvqa` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 8.32% | 9.07% | 0.832 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.70% | 8.92% | 0.832 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.01% | 6.11% | 0.832 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.12% | 5.00% | *0.841* | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.03% | 4.89% | **0.842** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.77% | 4.74% | 0.832 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.70%** | 4.86% | 0.832 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.27% | 8.50% | 0.832 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | *4.71%* | 5.34% | 0.834 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.01% | 5.00% | 0.834 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.78% | 4.02% | 0.835 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 11.80% | - | 0.811 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.20% | - | 0.774 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 26.60% | - | 0.814 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.40% | - | 0.829 | `VALID` |

### Benchmark: `vizwiz-vqa`
**Dataset**: `vizwiz-vqa` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 16.71% | 16.71% | 0.835 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 15.56% | 16.03% | 0.835 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.76% | 7.69% | 0.835 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **4.34%** | 4.84% | 0.833 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | *5.55%* | 4.50% | 0.826 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.26% | 7.81% | 0.835 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 7.50% | 8.37% | 0.835 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 15.38% | 15.46% | 0.835 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.46% | 7.43% | 0.836 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.55% | 7.40% | *0.836* | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.76% | 6.68% | **0.837** | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.80% | - | 0.723 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.40% | - | 0.701 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.30% | - | 0.757 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.70% | - | 0.768 | `VALID` |

### Benchmark: `vqav2`
**Dataset**: `vqav2` | **Model**: M3-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 8.51% | 8.06% | 0.756 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 10.48% | 10.51% | 0.756 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 7.13% | 9.71% | 0.756 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.29% | 6.13% | 0.793 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.46% | 6.59% | 0.795 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.69%** | 5.45% | 0.769 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 5.11% | 8.21% | 0.762 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.63% | 9.28% | 0.767 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.15% | 9.16% | 0.761 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.31% | 8.10% | 0.768 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *4.56%* | 4.82% | 0.783 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.40% | - | 0.787 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 11.40% | - | 0.690 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 6.20% | - | *0.816* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 5.80% | - | **0.819** | `VALID` |

---

## Section 2: Macro-Average Summary Table (15 Methods Across All 14 Datasets)

Macro-averaged evaluation metrics of 15 benchmark methods across all 14 datasets for **M3-LLaVA**.

| Calibration Method | Category | Regime / Sampling | Macro ECE (%) $\downarrow$ | Macro Adaptive ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 40.33% | 40.33% | 0.671 |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 21.93% | 22.68% | 0.671 |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *3.24%* | 5.70% | 0.682 |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.41% | 5.39% | **0.718** |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 4.59% | 5.93% | *0.703* |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.22%** | 4.95% | 0.678 |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 3.45% | 5.83% | 0.669 |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 21.86% | 22.87% | 0.685 |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 4.32% | 5.83% | 0.702 |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.94% | 5.59% | 0.695 |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.11% | 5.20% | 0.698 |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.24% | - | 0.675 |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.25% | - | 0.641 |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 24.74% | - | 0.694 |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 17.34% | - | 0.698 |

---

## Section 3: Win-Count & Comparative Analysis

### A. Win-Count Summary
- **Overall Best ECE (#1 across ALL evaluated methods)**: Our VCPS methods (**VCPS-5D** / **VCPS-17D**) achieve the absolute lowest ECE on **2 out of 14 datasets** (14.3% win rate).
  - **Specific Datasets Won in ECE**: `chartqa` (VCPS-17D (Our Method): **2.67%**), `pope` (VCPS-17D (Our Method): **3.38%**).
- **Win Rate vs. UMPIRE Multi-Pass Baselines in ECE**: VCPS trajectory calibration achieves lower ECE than all four UMPIRE multi-pass baselines on **14 out of 14 datasets**.
- **Overall Best AUROC (#1 across ALL evaluated methods)**: Our VCPS methods achieve the highest selective prediction AUROC on **4 out of 14 datasets**.
  - **Specific Datasets Won in AUROC**: `chartqa` (VCPS-17D (Our Method): **0.838**), `mmbench` (VCPS-17D (Our Method): **0.693**), `pope` (VCPS-17D (Our Method): **0.765**), `vizwiz-vqa` (VCPS-17D (Our Method): **0.837**).

### B. Detailed Win Breakdown Table
| Dataset | Lowest ECE Method | Best ECE (%) | Highest AUROC Method | Best AUROC | VCPS Win Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `ai2d` | Spline Calibration | **0.74%** | Trajectory LR | **0.645** | `Competitive` |
| `chartqa` | VCPS-17D (Our Method) | **2.67%** | VCPS-17D (Our Method) | **0.838** | `Top ECE, Top AUROC` |
| `docvqa` | Spline Calibration | **1.28%** | `umpire` | **0.834** | `Competitive` |
| `gqa` | Platt Scaling (1D) | **4.08%** | `umpire` | **0.712** | `Competitive` |
| `infographicvqa` | Trajectory LR | **0.01%** | Quadratic Platt (Logit-Only) | **0.696** | `Competitive` |
| `lego-puzzles` | Platt Scaling (1D) | **0.71%** | Quadratic Platt (Logit-Only) | **0.668** | `Competitive` |
| `mmbench` | Spline Calibration | **1.46%** | VCPS-17D (Our Method) | **0.693** | `Top AUROC` |
| `mmmu` | Spline Calibration | **0.24%** | Trajectory LR | **0.645** | `Competitive` |
| `pope` | VCPS-17D (Our Method) | **3.38%** | VCPS-17D (Our Method) | **0.765** | `Top ECE, Top AUROC` |
| `scienceqa` | Trajectory LR | **2.60%** | Adaptive TS (ATS) | **0.705** | `Competitive` |
| `seedbench` | Adaptive TS (ATS) | **1.09%** | `eigen_score` | **0.655** | `Competitive` |
| `textvqa` | Spline Calibration | **3.70%** | Trajectory LR (No Bias) | **0.842** | `Competitive` |
| `vizwiz-vqa` | Trajectory LR | **4.34%** | VCPS-17D (Our Method) | **0.837** | `Top AUROC` |
| `vqav2` | Quadratic Platt (Logit-Only) | **3.69%** | `umpire` | **0.819** | `Competitive` |
