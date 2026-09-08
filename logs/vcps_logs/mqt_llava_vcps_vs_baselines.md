# VCPS Trajectory Calibration vs. Baselines: MQT-LLaVA

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
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 24.34% | 24.32% | 0.705 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 11.04% | 13.01% | 0.705 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 7.89% | 9.51% | 0.705 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 13.31% | 12.45% | 0.704 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 13.22% | 12.99% | 0.680 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **6.24%** | 9.06% | 0.705 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *6.38%* | 8.64% | 0.705 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 11.09% | 13.92% | 0.706 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 8.95% | 11.34% | **0.707** | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.78% | 12.43% | *0.706* | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.79% | 12.30% | *0.706* | `VALID` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 2.83% | 8.03% | 0.533 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 1.60% | 8.98% | 0.716 | `SCRAMBLED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 9.14% | 11.82% | 0.348 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 2.86% | 8.04% | 0.533 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.26%** | 4.71% | 0.554 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 41.53% | 41.53% | 0.467 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 2.76% | 7.90% | 0.578 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.84% | 6.98% | 0.526 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *0.42%* | 7.05% | 0.619 | `COLLAPSED` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 0.47% | 4.56% | 0.644 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 2.37% | 3.09% | 0.786 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.92% | 9.33% | 0.407 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 1.49% | 4.86% | 0.699 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.05%** | 4.22% | 0.623 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 46.62% | 46.62% | 0.644 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.54% | 5.06% | 0.666 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.37% | 5.04% | 0.665 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *0.23%* | 3.05% | 0.677 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.70% | - | 0.805 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.30% | - | 0.816 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 33.50% | - | *0.839* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.00% | - | **0.853** | `VALID` |

### Benchmark: `gqa`
**Dataset**: `gqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 13.64% | 10.19% | 0.757 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.21% | 10.61% | 0.757 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.56% | 10.60% | 0.757 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 10.94% | 10.13% | **0.761** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 9.54% | 9.92% | 0.756 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 12.09% | 10.02% | 0.757 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **8.33%** | 10.31% | 0.757 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | *8.39%* | 10.51% | 0.757 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 9.63% | 10.49% | 0.757 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.77% | 10.42% | 0.757 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.55% | 10.20% | *0.758* | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.80% | - | 0.698 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.00% | - | 0.624 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.70% | - | 0.696 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.70% | - | 0.715 | `VALID` |

### Benchmark: `infographicvqa`
**Dataset**: `infographicvqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 75.51% | 75.51% | 0.769 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 52.63% | 52.63% | 0.769 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 0.28% | 1.11% | 0.769 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 0.31% | 0.95% | **0.950** | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 0.62% | 1.50% | 0.613 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *0.28%* | 1.09% | 0.769 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.27%** | 1.08% | *0.864* | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 52.61% | 52.61% | 0.769 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.29% | 1.11% | 0.769 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.29% | 1.10% | 0.769 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.29% | 1.10% | 0.769 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 34.10% | - | 0.615 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 47.50% | - | 0.648 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 47.80% | - | 0.648 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 26.10% | - | 0.646 | `VALID` |

### Benchmark: `lego-puzzles`
**Dataset**: `lego-puzzles` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 34.60% | 33.80% | 0.510 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 23.68% | 23.68% | 0.510 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *0.60%* | 12.43% | 0.490 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 9.25% | 9.41% | 0.587 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 8.99% | 12.96% | **0.629** | `SCRAMBLED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 5.99% | 7.21% | 0.582 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.15%** | 8.21% | 0.542 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 23.68% | 23.68% | 0.511 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 3.85% | 7.93% | 0.554 | `SCRAMBLED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.31% | 11.99% | *0.627* | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.23% | 7.40% | 0.611 | `SCRAMBLED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 25.30% | - | 0.531 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 45.20% | - | 0.533 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.40% | - | 0.502 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.60% | - | 0.537 | `VALID` |

### Benchmark: `mmbench`
**Dataset**: `mmbench` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 15.38% | 15.27% | 0.828 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.81% | 10.24% | 0.828 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.11% | 8.38% | 0.828 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 6.33% | 4.25% | *0.844* | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.05% | 5.67% | **0.845** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.87%** | 7.38% | 0.828 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *5.37%* | 7.75% | 0.828 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 9.98% | 10.13% | 0.831 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 8.07% | 5.96% | 0.830 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.79% | 6.67% | 0.831 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.91% | 6.68% | 0.838 | `VALID` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 0.16% | 2.14% | 0.315 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 0.23% | 2.00% | **0.803** | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 2.12% | 4.00% | 0.497 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **0.12%** | 2.15% | 0.432 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *0.12%* | 2.13% | 0.278 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 49.31% | 49.31% | 0.315 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.16% | 2.15% | 0.350 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.15% | 2.15% | 0.350 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.15% | 2.14% | 0.394 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.10% | - | 0.584 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 44.40% | - | 0.588 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 45.50% | - | *0.627* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 37.70% | - | 0.598 | `VALID` |

### Benchmark: `pope`
**Dataset**: `pope` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 7.04% | 6.18% | 0.818 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.86% | 5.44% | 0.818 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *4.37%* | 4.79% | 0.818 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | **4.31%** | 6.16% | 0.817 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.74% | 6.58% | **0.820** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.54% | 4.91% | 0.818 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 5.40% | 6.42% | 0.818 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 5.80% | 6.62% | 0.816 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 4.46% | 6.01% | 0.818 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.44% | 7.14% | 0.817 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.44% | 5.93% | *0.818* | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.90% | - | 0.683 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.70% | - | 0.368 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.10% | - | 0.616 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.70% | - | 0.693 | `VALID` |

### Benchmark: `scienceqa`
**Dataset**: `scienceqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 23.29% | 23.04% | 0.775 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 10.47% | 10.84% | 0.775 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 7.75% | 6.78% | 0.775 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 4.76% | 5.34% | *0.782* | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.48% | 8.60% | 0.776 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 5.00% | 6.37% | 0.771 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **2.84%** | 7.04% | 0.773 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 10.10% | 10.49% | 0.777 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.47% | 5.88% | 0.779 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.16% | 5.53% | 0.779 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *4.47%* | 5.37% | **0.782** | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.80% | - | 0.581 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.50% | - | 0.568 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.00% | - | 0.588 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.20% | - | 0.581 | `VALID` |

### Benchmark: `seedbench`
**Dataset**: `seedbench` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 18.56% | 17.74% | 0.733 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.68% | 10.44% | 0.733 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.51% | 8.62% | 0.733 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.25% | 9.60% | **0.740** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | *5.92%* | 10.35% | 0.732 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.38% | 8.50% | 0.733 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.37%** | 8.82% | 0.733 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 6.26% | 9.95% | 0.736 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 7.89% | 8.47% | 0.738 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.80% | 9.43% | 0.740 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.63% | 10.03% | *0.740* | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.10% | - | 0.610 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.00% | - | 0.599 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.00% | - | 0.639 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.70% | - | 0.605 | `VALID` |

### Benchmark: `textvqa`
**Dataset**: `textvqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 31.24% | 30.96% | 0.681 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 17.62% | 18.67% | 0.681 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.78% | 11.82% | 0.681 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.46% | 7.00% | 0.759 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 9.00% | 7.11% | 0.754 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.55% | 8.92% | 0.681 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.29%** | 5.43% | 0.682 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 18.55% | 18.34% | 0.694 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | *5.27%* | 7.44% | 0.714 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.13% | 7.23% | 0.729 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.28% | 7.36% | 0.737 | `VALID` |
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
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.70% | 5.76% | 0.802 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.25% | 5.07% | 0.798 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 9.88% | 10.66% | 0.759 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *4.92%* | 4.47% | 0.803 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **2.00%** | 4.29% | 0.802 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 28.84% | 28.45% | 0.801 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.96% | 5.53% | 0.803 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.07% | 5.62% | **0.804** | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.35% | 5.63% | *0.803* | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.50% | - | 0.718 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.70% | - | 0.689 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.20% | - | 0.744 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.30% | - | 0.754 | `VALID` |

### Benchmark: `vqav2`
**Dataset**: `vqav2` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 12.14% | 10.93% | 0.746 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 15.10% | 16.04% | 0.746 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 14.26% | 14.95% | 0.746 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 8.70% | 9.75% | **0.802** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | *6.80%* | 11.39% | *0.797* | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **6.25%** | 8.39% | 0.765 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 7.18% | 10.70% | 0.747 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 12.65% | 13.21% | 0.766 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 9.38% | 12.40% | 0.751 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 10.03% | 10.42% | 0.763 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.79% | 7.74% | 0.773 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.30% | - | 0.767 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.70% | - | 0.699 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.20% | - | 0.773 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 7.80% | - | 0.790 | `VALID` |

---

## Section 2: Macro-Average Summary Table (15 Methods Across All 14 Datasets)

Macro-averaged evaluation metrics of 15 benchmark methods across all 14 datasets for **MQT-LLaVA**.

| Calibration Method | Category | Regime / Sampling | Macro ECE (%) $\downarrow$ | Macro Adaptive ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 32.32% | 31.76% | 0.682 |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 23.45% | 24.13% | 0.682 |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.88% | 7.82% | 0.686 |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.86% | 6.73% | **0.775** |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 7.17% | 8.78% | 0.672 |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.90% | 6.53% | 0.706 |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.21%** | 6.41% | 0.693 |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 23.24% | 23.96% | 0.685 |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.26% | 6.98% | 0.701 |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.28% | 7.30% | 0.705 |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *4.89%* | 6.57% | *0.716* |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.77% | - | 0.666 |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 33.37% | - | 0.639 |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.57% | - | 0.689 |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 21.27% | - | 0.691 |

---

## Section 3: Win-Count & Comparative Analysis

### A. Win-Count Summary
- **Overall Best ECE (#1 across ALL evaluated methods)**: Our VCPS methods (**VCPS-5D** / **VCPS-17D**) achieve the absolute lowest ECE on **0 out of 14 datasets** (0.0% win rate).
  - **Specific Datasets Won in ECE**: .
- **Win Rate vs. UMPIRE Multi-Pass Baselines in ECE**: VCPS trajectory calibration achieves lower ECE than all four UMPIRE multi-pass baselines on **13 out of 14 datasets**.
- **Overall Best AUROC (#1 across ALL evaluated methods)**: Our VCPS methods achieve the highest selective prediction AUROC on **2 out of 14 datasets**.
  - **Specific Datasets Won in AUROC**: `scienceqa` (VCPS-17D (Our Method): **0.782**), `vizwiz-vqa` (VCPS-5D (Our Method): **0.804**).

### B. Detailed Win Breakdown Table
| Dataset | Lowest ECE Method | Best ECE (%) | Highest AUROC Method | Best AUROC | VCPS Win Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `ai2d` | Quadratic Platt (Logit-Only) | **6.24%** | Residual Calibrator | **0.707** | `Competitive` |
| `chartqa` | Spline Calibration | **0.26%** | `umpire` | **0.861** | `Competitive` |
| `docvqa` | Spline Calibration | **0.05%** | `umpire` | **0.853** | `Competitive` |
| `gqa` | Spline Calibration | **8.33%** | Trajectory LR | **0.761** | `Competitive` |
| `infographicvqa` | Spline Calibration | **0.27%** | Trajectory LR | **0.950** | `Competitive` |
| `lego-puzzles` | Spline Calibration | **0.15%** | Trajectory LR (No Bias) | **0.629** | `Competitive` |
| `mmbench` | Quadratic Platt (Logit-Only) | **3.87%** | Trajectory LR (No Bias) | **0.845** | `Competitive` |
| `mmmu` | Quadratic Platt (Logit-Only) | **0.12%** | Trajectory LR | **0.803** | `Competitive` |
| `pope` | Trajectory LR | **4.31%** | Trajectory LR (No Bias) | **0.820** | `Competitive` |
| `scienceqa` | Spline Calibration | **2.84%** | VCPS-17D (Our Method) | **0.782** | `Top AUROC` |
| `seedbench` | Spline Calibration | **3.37%** | Trajectory LR | **0.740** | `Competitive` |
| `textvqa` | Spline Calibration | **3.29%** | `umpire` | **0.818** | `Competitive` |
| `vizwiz-vqa` | Spline Calibration | **2.00%** | VCPS-5D (Our Method) | **0.804** | `Top AUROC` |
| `vqav2` | Quadratic Platt (Logit-Only) | **6.25%** | Trajectory LR | **0.802** | `Competitive` |
