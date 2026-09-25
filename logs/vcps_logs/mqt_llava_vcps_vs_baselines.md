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
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 27.15% | 26.92% | *0.746* | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 13.31% | 13.10% | *0.746* | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.92% | 8.25% | *0.746* | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 6.11% | 6.94% | 0.739 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 8.60% | 7.98% | 0.731 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 6.27% | 6.53% | 0.739 | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | **4.61%** | 6.21% | 0.737 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 5.56% | 8.38% | *0.746* | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 5.18% | 8.38% | *0.746* | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 13.38% | 14.08% | **0.747** | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | *4.87%* | 7.05% | 0.745 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.80% | 6.99% | 0.746 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.84% | 7.50% | 0.736 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 32.40% | - | 0.611 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 35.70% | - | 0.594 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 31.50% | - | 0.639 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 31.80% | - | 0.612 | `VALID` |

### Benchmark: `chartqa`
**Dataset**: `chartqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 71.30% | 71.30% | 0.491 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 44.88% | 44.88% | 0.491 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.06% | 4.27% | 0.509 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 2.05% | 4.43% | 0.603 | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 1.80% | 3.16% | 0.639 | `COLLAPSED` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 2.44% | 3.92% | 0.603 | `COLLAPSED` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 2.70% | 3.63% | 0.606 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 1.28% | 4.18% | 0.536 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.05%** | 3.06% | 0.500 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 44.87% | 44.87% | 0.491 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 2.60% | 4.61% | 0.531 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *0.45%* | 3.61% | 0.559 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.63% | 5.93% | 0.601 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.80% | - | 0.710 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 37.70% | - | 0.830 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 24.90% | - | *0.855* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.00% | - | **0.861** | `VALID` |

### Benchmark: `docvqa`
**Dataset**: `docvqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 53.30% | 53.30% | 0.517 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 47.07% | 47.07% | 0.517 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 0.09% | 3.67% | 0.517 | `COLLAPSED` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 1.12% | 2.06% | 0.594 | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 1.22% | 3.43% | 0.552 | `COLLAPSED` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 1.08% | 2.06% | 0.594 | `COLLAPSED` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 1.07% | 2.16% | 0.735 | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 0.93% | 2.65% | 0.595 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.06%** | 2.68% | 0.542 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 47.06% | 47.06% | 0.517 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.09% | 2.75% | 0.528 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | *0.08%* | 2.73% | 0.529 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.33% | 3.08% | 0.608 | `COLLAPSED` |
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
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 13.10% | 13.26% | **0.763** | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 13.09% | 13.09% | 0.763 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 13.81% | 14.66% | *0.763* | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 8.70% | 9.18% | 0.757 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 12.09% | 10.02% | 0.757 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *8.33%* | 10.31% | 0.757 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | **7.58%** | 11.99% | 0.759 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 9.41% | 12.58% | 0.759 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 9.27% | 11.88% | 0.759 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.55% | 10.20% | 0.758 | `VALID` |
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
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 0.30% | 1.06% | 0.915 | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 0.43% | 1.31% | 0.573 | `COLLAPSED` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 0.30% | 1.08% | *0.930* | `COLLAPSED` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 0.31% | 0.97% | **0.945** | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *0.28%* | 1.09% | 0.769 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.27%** | 1.08% | 0.864 | `COLLAPSED` |
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
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.81% | 8.67% | 0.593 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 8.52% | 9.84% | 0.594 | `SCRAMBLED` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 7.81% | 8.67% | 0.593 | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 8.14% | 10.85% | 0.571 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 5.99% | 7.21% | 0.582 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **0.15%** | 8.21% | 0.542 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 23.68% | 23.68% | 0.511 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 1.32% | 10.13% | 0.544 | `SCRAMBLED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.39% | 9.48% | **0.613** | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.23% | 7.40% | *0.611* | `SCRAMBLED` |
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
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 9.37% | 7.48% | 0.818 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 9.77% | 7.19% | 0.824 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 10.42% | 7.51% | 0.819 | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 5.40% | 5.82% | **0.848** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **3.87%** | 7.38% | 0.828 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *5.37%* | 7.75% | 0.828 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 9.05% | 10.14% | 0.829 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 7.52% | 7.74% | 0.829 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.78% | 7.63% | 0.829 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.91% | 6.68% | *0.838* | `VALID` |
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
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 0.23% | 2.04% | 0.721 | `COLLAPSED` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 0.16% | 1.96% | **0.803** | `COLLAPSED` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 0.23% | 2.04% | 0.724 | `COLLAPSED` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 0.36% | 2.08% | *0.800* | `COLLAPSED` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | **0.12%** | 2.15% | 0.432 | `COLLAPSED` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *0.12%* | 2.13% | 0.278 | `COLLAPSED` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 49.31% | 49.31% | 0.315 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 0.15% | 2.15% | 0.365 | `COLLAPSED` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.15% | 2.15% | 0.365 | `COLLAPSED` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 0.15% | 2.14% | 0.394 | `COLLAPSED` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.10% | - | 0.584 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 44.40% | - | 0.588 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 45.50% | - | 0.627 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 37.70% | - | 0.598 | `VALID` |

### Benchmark: `pope`
**Dataset**: `pope` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 4.01% | 4.30% | **0.876** | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | *3.52%* | 4.93% | **0.876** | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 3.97% | 4.64% | **0.876** | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 3.63% | 4.46% | 0.873 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 3.74% | 4.97% | 0.873 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 3.55% | 4.48% | 0.873 | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 5.55% | 3.78% | 0.873 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.04% | 4.69% | **0.876** | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | 5.51% | 6.50% | **0.876** | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 3.75% | 4.90% | *0.874* | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 4.01% | 4.65% | 0.874 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 3.84% | 4.49% | 0.874 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | **3.48%** | 4.22% | 0.874 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.90% | - | 0.683 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 16.70% | - | 0.368 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.10% | - | 0.616 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.70% | - | 0.693 | `VALID` |

### Benchmark: `scienceqa`
**Dataset**: `scienceqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 23.65% | 23.17% | 0.760 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 9.32% | 10.23% | 0.760 | `VALID` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 7.99% | 6.91% | 0.760 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 6.61% | 6.61% | 0.769 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | *4.76%* | 5.00% | **0.773** | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 6.46% | 5.72% | *0.769* | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 6.34% | 6.65% | 0.759 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 4.99% | 5.79% | 0.756 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **4.21%** | 5.71% | 0.760 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 8.73% | 9.38% | 0.761 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 6.38% | 6.55% | 0.762 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.87% | 5.39% | 0.763 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.29% | 5.32% | 0.766 | `VALID` |
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
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.80% | 8.13% | 0.740 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.16% | 9.49% | 0.740 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 6.66% | 9.67% | *0.741* | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | *5.32%* | 8.42% | **0.745** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.38% | 8.50% | 0.733 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.37%** | 8.82% | 0.733 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 6.04% | 9.65% | 0.736 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 7.05% | 6.77% | 0.735 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.66% | 8.69% | 0.737 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.63% | 10.03% | 0.740 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.10% | - | 0.610 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 30.00% | - | 0.599 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.00% | - | 0.639 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 27.70% | - | 0.605 | `VALID` |

### Benchmark: `textvqa`
**Dataset**: `textvqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 32.27% | 31.91% | 0.684 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 18.21% | 18.32% | 0.684 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 8.95% | 10.46% | 0.684 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.25% | 7.77% | 0.719 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.40% | 5.87% | 0.718 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | *5.25%* | 7.76% | 0.719 | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 11.12% | 8.41% | 0.755 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 7.78% | 8.92% | 0.679 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.19%** | 4.21% | 0.684 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 18.36% | 18.36% | 0.689 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 8.61% | 11.21% | 0.699 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 9.04% | 10.63% | 0.707 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.62% | 6.42% | 0.737 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 14.80% | - | 0.793 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 31.50% | - | 0.789 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 28.30% | - | *0.802* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.10% | - | **0.818** | `VALID` |

### Benchmark: `vizwiz-vqa`
**Dataset**: `vizwiz-vqa` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 30.66% | 30.66% | 0.797 | `COLLAPSED` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 29.41% | 29.41% | 0.797 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 6.27% | 7.62% | 0.797 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 6.07% | 8.17% | 0.795 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.32% | 7.10% | 0.791 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 6.03% | 8.17% | 0.795 | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | **4.28%** | 4.58% | **0.801** | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | 5.26% | 6.91% | 0.796 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | *4.55%* | 7.66% | 0.797 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 29.11% | 29.11% | 0.796 | `COLLAPSED` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 5.59% | 7.85% | 0.797 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 6.12% | 7.59% | 0.797 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 5.82% | 6.64% | *0.799* | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 18.50% | - | 0.718 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.70% | - | 0.689 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 39.20% | - | 0.744 | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 19.30% | - | 0.754 | `VALID` |

### Benchmark: `vqav2`
**Dataset**: `vqav2` | **Model**: MQT-LLaVA
| Calibration Method | Category | Regime / Sampling | ECE (%) $\downarrow$ | Adaptive ECE (%) $\downarrow$ | AUROC $\uparrow$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 10.49% | 10.31% | 0.641 | `VALID` |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 13.87% | 13.65% | 0.641 | `COLLAPSED` |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 10.54% | 10.94% | 0.641 | `VALID` |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 7.17% | 7.64% | 0.739 | `VALID` |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 6.78% | 7.71% | 0.737 | `VALID` |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 7.16% | 7.64% | 0.739 | `VALID` |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 6.13% | 8.03% | 0.750 | `VALID` |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *3.83%* | 4.60% | 0.723 | `VALID` |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.23%** | 5.65% | 0.681 | `VALID` |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 12.36% | 12.03% | 0.658 | `VALID` |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 10.04% | 9.24% | 0.647 | `VALID` |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 8.58% | 8.83% | 0.657 | `VALID` |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 7.95% | 8.01% | 0.674 | `VALID` |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 12.30% | - | 0.767 | `VALID` |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 13.70% | - | 0.699 | `VALID` |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 10.20% | - | *0.773* | `VALID` |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 7.80% | - | **0.790** | `VALID` |

---

## Section 2: Macro-Average Summary Table (15 Methods Across All 14 Datasets)

Macro-averaged evaluation metrics of 15 benchmark methods across all 14 datasets for **MQT-LLaVA**.

| Calibration Method | Category | Regime / Sampling | Macro ECE (%) $\downarrow$ | Macro Adaptive ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0$) | 33.05% | 32.62% | 0.673 |
| Temperature Scaling (TS) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 23.64% | 24.18% | 0.673 |
| Platt Scaling (1D) | Classic Post-Hoc Calibrator | Single-Pass ($T=0.0$) | 5.36% | 7.15% | 0.673 |
| Trajectory LR | Linear Trajectory Baseline | Single-Pass ($T=0.0$) | 5.33% | 6.34% | 0.741 |
| Trajectory LR (No Bias) | Linear Trajectory Baseline (Zero-Bias) | Single-Pass ($T=0.0$) | 5.48% | 6.29% | 0.722 |
| Trajectory Platt (5D) | Trajectory Platt Baseline (5D) | Single-Pass ($T=0.0$) | 5.53% | 6.42% | *0.743* |
| Trajectory Platt (17D) | Trajectory Platt Baseline (17D) | Single-Pass ($T=0.0$) | 5.00% | 5.77% | **0.763** |
| Quadratic Platt (Logit-Only) | Logit-Only Polynomial Baseline | Single-Pass ($T=0.0$) | *4.53%* | 5.89% | 0.701 |
| Spline Calibration | Non-Parametric Calibrator | Single-Pass ($T=0.0$) | **3.11%** | 5.87% | 0.685 |
| Adaptive TS (ATS) | Adaptive Calibrator | Single-Pass ($T=0.0$) | 23.28% | 24.08% | 0.675 |
| Residual Calibrator | Feature-Aided Calibrator | Single-Pass ($T=0.0$) | 4.85% | 6.74% | 0.684 |
| VCPS-5D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.95% | 6.51% | 0.693 |
| VCPS-17D (Our Method) | Proposed Trajectory Calibration | Single-Pass ($T=0.0$) | 4.77% | 6.05% | 0.707 |
| `ln_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 22.77% | - | 0.666 |
| `semantic_entropy` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 33.37% | - | 0.639 |
| `eigen_score` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 29.57% | - | 0.689 |
| `umpire` | UMPIRE Multi-Pass Baseline | Multi-Pass ($T=0.5, K=10$) | 21.27% | - | 0.691 |

---

## Section 3: Win-Count & Comparative Analysis

### A. Win-Count Summary
- **Overall Best ECE (#1 across ALL evaluated methods)**: Our VCPS methods (**VCPS-5D** / **VCPS-17D**) achieve the absolute lowest ECE on **1 out of 14 datasets** (7.1% win rate).
  - **Specific Datasets Won in ECE**: `pope` (VCPS-17D (Our Method): **3.48%**).
- **Win Rate vs. UMPIRE Multi-Pass Baselines in ECE**: VCPS trajectory calibration achieves lower ECE than all four UMPIRE multi-pass baselines on **13 out of 14 datasets**.
- **Overall Best AUROC (#1 across ALL evaluated methods)**: Our VCPS methods achieve the highest selective prediction AUROC on **1 out of 14 datasets**.
  - **Specific Datasets Won in AUROC**: `lego-puzzles` (VCPS-5D (Our Method): **0.613**).

### B. Detailed Win Breakdown Table
| Dataset | Lowest ECE Method | Best ECE (%) | Highest AUROC Method | Best AUROC | VCPS Win Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `ai2d` | Trajectory Platt (17D) | **4.61%** | Adaptive TS (ATS) | **0.747** | `Competitive` |
| `chartqa` | Spline Calibration | **0.05%** | `umpire` | **0.861** | `Competitive` |
| `docvqa` | Spline Calibration | **0.06%** | `umpire` | **0.853** | `Competitive` |
| `gqa` | Adaptive TS (ATS) | **7.58%** | Trajectory LR | **0.763** | `Competitive` |
| `infographicvqa` | Spline Calibration | **0.27%** | Trajectory Platt (17D) | **0.945** | `Competitive` |
| `lego-puzzles` | Spline Calibration | **0.15%** | VCPS-5D (Our Method) | **0.613** | `Top AUROC` |
| `mmbench` | Quadratic Platt (Logit-Only) | **3.87%** | Trajectory Platt (17D) | **0.848** | `Competitive` |
| `mmmu` | Quadratic Platt (Logit-Only) | **0.12%** | Trajectory LR (No Bias) | **0.803** | `Competitive` |
| `pope` | VCPS-17D (Our Method) | **3.48%** | Naive Confidence (NC) | **0.876** | `Top ECE` |
| `scienceqa` | Spline Calibration | **4.21%** | Trajectory LR (No Bias) | **0.773** | `Competitive` |
| `seedbench` | Spline Calibration | **3.37%** | Trajectory Platt (17D) | **0.745** | `Competitive` |
| `textvqa` | Spline Calibration | **3.19%** | `umpire` | **0.818** | `Competitive` |
| `vizwiz-vqa` | Trajectory Platt (17D) | **4.28%** | Trajectory Platt (17D) | **0.801** | `Competitive` |
| `vqav2` | Spline Calibration | **3.23%** | `umpire` | **0.790** | `Competitive` |
