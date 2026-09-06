# Trajectory Calibration: Elastic Multi-Scale Uncertainty Quantification for Multimodal LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 17 Passed](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)]()
[![Code Quality: Modularity <= 200 LOC](https://img.shields.io/badge/modularity-%E2%89%A4200%20LOC-success.svg)]()
[![Platform: Windows | Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

A clean, modular implementation of **Elastic Trajectory Uncertainty Calibration** and **Varying-Coefficient Platt Scaling (VCPS)** for Matryoshka Multimodal Models (**M3-LLaVA** and **MQT-LLaVA**).

---

## Abstract and Motivation

Multimodal Large Language Models (MLLMs) frequently suffer from overconfidence and severe miscalibration. Standard uncertainty quantification (UQ) techniques—such as multi-rollout sampling or Monte Carlo dropout—require **5x to 25x repetitive inference calls**, creating unacceptable computational overhead for real-time vision-language systems.

This repository implements **Trajectory Uncertainty Calibration**:
1. **Single-Pass Elastic Signatures**: By leveraging Matryoshka visual token compression ($m \in \{1, 9, 36, 144, 576\}$ for M3 or $\{1, 9, 36, 144, 256\}$ for MQT), we capture the model's confidence trajectory across visual granularities in a **single forward pass**.
2. **Varying-Coefficient Platt Scaling (VCPS)**: Dynamically modulates both the calibration slope $a(\mathbf{z})$ and intercept $b(\mathbf{z})$ as generalized linear functions of multi-scale trajectory signatures:

$$\text{logit}(p(\mathbf{x})) = a(\mathbf{z}) \cdot x_1 + b(\mathbf{z})$$

$$a(\mathbf{z}) = \exp(a_0 + \boldsymbol{\gamma}^T \mathbf{z}_{\text{slope}}), \quad b(\mathbf{z}) = b_0 + \mathbf{w}^T \mathbf{z}_{\text{intercept}}$$

- **VCPS-17D (Full Signature Binding)**: Dynamically binds all 17 non-anchor trajectory signatures ($x_3, x_4, \dots, x_{22}$) for both dynamic slope $\boldsymbol{\gamma}$ and dynamic intercept $\mathbf{w}$, optimizing **36 parameters** ($1 + 17 + 1 + 17 = 2K + 2$): scalar log-slope base $a_0$, 17 slope weights $\boldsymbol{\gamma}$, scalar intercept base $b_0$, and 17 intercept weights $\mathbf{w}$ across an 18-D input ($x_1 + \mathbf{z}_{17}$).
- **VCPS-5D (Stepwise Subset)**: Binds 5 forward-stepwise selected trajectory signatures for both dynamic slope $\boldsymbol{\gamma}$ and dynamic intercept $\mathbf{w}$, optimizing **12 parameters** ($1 + 5 + 1 + 5 = 2K + 2$): scalar log-slope base $a_0$, 5 slope weights $\boldsymbol{\gamma}$, scalar intercept base $b_0$, and 5 intercept weights $\mathbf{w}$ across a 6-D input ($x_1 + \mathbf{z}_5$).

3. **Pareto Dominance**: Outperforms standard post-hoc temperature scaling while maintaining **1x inference cost**.

---

## Key Features

- **Strict Modularity**: Every source file in `src/trajectory_calibration/` is structured as a clean, single-responsibility module.
- **Primary Calibration Method**: Varying-Coefficient Platt Scaling (**VCPS-17D** with full 36-parameter dynamic binding across all 17 trajectory signatures, and **VCPS-5D** with stepwise subset binding) with exact analytical gradients and L-BFGS-B optimization.
- **Canonical UQ Baseline Suite**:
  - **UQLM White-Box Scorers** ([CVS Health UQLM](https://github.com/cvs-health/uqlm)): Sequence Probability (Joint / Length-Normalized), Min Token Probability, Mean Token Negentropy, and Top-1/Top-2 Probability Margin.
  - **Semantic Entropy & NLI Clustering** ([Kuhn et al., 2023 / UMPIRE OpenReview](https://openreview.net/forum?id=c9TWeKZQR4)): DeBERTa-v2-xlarge bidirectional NLI entailment clustering, LogSumExp cluster aggregation, and Cluster Assignment Entropy.
  - **EigenScore & LogDet Spectral Volume** ([Chen et al., 2024 / UMPIRE](https://openreview.net/forum?id=c9TWeKZQR4)): SVD on covariance and Gram matrices ($\frac{1}{K}\sum \log_{10} s_i$) and matrix log-determinant volume metrics.
- **Classic Post-Hoc Calibrators**: Global Temperature Scaling (Guo et al.), 1D Platt Scaling, Trajectory LR (No Bias), Monotonic Spline Calibration (PCHIP), Quadratic Platt, and Adaptive Temperature Scaling (ATS / Thermometer).
- **Hardened VLM Inference Wrapper**:
  - Full `sys.modules` namespace purge preventing package shadowing between M3 and MQT.
  - FlashAttention-friendly fast inference (`output_attentions=False`).
  - First-token logit NaN/Inf clamping and Vicuna-1.5 `llava_v1` conversation mode enforcement.
- **Multi-Dataset Support**: 14 vision-language benchmarks with non-withheld validation splits and 10-annotator soft consensus $\min(1.0, \text{matches}/3.0)$.

---

## Clean Modular Architecture ($\le 200$ LOC Per File)

```
src/trajectory_calibration/
├── utils/
│   ├── math.py              (39 LOC)  - Stable sigmoid, get_logits, safe_clip_probs
│   ├── config.py            (41 LOC)  - Central Config dataclass and ARCH_SCALES
│   ├── helpers.py           (70 LOC)  - safe_torch_load, set_seed, clean_text
│   └── __init__.py          (18 LOC)  - Re-exports
├── features/
│   ├── definitions.py       (30 LOC)  - FEATURE_NAMES and 18 signature FEATURE_KEYS
│   ├── extractor.py         (120 LOC) - compute_features_from_sample (18-D vector: 1 anchor + 17 signatures)
│   ├── loader.py            (115 LOC) - find_feature_file, load_dataset_features, splits
│   ├── selection.py         (89 LOC)  - select_best_5d_subset (x1-anchored), VIF filters
│   ├── diagnostics.py       (65 LOC)  - evaluate_model_diagnostics (Health panel)
│   ├── synthetic.py         (45 LOC)  - generate_mock_df for offline tests
│   ├── trajectory.py        (38 LOC)  - Backwards-compatible facade
│   └── __init__.py          (35 LOC)  - Re-exports
├── calibrators/
│   ├── classic.py           (178 LOC) - NC, TS (bounded), 1D Platt, Spline, ATS
│   ├── proxies.py           (163 LOC) - MSSC, MSE-EIGEN, UQLM proxy baseline estimators
│   ├── vcps.py              - VaryingCoefficientPlattScaler (VCPS-17D 36 params, VCPS-5D 12 params, Analytical L-BFGS-B)
│   ├── residual.py          (138 LOC) - ResidualTrajectoryCalibrator, AURC (trapezoid)
│   ├── adaptation.py        (94 LOC)  - Saerens-EM (2002), Target Intercept, Beta Calib
│   ├── baselines.py         (40 LOC)  - Facade re-exporting classic and proxy estimators
│   └── __init__.py          (74 LOC)  - Re-exports
├── metrics/
│   ├── ece.py               (145 LOC) - Equal-width ECE, MCE, Adaptive ECE, KDE-ECE
│   ├── scoring.py           (66 LOC)  - Brier Score, NLL, Prediction Std, AUROC
│   ├── murphy.py            (59 LOC)  - Murphy (1973) Brier Decomposition
│   ├── statistical.py       (71 LOC)  - Logistic slope/intercept, Bootstrap CIs
│   ├── calibration.py       (38 LOC)  - Backwards-compatible facade
│   └── __init__.py          (31 LOC)  - Re-exports
├── uq/
│   ├── whitebox.py          (107 LOC) - UQLM: SequenceProb, MinProb, TokenEntropy, Margin
│   ├── semantic_entropy.py  (165 LOC) - Kuhn/UMPIRE: DeBERTa NLI clustering, LogSumExp
│   ├── eigenscore.py        (92 LOC)  - Chen/UMPIRE: SVD on covariance & LogDet volume
│   └── __init__.py          (46 LOC)  - Re-exports
└── vlm/
    ├── registry.py          (154 LOC) - 14-benchmark DATASET_REGISTRY & HF loader
    ├── evaluators.py        (95 LOC)  - evaluate_accuracy (open, list_soft, mc_index, mc_letter)
    ├── formatting.py        (87 LOC)  - format_question, load_image_from_sample
    ├── patches.py           (48 LOC)  - Protected transformers >= 4.38 monkey-patches
    ├── llava_compat.py      (131 LOC) - Namespace purge router and module resolver
    ├── wrapper.py           (165 LOC) - UnifiedVLMWrapper (FlashAttention, NaN guards)
    ├── datasets.py          (23 LOC)  - Facade re-exporting registry & evaluators
    └── __init__.py          (25 LOC)  - Re-exports
```

---

## Installation and Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/pika-droid/trajectory_calibration.git
cd trajectory_calibration

# Create Python 3.12 virtual environment & install in editable mode
python -m venv .venv
.\.venv\Scripts\pip install -e .
```

### 2. Fast CPU Smoke Test (~3 seconds)

Runs the entire calibrator pipeline across all 15 benchmark methods on synthetic pilot features with zero GPU requirements:

```bash
# Windows
.\.venv\Scripts\python scripts/run_mock.py

# Linux / MacOS
./.venv/bin/python scripts/run_mock.py
```

### 3. Run Pytest Suite

```bash
.\.venv\Scripts\python -m pytest tests/ -v
```

### 4. VCPS Python API Usage

```python
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import FEATURE_KEYS

# VCPS-17D: Dynamically binds all 17 trajectory signatures (36 parameters total)
vcps_17d = VaryingCoefficientPlattScaler(feature_set="17d")
vcps_17d.fit(X_train_18d, y_train, feature_names=FEATURE_KEYS)
probs_17d = vcps_17d.predict_proba(X_test_18d)

# VCPS-5D: Binds stepwise-selected 5-signature subset (12 parameters total, 6-feature input: x1 + 5 signatures)
vcps_5d = VaryingCoefficientPlattScaler(feature_set="5d")
vcps_5d.fit(X_train_6d, y_train, feature_names=best_5d_keys)
probs_5d = vcps_5d.predict_proba(X_test_6d)
```

---

## Empirical Benchmark Results

Evaluated across all 14 vision-language benchmarks on single-pass feature matrices ($T_{\text{gen}} = 0.00$, $1\times$ inference cost). Best results are **bolded**, second-best are *italicized*.

> [!NOTE]
> **Single-Pass vs. Multi-Rollout UQ**: Multi-rollout sampling algorithms (such as Kuhn Semantic Entropy, Chen EigenScore, and UQLM Token Negentropy) require drawing $M$ stochastic response rollouts per sample ($M \ge 5$) or full-vocabulary logit matrices.

###### M3-LLaVA: Adaptive ECE (%) [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | gqa | infographicvqa | lego-puzzles | mmbench | mmmu | pope | scienceqa | seedbench | textvqa | vizwiz-vqa | vqav2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 41.80% | 70.70% | 73.52% | 32.43% | 84.55% | 73.16% | 28.58% | 78.76% | 4.30% | 10.66% | 32.40% | 9.07% | 16.71% | 8.06% |
| Temperature Scaling (TS) | 9.38% | 45.07% | 48.86% | 9.90% | 53.91% | 34.71% | 7.88% | 52.21% | 4.23% | 9.76% | 6.16% | 8.92% | 16.03% | 10.51% |
| Platt Scaling (1D) | **5.19%** | 2.35% | 2.39% | 6.70% | 1.13% | 11.89% | 5.50% | *3.44%* | 4.33% | 7.46% | *5.98%* | 6.11% | 7.69% | 9.71% |
| Trajectory LR | 9.58% | 4.01% | *2.36%* | 5.79% | **1.09%** | 11.23% | 6.37% | 3.64% | 5.64% | 6.43% | **5.74%** | 5.00% | *4.84%* | *6.13%* |
| Trajectory LR (No Bias) | 8.58% | 4.27% | 3.99% | 8.51% | 1.18% | 12.14% | 5.60% | 4.99% | 5.65% | 6.33% | 6.32% | 4.89% | **4.50%** | 6.59% |
| Spline Calibration | *7.68%* | 3.80% | **1.93%** | 6.44% | 1.23% | 12.33% | *4.81%* | **3.03%** | *3.82%* | 7.51% | 7.60% | *4.86%* | 8.37% | 8.21% |
| Adaptive TS (ATS) | 10.25% | 45.05% | 48.84% | 9.41% | 53.89% | 34.66% | 9.89% | 52.19% | 3.87% | 8.47% | 10.36% | 8.50% | 15.46% | 9.28% |
| Residual Calibrator | 8.13% | 2.27% | 2.40% | 5.74% | *1.13%* | *8.07%* | 7.56% | 3.46% | **3.81%** | *6.05%* | 10.21% | 5.34% | 7.43% | 9.16% |
| **VCPS-5D (Our Method)** | 9.85% | **2.19%** | 2.65% | *5.41%* | 1.31% | 8.13% | **4.54%** | 3.48% | 4.03% | **5.88%** | 7.58% | 5.00% | 7.40% | 8.10% |
| **VCPS-17D (Our Method)** | 9.58% | *2.23%* | 3.11% | **5.26%** | 1.22% | **6.49%** | 5.39% | 3.47% | 3.86% | 6.62% | 8.23% | **4.02%** | 6.68% | **4.52%** |

- **VCPS vs. Global Temperature Scaling (TS)**:
  - **Family Win**: VCPS beats TS on **12 / 14 datasets** (all except `ai2d`, `seedbench`).
  - **Separate Counts**: VCPS-17D alone beats TS on **12 / 14 datasets**; VCPS-5D alone beats TS on **12 / 14 datasets**.
- **VCPS vs. 1D Platt Scaling**:
  - **Family Win**: VCPS beats 1D Platt Scaling on **9 / 14 datasets**:
    `chartqa`, `gqa`, `lego-puzzles`, `mmbench`, `pope`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
  - **Separate Counts**: VCPS-17D alone beats 1D Platt on **9 / 14 datasets**; VCPS-5D alone beats 1D Platt on **9 / 14 datasets**.
- **Our Trajectory Methods (VCPS-5D, VCPS-17D, Residual Calibrator)** win the #1 lowest Adaptive ECE on **8 / 14 benchmarks**:
  `chartqa` (VCPS-5D: **2.19%**), `gqa` (VCPS-17D: **5.26%**), `lego-puzzles` (VCPS-17D: **6.49%**), `mmbench` (VCPS-5D: **4.54%**), `pope` (Residual: **3.81%**), `scienceqa` (VCPS-5D: **5.88%**), `textvqa` (VCPS-17D: **4.02%**), `vqav2` (VCPS-17D: **4.52%**).
---

### MQT-LLaVA: Adaptive ECE (%) [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | gqa | infographicvqa | lego-puzzles | mmbench | mmmu | pope | scienceqa | seedbench | textvqa | vizwiz-vqa | vqav2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 24.32% | 62.99% | 51.18% | 10.19% | 75.51% | 33.80% | 15.27% | 52.24% | 6.18% | 23.04% | 17.74% | 30.96% | 30.26% | 10.93% |
| Temperature Scaling (TS) | 13.01% | 41.54% | 46.63% | 10.61% | 52.63% | 23.68% | 10.24% | 49.32% | *5.44%* | 10.84% | 10.44% | 18.67% | 28.77% | 16.04% |
| Platt Scaling (1D) | *9.51%* | 8.03% | 4.56% | 10.60% | 1.11% | 12.43% | 8.38% | 2.14% | **4.79%** | 6.78% | *8.62%* | 11.82% | 5.76% | 14.95% |
| Trajectory LR | 12.45% | 7.42% | *4.05%* | *10.13%* | **0.95%** | 9.41% | *7.15%* | **2.00%** | 6.16% | **5.34%** | 9.60% | *7.00%* | *5.07%* | *9.75%* |
| Trajectory LR (No Bias) | 12.99% | 11.53% | 8.93% | **9.92%** | 1.50% | 12.96% | **6.05%** | 4.00% | 6.58% | 8.60% | 10.35% | 7.11% | 10.66% | 11.39% |
| Spline Calibration | **8.64%** | **4.71%** | 4.22% | 10.31% | *1.08%* | 8.21% | 7.75% | *2.13%* | 6.42% | 7.04% | 8.82% | **5.43%** | **4.29%** | 10.70% |
| Adaptive TS (ATS) | 13.92% | 41.53% | 46.62% | 10.51% | 52.61% | 23.68% | 10.53% | 49.31% | 6.62% | 10.49% | 9.95% | 18.34% | 28.45% | 13.21% |
| Residual Calibrator | 11.34% | 7.30% | 4.38% | 10.49% | 1.11% | *7.93%* | 7.27% | 2.15% | 6.01% | 5.88% | **8.47%** | 7.44% | 5.53% | 12.40% |
| **VCPS-5D (Our Method)** | 12.43% | *5.83%* | 4.37% | 10.42% | 1.10% | 11.99% | 7.57% | 2.15% | 7.14% | 5.53% | 9.43% | 7.23% | 5.62% | 10.42% |
| **VCPS-17D (Our Method)** | 12.31% | 7.61% | **3.04%** | 10.72% | 1.10% | **6.30%** | 8.61% | 2.14% | 5.92% | *5.38%* | 9.01% | 7.36% | 5.63% | **7.74%** |

- **VCPS vs. Global Temperature Scaling (TS)**:
  - **Family Win**: VCPS beats TS on **13 / 14 datasets** (all except `pope`).
  - **Separate Counts**: VCPS-17D alone beats TS on **12 / 14 datasets**; VCPS-5D alone beats TS on **13 / 14 datasets**.
- **VCPS vs. 1D Platt Scaling**:
  - **Family Win**: VCPS beats 1D Platt Scaling on **10 / 14 datasets**:
    `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
  - **Separate Counts**: VCPS-17D alone beats 1D Platt on **8 / 14 datasets**; VCPS-5D alone beats 1D Platt on **10 / 14 datasets**.
- **Our Trajectory Methods (VCPS-5D, VCPS-17D, Residual Calibrator)** win the #1 lowest Adaptive ECE on **4 / 14 benchmarks**:
  `docvqa` (VCPS-17D: **3.04%**), `lego-puzzles` (VCPS-17D: **6.30%**), `seedbench` (Residual: **8.47%**), `vqav2` (VCPS-17D: **7.74%**).
---

## Benchmark Logs & Multi-Pass Baselines (UMPIRE Paper)

Comprehensive, structured Markdown log reports detailing all per-dataset evaluations, macro summaries, and comparative analyses are available in the [`logs/`](logs/) directory:

### 1. UMPIRE Multi-Pass Baseline Logs ($T=0.5, K=10$ rollouts)
Evaluates `ln_entropy`, `semantic_entropy` (DeBERTa-v2 NLI clustering), `eigen_score` (SVD covariance dispersion), and `umpire` (Incoherence-adjusted Semantic Volume) as formulated in Lau et al. (*arXiv:2602.24195*):
- **M3-LLaVA**: [`logs/umpire_baselines/m3_llava_umpire_baselines.md`](logs/umpire_baselines/m3_llava_umpire_baselines.md)
- **MQT-LLaVA**: [`logs/umpire_baselines/mqt_llava_umpire_baselines.md`](logs/umpire_baselines/mqt_llava_umpire_baselines.md)

### 2. VCPS vs. All Baselines Unified Comparison Logs
Compares single-pass greedy calibration ($T=0.0$, $1\times$ compute) against classic post-hoc calibrators (TS, Platt, Spline, ATS, Residual) and multi-pass sampling baselines ($T=0.5, K=10$ rollouts):
- **M3-LLaVA**: [`logs/vcps_logs/m3_llava_vcps_vs_baselines.md`](logs/vcps_logs/m3_llava_vcps_vs_baselines.md)
- **MQT-LLaVA**: [`logs/vcps_logs/mqt_llava_vcps_vs_baselines.md`](logs/vcps_logs/mqt_llava_vcps_vs_baselines.md)

### Macro-Average Comparison Across 15 Methods on 14 Datasets ($T_{\text{gen}} = 0.00$, $1\times$ Compute)

| Model | Calibration Method | Paradigm / Regime | Sampling | Macro ECE (%) $\downarrow$ | Macro Ada-ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **M3-LLaVA** | Quadratic Platt (Logit-Only) | Polynomial Logit | Greedy ($T=0.0, K=1$) | **3.22%** | **4.95%** | 0.678 |
|  | **VCPS-17D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 4.18% | *5.05%* | 0.698 |
|  | Spline Calibration | Non-Parametric (Isotonic) | Greedy ($T=0.0, K=1$) | 3.45% | 5.83% | 0.669 |
|  | **VCPS-5D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 3.95% | 5.40% | 0.694 |
|  | Trajectory LR | Linear Trajectory | Greedy ($T=0.0, K=1$) | 4.49% | 5.56% | **0.718** |
|  | Platt Scaling (1D) | Classic Linear Post-Hoc | Greedy ($T=0.0, K=1$) | *3.24%* | 5.70% | 0.682 |
|  | Residual Calibrator | Feature-Aided | Greedy ($T=0.0, K=1$) | 4.20% | 5.77% | 0.702 |
|  | Trajectory LR (No Bias) | Linear Trajectory (Zero-Bias) | Greedy ($T=0.0, K=1$) | 4.34% | 5.97% | *0.703* |
|  | Temperature Scaling (TS) | Classic Post-Hoc | Greedy ($T=0.0, K=1$) | 21.93% | 22.68% | 0.671 |
|  | Adaptive TS (ATS) | Adaptive Calibrator | Greedy ($T=0.0, K=1$) | 21.86% | 22.87% | 0.685 |
|  | Naive Confidence (NC) | Uncalibrated Baseline | Greedy ($T=0.0, K=1$) | 40.33% | 40.33% | 0.671 |
|  | `umpire` | Multi-Pass Semantic Volume | Stochastic ($T=0.5, K=10$) | 17.34% | - | 0.698 |
|  | `eigen_score` | SVD Covariance Dispersion | Stochastic ($T=0.5, K=10$) | 24.74% | - | 0.694 |
|  | `ln_entropy` | Predictive Entropy | Stochastic ($T=0.5, K=10$) | 19.24% | - | 0.675 |
|  | `semantic_entropy` | DeBERTa NLI Clustering | Stochastic ($T=0.5, K=10$) | 28.25% | - | 0.641 |
| **MQT-LLaVA** | Quadratic Platt (Logit-Only) | Polynomial Logit | Greedy ($T=0.0, K=1$) | 4.90% | *6.53%* | 0.706 |
|  | **VCPS-17D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | *4.73%* | 6.63% | *0.716* |
|  | Spline Calibration | Non-Parametric (Isotonic) | Greedy ($T=0.0, K=1$) | **3.21%** | **6.41%** | 0.693 |
|  | **VCPS-5D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 5.28% | 7.23% | 0.707 |
|  | Trajectory LR | Linear Trajectory | Greedy ($T=0.0, K=1$) | 6.19% | 6.89% | **0.775** |
|  | Platt Scaling (1D) | Classic Linear Post-Hoc | Greedy ($T=0.0, K=1$) | 5.88% | 7.82% | 0.686 |
|  | Residual Calibrator | Feature-Aided | Greedy ($T=0.0, K=1$) | 5.17% | 6.98% | 0.702 |
|  | Trajectory LR (No Bias) | Linear Trajectory (Zero-Bias) | Greedy ($T=0.0, K=1$) | 7.27% | 8.75% | 0.676 |
|  | Temperature Scaling (TS) | Classic Post-Hoc | Greedy ($T=0.0, K=1$) | 23.45% | 24.13% | 0.682 |
|  | Adaptive TS (ATS) | Adaptive Calibrator | Greedy ($T=0.0, K=1$) | 23.15% | 23.98% | 0.685 |
|  | Naive Confidence (NC) | Uncalibrated Baseline | Greedy ($T=0.0, K=1$) | 32.32% | 31.76% | 0.682 |
|  | `umpire` | Multi-Pass Semantic Volume | Stochastic ($T=0.5, K=10$) | 21.27% | - | 0.691 |
|  | `eigen_score` | SVD Covariance Dispersion | Stochastic ($T=0.5, K=10$) | 29.57% | - | 0.689 |
|  | `ln_entropy` | Predictive Entropy | Stochastic ($T=0.5, K=10$) | 22.77% | - | 0.666 |
|  | `semantic_entropy` | DeBERTa NLI Clustering | Stochastic ($T=0.5, K=10$) | 33.37% | - | 0.639 |
---

### Decoding Temperature Robustness Study ($T \in \{0.0, 0.3, 0.6, 1.0, 1.5\}$)

To rigorously evaluate zero-shot calibration stability under generation temperature shifts, calibrators trained at greedy decoding ($T=0.0$) are evaluated across stochastic sampling temperatures on **4 representative benchmark archetypes**:
1. **`pope`**: Binary object hallucination grounding (Yes/No).
2. **`scienceqa`**: Multimodal multiple-choice science reasoning.
3. **`textvqa`**: Fine-grained visual OCR and scene text comprehension.
4. **`vizwiz-vqa`**: Real-world assistive vision questions with unconstrained answers.

#### Macro Transfer ECE (%) Across Temperatures (Trained at $T=0.0$)

| Model | Calibration Method | $T=0.0$ | $T=0.3$ | $T=0.6$ | $T=1.0$ | $T=1.5$ | Mean ECE $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M3-LLaVA** | Spline Calibration (PCHIP) | *4.53%* | **16.10%** | **7.75%** | 6.54% | 10.85% | **9.15%** |
|  | Quadratic Platt (Logit-Only) | 4.71% | *19.49%* | 9.57% | 5.31% | **7.54%** | *9.32%* |
|  | Platt Scaling (1D) | 4.77% | 19.62% | 9.65% | **4.75%** | *8.36%* | 9.43% |
|  | Residual Calibrator | 4.73% | 20.04% | 9.78% | *5.27%* | 9.54% | 9.87% |
|  | **VCPS-5D (Our Method)** | 4.66% | 19.89% | *9.42%* | 5.65% | 10.00% | 9.93% |
|  | **VCPS-17D (Our Method)** | 4.63% | 20.02% | 9.78% | 7.08% | 12.00% | 10.70% |
|  | Trajectory LR (No Bias) | 4.58% | 21.54% | 13.00% | 8.75% | 13.52% | 12.28% |
|  | Trajectory LR | **4.11%** | 21.65% | 12.72% | 10.71% | 15.00% | 12.84% |
|  | Temperature Scaling (TS) | 9.16% | 22.14% | 13.87% | 9.41% | 11.33% | 13.18% |
|  | Adaptive TS (ATS) | 8.79% | 22.61% | 13.76% | 9.03% | 11.72% | 13.18% |
|  | Naive Confidence (NC) | 10.03% | 23.98% | 16.05% | 9.82% | 9.73% | 13.92% |
| **MQT-LLaVA** | Spline Calibration (PCHIP) | **3.38%** | **16.16%** | *9.72%* | 5.38% | 8.49% | **8.63%** |
|  | Trajectory LR (No Bias) | 7.78% | *17.63%* | **9.33%** | *5.31%* | **6.62%** | *9.33%* |
|  | Trajectory LR | *5.45%* | 21.76% | 10.36% | **4.78%** | 7.67% | 10.00% |
|  | Quadratic Platt (Logit-Only) | 5.50% | 21.60% | 10.20% | 6.13% | *7.53%* | 10.19% |
|  | **VCPS-17D (Our Method)** | 6.00% | 23.48% | 12.26% | 5.61% | 7.90% | 11.05% |
|  | **VCPS-5D (Our Method)** | 5.70% | 24.17% | 12.78% | 5.86% | 7.89% | 11.28% |
|  | Residual Calibrator | 5.54% | 24.29% | 12.97% | 6.30% | 7.98% | 11.42% |
|  | Platt Scaling (1D) | 7.15% | 24.62% | 13.47% | 7.46% | 8.74% | 12.29% |
|  | Temperature Scaling (TS) | 15.68% | 28.00% | 21.90% | 19.27% | 20.85% | 21.14% |
|  | Adaptive TS (ATS) | 15.82% | 28.82% | 22.41% | 19.40% | 20.19% | 21.33% |
|  | Naive Confidence (NC) | 22.96% | 40.62% | 34.40% | 27.43% | 20.96% | 29.27% |
---

### Leave-One-Dataset-Out (LODO) Cross-Domain Generalization

Evaluates zero-shot transfer by training calibrators on 13 pooled benchmarks and testing on the held-out 14th benchmark across all 14 datasets:
- **M3-LLaVA**: **VCPS-17D** achieves **$27.25\%$ Macro ECE** ($27.47\%$ Ada-ECE) and top discrimination (**$0.695$ Macro AUROC**), outperforming 1D Platt Scaling ($28.34\%$ ECE, $29.41\%$ Ada-ECE, $0.677$ AUROC). **VCPS-5D** achieves **$27.69\%$ Macro ECE** ($28.25\%$ Ada-ECE) and **$0.694$ Macro AUROC**.
- **MQT-LLaVA**: **VCPS-17D** achieves **$20.59\%$ Macro ECE** ($20.63\%$ Ada-ECE) and top discrimination (**$0.709$ Macro AUROC**), outperforming 1D Platt Scaling ($22.64\%$ ECE, $22.76\%$ Ada-ECE, $0.698$ AUROC). **VCPS-5D** achieves **$21.10\%$ Macro ECE** ($21.14\%$ Ada-ECE) and **$0.706$ Macro AUROC**.
- Full publication table available in [`dataset_tables/lodo_cross_dataset.tex`](dataset_tables/lodo_cross_dataset.tex).

---

## Baseline Repositories & Modifications

The [`baseline_repo/`](baseline_repo/) directory contains adapted baseline frameworks:
- **`baseline_repo/UMPIRE/`**: Forked and hardened implementation of Lau et al. (*arXiv:2602.24195*) with batch NLI clustering, multi-GPU rollout generation, and evaluation pipelines for M3-LLaVA and MQT-LLaVA.
- **`baseline_repo/eigenscore/`**: Implementation of Chen et al. (*EigenScore*) SVD covariance dispersion UQ.
- **`baseline_repo/semantic_uncertainty/`**: Reference implementation of Kuhn et al. (*Semantic Entropy*).

---

## 18-D Trajectory Feature Space (1 Base Anchor + 17 Trajectory Signatures: $x_1, x_3 \dots x_{22}$)

| Key | Feature Name | Formula | Scientific Meaning |
| :--- | :--- | :--- | :--- |
| **`x1`** | **Final Logit** | $\ln(c_{\text{fine}} / (1 - c_{\text{fine}}))$ | Primary uncalibrated confidence anchor ($m=576/256$) |
| **`x3`** | **Confidence Gain** | $c_{\text{fine}} - c_9$ | Visual resolution sensitivity (fine minus coarse) |
| **`x4`** | **Monotonicity Count** | $\sum_{i=1}^4 \mathbb{I}(c_{m_{i+1}} > c_{m_i})$ | Monotonic confidence trajectory consistency |
| **`x6`** | **Confidence Variance** | $\text{Var}([c_1, c_9, c_{36}, c_{144}, c_{\text{fine}}])$ | Fluctuation/dispersion across visual scales |
| **`x8`** | **Scale Dip Depth** | $\max(0, \max(c_1, c_9) - \min(c_{36}, c_{144}))$ | Mid-scale visual confusion indicator |
| **`x9`** | **Log-Scale Slope** | $\frac{\sum (\ln m_i - \overline{\ln m})(c_{m_i} - \bar{c})}{\sum (\ln m_i - \overline{\ln m})^2}$ | Logarithmic rate of confidence growth |
| **`x10`** | **Logprob Gain** | $\ln c_{\text{fine}} - \ln c_9$ | Probability magnitude shift in log-space |
| **`x11`** | **Logprob Variance** | $\text{Var}([\ln c_1, \dots, \ln c_{\text{fine}}])$ | Log-likelihood stability across scales |
| **`x12`** | **Logprob Acceleration** | $(\ln c_{\text{fine}} - \ln c_{144}) - (\ln c_{144} - \ln c_{36})$ | Discrete 2nd derivative of log-confidence |
| **`x13`** | **Answer Stability** | $1 / \text{UniqueAnswers}$ | Inverse count of distinct decoded strings across 5 scales |
| **`x14`** | **Relative Gain Ratio** | $c_{\text{fine}} / (c_9 + \epsilon)$ | Multiplicative confidence enhancement ratio |
| **`x15`** | **Mid-Fine Contrast** | $(c_{\text{fine}} - c_{144}) - (c_{144} - c_9)$ | Convexity of mid-to-fine transition |
| **`x17`** | **End-Scale Spike** | $c_{\text{fine}} - \frac{1}{4}\sum_{i=1}^4 c_{m_i}$ | Sudden fine-scale confidence jump |
| **`x18`** | **Entropy Slope** | OLS slope of binary entropy $H(c_m)$ vs $\ln m$ | Rate of information gain with resolution |
| **`x19`** | **Margin Growth** | $\text{margin}_{\text{fine}} / (\text{margin}_9 + \epsilon)$ | Top-1 vs Top-2 separation growth |
| **`x20`** | **Answer Flip Freq** | $\frac{1}{4} \sum_{i=1}^4 \mathbb{I}(\text{ans}(m_i) \neq \text{ans}(m_{i+1}))$ | Textual prediction volatility across scales |
| **`x21`** | **Logit Trajectory Convexity** | $(m_{\text{fine}} - m_{144}) - (m_{144} - m_{36})$ | High-resolution margin/logit curve convexity |
| **`x22`** | **Jump Ratio** | $(c_{\text{fine}} - c_1) / (c_{\text{fine}} + \epsilon)$ | Relative span from single-token to full scale |

---

## Model Diagnostics & Invariant Rules

1. **$x_1$ Anchor Invariant**: $x_1$ (Final Logit) must ALWAYS be retained as the root feature. Ablating $x_1$ causes prediction collapse ($\sigma_p < 0.02$).
2. **Diagnostic Panel**: Always computes ECE, MCE, Brier, Brier Gain, $\sigma_p$, AUROC, and Spearman $\rho$, tagging status:
   - `COLLAPSED` if $\sigma_p < 0.02$ or $\text{BrierGain} \le 0.001$.
   - `SCRAMBLED` if $\rho < 0.10$.
   - `VALID` otherwise.
3. **Protected Monkeypatches**: Never remove the `transformers >= 4.38` monkeypatches in `vlm/patches.py` (`cache_position`, `num_logits_to_keep`, `matryoshka_vis_token_scale`).
4. **Temperature Bounds**: Standardized to $T \in [0.01, 20.0]$ globally.
5. **Numerical Stability**: Logits bounded to $[-35.0, 35.0]$, probabilities clipped to $[10^{-7}, 1 - 10^{-7}]$.
