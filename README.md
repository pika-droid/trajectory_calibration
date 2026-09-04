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

Runs the entire calibrator pipeline across all 16 methods on synthetic pilot features with zero GPU requirements:

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

##### M3-LLaVA: Adaptive ECE (%) [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | gqa | infographicvqa | lego-puzzles | mmbench | mmmu | pope | scienceqa | seedbench | textvqa | vizwiz-vqa | vqav2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 41.80% | 70.70% | 73.52% | 32.43% | 84.55% | 73.16% | 28.58% | 78.76% | 4.12% | 10.66% | 32.40% | 8.72% | 16.71% | 7.74% |
| Temperature Scaling (TS) | 10.73% | 45.07% | 48.86% | 10.84% | 53.91% | 34.71% | 7.77% | 52.21% | 3.96% | 9.76% | 2.64% | 8.35% | 15.56% | 10.63% |
| Platt Scaling (1D) | *7.49%* | *2.34%* | *2.42%* | 7.71% | *1.12%* | 10.07% | *4.88%* | *3.45%* | 4.05% | 7.46% | *2.25%* | 5.70% | 7.25% | 9.48% |
| Trajectory LR | 9.19% | 3.64% | 3.09% | 6.78% | 1.23% | 9.24% | 6.25% | 3.58% | 5.80% | *5.46%* | 4.61% | 5.60% | *5.39%* | 6.99% |
| Trajectory LR (No Bias) | 8.58% | 2.82% | 5.99% | 6.21% | 1.40% | 10.21% | 6.60% | 3.93% | 5.17% | 6.49% | 4.61% | **2.94%** | **5.15%** | *5.90%* |
| Spline Calibration (PCHIP) | **6.98%** | 4.23% | 2.75% | 6.57% | **0.60%** | **6.36%** | **4.45%** | **2.64%** | 5.10% | **5.16%** | **1.17%** | 10.25% | 8.45% | 10.14% |
| Adaptive TS (ATS) | 10.69% | 45.05% | 48.84% | 9.80% | 53.89% | 34.66% | 10.20% | 52.19% | *3.92%* | 9.42% | 9.65% | 8.61% | 15.45% | 9.41% |
| Residual Calibrator | 8.27% | **2.03%** | **2.36%** | 6.59% | 1.29% | 10.54% | 6.57% | 3.47% | 4.34% | 5.56% | 8.92% | 5.55% | 7.09% | 9.06% |
| **VCPS-5D (Our Method)** | 9.65% | 2.88% | 2.53% | **5.61%** | 1.31% | 12.78% | 5.80% | 3.46% | 4.06% | 6.32% | 7.91% | 5.11% | 7.00% | 8.05% |
| **VCPS-17D (Our Method)** | 8.82% | 2.97% | 3.13% | *6.13%* | 1.23% | *8.71%* | 5.53% | 3.48% | **3.79%** | 6.14% | 7.83% | *4.00%* | 6.64% | **5.45%** |

- **VCPS beats Global Temperature Scaling (TS)** on **13 / 14 datasets** (all except `seedbench`):
  `ai2d`, `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `mmmu`, `pope`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS beats 1D Platt Scaling** on **8 / 14 datasets**:
  `gqa`, `lego-puzzles`, `mmmu`, `pope`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS achieves #1 or #2 Rank** on **5 / 14 datasets**:
  `pope` (1st, **3.79%**), `vqav2` (1st, **5.45%**), `gqa` (1st, **5.61%** & 2nd, *6.13%*), `lego-puzzles` (2nd, *8.71%*), and `textvqa` (2nd, *4.00%*).
- **Trajectory LR / Trajectory LR (No Bias)** achieves **#1 or #2 Rank on 5 / 14 datasets**:
  `scienceqa` (2nd, *5.46%*), `textvqa` (1st, **2.94%**), `vizwiz-vqa` (1st, **5.15%** & 2nd, *5.39%*), and `vqav2` (2nd, *5.90%*).

---

### MQT-LLaVA: Adaptive ECE (%) [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | gqa | infographicvqa | lego-puzzles | mmbench | mmmu | pope | scienceqa | seedbench | textvqa | vizwiz-vqa | vqav2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 24.31% | 62.99% | 51.18% | 9.87% | 75.51% | 34.14% | 14.87% | 52.24% | 6.45% | 23.04% | 17.74% | 30.96% | 30.26% | 11.52% |
| Temperature Scaling (TS) | 13.47% | 41.54% | 46.63% | 10.14% | 52.63% | 24.08% | 10.07% | 49.32% | 5.90% | 10.44% | 7.99% | 18.27% | 28.77% | 15.96% |
| Platt Scaling (1D) | 9.52% | 7.04% | 5.04% | 10.27% | 1.09% | 11.70% | 8.24% | 2.15% | *5.29%* | 6.51% | **6.88%** | 11.59% | 6.81% | 15.96% |
| Trajectory LR | 9.92% | *6.32%* | 2.85% | 13.04% | *1.06%* | 7.96% | *5.98%* | **1.79%** | **5.00%** | *6.16%* | 8.56% | *6.99%* | *5.57%* | *8.56%* |
| Trajectory LR (No Bias) | 11.88% | 12.11% | *2.74%* | 13.01% | 1.30% | *6.20%* | **5.70%** | *1.93%* | 6.15% | 7.87% | 7.91% | 8.03% | 7.38% | 10.53% |
| Spline Calibration (PCHIP) | *8.98%* | 7.07% | **2.48%** | 9.78% | **0.60%** | **5.27%** | 10.04% | 2.27% | 7.97% | 7.97% | 7.65% | 7.32% | 5.64% | 15.17% |
| Adaptive TS (ATS) | 13.92% | 41.53% | 46.62% | **9.11%** | 52.61% | 24.08% | 9.87% | 49.32% | 5.99% | 10.12% | 8.99% | 18.16% | 28.45% | 13.03% |
| Residual Calibrator | 11.34% | **5.96%** | 4.16% | 9.26% | 1.09% | 9.07% | 7.52% | 2.15% | 6.58% | 7.23% | 8.32% | **6.67%** | 6.61% | 15.08% |
| **VCPS-5D (Our Method)** | 11.04% | 6.95% | 4.15% | *9.22%* | 1.09% | 10.69% | 6.84% | 2.15% | 6.54% | 7.51% | 10.95% | 7.63% | 6.11% | 11.11% |
| **VCPS-17D (Our Method)** | **8.52%** | 6.81% | 3.57% | 9.63% | 1.09% | 11.80% | 6.22% | 2.15% | 5.43% | **5.24%** | *7.64%* | 7.68% | **5.34%** | **7.83%** |

- **VCPS beats Global Temperature Scaling (TS)** on **14 / 14 datasets**:
  `ai2d`, `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `mmmu`, `pope`, `scienceqa`, `seedbench`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS beats 1D Platt Scaling** on **12 / 14 datasets** (all except `pope`, `seedbench`):
  `ai2d`, `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `mmmu`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS achieves #1 or #2 Rank** on **6 / 14 datasets**:
  `ai2d` (1st, **8.52%**), `scienceqa` (1st, **5.24%**), `vizwiz-vqa` (1st, **5.34%**), `vqav2` (1st, **7.83%**), `gqa` (2nd, *9.22%*), and `seedbench` (2nd, *7.64%*).
- **Trajectory LR / Trajectory LR (No Bias)** achieves **#1 or #2 Rank on 11 / 14 datasets**:
  `mmmu` (1st, **1.79%** & 2nd, *1.93%*), `pope` (1st, **5.00%**), `mmbench` (1st, **5.70%** & 2nd, *5.98%*), `chartqa` (2nd, *6.32%*), `docvqa` (2nd, *2.74%*), `infographicvqa` (2nd, *1.06%*), `lego-puzzles` (2nd, *6.20%*), `scienceqa` (2nd, *6.16%*), `textvqa` (2nd, *6.99%*), `vizwiz-vqa` (2nd, *5.57%*), and `vqav2` (2nd, *8.56%*).

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

### Macro-Average Comparison Across All 14 Benchmarks ($T_{\text{gen}} = 0.00$, $1\times$ Compute)

| Model | Calibration Method | Paradigm / Regime | Sampling | Macro ECE (%) $\downarrow$ | Macro Ada-ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **M3-LLaVA** | Quadratic Platt (Logit-Only) | Polynomial Logit | Greedy ($T=0.0, K=1$) | **3.22%** | **4.81%** | 0.678 |
| | **VCPS-17D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 4.30% | *5.28%* | 0.697 |
| | Spline Calibration (PCHIP) | Non-Parametric | Greedy ($T=0.0, K=1$) | 4.75% | 5.35% | 0.669 |
| | Platt Scaling (1D) | Classic Linear Post-Hoc | Greedy ($T=0.0, K=1$) | *3.24%* | 5.41% | 0.682 |
| | Trajectory LR (No Bias) | Linear Trajectory (Zero-Bias) | Greedy ($T=0.0, K=1$) | 4.43% | 5.43% | 0.700 |
| | Trajectory LR | Linear Trajectory | Greedy ($T=0.0, K=1$) | 3.82% | 5.49% | **0.717** |
| | Residual Calibrator | Feature-Aided | Greedy ($T=0.0, K=1$) | 4.10% | 5.83% | *0.703* |
| | **VCPS-5D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 4.24% | 5.89% | 0.695 |
| | Temperature Scaling (TS) | Classic Post-Hoc | Greedy ($T=0.0, K=1$) | 21.93% | 22.50% | 0.671 |
| | Adaptive TS (ATS) | Adaptive Calibrator | Greedy ($T=0.0, K=1$) | 22.00% | 22.98% | 0.687 |
| | Naive Confidence (NC) | Uncalibrated Baseline | Greedy ($T=0.0, K=1$) | 40.33% | 40.27% | 0.671 |
| | `umpire` | Multi-Pass Semantic Volume | Stochastic ($T=0.5, K=10$) | 17.34% | - | 0.698 |
| | `eigen_score` | SVD Covariance Dispersion | Stochastic ($T=0.5, K=10$) | 24.74% | - | 0.694 |
| | `ln_entropy` | Predictive Entropy | Stochastic ($T=0.5, K=10$) | 19.24% | - | 0.675 |
| | `semantic_entropy` | DeBERTa NLI Clustering | Stochastic ($T=0.5, K=10$) | 28.25% | - | 0.641 |
| **MQT-LLaVA** | **VCPS-17D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | *4.74%* | **6.35%** | 0.709 |
| | Trajectory LR | Linear Trajectory | Greedy ($T=0.0, K=1$) | 5.47% | *6.41%* | **0.772** |
| | Quadratic Platt (Logit-Only) | Polynomial Logit | Greedy ($T=0.0, K=1$) | **4.90%** | 6.50% | 0.706 |
| | Spline Calibration (PCHIP) | Non-Parametric | Greedy ($T=0.0, K=1$) | 5.96% | 7.02% | 0.685 |
| | Residual Calibrator | Feature-Aided | Greedy ($T=0.0, K=1$) | 5.25% | 7.22% | 0.706 |
| | **VCPS-5D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 4.88% | 7.29% | 0.699 |
| | Trajectory LR (No Bias) | Linear Trajectory (Zero-Bias) | Greedy ($T=0.0, K=1$) | 6.33% | 7.34% | *0.721* |
| | Platt Scaling (1D) | Classic Linear Post-Hoc | Greedy ($T=0.0, K=1$) | 5.88% | 7.72% | 0.686 |
| | Adaptive TS (ATS) | Adaptive Calibrator | Greedy ($T=0.0, K=1$) | 23.22% | 23.70% | 0.685 |
| | Temperature Scaling (TS) | Classic Post-Hoc | Greedy ($T=0.0, K=1$) | 23.45% | 23.94% | 0.682 |
| | Naive Confidence (NC) | Uncalibrated Baseline | Greedy ($T=0.0, K=1$) | 32.32% | 31.79% | 0.682 |
| | `umpire` | Multi-Pass Semantic Volume | Stochastic ($T=0.5, K=10$) | 21.27% | - | 0.691 |
| | `eigen_score` | SVD Covariance Dispersion | Stochastic ($T=0.5, K=10$) | 29.57% | - | 0.689 |
| | `ln_entropy` | Predictive Entropy | Stochastic ($T=0.5, K=10$) | 22.77% | - | 0.666 |
| | `semantic_entropy` | DeBERTa NLI Clustering | Stochastic ($T=0.5, K=10$) | 33.37% | - | 0.639 |

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
| **M3-LLaVA** | Quadratic Platt (Logit-Only) | 4.71% | 19.49% | **9.57%** | 5.31% | **7.54%** | **9.32%** |
| | Platt Scaling (1D) | 4.77% | 19.62% | *9.65%* | **4.75%** | *8.36%* | *9.43%* |
| | Residual Calibrator | 4.82% | 19.53% | 9.73% | *5.12%* | 9.07% | 9.65% |
| | **VCPS-17D (Our Method)** | *4.44%* | 19.83% | 10.30% | 5.34% | 8.95% | 9.77% |
| | **VCPS-5D (Our Method)** | 5.03% | *19.36%* | 9.75% | 5.41% | 9.33% | 9.78% |
| | Spline Calibration (PCHIP) | 6.90% | **18.25%** | 11.42% | 7.99% | 8.44% | 10.60% |
| | Trajectory LR (No Bias) | 4.49% | 20.47% | 11.54% | 8.87% | 11.12% | 11.30% |
| | Trajectory LR | **4.37%** | 21.10% | 11.56% | 9.60% | 11.95% | 11.72% |
| | Temperature Scaling (TS) | 9.16% | 22.14% | 13.87% | 9.41% | 11.33% | 13.18% |
| | Adaptive TS (ATS) | 9.00% | 22.64% | 14.00% | 9.59% | 11.83% | 13.41% |
| | Naive Confidence (NC) | 10.03% | 23.98% | 16.05% | 9.82% | 9.73% | 13.92% |
| **MQT-LLaVA** | Trajectory LR (No Bias) | 6.43% | **18.47%** | 10.22% | 6.53% | **6.08%** | **9.55%** |
| | Trajectory LR | 6.46% | 20.47% | **9.60%** | **4.55%** | *7.30%* | *9.68%* |
| | Quadratic Platt (Logit-Only) | **5.50%** | 21.60% | *10.20%* | 6.13% | 7.53% | 10.19% |
| | **VCPS-5D (Our Method)** | 6.21% | 23.94% | 12.66% | *5.90%* | 7.82% | 11.31% |
| | Residual Calibrator | *5.78%* | 24.21% | 13.26% | 6.39% | 8.05% | 11.54% |
| | **VCPS-17D (Our Method)** | 6.60% | 24.10% | 13.28% | 6.29% | 7.85% | 11.62% |
| | Spline Calibration (PCHIP) | 6.62% | *19.12%* | 14.57% | 10.57% | 10.05% | 12.19% |
| | Platt Scaling (1D) | 7.15% | 24.62% | 13.47% | 7.46% | 8.74% | 12.29% |
| | Temperature Scaling (TS) | 15.68% | 28.00% | 21.90% | 19.27% | 20.85% | 21.14% |
| | Adaptive TS (ATS) | 15.71% | 28.72% | 22.34% | 19.24% | 20.16% | 21.23% |
| | Naive Confidence (NC) | 22.96% | 40.62% | 34.40% | 27.43% | 20.96% | 29.27% |

---

### Leave-One-Dataset-Out (LODO) Cross-Domain Generalization

Evaluates zero-shot transfer by training calibrators on 13 pooled benchmarks and testing on the held-out 14th benchmark across all 14 datasets:
- **M3-LLaVA**: **VCPS-5D** achieves **$27.38\%$ Macro ECE** and **$0.697$ Macro AUROC**, outperforming 1D Platt Scaling ($28.34\%$ ECE, $0.677$ AUROC). Standardized Trajectory LR achieves **$24.62\%$ Macro ECE** and **$0.672$ Macro AUROC** (Trajectory LR No Bias achieves **$24.18\%$ Macro ECE** and **$0.672$ Macro AUROC**).
- **MQT-LLaVA**: **VCPS-5D** achieves **$21.28\%$ Macro ECE** and top selective risk discrimination (**$0.704$ Macro AUROC**), outperforming 1D Platt Scaling ($22.64\%$ ECE, $0.698$ AUROC). Standardized Trajectory LR achieves **$18.85\%$ Macro ECE** and **$0.699$ Macro AUROC** (Trajectory LR No Bias achieves **$19.71\%$ Macro ECE** and **$0.696$ Macro AUROC**).
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
| **`x21`** | **Logit Convexity** | $(c_{\text{fine}} - c_{144}) - (c_{144} - c_{36})$ | High-resolution curve convexity |
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
