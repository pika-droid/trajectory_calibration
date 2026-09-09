# Trajectory Calibration: Elastic Multi-Scale Uncertainty Quantification for Multimodal LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 74 Passed](https://img.shields.io/badge/tests-74%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checker: Pyright](https://img.shields.io/badge/type%20checker-pyright-2b5797.svg)](https://github.com/microsoft/pyright)
[![Tested with: Hypothesis](https://img.shields.io/badge/property%20testing-hypothesis-0a9edc.svg)](https://hypothesis.readthedocs.io/)
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

- **VCPS-17D (Full Signature Binding)**: Dynamically binds all 16 non-anchor trajectory signatures ($x_2, x_3, \dots, x_{17}$) for both dynamic slope $\boldsymbol{\gamma}$ and dynamic intercept $\mathbf{w}$, optimizing **34 parameters** ($1 + 16 + 1 + 16 = 2K + 2$): scalar log-slope base $a_0$, 16 slope weights $\boldsymbol{\gamma}$, scalar intercept base $b_0$, and 16 intercept weights $\mathbf{w}$ across a 17-D input ($x_1 + \mathbf{z}_{16}$).
- **VCPS-5D (Stepwise Subset)**: Binds 4 forward-stepwise selected trajectory signatures for both dynamic slope $\boldsymbol{\gamma}$ and dynamic intercept $\mathbf{w}$, optimizing **10 parameters** ($1 + 4 + 1 + 4 = 2K + 2$): scalar log-slope base $a_0$, 4 slope weights $\boldsymbol{\gamma}$, scalar intercept base $b_0$, and 4 intercept weights $\mathbf{w}$ across a 5-D input ($x_1 + \mathbf{z}_4$).

3. **Pareto Dominance**: Outperforms standard post-hoc temperature scaling while maintaining **1x inference cost**.

---

## Key Features

- **Strict Modularity**: Every source file in `src/trajectory_calibration/` is structured as a clean, single-responsibility module.
- **Primary Calibration Method**: Varying-Coefficient Platt Scaling (**VCPS-17D** with full 34-parameter dynamic binding across all 16 trajectory signatures, and **VCPS-5D** with stepwise subset binding) with exact analytical gradients and L-BFGS-B optimization.
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
│   ├── definitions.py       (34 LOC)  - FEATURE_NAMES, FEATURE_KEYS (17D), CANONICAL_5D_KEYS
│   ├── extractor.py         (120 LOC) - compute_features_from_sample (17-D vector: 1 anchor + 16 signatures)
│   ├── loader.py            (115 LOC) - find_feature_file, load_dataset_features, splits
│   ├── selection.py         (89 LOC)  - select_best_5d_subset (x1-anchored), VIF filters
│   ├── diagnostics.py       (65 LOC)  - evaluate_model_diagnostics (Health panel)
│   ├── synthetic.py         (45 LOC)  - generate_mock_df for offline tests
│   ├── trajectory.py        (38 LOC)  - Backwards-compatible facade
│   └── __init__.py          (35 LOC)  - Re-exports
├── calibrators/
│   ├── classic.py           (178 LOC) - NC, TS (bounded), 1D Platt, Spline, ATS
│   ├── proxies.py           (163 LOC) - MSSC, MSE-EIGEN, UQLM proxy baseline estimators
│   ├── vcps.py              - VaryingCoefficientPlattScaler (VCPS-17D 34 params, VCPS-5D 10 params, Analytical L-BFGS-B)
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

Runs the entire calibrator pipeline across all 17 benchmark methods on synthetic pilot features with zero GPU requirements:

```bash
# Windows
.\.venv\Scripts\python scripts/run_mock.py

# Linux / MacOS
./.venv/bin/python scripts/run_mock.py
```

### 3. Unified End-to-End Benchmark & Reporting Pipeline

To execute all calibration benchmarks (comprehensive 14-dataset evaluation, 10-seed CV, temperature transfer, LODO, feature ablation, and VCPS dynamic slope analysis) across all architectures (`m3`, `mqt`) and decoding temperatures, followed by programmatic generation of all markdown logs, LaTeX tables, publication figures, and verified `README.md` tables with real-time flushed output:

```bash
# Run complete end-to-end benchmark & documentation pipeline
uv run python scripts/run_pipeline.py

# Inspect planned stages and commands without execution
uv run python scripts/run_pipeline.py --dry-run

# Run only table, log, figure, and README generation from existing experiment CSVs
uv run python scripts/run_pipeline.py --skip-calibration
```

### 4. Run Pytest Suite

```bash
uv run pytest
```

### 5. VCPS Python API Usage

```python
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS, FEATURE_KEYS

# VCPS-17D: Dynamically binds all 16 trajectory signatures (34 parameters total)
vcps_17d = VaryingCoefficientPlattScaler(feature_set="17d")
vcps_17d.fit(X_train_17d, y_train, feature_names=FEATURE_KEYS)
probs_17d = vcps_17d.predict_proba(X_test_17d)

# VCPS-5D: Binds canonical 4-signature subset (10 parameters total, 5-feature input: x1 + 4 signatures)
vcps_5d = VaryingCoefficientPlattScaler(feature_set="5d")
vcps_5d.fit(X_train_5d, y_train, feature_names=CANONICAL_5D_KEYS)
probs_5d = vcps_5d.predict_proba(X_test_5d)
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
| Platt Scaling (1D) | **5.19%** | 2.35% | 2.39% | 6.70% | 1.13% | 11.89% | 5.50% | *3.44%* | 4.33% | 7.46% | 5.98% | 6.11% | 7.69% | 9.71% |
| Trajectory LR | 8.73% | 3.27% | 2.94% | *4.61%* | **1.02%** | 10.29% | 5.28% | 3.70% | 4.74% | **4.65%** | 5.61% | 5.88% | 7.07% | 7.87% |
| Trajectory LR (No Bias) | 8.94% | 6.12% | 5.36% | 7.34% | 1.44% | 9.75% | 7.02% | 3.51% | 4.73% | 6.33% | 5.99% | 6.59% | *6.22%* | 6.48% |
| Trajectory Platt (5D) | 9.53% | 3.27% | 2.93% | **4.47%** | *1.03%* | 10.29% | 5.29% | 3.65% | 4.73% | *4.65%* | *5.61%* | 5.87% | 7.09% | 7.87% |
| Trajectory Platt (17D) | 9.38% | 3.55% | *2.28%* | 6.19% | 1.29% | 9.85% | 5.20% | 3.60% | 5.52% | 5.44% | **4.57%** | 6.02% | **4.08%** | **4.69%** |
| Spline Calibration | *7.68%* | 3.80% | **1.93%** | 6.44% | 1.23% | 12.33% | **4.81%** | **3.03%** | 3.82% | 7.51% | 7.60% | *4.86%* | 8.37% | 8.21% |
| Adaptive TS (ATS) | 10.15% | 45.05% | 48.84% | 10.96% | 53.89% | 34.66% | 9.95% | 52.19% | **3.76%** | 9.41% | 8.52% | 8.46% | 15.40% | 9.07% |
| Residual Calibrator | 8.37% | 2.82% | 2.42% | 8.55% | 1.13% | **8.53%** | 7.76% | 3.46% | *3.80%* | 5.44% | 9.33% | 6.14% | 7.38% | 7.77% |
| **VCPS-5D (Our Method)** | 11.36% | **2.03%** | 2.53% | 5.98% | 1.31% | 9.94% | 5.38% | 3.45% | 4.01% | 4.83% | 6.82% | 6.12% | 7.40% | 6.98% |
| **VCPS-17D (Our Method)** | 9.57% | *2.22%* | 3.11% | 5.25% | 1.22% | *9.19%* | *4.99%* | 3.47% | 3.86% | 6.62% | 7.82% | **4.02%** | 6.68% | *4.82%* |

- **VCPS vs. Global Temperature Scaling (TS)**:
  - **Family Win**: VCPS beats TS on **12 / 14 datasets** (all except `ai2d`, `seedbench`).
  - **Separate Counts**: VCPS-17D alone beats TS on **12 / 14 datasets**; VCPS-5D alone beats TS on **12 / 14 datasets**.
- **VCPS vs. 1D Platt Scaling**:
  - **Family Win**: VCPS beats 1D Platt Scaling on **9 / 14 datasets**:
    `chartqa`, `gqa`, `lego-puzzles`, `mmbench`, `pope`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
  - **Separate Counts**: VCPS-17D alone beats 1D Platt on **9 / 14 datasets**; VCPS-5D alone beats 1D Platt on **8 / 14 datasets**.
- **Trajectory Platt vs. Global Temperature Scaling (TS)**:
  - **Family Win**: Trajectory Platt beats TS on **12 / 14 datasets** (all except `ai2d`, `pope`).
  - **Separate Counts**: Trajectory Platt (17D) alone beats TS on **12 / 14 datasets**; Trajectory Platt (5D) alone beats TS on **12 / 14 datasets**.
- **Trajectory Platt vs. 1D Platt Scaling**:
  - **Family Win**: Trajectory Platt beats 1D Platt Scaling on **10 / 14 datasets**:
    `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `scienceqa`, `seedbench`, `textvqa`, `vizwiz-vqa`, `vqav2`.
  - **Separate Counts**: Trajectory Platt (17D) alone beats 1D Platt on **9 / 14 datasets**; Trajectory Platt (5D) alone beats 1D Platt on **9 / 14 datasets**.
- **Our Trajectory Methods (VCPS-5D (Our Method), VCPS-17D (Our Method), Trajectory Platt (5D), Trajectory Platt (17D), Residual Calibrator)** win the #1 lowest Adaptive ECE on **7 / 14 benchmarks**:
  `chartqa` (VCPS-5D: **2.03%**), `gqa` (TP-5D: **4.47%**), `lego-puzzles` (Residual: **8.53%**), `seedbench` (TP-17D: **4.57%**), `textvqa` (VCPS-17D: **4.02%**), `vizwiz-vqa` (TP-17D: **4.08%**), `vqav2` (TP-17D: **4.69%**).
---

### MQT-LLaVA: Adaptive ECE (%) [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | gqa | infographicvqa | lego-puzzles | mmbench | mmmu | pope | scienceqa | seedbench | textvqa | vizwiz-vqa | vqav2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 24.32% | 62.99% | 51.18% | *10.19%* | 75.51% | 33.80% | 15.27% | 52.24% | 6.18% | 23.04% | 17.74% | 30.96% | 30.26% | 10.93% |
| Temperature Scaling (TS) | 13.01% | 41.54% | 46.63% | 10.61% | 52.63% | 23.68% | 10.24% | 49.32% | 5.44% | 10.84% | 10.44% | 18.67% | 28.77% | 16.04% |
| Platt Scaling (1D) | *9.51%* | 8.03% | 4.56% | 10.60% | 1.11% | 12.43% | 8.38% | 2.14% | *4.79%* | 6.78% | 8.62% | 11.82% | 5.76% | 14.95% |
| Trajectory LR | 13.53% | 6.79% | *3.59%* | 13.93% | 1.09% | 9.07% | 10.04% | **2.03%** | 5.55% | *5.29%* | 9.67% | 7.54% | 6.81% | **6.67%** |
| Trajectory LR (No Bias) | 9.91% | 12.28% | 6.20% | 13.20% | 1.79% | **6.98%** | 9.46% | 3.15% | 5.81% | 7.99% | 10.14% | 9.49% | 5.80% | 9.44% |
| Trajectory Platt (5D) | 12.48% | 6.80% | 3.60% | 13.94% | 1.10% | 8.07% | 10.11% | *2.04%* | 5.09% | **5.09%** | 9.65% | 7.54% | 6.81% | *6.68%* |
| Trajectory Platt (17D) | 13.62% | 5.12% | 3.96% | **9.18%** | **0.97%** | 10.85% | **5.82%** | 2.08% | **4.21%** | 5.69% | 8.42% | 7.74% | **3.46%** | 7.94% |
| Spline Calibration | **8.64%** | **4.71%** | 4.22% | 10.31% | *1.08%* | 8.21% | 7.75% | 2.13% | 6.42% | 7.04% | 8.82% | **5.43%** | *4.29%* | 10.70% |
| Adaptive TS (ATS) | 14.95% | 41.53% | 46.62% | 11.45% | 52.61% | 23.68% | 10.13% | 49.31% | 6.05% | 10.04% | 10.41% | 18.05% | 28.61% | 12.47% |
| Residual Calibrator | 11.91% | 7.15% | 4.43% | 12.55% | 1.11% | 8.40% | 8.13% | 2.15% | 5.27% | 5.61% | *6.48%* | 8.73% | 5.82% | 11.73% |
| **VCPS-5D (Our Method)** | 11.36% | *4.76%* | 4.42% | 11.40% | 1.10% | 9.42% | 7.66% | 2.15% | 4.82% | 5.50% | **6.38%** | 8.62% | 5.18% | 11.79% |
| **VCPS-17D (Our Method)** | 12.30% | 7.05% | **3.05%** | 10.20% | 1.10% | *7.40%* | *6.68%* | 2.14% | 5.93% | 5.37% | 10.03% | *7.36%* | 5.63% | 7.74% |

- **VCPS vs. Global Temperature Scaling (TS)**:
  - **Family Win**: VCPS beats TS on **14 / 14 datasets**.
  - **Separate Counts**: VCPS-17D alone beats TS on **13 / 14 datasets**; VCPS-5D alone beats TS on **13 / 14 datasets**.
- **VCPS vs. 1D Platt Scaling**:
  - **Family Win**: VCPS beats 1D Platt Scaling on **11 / 14 datasets**:
    `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `scienceqa`, `seedbench`, `textvqa`, `vizwiz-vqa`, `vqav2`.
  - **Separate Counts**: VCPS-17D alone beats 1D Platt on **10 / 14 datasets**; VCPS-5D alone beats 1D Platt on **10 / 14 datasets**.
- **Trajectory Platt vs. Global Temperature Scaling (TS)**:
  - **Family Win**: Trajectory Platt beats TS on **14 / 14 datasets**.
  - **Separate Counts**: Trajectory Platt (17D) alone beats TS on **13 / 14 datasets**; Trajectory Platt (5D) alone beats TS on **13 / 14 datasets**.
- **Trajectory Platt vs. 1D Platt Scaling**:
  - **Family Win**: Trajectory Platt beats 1D Platt Scaling on **13 / 14 datasets**:
    `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `mmmu`, `pope`, `scienceqa`, `seedbench`, `textvqa`, `vizwiz-vqa`, `vqav2`.
  - **Separate Counts**: Trajectory Platt (17D) alone beats 1D Platt on **13 / 14 datasets**; Trajectory Platt (5D) alone beats 1D Platt on **8 / 14 datasets**.
- **Our Trajectory Methods (VCPS-5D (Our Method), VCPS-17D (Our Method), Trajectory Platt (5D), Trajectory Platt (17D), Residual Calibrator)** win the #1 lowest Adaptive ECE on **8 / 14 benchmarks**:
  `docvqa` (VCPS-17D: **3.05%**), `gqa` (TP-17D: **9.18%**), `infographicvqa` (TP-17D: **0.97%**), `mmbench` (TP-17D: **5.82%**), `pope` (TP-17D: **4.21%**), `scienceqa` (TP-5D: **5.09%**), `seedbench` (VCPS-5D: **6.38%**), `vizwiz-vqa` (TP-17D: **3.46%**).
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

### Macro-Average Comparison Across 17 Methods on 14 Datasets ($T_{\text{gen}} = 0.00$, $1\times$ Compute)

| Model | Calibration Method | Paradigm / Regime | Sampling | Macro ECE (%) $\downarrow$ | Macro Ada-ECE (%) $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **M3-LLaVA** | Quadratic Platt (Logit-Only) | Polynomial Logit | Greedy ($T=0.0, K=1$) | **3.22%** | **4.95%** | 0.678 |
|  | **VCPS-17D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 4.11% | 5.20% | 0.698 |
|  | Spline Calibration | Non-Parametric (Isotonic) | Greedy ($T=0.0, K=1$) | 3.45% | 5.83% | 0.669 |
|  | **VCPS-5D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 4.01% | 5.58% | 0.695 |
|  | Trajectory LR | Linear Trajectory | Greedy ($T=0.0, K=1$) | 4.13% | 5.40% | *0.721* |
|  | Platt Scaling (1D) | Classic Linear Post-Hoc | Greedy ($T=0.0, K=1$) | *3.24%* | 5.70% | 0.682 |
|  | Residual Calibrator | Feature-Aided | Greedy ($T=0.0, K=1$) | 4.09% | 5.92% | 0.701 |
|  | Trajectory LR (No Bias) | Linear Trajectory (Zero-Bias) | Greedy ($T=0.0, K=1$) | 4.65% | 6.13% | 0.692 |
|  | Trajectory Platt (5D) | Platt on 5D features | Greedy ($T=0.0, K=1$) | 4.09% | 5.45% | **0.722** |
|  | Trajectory Platt (17D) | Platt on 17D features | Greedy ($T=0.0, K=1$) | 4.18% | *5.12%* | 0.715 |
|  | Temperature Scaling (TS) | Classic Post-Hoc | Greedy ($T=0.0, K=1$) | 21.93% | 22.68% | 0.671 |
|  | Adaptive TS (ATS) | Adaptive Calibrator | Greedy ($T=0.0, K=1$) | 21.87% | 22.88% | 0.684 |
|  | Naive Confidence (NC) | Uncalibrated Baseline | Greedy ($T=0.0, K=1$) | 40.33% | 40.33% | 0.671 |
|  | `umpire` | Multi-Pass Semantic Volume | Stochastic ($T=0.5, K=10$) | 17.34% | - | 0.698 |
|  | `eigen_score` | SVD Covariance Dispersion | Stochastic ($T=0.5, K=10$) | 24.74% | - | 0.694 |
|  | `ln_entropy` | Predictive Entropy | Stochastic ($T=0.5, K=10$) | 19.24% | - | 0.675 |
|  | `semantic_entropy` | DeBERTa NLI Clustering | Stochastic ($T=0.5, K=10$) | 28.25% | - | 0.641 |
| **MQT-LLaVA** | Quadratic Platt (Logit-Only) | Polynomial Logit | Greedy ($T=0.0, K=1$) | 4.90% | 6.53% | 0.706 |
|  | **VCPS-17D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | *4.89%* | 6.57% | 0.716 |
|  | Spline Calibration | Non-Parametric (Isotonic) | Greedy ($T=0.0, K=1$) | **3.21%** | *6.41%* | 0.693 |
|  | **VCPS-5D (Our Method)** | Trajectory Calibration | Greedy ($T=0.0, K=1$) | 5.30% | 6.75% | 0.701 |
|  | Trajectory LR | Linear Trajectory | Greedy ($T=0.0, K=1$) | 5.87% | 7.26% | 0.755 |
|  | Platt Scaling (1D) | Classic Linear Post-Hoc | Greedy ($T=0.0, K=1$) | 5.88% | 7.82% | 0.686 |
|  | Residual Calibrator | Feature-Aided | Greedy ($T=0.0, K=1$) | 5.36% | 7.11% | 0.699 |
|  | Trajectory LR (No Bias) | Linear Trajectory (Zero-Bias) | Greedy ($T=0.0, K=1$) | 6.40% | 7.97% | 0.640 |
|  | Trajectory Platt (5D) | Platt on 5D features | Greedy ($T=0.0, K=1$) | 5.92% | 7.07% | *0.756* |
|  | Trajectory Platt (17D) | Platt on 17D features | Greedy ($T=0.0, K=1$) | 5.20% | **6.36%** | **0.771** |
|  | Temperature Scaling (TS) | Classic Post-Hoc | Greedy ($T=0.0, K=1$) | 23.45% | 24.13% | 0.682 |
|  | Adaptive TS (ATS) | Adaptive Calibrator | Greedy ($T=0.0, K=1$) | 22.97% | 23.99% | 0.684 |
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
| **M3-LLaVA** | Spline Calibration (PCHIP) | **4.53%** | **16.10%** | **7.75%** | 6.54% | 10.85% | **9.15%** |
|  | Quadratic Platt (Logit-Only) | 4.71% | *19.49%* | *9.57%* | *5.31%* | **7.54%** | *9.32%* |
|  | Platt Scaling (1D) | 4.77% | 19.62% | 9.65% | **4.75%** | *8.36%* | 9.43% |
|  | Residual Calibrator | 4.71% | 19.89% | 10.24% | 6.25% | 11.02% | 10.42% |
|  | **VCPS-17D (Our Method)** | *4.63%* | 20.02% | 9.78% | 7.08% | 12.00% | 10.70% |
|  | **VCPS-5D (Our Method)** | 4.79% | 20.17% | 10.63% | 6.82% | 11.79% | 10.84% |
|  | Temperature Scaling (TS) | 9.16% | 22.14% | 13.87% | 9.41% | 11.33% | 13.18% |
|  | Adaptive TS (ATS) | 8.83% | 22.51% | 14.38% | 9.41% | 11.63% | 13.35% |
|  | Trajectory Platt (17D) | 4.95% | 21.18% | 13.88% | 12.12% | 16.29% | 13.68% |
|  | Naive Confidence (NC) | 10.03% | 23.98% | 16.05% | 9.82% | 9.73% | 13.92% |
|  | Trajectory LR (No Bias) | 4.79% | 21.08% | 13.73% | 13.04% | 17.56% | 14.04% |
|  | Trajectory Platt (5D) | 4.71% | 21.14% | 14.03% | 15.45% | 19.82% | 15.03% |
|  | Trajectory LR | 4.81% | 21.23% | 14.13% | 15.44% | 19.74% | 15.07% |
| **MQT-LLaVA** | Spline Calibration (PCHIP) | **3.38%** | **16.16%** | **9.72%** | **5.38%** | 8.49% | **8.63%** |
|  | Trajectory Platt (5D) | 5.32% | *20.59%* | 10.76% | 5.71% | **7.46%** | *9.97%* |
|  | Trajectory LR | 5.22% | 20.76% | 10.65% | 5.80% | 7.58% | 10.00% |
|  | Trajectory Platt (17D) | *5.17%* | 21.83% | 10.38% | *5.41%* | 7.61% | 10.08% |
|  | Quadratic Platt (Logit-Only) | 5.50% | 21.60% | *10.20%* | 6.13% | *7.53%* | 10.19% |
|  | **VCPS-17D (Our Method)** | 6.13% | 23.51% | 12.24% | 5.59% | 7.71% | 11.04% |
|  | **VCPS-5D (Our Method)** | 5.99% | 24.04% | 13.15% | 6.23% | 8.12% | 11.51% |
|  | Residual Calibrator | 6.23% | 24.49% | 13.39% | 6.70% | 8.38% | 11.84% |
|  | Platt Scaling (1D) | 7.15% | 24.62% | 13.47% | 7.46% | 8.74% | 12.29% |
|  | Trajectory LR (No Bias) | 5.91% | 26.70% | 16.72% | 9.62% | 9.23% | 13.64% |
|  | Adaptive TS (ATS) | 15.41% | 28.21% | 21.98% | 18.88% | 20.75% | 21.05% |
|  | Temperature Scaling (TS) | 15.68% | 28.00% | 21.90% | 19.27% | 20.85% | 21.14% |
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

## 17-D Trajectory Feature Space (1 Base Anchor + 16 Trajectory Signatures: $x_1 \dots x_{17}$)

| Key | Feature Name | Formula | Scientific Meaning |
| :--- | :--- | :--- | :--- |
| **`x1`** | **Final Logit** | $\ln(c_{\text{fine}} / (1 - c_{\text{fine}}))$ | Primary uncalibrated confidence anchor ($m=576/256$) |
| **`x2`** | **Monotonicity Count** | $\sum_{i=1}^4 \mathbb{I}(c_{m_{i+1}} > c_{m_i})$ | Monotonic confidence trajectory consistency |
| **`x3`** | **Discrete Answer Stability** | $1 / \text{UniqueAnswers}$ | Inverse count of distinct decoded strings across 5 scales |
| **`x4`** | **Scale Entropy Slope** | OLS slope of binary entropy $H(c_m)$ vs $\ln m$ | Rate of information gain with resolution |
| **`x5`** | **Logprob Variance** | $\text{Var}([\ln c_1, \dots, \ln c_{\text{fine}}])$ | Log-likelihood stability across scales |
| **`x6`** | **Scale Dip Depth** | $\max(0, \max(c_1, c_9) - \min(c_{36}, c_{144}))$ | Mid-scale visual confusion indicator |
| **`x7`** | **Answer Flip Frequency** | $\frac{1}{4} \sum_{i=1}^4 \mathbb{I}(\text{ans}(m_i) \neq \text{ans}(m_{i+1}))$ | Textual prediction volatility across scales |
| **`x8`** | **Mid-Fine Gain Contrast** | $(c_{\text{fine}} - c_{144}) - (c_{144} - c_9)$ | Second discrete derivative on confidences |
| **`x9`** | **Log-Scale Slope** | $\frac{\sum (\ln m_i - \overline{\ln m})(c_{m_i} - \bar{c})}{\sum (\ln m_i - \overline{\ln m})^2}$ | Logarithmic rate of confidence growth |
| **`x10`** | **Confidence Gain** | $c_{\text{fine}} - c_9$ | Visual resolution sensitivity (fine minus coarse) |
| **`x11`** | **Logprob Acceleration** | $(\ln c_{\text{fine}} - \ln c_{144}) - (\ln c_{144} - \ln c_{36})$ | Discrete 2nd derivative of log-confidence |
| **`x12`** | **First-to-Final Jump Ratio** | $(c_{\text{fine}} - c_1) / (c_{\text{fine}} + \epsilon)$ | Relative span from single-token to full scale |
| **`x13`** | **Confidence Variance** | $\text{Var}([c_1, c_9, c_{36}, c_{144}, c_{\text{fine}}])$ | Fluctuation/dispersion across visual scales |
| **`x14`** | **Relative Gain Ratio** | $c_{\text{fine}} / (c_9 + \epsilon)$ | Multiplicative confidence enhancement ratio |
| **`x15`** | **End-Scale Spike Ratio** | $c_{\text{fine}} - \frac{1}{4}\sum_{i=1}^4 c_{m_i}$ | Sudden fine-scale confidence jump |
| **`x16`** | **Logprob Gain** | $\ln c_{\text{fine}} - \ln c_9$ | Probability magnitude shift in log-space |
| **`x17`** | **Relative Margin Growth** | $\text{margin}_{\text{fine}} / (\text{margin}_9 + \epsilon)$ | Top-1 vs Top-2 separation growth |

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

---

## Development & Quality Toolchain

The project enforces strict code hygiene, fast Rust-based linting/formatting, static typing, and property testing:

```powershell
# 1. Install / sync dependencies
uv sync --extra dev

# 2. Install pre-commit git hooks
uv run pre-commit install

# 3. Format and lint code (Astral Ruff)
uv run ruff format .
uv run ruff check --fix .

# 4. Static type checking (Pyright)
uv run pyright

# 5. Run unit & Hypothesis property-based tests
uv run pytest
```
