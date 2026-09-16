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
- **VCPS-5D (Canonical 5D Subset)**: Binds the top 4 canonical trajectory signatures ($x_2, x_3, x_4, x_5$) for both dynamic slope $\boldsymbol{\gamma}$ and dynamic intercept $\mathbf{w}$, optimizing **10 parameters** ($1 + 4 + 1 + 4 = 2K + 2$): scalar log-slope base $a_0$, 4 slope weights $\boldsymbol{\gamma}$, scalar intercept base $b_0$, and 4 intercept weights $\mathbf{w}$ across a 5-D input ($x_1 + \mathbf{z}_4$).

3. **Pareto Dominance**: Outperforms standard post-hoc temperature scaling while maintaining **1x inference cost**.

---

## Key Features

- **Strict Modularity**: Every source file in `src/trajectory_calibration/` is structured as a clean, single-responsibility module.
- **Primary Calibration Method**: Varying-Coefficient Platt Scaling (**VCPS-17D** with full 34-parameter dynamic binding across all 16 trajectory signatures, and **VCPS-5D** with canonical 10-parameter 5D binding) with exact analytical gradients and L-BFGS-B optimization.
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

###### M3-LLaVA: Adaptive ECE (%) Across Core 7 Datasets [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | scienceqa | textvqa | vizwiz-vqa | vqav2 | Macro Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | 41.80% | 70.70% | 73.52% | 10.66% | 9.07% | 16.71% | *8.06%* | 32.93% |
| Temperature Scaling (TS) | 9.38% | 45.07% | 48.86% | 9.76% | 8.92% | 16.03% | 10.51% | 21.22% |
| Platt Scaling (1D) | **5.19%** | **2.35%** | **2.39%** | *7.46%* | *6.11%* | *7.69%* | 9.71% | *5.84%* |
| **Trajectory Platt (5D) [Our Method]** | *8.83%* | *3.88%* | *2.39%* | **3.97%** | **5.91%** | **6.75%** | **7.19%** | **5.56%** |

- **Trajectory Platt (5D) vs. Global Temperature Scaling (TS)**:
  - Trajectory Platt (5D) beats TS on **7 / 7 Core datasets**.
- **Trajectory Platt (5D) vs. 1D Platt Scaling**:
  - Trajectory Platt (5D) beats 1D Platt Scaling on **4 / 7 Core datasets**: `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **Trajectory Platt (5D)** achieves the #1 lowest Adaptive ECE on **4 / 7 Core benchmarks**: `scienceqa` (**3.97%**), `textvqa` (**5.91%**), `vizwiz-vqa` (**6.75%**), `vqav2` (**7.19%**).
---

### MQT-LLaVA: Adaptive ECE (%) Across Core 7 Datasets [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | scienceqa | textvqa | vizwiz-vqa | vqav2 | Macro Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Confidence (NC) | 24.32% | 62.99% | 51.18% | 23.04% | 30.96% | 30.26% | *10.93%* | 33.38% |
| Temperature Scaling (TS) | *13.01%* | 41.54% | 46.63% | 10.84% | 18.67% | 28.77% | 16.04% | 25.07% |
| Platt Scaling (1D) | **9.51%** | *8.03%* | *4.56%* | *6.78%* | *11.82%* | **5.76%** | 14.95% | *8.77%* |
| **Trajectory Platt (5D) [Our Method]** | 13.41% | **6.27%** | **3.86%** | **5.46%** | **7.83%** | *5.98%* | **8.52%** | **7.33%** |

- **Trajectory Platt (5D) vs. Global Temperature Scaling (TS)**:
  - Trajectory Platt (5D) beats TS on **6 / 7 Core datasets** (all except `ai2d`).
- **Trajectory Platt (5D) vs. 1D Platt Scaling**:
  - Trajectory Platt (5D) beats 1D Platt Scaling on **5 / 7 Core datasets**: `chartqa`, `docvqa`, `scienceqa`, `textvqa`, `vqav2`.
- **Trajectory Platt (5D)** achieves the #1 lowest Adaptive ECE on **5 / 7 Core benchmarks**: `chartqa` (**6.27%**), `docvqa` (**3.86%**), `scienceqa` (**5.46%**), `textvqa` (**7.83%**), `vqav2` (**8.52%**).
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

### Macro-Average Calibration Benchmark Across Core 7 Datasets ($T_{\text{gen}} = 0.00$, $1\times$ Compute)

| Model | Calibration Method | Regime / Sampling | Macro ECE (%) $\downarrow$ | Macro Ada-ECE (%) $\downarrow$ | Macro Brier $\downarrow$ | Macro AUROC $\uparrow$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **M3-LLaVA** | Naive Confidence (NC) | Single-Pass ($T = 0.0$) | 32.95% | 32.93% | 0.3317 | *0.756* |
|  | Temperature Scaling (TS) | Single-Pass ($T = 0.0$) | 20.90% | 21.22% | 0.2103 | *0.756* |
|  | Platt Scaling (1D) | Single-Pass ($T = 0.0$) | **4.40%** | *5.84%* | *0.1383* | *0.756* |
|  | **Trajectory Platt (5D) [Our Method]** | Single-Pass ($T = 0.0$) | *4.80%* | **5.56%** | **0.1357** | **0.767** |
| **MQT-LLaVA** | Naive Confidence (NC) | Single-Pass ($T = 0.0$) | 33.63% | 33.38% | 0.3148 | 0.689 |
|  | Temperature Scaling (TS) | Single-Pass ($T = 0.0$) | 24.45% | 25.07% | 0.2273 | 0.689 |
|  | Platt Scaling (1D) | Single-Pass ($T = 0.0$) | *7.10%* | *8.77%* | *0.1517* | *0.698* |
|  | **Trajectory Platt (5D) [Our Method]** | Single-Pass ($T = 0.0$) | **5.96%** | **7.33%** | **0.1461** | **0.726** |

### VQAv2 Benchmark: Single-Pass Trajectory Calibration ($1\times$) vs. Multi-Rollout UMPIRE Suite ($10\times$)

| Model | Calibration Method | Paradigm / Regime | Sampling | ECE (%) $\downarrow$ | Ada-ECE (%) $\downarrow$ | Brier $\downarrow$ | AUROC $\uparrow$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **M3-LLaVA** | Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0, K=1$) | 8.51% | *8.06%* | *0.1427* | 0.756 |
|  | Temperature Scaling (TS) | Classic Post-Hoc | Single-Pass ($T=0.0, K=1$) | 10.48% | 10.51% | 0.1495 | 0.756 |
|  | Platt Scaling (1D) | Classic Linear Post-Hoc | Single-Pass ($T=0.0, K=1$) | 7.13% | 9.71% | 0.1455 | 0.756 |
|  | **Trajectory Platt (5D) [Our Method]** | Trajectory Calibration | Single-Pass ($T=0.0, K=1$) | 6.30% | **7.19%** | **0.1418** | 0.787 |
|  | `LN-Entropy` | Predictive Entropy | Multi-Pass ($T=0.5, K=10$) | 12.40% | - | - | 0.787 |
|  | `Semantic Entropy` | DeBERTa NLI Clustering | Multi-Pass ($T=0.5, K=10$) | 11.40% | - | - | 0.690 |
|  | `EigenScore` | SVD Covariance Dispersion | Multi-Pass ($T=0.5, K=10$) | *6.20%* | - | - | *0.816* |
|  | `UMPIRE` | Multi-Pass Semantic Volume | Multi-Pass ($T=0.5, K=10$) | **5.80%** | - | - | **0.819** |
| **MQT-LLaVA** | Naive Confidence (NC) | Uncalibrated Baseline | Single-Pass ($T=0.0, K=1$) | 12.14% | *10.93%* | **0.1439** | 0.746 |
|  | Temperature Scaling (TS) | Classic Post-Hoc | Single-Pass ($T=0.0, K=1$) | 15.10% | 16.04% | 0.1597 | 0.746 |
|  | Platt Scaling (1D) | Classic Linear Post-Hoc | Single-Pass ($T=0.0, K=1$) | 14.26% | 14.95% | 0.1588 | 0.746 |
|  | **Trajectory Platt (5D) [Our Method]** | Trajectory Calibration | Single-Pass ($T=0.0, K=1$) | *8.48%* | **8.52%** | *0.1514* | **0.791** |
|  | `LN-Entropy` | Predictive Entropy | Multi-Pass ($T=0.5, K=10$) | 12.30% | - | - | 0.767 |
|  | `Semantic Entropy` | DeBERTa NLI Clustering | Multi-Pass ($T=0.5, K=10$) | 13.70% | - | - | 0.699 |
|  | `EigenScore` | SVD Covariance Dispersion | Multi-Pass ($T=0.5, K=10$) | 10.20% | - | - | 0.773 |
|  | `UMPIRE` | Multi-Pass Semantic Volume | Multi-Pass ($T=0.5, K=10$) | **7.80%** | - | - | *0.790* |
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
| **M3-LLaVA** | Platt Scaling (1D) | **4.77%** | *19.62%* | *9.65%* | **4.75%** | **8.36%** | **9.43%** |
|  | **Trajectory Platt (5D) [Our Method]** | *4.80%* | **18.84%** | **9.17%** | *5.92%* | 9.77% | *9.70%* |
|  | Temperature Scaling (TS) | 9.16% | 22.14% | 13.87% | 9.41% | 11.33% | 13.18% |
|  | Naive Confidence (NC) | 10.03% | 23.98% | 16.05% | 9.82% | *9.73%* | 13.92% |
| **MQT-LLaVA** | **Trajectory Platt (5D) [Our Method]** | **4.90%** | **20.65%** | **10.41%** | **5.00%** | **7.23%** | **9.64%** |
|  | Platt Scaling (1D) | *7.15%* | *24.62%* | *13.47%* | *7.46%* | *8.74%* | *12.29%* |
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
| **`x2`** | **Discrete Answer Stability** | $1 / \text{UniqueAnswers}$ | Inverse count of distinct decoded strings across 5 scales |
| **`x3`** | **Scale Entropy Slope** | OLS slope of binary entropy $H(c_m)$ vs $\ln m$ | Rate of information gain with resolution |
| **`x4`** | **Answer Flip Frequency** | $\frac{1}{4} \sum_{i=1}^4 \mathbb{I}(\text{ans}(m_i) \neq \text{ans}(m_{i+1}))$ | Textual prediction volatility across scales |
| **`x5`** | **Monotonicity Count** | $\sum_{i=1}^4 \mathbb{I}(c_{m_{i+1}} > c_{m_i})$ | Monotonic confidence trajectory consistency |
| **`x6`** | **Logprob Variance** | $\text{Var}([\ln c_1, \dots, \ln c_{\text{fine}}])$ | Log-likelihood stability across scales |
| **`x7`** | **Mid-Fine Gain Contrast** | $(c_{\text{fine}} - c_{144}) - (c_{144} - c_9)$ | Second discrete derivative on confidences |
| **`x8`** | **Confidence Variance** | $\text{Var}([c_1, c_9, c_{36}, c_{144}, c_{\text{fine}}])$ | Fluctuation/dispersion across visual scales |
| **`x9`** | **End-Scale Spike Ratio** | $c_{\text{fine}} - \frac{1}{4}\sum_{i=1}^4 c_{m_i}$ | Sudden fine-scale confidence jump |
| **`x10`** | **Scale Dip Depth** | $\max(0, \max(c_1, c_9) - \min(c_{36}, c_{144}))$ | Mid-scale visual confusion indicator |
| **`x11`** | **First-to-Final Jump Ratio** | $(c_{\text{fine}} - c_1) / (c_{\text{fine}} + \epsilon)$ | Relative span from single-token to full scale |
| **`x12`** | **Log-Scale Slope** | $\frac{\sum (\ln m_i - \overline{\ln m})(c_{m_i} - \bar{c})}{\sum (\ln m_i - \overline{\ln m})^2}$ | Logarithmic rate of confidence growth |
| **`x13`** | **Relative Gain Ratio** | $c_{\text{fine}} / (c_9 + \epsilon)$ | Multiplicative confidence enhancement ratio |
| **`x14`** | **Logprob Acceleration** | $(\ln c_{\text{fine}} - \ln c_{144}) - (\ln c_{144} - \ln c_{36})$ | Discrete 2nd derivative of log-confidence |
| **`x15`** | **Confidence Gain** | $c_{\text{fine}} - c_9$ | Visual resolution sensitivity (fine minus coarse) |
| **`x16`** | **Relative Margin Growth** | $\text{margin}_{\text{fine}} / (\text{margin}_9 + \epsilon)$ | Top-1 vs Top-2 separation growth |
| **`x17`** | **Logprob Gain** | $\ln c_{\text{fine}} - \ln c_9$ | Probability magnitude shift in log-space |

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
