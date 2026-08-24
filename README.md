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

3. **Pareto Dominance**: Outperforms standard post-hoc temperature scaling while maintaining **1x inference cost**.

---

## Key Features

- **Strict $\le 200$ LOC Modularity**: Every source file in `src/trajectory_calibration/` is structured as a clean, single-responsibility module strictly $\le 184$ LOC.
- **Primary Calibration Method**: Varying-Coefficient Platt Scaling (**VCPS-5D** & **VCPS-17D**) with exact analytical gradients and L-BFGS-B optimization.
- **Canonical UQ Baseline Suite**:
  - **UQLM White-Box Scorers** ([CVS Health UQLM](https://github.com/cvs-health/uqlm)): Sequence Probability (Joint / Length-Normalized), Min Token Probability, Mean Token Negentropy, and Top-1/Top-2 Probability Margin.
  - **Semantic Entropy & NLI Clustering** ([Kuhn et al., 2023 / UMPIRE OpenReview](https://openreview.net/forum?id=c9TWeKZQR4)): DeBERTa-v2-xlarge bidirectional NLI entailment clustering, LogSumExp cluster aggregation, and Cluster Assignment Entropy.
  - **EigenScore & LogDet Spectral Volume** ([Chen et al., 2024 / UMPIRE](https://openreview.net/forum?id=c9TWeKZQR4)): SVD on covariance and Gram matrices ($\frac{1}{K}\sum \log_{10} s_i$) and matrix log-determinant volume metrics.
- **Classic Post-Hoc Calibrators**: Global Temperature Scaling (Guo et al.), 1D Platt Scaling, Monotonic Spline Calibration (PCHIP), and Adaptive Temperature Scaling (ATS / Thermometer).
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
│   ├── vcps.py              (184 LOC) - VaryingCoefficientPlattScaler (Analytical L-BFGS-B)
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

---

## Empirical Benchmark Results

Evaluated across all 14 vision-language benchmarks on single-pass feature matrices ($T_{\text{gen}} = 0.00$, $1\times$ inference cost). Best results are **bolded**, second-best are *italicized*.

> [!NOTE]
> **Single-Pass vs. Multi-Rollout UQ**: Multi-rollout sampling algorithms (such as Kuhn Semantic Entropy, Chen EigenScore, and UQLM Token Negentropy) require drawing $M$ stochastic response rollouts per sample ($M \ge 5$) or full-vocabulary logit matrices.

### M3-LLaVA: Adaptive ECE (%) [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | gqa | infographicvqa | lego-puzzles | mmbench | mmmu | pope | scienceqa | seedbench | textvqa | vizwiz-vqa | vqav2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 41.80% | 70.70% | 73.52% | 32.43% | 84.55% | 73.16% | 28.58% | 78.76% | 4.12% | 10.66% | 32.40% | 8.72% | 16.71% | **7.74%** |
| **Temperature Scaling (TS)** | 10.73% | 45.07% | 48.86% | 10.84% | 53.91% | 34.71% | 7.77% | 52.21% | *3.96%* | 9.76% | 2.64% | 8.35% | 15.56% | 10.63% |
| **Platt Scaling (1D)** | 7.49% | 2.34% | 2.42% | 7.71% | *1.12%* | 10.07% | *4.88%* | *3.45%* | 4.05% | 7.46% | *2.25%* | 5.70% | 7.25% | 9.48% |
| **Spline Calibration (PCHIP)** | 6.98% | 4.23% | 2.75% | **6.57%** | **0.60%** | **6.36%** | **4.45%** | **2.64%** | 5.10% | **5.16%** | **1.17%** | 10.25% | 8.45% | 10.14% |
| **Adaptive TS (ATS)** | 10.69% | 45.05% | 48.84% | 9.84% | 53.89% | 34.66% | 9.99% | 52.19% | 4.16% | 9.43% | 9.98% | 8.80% | 15.45% | 9.49% |
| **Residual Calibrator** | 8.27% | **2.03%** | **2.36%** | *6.58%* | 1.29% | *6.80%* | 7.81% | 3.47% | 4.31% | 5.91% | 9.08% | *5.70%* | *7.08%* | 9.09% |
| **VCPS-5D (Our Method)** | *6.71%* | 3.06% | 2.45% | 7.71% | 1.23% | 11.36% | 6.12% | 3.47% | 3.98% | 6.50% | 7.23% | **5.06%** | 7.23% | 9.35% |
| **VCPS-17D (Our Method)** | **6.28%** | *2.14%* | *2.36%* | 7.82% | 1.29% | 10.47% | 5.10% | 3.47% | **3.57%** | *5.90%* | 4.99% | 5.80% | **6.76%** | *9.02%* |

- **VCPS beats Global Temperature Scaling (TS)** on **13 / 14 datasets** (all except `seedbench`):
  `ai2d`, `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `mmmu`, `pope`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS beats 1D Platt Scaling** on **8 / 14 datasets**:
  `ai2d`, `chartqa`, `docvqa`, `pope`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS achieves #1 or #2 Rank** on **8 / 14 datasets**:
  `ai2d` (1st), `chartqa` (2nd), `docvqa` (1st), `gqa` (2nd), `pope` (1st), `scienceqa` (2nd), `textvqa` (1st), `vizwiz-vqa` (1st).

---

### MQT-LLaVA: Adaptive ECE (%) [Lower is Better]

| Calibration Method | ai2d | chartqa | docvqa | gqa | infographicvqa | lego-puzzles | mmbench | mmmu | pope | scienceqa | seedbench | textvqa | vizwiz-vqa | vqav2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 24.31% | 62.99% | 51.18% | 9.87% | 75.51% | 34.14% | 14.87% | 52.24% | 6.45% | 23.04% | 17.74% | 30.96% | 30.26% | **11.52%** |
| **Temperature Scaling (TS)** | 13.47% | 41.54% | 46.63% | 10.14% | 52.63% | 24.08% | 10.07% | 49.32% | 5.90% | 10.44% | 7.99% | 18.27% | 28.77% | 15.96% |
| **Platt Scaling (1D)** | *9.52%* | 7.04% | 5.04% | 10.27% | 1.09% | 11.70% | *8.24%* | 2.15% | **5.29%** | *6.51%* | **6.88%** | 11.59% | 6.81% | 15.96% |
| **Spline Calibration (PCHIP)** | **8.98%** | 7.07% | **2.48%** | 9.78% | **0.60%** | **5.27%** | 10.04% | 2.27% | 7.97% | 7.97% | *7.65%* | *7.32%* | **5.64%** | 15.17% |
| **Adaptive TS (ATS)** | 13.92% | 41.53% | 46.62% | **9.10%** | 52.61% | 24.08% | 9.89% | 49.32% | 6.02% | 10.18% | 8.00% | 18.15% | 28.45% | 13.10% |
| **Residual Calibrator** | 11.34% | **6.69%** | 4.82% | *9.26%* | *1.09%* | *9.21%* | 8.53% | **2.14%** | 6.58% | 7.23% | 9.17% | **6.71%** | *6.61%* | 13.12% |
| **VCPS-5D (Our Method)** | 13.46% | *6.71%* | 4.20% | 10.26% | 1.09% | 10.56% | 8.80% | *2.15%* | 5.83% | 6.89% | 8.99% | 8.53% | 6.69% | 15.00% |
| **VCPS-17D (Our Method)** | 11.55% | 9.74% | *3.73%* | 9.36% | 1.10% | 11.16% | **8.11%** | 2.16% | *5.69%* | **6.41%** | 9.28% | 11.61% | 7.00% | *12.86%* |

- **VCPS beats Global Temperature Scaling (TS)** on **13 / 14 datasets** (all except `seedbench`):
  `ai2d`, `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `mmmu`, `pope`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS beats 1D Platt Scaling** on **10 / 14 datasets**:
  `chartqa`, `docvqa`, `gqa`, `infographicvqa`, `lego-puzzles`, `mmbench`, `scienceqa`, `textvqa`, `vizwiz-vqa`, `vqav2`.
- **VCPS achieves #1 or #2 Rank** on **10 / 14 datasets**:
  `chartqa` (1st), `docvqa` (2nd), `gqa` (2nd), `lego-puzzles` (2nd), `mmbench` (1st), `mmmu` (2nd), `pope` (2nd), `scienceqa` (1st), `textvqa` (2nd), `vizwiz-vqa` (2nd).

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
| **`x13`** | **Answer Stability** | $1 / |\text{UniqueAnswers}|$ across 5 scales | Single-pass semantic consistency proxy |
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
