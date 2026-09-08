# Domain Model: Trajectory Calibration

## Context Boundaries

This project focuses on **Elastic Multi-Scale Uncertainty Quantification (UQ)** and **Varying-Coefficient Platt Scaling (VCPS)** for Matryoshka Multimodal Language Models (**M3-LLaVA** and **MQT-LLaVA**).

---

## Ubiquitous Language & Core Terminology

### 1. Model Architectures
- **M3-LLaVA**: Matryoshka Multimodal LLaVA using 2D spatial average pooling over visual tokens across scales  \in \{1, 9, 36, 144, 576\}$.
- **MQT-LLaVA**: Matryoshka Query Transformer LLaVA using fixed token budgets across scales  \in \{1, 9, 36, 144, 256\}$.

### 2. Features & Trajectories
- **$\ell$ Anchor Invariant ($x_1$)**: The base uncalibrated log-odds anchor $\ell = x_1 = \ln(c_{\text{fine}} / (1 - c_{\text{fine}}))$ at fine visual scale. It must never be ablated or standardized with mean subtraction, preserving $\ell = 0 \iff c = 0.5$.
- **Trajectory Signatures ($\mathbf{z} = [x_2, \dots, x_{17}]$)**: 16 scalar signatures capturing cross-scale monotonicity counts, answer stability, entropy decay, variance, log-scale slope, acceleration, and confidence gains across visual resolutions, ordered contiguously by calibration importance.
- **VCPS-17D**: Varying-Coefficient Platt Scaling operating over the 17-D feature representation ($x_1$ anchor + $\mathbf{z} \in \mathbb{R}^{16}$) binding all 16 trajectory signatures into dynamic slope $a(\mathbf{z}) = \exp(\alpha_0 + \mathbf{z}^T \boldsymbol{\gamma})$ and dynamic intercept $b(\mathbf{z}) = b_0 + \mathbf{z}^T \mathbf{w}$ (34 parameters total).
- **VCPS-5D**: Parsimonious subset binding 5 signatures (defaulting to top 5 importance-ranked signatures $\mathbf{z}_{1..5} = [x_2, \dots, x_6]$, or forward-stepwise selected signatures) into slope and intercept (12 parameters total).
- **Trajectory Platt Scaler**: Baseline calibrator generalizing 1D Platt scaling to multi-dimensional trajectory features via $p = \sigma(\mathbf{a}^T \mathbf{z} + b)$, optimizing explicit `torch.nn.Parameter` objects $\mathbf{a}$ and $b$ via `torch.optim.LBFGS` (benchmarking 5D and 17D variants).

### 3. Evaluation Metrics & Benchmarks
- **Adaptive ECE (Ada-ECE)**: Equal-frequency (quantile partitioned) Expected Calibration Error, preventing empty bin collapse. Lower is better.
- **ECE**: Standard equal-width binning Expected Calibration Error.
- **AUROC**: Area Under the Receiver Operating Characteristic curve measuring selective prediction discrimination between correct and incorrect answers. Higher is better.
- **LODO (Leave-One-Dataset-Out)**: 14-fold cross-dataset evaluation protocol where calibrators train on 13 pooled benchmarks and evaluate zero-shot on the 14th held-out benchmark.
- **14 Vision-Language Benchmarks**: ai2d, chartqa, docvqa, gqa, infographicvqa, lego-puzzles, mmbench, mmmu, pope, scienceqa, seedbench, textvqa, vizwiz-vqa, vqav2.
- **17 Benchmark Calibration Methods**: Naive Confidence (NC), Temperature Scaling (TS), Platt Scaling (1D), Trajectory LR, Trajectory LR (No Bias), Trajectory Platt (5D), Trajectory Platt (17D), Quadratic Platt (Logit-Only), Spline Calibration, Adaptive TS (ATS), LN-Entropy, Semantic Entropy, EigenScore, UMPIRE, Residual Calibrator, VCPS-5D (Our Method), VCPS-17D (Our Method).

### 4. Ranking & Presentation Rules (Option A Adopted)
- **Universal Application**: Apply Rank 1 (**bold**) and Rank 2 (*italic*) formatting to all metric columns across all tables:
  - Per-Architecture Ada-ECE Tables: every dataset column (lower is better $\\downarrow$).
  - Macro-Average Summary Table: Macro ECE ($\\downarrow$), Macro Ada-ECE ($\\downarrow$), and Macro AUROC ($\\uparrow$). Multi-pass baselines reporting '-' for Ada-ECE are skipped during Ada-ECE ranking.
  - Temperature Robustness Table: across all temperature columns and Mean ECE ($\\downarrow$).
- **LaTeX Tables**: Format Rank 1 as \\textbf{...} and Rank 2 as \\textit{...}.
- **Comparative Metrics**: Win statistics (e.g., VCPS vs. TS, VCPS vs. 1D Platt, #1 rankings across 14 benchmarks) must be programmatically tallied and updated in prose directly below the benchmark tables.\n- **Win Reporting Convention**: Report both the **Family Win** ($\\min(\\text{VCPS-17D}, \\text{VCPS-5D}) < \\text{Baseline}$) AND **Separate Counts** (VCPS-17D vs. Baseline, and VCPS-5D vs. Baseline) for complete transparency under each benchmark table.
