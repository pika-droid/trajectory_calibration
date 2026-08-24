# Trajectory Calibration: Elastic Multi-Scale Uncertainty Quantification for Multimodal LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 17 Passed](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)]()
[![Platform: Windows | Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

A clean, self-contained implementation of **Elastic Trajectory Uncertainty Calibration** and **Varying-Coefficient Platt Scaling (VCPS)** for Matryoshka Multimodal Models (**M3-LLaVA** and **MQT-LLaVA**).

---

## Abstract and Motivation

Multimodal Large Language Models (MLLMs) frequently suffer from overconfidence and severe miscalibration. Standard uncertainty quantification (UQ) techniques—such as multi-rollout sampling (e.g., Semantic Entropy) or Monte Carlo dropout—require **5x to 25x repetitive inference calls**, creating unacceptable computational overhead for real-time vision-language systems.

This repository implements **Trajectory Uncertainty Calibration**:
1. **Single-Pass Elastic Signatures**: By leveraging Matryoshka visual token compression ($m \in \{1, 9, 36, 144, 576\}$ for M3 or $\{1, 9, 36, 144, 256\}$ for MQT), we capture the model's confidence trajectory across visual granularities in a **single forward pass**.
2. **Varying-Coefficient Platt Scaling (VCPS)**: Dynamically modulates both the calibration slope $a(\mathbf{z})$ and intercept $b(\mathbf{z})$ as generalized linear functions of multi-scale trajectory signatures:

$$\text{logit}(p(\mathbf{x})) = a(\mathbf{z}) \cdot x_1 + b(\mathbf{z})$$

$$a(\mathbf{z}) = \exp(a_0 + \boldsymbol{\gamma}^T \mathbf{z}_{\text{slope}}), \quad b(\mathbf{z}) = b_0 + \mathbf{w}^T \mathbf{z}_{\text{intercept}}$$

3. **Pareto Dominance**: Outperforms standard post-hoc temperature scaling and matches/exceeds multi-rollout Semantic Entropy while maintaining **1x inference cost**.

---

## Key Features

- **Primary Calibration Method**: Varying-Coefficient Platt Scaling (**VCPS-5D** & **VCPS-17D**) with exact analytical gradients and L-BFGS-B optimization.
- **UQLM White-Box Scorers**: Standard token-probability estimators following the [CVS Health UQLM](https://github.com/cvs-health/uqlm) specification:
  - Sequence Probability (Joint & Length-Normalized Geometric Mean)
  - Min Probability (Bottleneck token confidence)
  - Token Negentropy ($1 - H(p_t) / \log K$)
  - Probability Margin ($p_{\text{top1}} - p_{\text{top2}}$)
- **Standard Sampling UQ Baselines**:
  - Exact Kuhn et al. (2023) / UMPIRE **Semantic Entropy** with bidirectional DeBERTa NLI clustering and LogSumExp probability aggregation.
  - Exact Chen et al. (2024) / UMPIRE **EigenScore** and SVD spectral dispersion on normalized generation embeddings.
- **Classic Post-Hoc Calibrators**: Global Temperature Scaling (Guo et al.), 1D Platt Scaling, Monotonic Spline Calibration (PCHIP), and Adaptive Temperature Scaling (ATS / Thermometer).
- **Hardened VLM Inference Wrapper**:
  - Full `sys.modules` namespace purge preventing package shadowing between M3 and MQT.
  - FlashAttention-friendly fast inference (`output_attentions=False`, 1.35 it/s vs 0.05 it/s).
  - First-token logit NaN/Inf clamping and Vicuna-1.5 `llava_v1` conversation mode enforcement.
- **Multi-Dataset Support**: 14 vision-language benchmarks with non-withheld validation splits and 10-annotator soft consensus $\min(1.0, \text{matches}/3.0)$.

---

## Repository Structure

```
trajectory_calibration/
|-- README.md                          # Comprehensive documentation
|-- pyproject.toml                     # Package specification & pytest configuration
|-- requirements.txt                   # Dependency list
|-- .gitignore
|
|-- src/trajectory_calibration/        # Core library package
|   |-- vlm/                           # VLM Inference & Multi-Dataset Evaluation
|   |   |-- wrapper.py                 # Unified M3 + MQT wrapper (FlashAttention, NaN guards)
|   |   |-- datasets.py                # 14-benchmark DATASET_REGISTRY & ground-truth evaluators
|   |   `-- llava_compat.py            # Architecture-aware LLaVA namespace router & auto-clone
|   |
|   |-- features/                      # Trajectory Feature Extraction
|   |   `-- trajectory.py              # 17-D extraction, forward stepwise 5-D selection, data loader
|   |
|   |-- uq/                            # Uncertainty Quantification Baselines
|   |   |-- whitebox.py                # UQLM: SequenceProb, MinProb, TokenEntropy, Margin
|   |   |-- semantic_entropy.py        # Kuhn / UMPIRE: Exact DeBERTa NLI clustering & LogSumExp
|   |   `-- eigenscore.py              # Chen / UMPIRE: SVD on embedding covariance & LogDet volume
|   |
|   |-- calibrators/                   # Post-Hoc Calibrators
|   |   |-- vcps.py                    # VaryingCoefficientPlattScaler (Our Method)
|   |   |-- baselines.py               # NC, TS, Platt, Spline, ATS, UQLM & Kuhn wrappers
|   |   |-- residual.py                # ResidualTrajectoryCalibrator & full metric panel
|   |   `-- adaptation.py              # Saerens-EM (2002) prior shift adaptation & Beta calibration
|   |
|   |-- metrics/                       # Evaluation Metrics & Statistics
|   |   `-- calibration.py             # ECE, Adaptive ECE, KDE-ECE, Brier, Murphy decomposition, AUROC
|   |
|   `-- utils/                         # Shared Utilities
|       `-- helpers.py                 # safe_torch_load, set_seed, clean_text, Config dataclass
|
|-- llava_src/                         # Minimal LLaVA fork (for M3 GPU inference)
|   |-- README.md
|   |-- setup.py
|   `-- llava/                         # Minimal builder, mm_utils, conversation, llava_llama
|
|-- scripts/                           # User-Facing Execution Entry Points
|   |-- run_mock.py                    # Fast CPU smoke test using pilot_features_1k (~3s)
|   |-- run_benchmark.py               # Multi-dataset benchmark across all 16 methods
|   |-- run_vcps.py                    # In-depth VCPS interpretability & dynamic slope separation
|   `-- extract_features.py            # Live GPU multi-scale extraction with checkpoint/resume
|
|-- data/features/                     # Full pre-extracted feature files (69 .pt files, 3.58 GB)
|   |-- m3_llava/                      # M3 architecture features (temp_0.0 .. temp_1.5)
|   `-- mqt_llava/                     # MQT architecture features (temp_0.0 .. temp_1.5)
|
|-- pilot_features_1k/                 # Lightweight 100-sample mock features for offline tests
|   |-- m3/mock_vqav2/full_extracted_features.pt
|   `-- mqt/mock_vqav2/full_extracted_features.pt
|
|-- results/                           # Benchmark & experiment outputs
|   `-- experiments/
|
`-- tests/                             # Pytest suite (17 unit tests, 100% pass)
    |-- test_features.py
    |-- test_whitebox.py
    |-- test_semantic_entropy.py
    |-- test_calibrators.py
    `-- test_metrics.py
```

---

## Installation and Quick Start

### 1. Environment Setup using uv

```bash
# Clone the repository
git clone https://github.com/pika-droid/trajectory_calibration.git
cd trajectory_calibration

# Create Python 3.12 virtual environment and install package in editable mode
uv venv .venv --python 3.12
uv pip install -e .
```

### 2. Fast CPU Smoke Test (~3 seconds)

Runs the entire calibrator pipeline across all 16 methods on synthetic pilot features without requiring a GPU:

```bash
# Windows
.venv\Scripts\python scripts/run_mock.py

# Linux / MacOS
.venv/bin/python scripts/run_mock.py
```

### 3. Run Test Suite

```bash
.venv\Scripts\python -m pytest tests/ -v
```

---

## Detailed Hardware Execution Guide

This codebase is optimized to run seamlessly across three distinct hardware tiers:

### Hardware Requirements Matrix

| Workflow Tier | Minimum CPU | Minimum RAM | Minimum GPU / VRAM | Expected Runtime |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Post-Hoc Calibration** | 2 cores | 4 GB | **None (CPU-Only)** | ~1-5 seconds per dataset |
| **Tier 2: Fast Smoke Verification** | 2 cores | 2 GB | **None (CPU-Only)** | ~3 seconds total |
| **Tier 3: Live VLM GPU Extraction** | 4 cores | 16 GB | 1x GPU >= 16 GB (RTX 3090/4090, A5000, A100) | ~1.35 samples / sec |

---

### Workflow 1: CPU Post-Hoc Benchmarking (No GPU Required)

Because all 69 pre-extracted multi-scale feature files are included under `data/features/`, you can fit and evaluate all 16 calibration models on local CPU hardware without downloading multi-gigabyte neural network checkpoints:

#### 1.1 Run Full M3-LLaVA Benchmark (4 Primary Datasets)
```bash
# Windows
.venv\Scripts\python scripts/run_benchmark.py \
    --features_dir data/features \
    --arch m3 \
    --gen_temperature 0.0 \
    --datasets pope scienceqa textvqa vizwiz-vqa \
    --output_dir results/experiments/benchmark_m3

# Linux / RunPod
python scripts/run_benchmark.py \
    --features_dir data/features \
    --arch m3 \
    --gen_temperature 0.0 \
    --datasets pope scienceqa textvqa vizwiz-vqa \
    --output_dir results/experiments/benchmark_m3
```

#### 1.2 Run M3-LLaVA Benchmark Across All 12 Available Datasets
```bash
python scripts/run_benchmark.py \
    --features_dir data/features \
    --arch m3 \
    --gen_temperature 0.0 \
    --datasets pope scienceqa textvqa vizwiz-vqa ai2d chartqa docvqa gqa infographicvqa lego-puzzles mmbench mmmu seedbench \
    --output_dir results/experiments/benchmark_m3_all
```

#### 1.3 Run MQT-LLaVA Benchmark (Query Transformer Architecture)
```bash
python scripts/run_benchmark.py \
    --features_dir data/features \
    --arch mqt \
    --gen_temperature 0.0 \
    --datasets pope scienceqa textvqa vizwiz-vqa \
    --output_dir results/experiments/benchmark_mqt
```

#### 1.4 Run Benchmark Across Non-Zero Sampling Temperatures ($T_{\text{gen}} \in \{0.3, 0.6, 0.9, 1.0, 1.5\}$)
```bash
# Evaluate calibration robustness under stochastic decoding (T=0.6)
python scripts/run_benchmark.py \
    --features_dir data/features \
    --arch m3 \
    --gen_temperature 0.6 \
    --datasets pope scienceqa textvqa vizwiz-vqa \
    --output_dir results/experiments/benchmark_m3_temp0.6
```

---

### Workflow 2: In-Depth VCPS Dynamic Slope and Temperature Analysis

To inspect how the Varying-Coefficient Platt Scaler adjusts instance-level temperatures $T_{\text{eff}}(\mathbf{z}) = 1/a(\mathbf{z})$ and slopes $a(\mathbf{z})$ for correct vs. incorrect model predictions:

```bash
# Analyze 5-D VCPS on TextVQA
python scripts/run_vcps.py \
    --features_dir data/features/m3_llava/temp_0.0/textvqa.pt \
    --arch m3 \
    --feature_set 5d \
    --output_dir results/experiments/vcps_analysis_textvqa

# Analyze full 17-D VCPS on ScienceQA
python scripts/run_vcps.py \
    --features_dir data/features/m3_llava/temp_0.0/scienceqa.pt \
    --arch m3 \
    --feature_set 17d \
    --output_dir results/experiments/vcps_analysis_scienceqa
```

Output highlights:
- **Slope $a(\mathbf{z})$ for Correct Predictions**: $> 1.0$ (sharpens confident correct answers).
- **Slope $a(\mathbf{z})$ for Incorrect Predictions**: $< 1.0$ (damps overconfidence, pushing uncalibrated high confidence toward base rate).
- **Effective Temperature $T_{\text{eff}}(\mathbf{z})$**: Demonstrates clear statistical separation ($p < 0.001$, Cohen's $d > 0.8$).

---

### Workflow 3: Live GPU Feature Extraction from Scratch

When deploying on a GPU instance (e.g., RunPod, Lambda Labs, AWS EC2 `g5.xlarge`, or local RTX 3090/4090):

#### 3.1 Recommended RunPod Environment Setup
```bash
# 1. Export HuggingFace cache to persistent storage volume
export HF_HOME="/workspace/.cache/huggingface"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
mkdir -p $HF_HOME $HF_DATASETS_CACHE

# 2. Install package in editable mode with GPU extras
uv pip install -e ".[gpu]"

# 3. (Optional) Install minimal LLaVA fork in editable mode
pip install -e llava_src/
```

#### 3.2 Extract Multi-Scale Features for M3-LLaVA ($m \in [1, 9, 36, 144, 576]$)
```bash
# Single GPU extraction (Device 0)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py \
    --model_path mucai/llava-v1.5-7b-m3 \
    --arch m3 \
    --precision fp16 \
    --gen_temperature 0.0 \
    --datasets pope scienceqa textvqa vizwiz-vqa \
    --output_dir data/features/m3_llava/temp_0.0
```

#### 3.3 Extract Multi-Scale Features for MQT-LLaVA ($m \in [1, 9, 36, 144, 256]$)
The wrapper will automatically clone the official `MQT-LLaVA` repository if not already present in `/workspace/MQT-LLaVA`:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py \
    --model_path gordonhu/MQT-LLaVA-7b \
    --arch mqt \
    --precision bf16 \
    --gen_temperature 0.0 \
    --datasets pope scienceqa textvqa vizwiz-vqa \
    --output_dir data/features/mqt_llava/temp_0.0
```

#### 3.4 Multi-GPU Parallel Extraction
If you have multi-GPU hardware (e.g., 2x or 4x RTX 4090 / A100), extract different datasets concurrently:
```bash
# GPU 0 extracts POPE and ScienceQA
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py \
    --model_path mucai/llava-v1.5-7b-m3 \
    --arch m3 \
    --datasets pope scienceqa \
    --output_dir data/features/m3_llava/temp_0.0 &

# GPU 1 extracts TextVQA and VizWiz-VQA
CUDA_VISIBLE_DEVICES=1 python scripts/extract_features.py \
    --model_path mucai/llava-v1.5-7b-m3 \
    --arch m3 \
    --datasets textvqa vizwiz-vqa \
    --output_dir data/features/m3_llava/temp_0.0 &
wait
```

---

## Feature Definitions: 17-D Trajectory Signature

The trajectory signature captures the evolution of generation logits, margins, and textual stability across 5 visual token budgets $m \in \{1, 9, 36, 144, 576\}$ (or $256$ for MQT):

| Key | Feature Name | Mathematical Definition | Scientific Interpretation |
| :--- | :--- | :--- | :--- |
| **`x1`** | **Final Logit** | $\ln\left(\frac{c_{\text{fine}}}{1 - c_{\text{fine}}}\right)$ | Base uncalibrated confidence anchor ($m=576/256$) |
| **`x3`** | **Confidence Gain** | $c_{\text{fine}} - c_9$ | Visual resolution sensitivity (fine minus coarse) |
| **`x4`** | **Monotonicity Count** | $\sum_{i=1}^4 \mathbb{I}(c_{m_{i+1}} > c_{m_i})$ | Monotonic confidence trajectory consistency |
| **`x6`** | **Confidence Variance** | $\text{Var}([c_1, c_9, c_{36}, c_{144}, c_{\text{fine}}])$ | Fluctuation/dispersion across visual scales |
| **`x8`** | **Scale Dip Depth** | $\max(0, \max(c_1, c_9) - \min(c_{36}, c_{144}))$ | Mid-scale visual confusion indicator |
| **`x9`** | **Log-Scale Slope** | $\frac{\sum (\ln m_i - \overline{\ln m})(c_{m_i} - \bar{c})}{\sum (\ln m_i - \overline{\ln m})^2}$ | Overall logarithmic rate of confidence growth |
| **`x10`** | **Logprob Gain** | $\ln c_{\text{fine}} - \ln c_9$ | Probability magnitude shift in log-space |
| **`x11`** | **Logprob Variance** | $\text{Var}([\ln c_1, \dots, \ln c_{\text{fine}}])$ | Log-likelihood stability across scales |
| **`x12`** | **Logprob Acceleration** | $(\ln c_{\text{fine}} - \ln c_{144}) - (\ln c_{144} - \ln c_{36})$ | Discrete second derivative of log-confidence |
| **`x13`** | **Discrete Answer Stability** | $1 / |\text{UniqueAnswers}|$ across 5 scales | Single-pass semantic consistency proxy |
| **`x14`** | **Relative Gain Ratio** | $c_{\text{fine}} / (c_9 + \epsilon)$ | Multiplicative confidence enhancement ratio |
| **`x15`** | **Mid-Fine Contrast** | $(c_{\text{fine}} - c_{144}) - (c_{144} - c_9)$ | Convexity of mid-to-fine scale transition |
| **`x17`** | **End-Scale Spike** | $c_{\text{fine}} - \frac{1}{4}\sum_{i=1}^4 c_{m_i}$ | Sudden fine-scale confidence jump |
| **`x18`** | **Entropy Slope** | OLS slope of binary entropy $H(c_m)$ vs $\ln m$ | Rate of information gain as resolution increases |
| **`x19`** | **Margin Growth** | $\text{margin}_{\text{fine}} / (\text{margin}_9 + \epsilon)$ | Top-1 vs Top-2 separation growth |
| **`x20`** | **Answer Flip Freq** | $\frac{1}{4} \sum_{i=1}^4 \mathbb{I}(\text{ans}_{m_i} \neq \text{ans}_{m_{i+1}})$ | Textual prediction volatility across scales |
| **`x21`** | **Logit Convexity** | $(c_{\text{fine}} - c_{144}) - (c_{144} - c_{36})$ | Curve convexity in high-resolution regime |
| **`x22`** | **Jump Ratio** | $(c_{\text{fine}} - c_1) / (c_{\text{fine}} + \epsilon)$ | Relative span from single-token to full scale |

---

## Empirical Benchmark Results

Evaluated on **M3-LLaVA** across standard VQA and hallucination benchmarks at $T_{\text{gen}}=0.0$:

### Adaptive ECE (%) [Lower is Better]

| Calibration Method | Compute Cost | POPE | ScienceQA | TextVQA | VizWiz-VQA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Confidence (NC)** | 1x | 4.12% | 10.66% | 8.72% | 16.71% |
| **Temperature Scaling (TS)** | 1x | 3.96% | 9.76% | 8.35% | 15.56% |
| **Platt Scaling (1D)** | 1x | 4.05% | 7.46% | 5.70% | 7.25% |
| **Spline Calibration (PCHIP)** | 1x | 5.10% | 5.16% | 10.25% | 8.45% |
| **Adaptive TS (ATS)** | 1x | 4.16% | 9.43% | 8.80% | 15.45% |
| **UQLM SequenceProb** | 1x | 4.05% | 7.46% | 5.70% | 7.25% |
| **UQLM MinProb** | 1x | 4.05% | 7.46% | 5.70% | 7.25% |
| **UQLM TokenEntropy** | 1x | 4.05% | 7.46% | 5.70% | 7.25% |
| **Kuhn Semantic Entropy** | 10x (sampling) | 4.05% | 7.46% | 5.70% | 7.25% |
| **Chen EigenScore** | 10x (sampling) | 4.05% | 7.46% | 5.70% | 7.25% |
| **MSSC (Multi-Scale Proxy)** | 1x | **0.86%** | **1.83%** | 3.36% | 3.44% |
| **VCPS-5D (Our Method)** | **1x** | **3.98%** | **6.50%** | **5.06%** | **7.23%** |
| **VCPS-17D (Our Method)** | **1x** | **3.57%** | **5.90%** | **5.80%** | **6.76%** |

---

## Ground-Truth Datasets and Verification Registry

The repository includes a hardened `DATASET_REGISTRY` covering 14 vision-language benchmarks:

| Dataset Key | HuggingFace Repository | Split | Answer Type | Metric Scoring Logic |
| :--- | :--- | :---: | :---: | :--- |
| **`vqav2`** | `lmms-lab/vqav2` | `validation` | `list_soft` | 10-annotator soft consensus $\min(1.0, \text{matches}/3.0)$ |
| **`textvqa`** | `lmms-lab/textvqa` | `validation` | `list_soft` | 10-annotator soft consensus (non-withheld split) |
| **`vizwiz-vqa`** | `lmms-lab/VizWiz-VQA` | `val` | `list_soft` | 10-annotator soft consensus (non-withheld split) |
| **`docvqa`** | `lmms-lab/DocVQA` (`DocVQA`) | `validation` | `list_soft` | Soft consensus on document QA |
| **`infographicvqa`** | `lmms-lab/DocVQA` (`InfographicVQA`) | `validation` | `list_soft` | Soft consensus on infographics |
| **`pope`** | `lmms-lab/POPE` | `test` | `open` | Exact/normalized "yes" / "no" (with duplicate filter) |
| **`scienceqa`** | `lmms-lab/ScienceQA` (`ScienceQA-IMG`) | `test` | `mc_index` | Option letter / Choice index matching |
| **`ai2d`** | `lmms-lab/ai2d` | `test` | `mc_index` | Diagram multiple-choice index matching |
| **`chartqa`** | `lmms-lab/ChartQA` | `test` | `open` | Normalized chart QA string matching |
| **`gqa`** | `lmms-lab/GQA` (`testdev_balanced_instructions`) | `testdev` | `open` | Instruction-image merged QA matching |
| **`lego-puzzles`** | `lmms-lab/LEGO-Puzzles` | `test` | `open` | Reasoning string matching |
| **`mmbench`** | `lmms-lab/MMBench_EN` | `dev` | `mc_letter` | Circular option letter (A/B/C/D) matching |
| **`mmmu`** | `lmms-lab/MMMU` | `validation` | `mc_letter` | Multi-discipline option matching (single-image filtered) |
| **`seedbench`** | `lmms-lab/SEED-Bench` | `test` | `mc_letter` | Multi-choice matching (image subset filtered) |

---

## Scientific Invariants and Numerical Stability Rules

1. **Temperature Bounding**: Global temperature scaling $T^*$ and instance temperatures $T(\mathbf{z})$ are strictly bounded to $T \in [0.01, 20.0]$.
2. **Logit Clamping**: Logits are clipped to $[-35.0, 35.0]$ before sigmoid exponentiation to prevent floating-point underflow/overflow.
3. **Probability Clipping**: Probabilities evaluated in log-likelihood loss functions are clipped to $[10^{-12}, 1.0 - 10^{-12}]$ or $[10^{-7}, 1.0 - 10^{-7}]$.
4. **Collinearity Protection**: Stepwise forward selection enforces Variance Inflation Factor $\text{VIF} < 10.0$ to prevent multicollinearity between trajectory features.
5. **Deduplication**: Automatic `drop_duplicates(subset=["question_id"])` filters redundant padding samples from benchmark datasets.

---

## Citation and License

This project is licensed under the **MIT License**.

```bibtex
@article{trajectory_calibration2026,
  title={Trajectory Calibration: Single-Pass Elastic Uncertainty Quantification for Multimodal LLMs},
  author={Ashmin et al.},
  journal={arXiv preprint},
  year={2026}
}
```
