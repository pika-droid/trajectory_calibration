# UMPIRE Multi-Pass Baseline Benchmark Logs: M3-LLaVA

> **Document Purpose**: Official baseline benchmark logs evaluating multi-pass uncertainty quantification methods from the UMPIRE paper (*Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume*, Lau et al., arXiv:2602.24195).

## Metadata & Benchmark Configuration
- **Architecture**: M3-LLaVA (7B parameters)
- **Decoding Mode**: Stochastic Autoregressive Generation
- **Sampling Temperature**: $T = 0.5$
- **Rollout Budget**: $K = 10$ stochastic rollout paths per query
- **NLI Cross-Encoder**: `microsoft/deberta-v2-xlarge-mnli` (EntailmentDeberta)
- **Evaluation Datasets**: 14 Standard Multimodal QA & Hallucination Benchmarks
- **Primary Data Source**: `results/umpire_eval/m3_llava_cumulative_summary.csv`

---

## Methodology & Metric Formulations

### A. Evaluated Multi-Pass Methods ($K=10, T=0.5$)

#### 1. `ln_entropy` (Length-Normalized Predictive Entropy)

$$\mathcal{H}_{\text{LN}}(y \mid x) = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p(y_t \mid y_{<t}, x)$$

Averaged token-level negative log-likelihood normalized by response token length across $K$ sampled trajectories.

#### 2. `semantic_entropy` (Kuhn Semantic Entropy)

$$\mathcal{H}_{\text{SE}}(\mathcal{C} \mid x) = -\sum_{c \in \mathcal{C}} p(c \mid x) \log p(c \mid x)$$

Rollouts are clustered into equivalence classes $\mathcal{C}$ using bidirectional NLI entailment checks via `microsoft/deberta-v2-xlarge-mnli` to capture semantic uncertainty invariant to phrasing.

#### 3. `eigen_score` (Chen EigenScore)

Computes the spectral dispersion of the normalized sentence embedding covariance matrix via singular value decomposition (SVD):

$$\mathbf{\Sigma} = \frac{1}{K} \sum_{k=1}^K (\mathbf{z}_k - \bar{\mathbf{z}})(\mathbf{z}_k - \bar{\mathbf{z}})^\top$$

$$\mathcal{E}_{\text{eigen}} = \frac{\sum_{i} \lambda_i^2}{\left(\sum_{i} \lambda_i\right)^2}$$

#### 4. `umpire` (Incoherence-adjusted Semantic Volume)

Combines multidimensional semantic volume (via differential entropy of embedding ellipsoid) with incoherence-based quadratic entropy penalty:

$$\mathcal{U} = \frac{1}{2} \log \det \left( \mathbf{\Sigma} + \epsilon \mathbf{I} \right) + \alpha_{\text{adaptive}} \cdot \sum_{i, j} \mathbf{D}_{ij}^2$$

where $\mathbf{D}_{ij}$ represents pairwise semantic contradiction distances.

### B. Evaluated Metrics
- **`auc` (AUROC) ($\uparrow$)**: Area Under Receiver Operating Characteristic Curve for selective risk/error prediction. Higher is better.
- **`cece` (Calibrated ECE) ($\downarrow$)**: Expected Calibration Error evaluated after development-set isotonic/temperature scaling. Lower is better.
- **`pearsonr` (CPC) ($\uparrow$)**: Calibration Pearson Correlation measuring linear correlation between uncertainty scores and empirical error rates. Higher is better.
- **`tpr_at_0.1_fpr` ($\uparrow$)**: True Positive Rate at a constrained 10% False Positive Rate budget. Higher is better.
- **`tpr_at_0.01_fpr` ($\uparrow$)**: True Positive Rate at a high-precision 1% False Positive Rate budget. Higher is better.
- **`aurac` ($\uparrow$)**: Area Under the Accuracy-Rejection Curve across confidence rejection thresholds. Higher is better.

> [!NOTE]
> **Highlighting Legend**: Across all tables, **bold** indicates the best performing method (e.g. highest AUROC, lowest ECE), while *italics* indicates the second-best performing method.

---

## Section 1: Per-Dataset Benchmark Results (14 Datasets)

### Benchmark: `ai2d`
- **Dataset Name**: `ai2d`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | **0.617** | 29.40% | **0.476** | *0.232* | **0.046** | **0.656** |
| `semantic_entropy` | 0.604 | 31.60% | 0.049 | 0.215 | 0.011 | 0.636 |
| `eigen_score` | 0.607 | **27.20%** | *0.264* | **0.247** | 0.016 | *0.642* |
| `umpire` | *0.608* | *28.60%* | 0.247 | 0.217 | *0.018* | 0.638 |

### Benchmark: `chartqa`
- **Dataset Name**: `chartqa`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.734 | *8.90%* | 0.714 | 0.318 | 0.020 | 0.219 |
| `semantic_entropy` | 0.790 | 25.80% | **0.786** | 0.475 | 0.050 | 0.229 |
| `eigen_score` | *0.818* | 19.30% | 0.230 | *0.554* | **0.185** | *0.241* |
| `umpire` | **0.828** | **8.80%** | *0.748* | **0.578** | *0.106* | **0.248** |

### Benchmark: `docvqa`
- **Dataset Name**: `docvqa`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.786 | *15.80%* | 0.603 | 0.379 | 0.096 | 0.172 |
| `semantic_entropy` | 0.809 | 37.10% | **0.750** | 0.524 | **0.269** | 0.164 |
| `eigen_score` | *0.830* | 29.70% | *0.684* | *0.532* | 0.164 | **0.182** |
| `umpire` | **0.834** | **10.40%** | 0.682 | **0.564** | *0.218* | *0.178* |

### Benchmark: `gqa`
- **Dataset Name**: `gqa`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.696* | 18.80% | 0.198 | 0.212 | 0.032 | **0.776** |
| `semantic_entropy` | 0.623 | 25.30% | **0.581** | 0.316 | *0.052* | 0.684 |
| `eigen_score` | 0.690 | *17.60%* | *0.559* | **0.332** | **0.071** | 0.741 |
| `umpire` | **0.712** | **17.40%** | 0.478 | *0.328* | **0.071** | *0.767* |

### Benchmark: `infographicvqa`
- **Dataset Name**: `infographicvqa`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.547 | **16.50%** | **0.310** | 0.055 | **0.025** | *0.022* |
| `semantic_entropy` | 0.534 | 39.20% | -0.211 | 0.032 | 0.003 | *0.022* |
| `eigen_score` | *0.561* | 34.80% | 0.061 | **0.078** | 0.009 | **0.023** |
| `umpire` | **0.562** | *16.60%* | *0.274* | *0.063* | *0.020* | **0.023** |

### Benchmark: `lego-puzzles`
- **Dataset Name**: `lego-puzzles`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.551 | 21.20% | *0.205* | 0.113 | 0.004 | 0.301 |
| `semantic_entropy` | *0.582* | 25.00% | 0.128 | *0.136* | *0.015* | **0.344** |
| `eigen_score` | 0.571 | *15.00%* | 0.157 | **0.159** | **0.025** | 0.309 |
| `umpire` | **0.588** | **11.00%** | **0.278** | *0.136* | 0.011 | *0.331* |

### Benchmark: `mmbench`
- **Dataset Name**: `mmbench`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.660* | 20.50% | -0.049 | *0.414* | 0.059 | **0.827** |
| `semantic_entropy` | 0.646 | 22.00% | *0.525* | 0.378 | **0.088** | 0.811 |
| `eigen_score` | **0.681** | **18.20%** | **0.740** | 0.389 | *0.087* | *0.823* |
| `umpire` | 0.658 | *19.30%* | 0.349 | **0.416** | 0.083 | 0.811 |

### Benchmark: `mmmu`
- **Dataset Name**: `mmmu`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.566 | **36.40%** | 0.130 | 0.104 | 0.030 | 0.380 |
| `semantic_entropy` | 0.568 | 47.60% | **0.628** | *0.132* | 0.067 | 0.376 |
| `eigen_score` | **0.602** | 50.10% | 0.536 | **0.198** | *0.074* | *0.400* |
| `umpire` | *0.592* | *39.70%* | *0.626* | **0.198** | **0.088** | **0.402** |

### Benchmark: `pope`
- **Dataset Name**: `pope`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.668* | **12.80%** | **0.618** | *0.388* | **0.061** | *0.901* |
| `semantic_entropy` | 0.387 | 14.90% | -0.091 | 0.024 | 0.024 | 0.785 |
| `eigen_score` | **0.678** | 16.50% | *0.417* | **0.398** | 0.027 | **0.910** |
| `umpire` | 0.666 | *13.50%* | 0.413 | *0.388* | *0.044* | 0.900 |

### Benchmark: `scienceqa`
- **Dataset Name**: `scienceqa`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.659* | *26.80%* | **0.241** | 0.312 | **0.036** | **0.707** |
| `semantic_entropy` | 0.633 | 30.00% | *-0.077* | **0.339** | **0.036** | 0.677 |
| `eigen_score` | 0.636 | 30.10% | -0.085 | 0.324 | 0.018 | 0.698 |
| `umpire` | **0.662** | **25.10%** | -0.101 | *0.334* | *0.031* | *0.702* |

### Benchmark: `seedbench`
- **Dataset Name**: `seedbench`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.644* | 23.30% | *0.245* | *0.334* | **0.060** | **0.764** |
| `semantic_entropy` | 0.629 | 25.00% | **0.473** | 0.329 | *0.057* | 0.743 |
| `eigen_score` | **0.655** | *22.70%* | 0.204 | **0.352** | *0.057* | *0.762* |
| `umpire` | *0.644* | **22.40%** | 0.048 | 0.329 | 0.055 | 0.751 |

### Benchmark: `textvqa`
- **Dataset Name**: `textvqa`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.811 | *11.80%* | **0.885** | 0.501 | 0.091 | *0.732* |
| `semantic_entropy` | 0.774 | 28.20% | 0.695 | 0.495 | *0.143* | 0.669 |
| `eigen_score` | *0.814* | 26.60% | 0.800 | *0.566* | 0.142 | 0.716 |
| `umpire` | **0.829** | **10.40%** | *0.815* | **0.570** | **0.149** | **0.737** |

### Benchmark: `vizwiz-vqa`
- **Dataset Name**: `vizwiz-vqa`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.723 | *14.80%* | **0.892** | 0.279 | 0.076 | *0.589* |
| `semantic_entropy` | 0.701 | 32.40% | *0.881* | 0.355 | 0.099 | 0.520 |
| `eigen_score` | *0.757* | 32.30% | 0.705 | **0.458** | *0.149* | 0.572 |
| `umpire` | **0.768** | **13.70%** | 0.790 | *0.456* | **0.156** | **0.599** |

### Benchmark: `vqav2`
- **Dataset Name**: `vqav2`
- **Architecture**: M3-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.787 | 12.40% | 0.662 | 0.351 | 0.059 | **0.924** |
| `semantic_entropy` | 0.690 | 11.40% | *0.826* | 0.444 | 0.125 | 0.851 |
| `eigen_score` | *0.816* | *6.20%* | 0.751 | *0.521* | *0.146* | **0.924** |
| `umpire` | **0.819** | **5.80%** | **0.900** | **0.529** | **0.157** | *0.923* |

---

## Section 2: Macro Mean Summary Table (Across All 14 Datasets)

The following table presents the unweighted macro-arithmetic mean of each uncertainty quantification method across all 14 evaluated multimodal benchmarks on **M3-LLaVA**.

| Method | Mean AUROC $\uparrow$ | Mean Calibrated ECE $\downarrow$ | Mean CPC $\uparrow$ | Mean TPR @ 10% FPR $\uparrow$ | Mean TPR @ 1% FPR $\uparrow$ | Mean AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.675 | *19.24%* | *0.438* | 0.285 | 0.050 | *0.569* |
| `semantic_entropy` | 0.641 | 28.25% | 0.424 | 0.300 | 0.074 | 0.536 |
| `eigen_score` | *0.694* | 24.74% | 0.430 | **0.365** | *0.084* | 0.567 |
| `umpire` | **0.698** | **17.34%** | **0.468** | *0.365* | **0.086** | **0.572** |
