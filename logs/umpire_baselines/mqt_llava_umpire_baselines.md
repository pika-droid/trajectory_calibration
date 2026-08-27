# UMPIRE Multi-Pass Baseline Benchmark Logs: MQT-LLaVA

> **Document Purpose**: Official baseline benchmark logs evaluating multi-pass uncertainty quantification methods from the UMPIRE paper (*Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume*, Lau et al., arXiv:2602.24195).

## Metadata & Benchmark Configuration
- **Architecture**: MQT-LLaVA (7B parameters)
- **Decoding Mode**: Stochastic Autoregressive Generation
- **Sampling Temperature**: $T = 0.5$
- **Rollout Budget**: $K = 10$ stochastic rollout paths per query
- **NLI Cross-Encoder**: `microsoft/deberta-v2-xlarge-mnli` (EntailmentDeberta)
- **Evaluation Datasets**: 14 Standard Multimodal QA & Hallucination Benchmarks
- **Primary Data Source**: `results/umpire_eval/mqt_llava_cumulative_summary.csv`

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
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.611 | 32.40% | **0.163** | **0.260** | **0.039** | 0.622 |
| `semantic_entropy` | 0.594 | 35.70% | *-0.125* | 0.238 | *0.028* | 0.596 |
| `eigen_score` | **0.639** | **31.50%** | -0.153 | 0.227 | 0.024 | **0.656** |
| `umpire` | *0.612* | *31.80%* | -0.258 | *0.239* | 0.023 | *0.644* |

### Benchmark: `chartqa`
- **Dataset Name**: `chartqa`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.710 | *14.80%* | 0.325 | 0.214 | 0.005 | 0.170 |
| `semantic_entropy` | 0.830 | 37.70% | *0.703* | 0.562 | 0.086 | 0.197 |
| `eigen_score` | *0.855* | 24.90% | 0.698 | *0.700* | *0.194* | *0.199* |
| `umpire` | **0.861** | **10.00%** | **0.708** | **0.704** | **0.315** | **0.206** |

### Benchmark: `docvqa`
- **Dataset Name**: `docvqa`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.805 | **12.70%** | **0.704** | 0.406 | 0.037 | 0.137 |
| `semantic_entropy` | 0.816 | 39.30% | *0.688* | 0.515 | *0.227* | 0.133 |
| `eigen_score` | *0.839* | 33.50% | 0.616 | *0.625* | 0.097 | *0.143* |
| `umpire` | **0.853** | *14.00%* | 0.671 | **0.649** | **0.235** | **0.148** |

### Benchmark: `gqa`
- **Dataset Name**: `gqa`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.698* | 19.80% | 0.278 | 0.226 | 0.036 | **0.777** |
| `semantic_entropy` | 0.624 | 30.00% | **0.804** | 0.314 | 0.088 | 0.667 |
| `eigen_score` | 0.696 | *18.70%* | 0.644 | *0.325* | *0.091* | 0.743 |
| `umpire` | **0.715** | **16.70%** | *0.798* | **0.336** | **0.103** | *0.770* |

### Benchmark: `infographicvqa`
- **Dataset Name**: `infographicvqa`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.615 | *34.10%* | *0.172* | *0.139* | **0.000** | *0.022* |
| `semantic_entropy` | **0.648** | 47.50% | **0.248** | **0.196** | **0.000** | 0.020 |
| `eigen_score` | **0.648** | 47.80% | 0.005 | 0.086 | **0.000** | **0.023** |
| `umpire` | *0.646* | **26.10%** | 0.132 | 0.073 | **0.000** | **0.023** |

### Benchmark: `lego-puzzles`
- **Dataset Name**: `lego-puzzles`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.531 | **25.30%** | **0.298** | **0.126** | **0.019** | **0.297** |
| `semantic_entropy` | *0.533* | 45.20% | 0.079 | *0.125* | *0.010* | 0.288 |
| `eigen_score` | 0.502 | 39.40% | 0.079 | 0.112 | 0.009 | 0.229 |
| `umpire` | **0.537** | *27.60%* | *0.219* | *0.125* | 0.006 | *0.295* |

### Benchmark: `mmbench`
- **Dataset Name**: `mmbench`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.613 | 22.20% | -0.176 | 0.289 | 0.056 | *0.794* |
| `semantic_entropy` | 0.596 | 23.30% | *0.401* | 0.259 | **0.066** | 0.775 |
| `eigen_score` | **0.675** | **20.90%** | **0.528** | **0.338** | *0.059* | **0.805** |
| `umpire` | *0.618* | *22.10%* | 0.033 | *0.321* | 0.056 | 0.775 |

### Benchmark: `mmmu`
- **Dataset Name**: `mmmu`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.584 | *39.10%* | 0.096 | 0.175 | 0.040 | 0.415 |
| `semantic_entropy` | 0.588 | 44.40% | 0.408 | 0.227 | 0.049 | 0.396 |
| `eigen_score` | **0.627** | 45.50% | *0.466* | *0.240* | *0.076* | **0.459** |
| `umpire` | *0.598* | **37.70%** | **0.584** | **0.241** | **0.077** | *0.429* |

### Benchmark: `pope`
- **Dataset Name**: `pope`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.683* | **13.90%** | *0.401* | **0.413** | **0.060** | **0.896** |
| `semantic_entropy` | 0.368 | 16.70% | -0.269 | 0.018 | 0.018 | 0.752 |
| `eigen_score` | 0.616 | 18.10% | **0.684** | 0.362 | *0.048* | 0.849 |
| `umpire` | **0.693** | *14.70%* | 0.062 | *0.404* | 0.039 | *0.895* |

### Benchmark: `scienceqa`
- **Dataset Name**: `scienceqa`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.581* | 30.80% | 0.143 | *0.252* | 0.022 | *0.678* |
| `semantic_entropy` | 0.568 | 32.50% | *0.153* | 0.235 | 0.019 | 0.665 |
| `eigen_score` | **0.588** | **29.00%** | 0.106 | 0.238 | *0.030* | **0.687** |
| `umpire` | *0.581* | *30.20%* | **0.191** | **0.259** | **0.031** | 0.672 |

### Benchmark: `seedbench`
- **Dataset Name**: `seedbench`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | *0.610* | 28.10% | **0.290** | 0.280 | 0.033 | *0.707* |
| `semantic_entropy` | 0.599 | 30.00% | -0.061 | *0.283* | 0.041 | 0.688 |
| `eigen_score` | **0.639** | **27.00%** | *0.203* | **0.293** | *0.042* | **0.721** |
| `umpire` | 0.605 | *27.70%* | 0.165 | 0.282 | **0.051** | 0.690 |

### Benchmark: `textvqa`
- **Dataset Name**: `textvqa`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.793 | *14.80%* | **0.785** | 0.366 | **0.095** | *0.641* |
| `semantic_entropy` | 0.789 | 31.50% | 0.305 | 0.475 | 0.064 | 0.600 |
| `eigen_score` | *0.802* | 28.30% | 0.452 | **0.531** | 0.077 | 0.626 |
| `umpire` | **0.818** | **12.10%** | *0.756* | *0.517* | *0.082* | **0.651** |

### Benchmark: `vizwiz-vqa`
- **Dataset Name**: `vizwiz-vqa`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.718 | **18.50%** | 0.624 | 0.242 | 0.045 | *0.550* |
| `semantic_entropy` | 0.689 | 39.70% | **0.872** | 0.308 | 0.115 | 0.477 |
| `eigen_score` | *0.744* | 39.20% | 0.399 | *0.436* | **0.168** | 0.538 |
| `umpire` | **0.754** | *19.30%* | *0.820* | **0.444** | *0.155* | **0.557** |

### Benchmark: `vqav2`
- **Dataset Name**: `vqav2`
- **Architecture**: MQT-LLaVA ($T=0.5, K=10$)

| Method | AUROC (auc) $\uparrow$ | Calibrated ECE (cece) $\downarrow$ | CPC (pearsonr) $\uparrow$ | TPR @ 10% FPR $\uparrow$ | TPR @ 1% FPR $\uparrow$ | AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.767 | 12.30% | *0.753* | 0.329 | 0.058 | *0.904* |
| `semantic_entropy` | 0.699 | 13.70% | 0.673 | 0.457 | 0.107 | 0.839 |
| `eigen_score` | *0.773* | *10.20%* | 0.611 | **0.490** | *0.110* | 0.887 |
| `umpire` | **0.790** | **7.80%** | **0.758** | *0.478* | **0.152** | **0.906** |

---

## Section 2: Macro Mean Summary Table (Across All 14 Datasets)

The following table presents the unweighted macro-arithmetic mean of each uncertainty quantification method across all 14 evaluated multimodal benchmarks on **MQT-LLaVA**.

| Method | Mean AUROC $\uparrow$ | Mean Calibrated ECE $\downarrow$ | Mean CPC $\uparrow$ | Mean TPR @ 10% FPR $\uparrow$ | Mean TPR @ 1% FPR $\uparrow$ | Mean AURAC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ln_entropy` | 0.666 | *22.77%* | 0.347 | 0.266 | 0.039 | *0.544* |
| `semantic_entropy` | 0.639 | 33.37% | 0.348 | 0.301 | 0.066 | 0.507 |
| `eigen_score` | *0.689* | 29.57% | *0.381* | *0.357* | *0.073* | 0.540 |
| `umpire` | **0.691** | **21.27%** | **0.403** | **0.362** | **0.095** | **0.547** |
