# Agent Operating Guide: Trajectory Calibration

Welcome, AI Agent / Developer! This document serves as the operational manual for working on the `trajectory_calibration` codebase.

---

## 1. Project Overview & Knowledge Base

**Trajectory Uncertainty Calibration for Multimodal Language Models (VLMs)**
- **Models Targeted**: M3-LLaVA (7B) and MQT-LLaVA (7B) autoregressive visual language models.
- **Core Innovation**: Varying-Coefficient Platt Scaling (**VCPS**) and Trajectory Platt Scaling leveraging autoregressive token logit trajectories (17D features) from a single greedy decoding pass ($T = 0, K = 1$) without requiring expensive multi-rollout sampling ($K = 10$).
- **Benchmark Scope**: **17 Calibration Methods** (11 single-pass post-hoc calibrators including Trajectory Platt 5D/17D, 2 VCPS variants, 4 multi-pass baselines) evaluated across **14 Vision-Language Benchmarks** (POPE, ScienceQA, TextVQA, VizWiz, GQA, AI2D, ChartQA, DocVQA, InfographicVQA, MMMU, MMBench, SEEDBench, LegoPuzzles, VQAv2).

### Consult Local Wiki & ADRs First
Before designing features, refactoring mathematical logic, or debugging, review the local documentation:
- **[`wiki/README.md`](file:///c:/Users/ashmi/OneDrive/Documents/trajectory_calibration/wiki/README.md)**: Local wiki navigation index and core invariant cheatsheet.
- **[`wiki/decisions.md`](file:///c:/Users/ashmi/OneDrive/Documents/trajectory_calibration/wiki/decisions.md)**: Exhaustive engineering log of 26 technical issues, root cause analyses, numerical precision fixes, and architectural choices.
- **[`docs/adr/`](file:///c:/Users/ashmi/OneDrive/Documents/trajectory_calibration/docs/adr/)**: Formal ADRs:
  - `0001`: Benchmark execution and reporting pipeline.
  - `0002`: Feature space trim to 17-D, $x_{21}$ ablation, and contiguous importance re-indexing.
  - `0003`: Standardized $K$-dimensional trajectory convention ($K = 1 + (K-1)$).
- **[`CONTEXT.md`](file:///c:/Users/ashmi/OneDrive/Documents/trajectory_calibration/CONTEXT.md)**: Ubiquitous language, domain models, and mathematical definitions.

---

## 2. Toolchain & Development Workflow

Standardized exclusively on the modern **Astral Rust toolchain** (`uv` + `ruff`). Always invoke commands via `uv run`:

| Tool | Purpose | Primary Command |
| :--- | :--- | :--- |
| **`uv`** | Package & environment manager | `uv sync --extra dev` |
| **`ruff`** | Linter (isort, bugbear, pyflakes, etc.) | `uv run ruff check --fix .` |
| **`ruff format`** | Code formatter (Black-compatible) | `uv run ruff format .` |
| **`pyright`** | Static type checker | `uv run pyright` |
| **`pytest`** | Unit test suite | `uv run pytest` |
| **`hypothesis`** | Property-based numerical stress tests | `uv run pytest tests/test_hypothesis_calibration.py` |
| **`pre-commit`** | Commit-time hygiene verification | `uv run pre-commit run --all-files` |

```powershell
# Setup environment & pre-commit hooks
uv sync --extra dev
uv run pre-commit install
```

---

## 3. Core Architectural & Mathematical Invariants

1. **$K$-D Trajectory Input Convention ($K = 1 + (K-1)$)**:
   - Total dimension $K = 1 \text{ Base Logit Anchor } (x_1) + (K - 1) \text{ Trajectory Signatures } (\mathbf{z} \in \mathbb{R}^{K-1})$.
   - **5D Representation**: $1 \text{ Anchor } + 4 \text{ Signatures } (x_2 \dots x_5) = 5\text{D}$ ($X \in \mathbb{R}^{N \times 5}$).
     - VCPS-5D: 10 parameters ($2K = 10$: $1 \text{ base slope } a_0 + 4 \text{ slope } \boldsymbol{\gamma} + 1 \text{ base intercept } b_0 + 4 \text{ intercept } \mathbf{w}$).
     - Trajectory Platt (5D): 6 parameters ($K + 1 = 6$: $\mathbf{a} \in \mathbb{R}^5 + b \in \mathbb{R}$).
     - `select_best_5d_subset` strictly selects 5 features ($x_1$ + 4 signatures).
   - **17D Representation**: $1 \text{ Anchor } + 16 \text{ Signatures } (x_2 \dots x_{17}) = 17\text{D}$ ($X \in \mathbb{R}^{N \times 17}$).
     - VCPS-17D: 34 parameters ($2K = 34$: $1 + 16 + 1 + 16 = 34$).
     - Trajectory Platt (17D): 18 parameters ($K + 1 = 18$: $\mathbf{a} \in \mathbb{R}^{17} + b \in \mathbb{R}$).
2. **Decoupled Scaling Invariant**:
   - $x_1$ is the raw uncalibrated fine-scale log-odds anchor ($\ell = \ln(c_{\text{fine}} / (1 - c_{\text{fine}}))$). It must **NEVER** be ablated or standardized with mean subtraction, preserving $x_1 = 0 \iff c = 0.5$.
   - Only trajectory conditioning signatures $\mathbf{z}$ are standardized.
3. **Probabilistic & Metric Invariants**:
   - Calibrators must output valid probabilities $\hat{p} \in [0.0, 1.0]$, never `NaN` or `Inf`.
   - Murphy decomposition identity $\text{Brier} = \text{Rel} - \text{Res} + \text{Unc} + \text{Within}$ must hold within $10^{-5}$ tolerance.
   - Use `safe_clip_probs(p)` or `get_logits(p)` to avoid infinite log-odds.

---

## 4. Universal Table Ranking & Reporting Standards

1. **Option A Universal Formatting**:
   - Highlight Rank 1 in **bold** (`**X.XX%**` / `\textbf{...}`) and Rank 2 in *italic* (`*X.XX%*` / `\textit{...}`) across all metric columns in all tables (per-dataset Ada-ECE, Macro-Average, and Temperature Robustness).
   - Directionality: Lower is better for ECE ($\downarrow$), Ada-ECE ($\downarrow$), Brier ($\downarrow$); Higher is better for AUROC ($\uparrow$).
   - Multi-pass baselines reporting `'-'` for Ada-ECE are skipped during Ada-ECE ranking.
2. **Dual Win Reporting**:
   - Under each benchmark table in `README.md`, always report both:
     - **Family Win**: $\min(\text{Ada-ECE}_{\text{VCPS-17D}}, \text{Ada-ECE}_{\text{VCPS-5D}}) < \text{Ada-ECE}_{\text{Baseline}}$ (and analogously for $\text{TP-17D} / \text{TP-5D}$).
     - **Separate Counts**: Explicit head-to-head counts (VCPS-17D alone vs. Baseline, and VCPS-5D alone vs. Baseline; TP-17D alone vs. Baseline, and TP-5D alone vs. Baseline).
3. **Automated Synchronization & Unified Pipeline**:
   - Run `uv run python scripts/update_readme_tables.py` after benchmark runs rather than manually editing Markdown tables (prevents regex backslash escape errors and hallucinations).
   - Alternatively, execute `uv run python scripts/run_pipeline.py` to run all calibration benchmarks across architectures/temperatures followed by automatic table, figure, and README generation.

---

## 5. Strict Engineering Guardrails

1. **Function LOC Constraint (< 200 LOC)**:
   - Every function and method in `src/` and `scripts/` MUST be strictly under 200 lines of code.
   - Enforced by `tests/test_audit_fixes.py::test_all_functions_under_200_loc`.
2. **Zero Pyright Diagnostics**:
   - `uv run pyright` must return **0 errors, 0 warnings**.
   - Use `@overload` signatures for mathematical primitives accepting scalar or array inputs.
   - Guard against `None` values on optional VLM generation scores and hidden states.
3. **Third-Party Code Isolation**:
   - Never modify or run formatting/linting fixes on `baseline_repo/` or `llava_src/`.
   - These are excluded in `.pre-commit-config.yaml`, `pyproject.toml`, and `.gitignore`.

---

## 6. Verification Checklist Before Committing

Always run this command sequence before committing or pushing changes:

```powershell
uv run pyright
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
uv run pytest
```
