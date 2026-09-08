# Agent Operating Guide: Trajectory Calibration

Welcome, AI Agent / Developer! This document serves as the operational manual for working on the `trajectory_calibration` codebase.

---

## 1. Project Overview

**Trajectory Uncertainty Calibration for Multimodal Language Models (VLMs)**
- **Models Targeted**: M3-LLaVA (7B) and MQT-LLaVA (7B) autoregressive visual language models.
- **Core Innovation**: Varying-Coefficient Platt Scaling (**VCPS**) and Trajectory Platt Scaling leveraging autoregressive token logit trajectories (17D features) from a single greedy decoding pass ($T = 0, K = 1$) without requiring expensive multi-rollout sampling ($K = 10$).
- **Benchmark Panel**: 14 vision-language benchmarks (POPE, ScienceQA, TextVQA, VizWiz, GQA, AI2D, ChartQA, DocVQA, InfographicVQA, MMMU, MMBench, SEEDBench, LegoPuzzles, VQAv2).

---

## 2. Toolchain & Development Workflow

This project is standardized on the modern **Astral Rust toolchain** (`uv` + `ruff`) along with strict static typing and property-based testing:

| Tool | Purpose | Primary Command |
| :--- | :--- | :--- |
| **`uv`** | Package & environment manager | `uv sync --extra dev` |
| **`ruff`** | Linter (isort, bugbear, pyflakes, etc.) | `uv run ruff check --fix .` |
| **`ruff format`** | Code formatter (Black-compatible) | `uv run ruff format .` |
| **`pyright`** | Static type checker | `uv run pyright` |
| **`pytest`** | Unit test suite | `uv run pytest` |
| **`hypothesis`** | Property-based numerical stress tests | `uv run pytest tests/test_hypothesis_calibration.py` |
| **`pre-commit`** | Commit-time hygiene verification | `uv run pre-commit run --all-files` |

### Environment Setup
```powershell
# Sync full dev dependencies
uv sync --extra dev

# Install git pre-commit hooks
uv run pre-commit install
```

---

## 3. Repository Structure

```
trajectory_calibration/
├── src/trajectory_calibration/     # Core library package
│   ├── calibrators/                # Calibration estimators (VCPS, Platt, TS, Spline, Residual)
│   ├── features/                   # 17D logit trajectory features & VIF selection
│   ├── metrics/                    # ECE, Adaptive ECE, Murphy decomposition, AUROC, Brier
│   ├── uq/                         # Semantic Entropy (Kuhn et al.), EigenScore, WhiteBox
│   ├── utils/                      # Math primitives (sigmoid, get_logits, safe_clip_probs)
│   └── vlm/                        # Model wrappers, dataset registries, multipass rollouts
├── tests/                          # 74 unit tests and Hypothesis property-based tests
│   ├── test_hypothesis_calibration.py # Mathematical invariant & numerical stability tests
│   ├── test_audit_fixes.py         # Architectural & LOC compliance tests
│   └── ...
├── scripts/                        # Pipelines for benchmarks, LODO, tables, and plotting
├── docs/                           # Architectural Decision Records (ADRs)
├── baseline_repo/                  # Vendored UMPIRE baseline (DO NOT MODIFY / EXCLUDED)
├── llava_src/                      # Vendored LLaVA source (DO NOT MODIFY / EXCLUDED)
├── .pre-commit-config.yaml         # Pre-commit configuration with Ruff hooks
└── pyproject.toml                  # Central project config (ruff, pyright, pytest)
```

---

## 4. Strict Engineering Guardrails

When modifying this repository, ensure all changes adhere to these rules:

1. **Function LOC Constraint (< 200 LOC)**:
   - Every function and method in `src/` and `scripts/` MUST be strictly under 200 lines of code.
   - Enforced by `tests/test_audit_fixes.py::test_all_functions_under_200_loc`.
   - If a function grows beyond 200 LOC, extract sub-helpers cleanly.

2. **Zero Pyright Diagnostics**:
   - `uv run pyright` must return **0 errors, 0 warnings**.
   - Use `@overload` signatures for mathematical primitives accepting scalar or array inputs.
   - Guard against `None` values on optional VLM generation scores and hidden states.

3. **Numerical Invariants in Calibration Math**:
   - Calibrators must strictly output probabilities $\hat{p} \in [0.0, 1.0]$.
   - Probabilities must never evaluate to `NaN` or `Inf`.
   - Log-odds conversions must use `get_logits(p, eps=...)` or `safe_clip_probs(p)` to avoid infinite values.
   - Murphy decomposition identity $\text{Brier} = \text{Rel} - \text{Res} + \text{Unc} + \text{Within}$ must hold within $10^{-5}$ tolerance.

4. **Third-Party Code Isolation**:
   - Never run formatting or linting fixes on `baseline_repo/` or `llava_src/`.
   - These are excluded in `.pre-commit-config.yaml`, `pyproject.toml`, and `.gitignore`.

---

## 5. Verification Checklist Before Committing

Always run this command sequence before committing or pushing changes:

```powershell
uv run pyright
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
uv run pytest
```
