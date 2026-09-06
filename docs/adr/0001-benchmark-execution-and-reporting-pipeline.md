# ADR 0001: End-to-End Benchmark Execution, Plotting, and Table Presentation Pipeline

## Status
Accepted

## Context
We need to execute the complete empirical benchmark suite across both M3-LLaVA and MQT-LLaVA architectures, regenerate all downstream publication artifacts (LaTeX tables, Markdown benchmark logs, Pareto/reliability plots), and update repository documentation (README.md) with exact numerical synchronization.

The user explicitly requested that in all tables:
1. The method with the best Ada-ECE value is bolded.
2. The second-best method is highlighted/italicized.
3. Summary text below the tables (win counts against baselines like TS and Platt, #1 rankings, and macro-average metrics) must be programmatically tallied and updated.

## Decision
1. **Experiment Pipeline Execution**:
   - Run 
un_benchmark.py for both m3 and mqt architectures at {\text{gen}} = 0.0$ across all 14 benchmarks.
   - Run 
un_lodo.py for 14-fold cross-domain zero-shot transfer.
   - Run 
un_temperature_study.py across temperatures  \in \{0.0, 0.3, 0.6, 1.0, 1.5\}$ on the 4 archetype datasets (pope, scienceqa, 	extvqa, izwiz-vqa).
   - Run 
un_ablation.py for univariate, LOO, and Pareto subset progression.
   - Run 
un_vcps.py for parameter analysis and coefficient interpretability.
   - Run 
un_benchmark_cv.py across 10 stratified random seeds (seeds 42–51).

2. **Artifact Generation & Plotting**:
   - Run generate_dataset_latex_tables.py to refresh all 19 .tex files in dataset_tables/.
   - Run generate_markdown_logs.py to update Markdown reports in logs/.
   - Run plot_benchmark_comparison.py to regenerate publication figures in 
esults/.

3. **Documentation Synchronization**:
   - Compute exact Rank 1 and Rank 2 values per dataset and macro averages from the freshly produced CSVs.
   - Format Rank 1 as **bold** and Rank 2 as *italic*.
   - Programmatically calculate win counts (VCPS vs TS, VCPS vs 1D Platt, #1 lowest Ada-ECE) and update the analytical bullet points directly beneath the tables in README.md.

4. **Git Commit Strategy**:
   - Automatically stage all updated artifacts (
esults/, dataset_tables/, logs/, README.md, CONTEXT.md, docs/) and create a structured git commit upon successful completion and verification.

## Consequences
- Guarantees 100% mathematical consistency across CSV result files, LaTeX tables, Markdown logs, plots, and README documentation.
- Eliminates any manual transcription errors or stale baseline comparison claims.
