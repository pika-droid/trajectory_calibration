#!/usr/bin/env python3
"""
Publication-Quality Qualitative Case Study Card Generator.

Renders standalone, high-resolution PNG cards demonstrating failure cases of 1D scalar
calibrators and their successful correction by Trajectory Platt (5D).
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image


def setup_card_canvas(title: str, subtitle: str) -> tuple[Figure, list[Axes]]:
    """Creates a 3-panel widescreen figure with publication styling."""
    fig = plt.figure(figsize=(18, 6.2), dpi=300, facecolor="#f8fafc")
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.1, 1.25, 1.25],
        wspace=0.32,
        left=0.04,
        right=0.96,
        top=0.80,
        bottom=0.15,
    )
    ax_img = fig.add_subplot(gs[0, 0])
    ax_traj = fig.add_subplot(gs[0, 1])
    ax_calib = fig.add_subplot(gs[0, 2])

    fig.text(0.04, 0.94, title, fontsize=15, fontweight="bold", color="#0f172a")
    fig.text(0.04, 0.88, subtitle, fontsize=10.5, color="#64748b")
    return fig, [ax_img, ax_traj, ax_calib]


def render_image_panel(
    ax: Axes,
    img_path: Path,
    question: str,
    gt_answer: str,
    pred_answer: str,
    is_correct: bool,
) -> None:
    """Renders the source diagram and question/answer metadata."""
    ax.axis("off")
    if img_path.exists():
        img = Image.open(img_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "Image Missing", ha="center", va="center")

    status_text = "CORRECT" if is_correct else "INCORRECT (ERROR)"
    ax.set_title(
        f"Input Question: {question}\nTarget: {gt_answer}  |  Prediction: {pred_answer} [{status_text}]",
        fontsize=9.2,
        fontweight="bold",
        color="#1e293b",
        pad=10,
        loc="center",
    )


def render_trajectory_panel(
    ax: Axes,
    progression: list[tuple[int, str, float]],
    stability: float,
    flips: float,
    entropy_slope: float,
) -> None:
    """Renders multi-scale token progression timeline and instability metrics."""
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
        spine.set_linewidth(1.2)

    ax.set_xlim(0, 7.5)
    ax.set_ylim(-1.2, len(progression) + 0.8)
    ax.axis("off")

    ax.text(
        0,
        len(progression) + 0.3,
        "Autoregressive Multi-Scale Trajectory",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )

    for i, (scale, ans, conf) in enumerate(reversed(progression)):
        y = i + 0.7
        ax.text(
            0.1,
            y,
            f"Scale {scale:3d}:",
            fontsize=9.2,
            fontweight="bold",
            color="#475569",
            va="center",
        )
        ax.text(1.8, y, f"'{ans}'", fontsize=9.5, fontweight="bold", color="#0f172a", va="center")

        bar_len = conf * 2.6
        rect = patches.FancyBboxPatch(
            (4.0, y - 0.16),
            bar_len,
            0.32,
            boxstyle="round,pad=0.03",
            facecolor="#38bdf8",
            edgecolor="none",
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            4.15 + bar_len,
            y,
            f"{conf * 100:.1f}%",
            fontsize=8.5,
            color="#0369a1",
            va="center",
            fontweight="bold",
        )

    summary_bg = patches.FancyBboxPatch(
        (0.1, -1.0),
        7.3,
        0.88,
        boxstyle="round,pad=0.08",
        facecolor="#f1f5f9",
        edgecolor="#cbd5e1",
        linewidth=1,
    )
    ax.add_patch(summary_bg)
    metrics_line1 = f"Answer Stability (x2) = {stability:.2f}  |  Flip Freq (x4) = {flips:.2f}"
    metrics_line2 = f"Entropy Slope (x3) = {entropy_slope:+.3f}"
    ax.text(
        3.75,
        -0.45,
        metrics_line1,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#334155",
    )
    ax.text(
        3.75,
        -0.75,
        metrics_line2,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#64748b",
    )


def render_calibration_panel(
    ax: Axes,
    raw_conf: float,
    ts_conf: float,
    p1d_conf: float,
    p5d_conf: float,
    is_correct: bool,
) -> None:
    """Renders comparative horizontal calibration bar chart with in-plot labels."""
    ax.set_facecolor("#ffffff")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")

    methods = [
        "Ground Truth Target",
        "Trajectory Platt 5D (Ours)",
        "Platt Scaling 1D (Baseline)",
        "Temperature Scaling (TS)",
        "Raw Confidence (Uncalibrated)",
    ]
    gt_val = 1.0 if is_correct else 0.0
    values = [gt_val, p5d_conf, p1d_conf, ts_conf, raw_conf]
    colors = [
        "#10b981" if is_correct else "#ef4444",
        "#059669",
        "#f59e0b",
        "#3b82f6",
        "#94a3b8",
    ]

    y_pos = [0.8, 2.0, 3.2, 4.4, 5.6]
    ax.set_ylim(0.2, 6.4)
    ax.set_yticks([])

    for y, method, val, col in zip(y_pos, methods, values, colors):
        ax.barh(y, val, color=col, height=0.45, edgecolor="none", zorder=3)
        # In-plot method label above bar
        label_suffix = ""
        if "1D" in method and not is_correct:
            label_suffix = "  [Overconfident Error]"
        elif "5D" in method and not is_correct:
            label_suffix = "  [Trajectory Corrected]"
        elif "5D" in method and is_correct:
            label_suffix = "  [High Confidence Preserved]"

        ax.text(
            0.01,
            y + 0.30,
            f"{method}{label_suffix}",
            fontsize=9.0,
            fontweight="bold",
            color="#1e293b",
            zorder=4,
        )
        # Percentage value at end of bar
        ax.text(
            val + 0.02,
            y,
            f"{val * 100:.1f}%",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color="#0f172a",
            zorder=4,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel(
        "Calibrated Confidence $\\hat{p}$",
        fontsize=10,
        fontweight="bold",
        color="#334155",
        labelpad=6,
    )
    ax.set_title(
        "Calibration Comparison vs. Outcome",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

    diff = (p1d_conf - p5d_conf) * 100
    if not is_correct and diff > 10:
        callout_text = f"Trajectory 5D slashes error confidence by -{diff:.1f}%!"
        callout_color = "#047857"
        bg_col = "#ecfdf5"
        edge_col = "#a7f3d0"
    elif is_correct:
        callout_text = f"Trajectory 5D maintains high confidence ({p5d_conf * 100:.1f}%)"
        callout_color = "#047857"
        bg_col = "#ecfdf5"
        edge_col = "#a7f3d0"
    else:
        callout_text = f"Difference: {diff:+.1f}%"
        callout_color = "#334155"
        bg_col = "#f1f5f9"
        edge_col = "#cbd5e1"

    ax.text(
        0.5,
        -0.20,
        callout_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=callout_color,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_col, edgecolor=edge_col),
    )


def generate_all_cards(output_dir: Path) -> list[Path]:
    """Generates all 3 individual qualitative case study cards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Card 1: ChartQA Airline Ranking Error (Sample 608)
    fig1, axes1 = setup_card_canvas(
        "Case Study 1: Subtle Diagram Ranking Error (ChartQA, M3-LLaVA 7B)",
        "Model predicts wrong airline with 85.2% confidence. Platt 1D stays fooled (78.1%), while Trajectory Platt 5D slashes error confidence to 19.9%.",
    )
    render_image_panel(
        axes1[0],
        Path("data/case_studies/images/chartqa_608.png"),
        question="which company has the largest percentage?",
        gt_answer="American Airlines",
        pred_answer="Delta",
        is_correct=False,
    )
    render_trajectory_panel(
        axes1[1],
        progression=[
            (1, "Southwest", 0.99),
            (9, "Delta", 0.97),
            (36, "Delta", 0.92),
            (144, "American", 0.52),
            (576, "Delta", 0.85),
        ],
        stability=0.33,
        flips=0.75,
        entropy_slope=0.081,
    )
    render_calibration_panel(
        axes1[2], raw_conf=0.852, ts_conf=0.716, p1d_conf=0.781, p5d_conf=0.199, is_correct=False
    )
    p1 = output_dir / "case_study_1_chartqa_error.png"
    fig1.savefig(p1, bbox_inches="tight", dpi=300)
    plt.close(fig1)
    generated.append(p1)

    # Card 2: ChartQA Overconfident Hallucination (Sample 771)
    fig2, axes2 = setup_card_canvas(
        "Case Study 2: Extreme Overconfident Sector Hallucination (ChartQA, M3-LLaVA 7B)",
        "Model claims 98.5% confidence in hallucinated 'Manufacturing' sector. 1D Platt stays at 89.8%, while Trajectory 5D slashes it to 34.4%.",
    )
    render_image_panel(
        axes2[0],
        Path("data/case_studies/images/chartqa_771.png"),
        question="Which sector has 5.52%?",
        gt_answer="Mining",
        pred_answer="Manufacturing",
        is_correct=False,
    )
    render_trajectory_panel(
        axes2[1],
        progression=[
            (1, "Transportation", 0.98),
            (9, "Finance", 0.93),
            (36, "Financials", 0.87),
            (144, "Financial", 0.50),
            (576, "Manufacturing", 0.98),
        ],
        stability=0.25,
        flips=0.75,
        entropy_slope=0.042,
    )
    render_calibration_panel(
        axes2[2], raw_conf=0.985, ts_conf=0.874, p1d_conf=0.898, p5d_conf=0.344, is_correct=False
    )
    p2 = output_dir / "case_study_2_chartqa_hallucination.png"
    fig2.savefig(p2, bbox_inches="tight", dpi=300)
    plt.close(fig2)
    generated.append(p2)

    # Card 3: ChartQA Grounded Correct Control (Sample 7)
    fig3, axes3 = setup_card_canvas(
        "Case Study 3: Complex Grounded Reasoning Control (ChartQA, M3-LLaVA 7B)",
        "Trajectory calibration preserves confident correct answers: invariant answers across scales (x2=1.0) maintain 98.3% confidence.",
    )
    render_image_panel(
        axes3[0],
        Path("data/case_studies/images/chartqa_7.png"),
        question="Is sum of Charismatic & Well-qualified > Strong leader?",
        gt_answer="Yes",
        pred_answer="Yes",
        is_correct=True,
    )
    render_trajectory_panel(
        axes3[1],
        progression=[
            (1, "Yes", 1.00),
            (9, "Yes", 1.00),
            (36, "Yes", 1.00),
            (144, "Yes", 1.00),
            (576, "Yes", 1.00),
        ],
        stability=1.00,
        flips=0.00,
        entropy_slope=-0.001,
    )
    render_calibration_panel(
        axes3[2], raw_conf=1.00, ts_conf=0.991, p1d_conf=0.995, p5d_conf=0.983, is_correct=True
    )
    p3 = output_dir / "case_study_3_chartqa_correct_control.png"
    fig3.savefig(p3, bbox_inches="tight", dpi=300)
    plt.close(fig3)
    generated.append(p3)

    return generated


if __name__ == "__main__":
    out = Path("results/experiments/benchmark/figures/case_studies")
    cards = generate_all_cards(out)
    for c in cards:
        print(f"Rendered: {c.resolve()}")
