#!/usr/bin/env python3
"""Render the two structural-falsifier panels from frozen results."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.environ["SOURCE_DATE_EPOCH"] = "1788652800"

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


INK = "#182230"
MUTED = "#667085"
GRID = "#D9DEE7"
BLUE = "#2F6FB0"
ORANGE = "#D95F2B"
GREEN = "#2E8B57"
GRAY = "#98A2B3"
WHITE = "#FFFFFF"


def configure_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Helvetica Neue", "Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Avenir Next",
            "mathtext.it": "Avenir Next:italic",
            "mathtext.bf": "Avenir Next:bold",
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.1,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.75,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.55,
            "lines.linewidth": 1.55,
            "lines.markersize": 5.3,
            "legend.fontsize": 7.0,
            "svg.fonttype": "none",
            "svg.hashsalt": "iclr2027-v3-figure3",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        },
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(length=2.5, width=0.65, pad=1.8)
    ax.axhline(0, color=MUTED, linewidth=0.72, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlim(0.82, 4.18)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(-0.06, 0.82)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Native exit $K$")


def series(results: dict, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depths = np.array([1, 2, 3, 4], dtype=float)
    rows = [results[f"{stem}_K{int(depth)}"] for depth in depths]
    estimate = np.array([row["estimate"] for row in rows], dtype=float)
    low = np.array([row["ci95"][0] for row in rows], dtype=float)
    high = np.array([row["ci95"][1] for row in rows], dtype=float)
    return depths, estimate, low, high


def draw_series(
    ax: plt.Axes,
    results: dict,
    stem: str,
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    x, y, low, high = series(results, stem)
    ax.errorbar(
        x,
        y,
        yerr=[y - low, high - y],
        fmt="none",
        ecolor=color,
        elinewidth=0.9,
        capsize=1.9,
        capthick=0.85,
        alpha=0.82,
        zorder=2,
    )
    sns.lineplot(
        x=x,
        y=y,
        estimator=None,
        errorbar=None,
        label=label,
        color=color,
        marker=marker,
        markeredgecolor=WHITE,
        markeredgewidth=0.6,
        linewidth=1.55,
        ax=ax,
        zorder=3,
    )


def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    result_path = repo_root / "reproduced/results/structural_falsifiers.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    results = payload["results"]
    output_dir = repo_root / "reproduced/figures/main"

    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 1.48), sharey=True)

    draw_series(
        axes[0], results, "on_effect_choice", label="On path", color=ORANGE, marker="o"
    )
    draw_series(
        axes[0], results, "off_effect_choice", label="Off path", color=BLUE, marker="s"
    )
    axes[0].set_ylabel("Choice effect  $C_K$")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.025), frameon=False,
                   ncol=2, handlelength=1.4, columnspacing=0.9,
                   handletextpad=0.4, borderaxespad=0.0)

    draw_series(
        axes[1], results, "graph_effect_choice_matched", label="Matched / local",
        color=GREEN, marker="o"
    )
    draw_series(
        axes[1], results, "graph_effect_choice_crossed", label="Crossed / nonlocal",
        color=ORANGE, marker="s"
    )
    axes[1].set_ylabel("")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.025), frameon=False,
                   ncol=2, handlelength=1.4, columnspacing=0.9,
                   handletextpad=0.4, borderaxespad=0.0)

    for ax in axes:
        clean_axis(ax)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.29, top=0.78, wspace=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg"):
        fig.savefig(output_dir / f"fig3_structural_stats.{extension}", bbox_inches=None)
    fig.savefig(output_dir / "fig3_structural_stats.png", dpi=360, bbox_inches=None)
    plt.close(fig)
    mpl.rcdefaults()


if __name__ == "__main__":
    main()
