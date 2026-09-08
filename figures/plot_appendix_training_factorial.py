#!/usr/bin/env python3
"""Render the Appendix K factorial trajectories from frozen run artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1788652800")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssf-matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


INK = "#182230"
MUTED = "#667085"
GRID = "#D9DEE7"
BLUE = "#2F6FB0"
VERMILION = "#D95F2B"
WHITE = "#FFFFFF"

CONDITIONS = (
    ("tied", "single", "Tied, final-only", BLUE, "o", "-", True),
    ("untied", "single", "Untied, final-only", VERMILION, "s", "-", True),
    ("tied", "multi", "Tied, every-exit", BLUE, "o", (0, (4, 2)), False),
    ("untied", "multi", "Untied, every-exit", VERMILION, "s", (0, (4, 2)), False),
)


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
            "font.size": 8.5,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.75,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.62,
            "lines.linewidth": 1.55,
            "lines.markersize": 5.2,
            "svg.fonttype": "none",
            "svg.hashsalt": "iclr2027-v3-appendix-k-factorial",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        },
    )


def seed_for(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_seed_means(values: np.ndarray, label: str) -> tuple[float, float]:
    rng = np.random.default_rng(seed_for(label))
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.55, alpha=0.62)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(length=2.6, width=0.7, pad=2.0)


def load_runs() -> dict[tuple[str, str], list[dict]]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    run_dir = repo_root / "data/analysis_ready/factorial/runs"
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("seed*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        key = (
            artifact["condition"]["tying"],
            artifact["condition"]["supervision"],
        )
        grouped[key].append(artifact)
    if set(grouped) != {(item[0], item[1]) for item in CONDITIONS}:
        raise RuntimeError("incomplete factorial artifact set")
    if any(len(rows) != 8 for rows in grouped.values()):
        raise RuntimeError("expected eight confirmation seeds per condition")
    return grouped


def draw_held_out(
    ax: plt.Axes,
    grouped: dict[tuple[str, str], list[dict]],
    supervision: str,
) -> None:
    depths = np.array([1, 2, 3, 4], dtype=float)
    for tying, _, _, color, marker, _, filled in [
        item for item in CONDITIONS if item[1] == supervision
    ]:
        runs = grouped[(tying, supervision)]
        means = []
        lows = []
        highs = []
        for depth in depths.astype(int):
            values = np.asarray(
                [run["evaluation"]["summary"][f"choice_effect_K{depth}"] for run in runs],
                dtype=float,
            )
            means.append(float(values.mean()))
            low, high = bootstrap_seed_means(values, f"{tying}-{supervision}-K{depth}")
            lows.append(low)
            highs.append(high)
        means_array = np.asarray(means)
        x = depths + (-0.025 if tying == "tied" else 0.025)
        ax.fill_between(x, lows, highs, color=color, alpha=0.11, linewidth=0)
        sns.lineplot(
            x=x,
            y=means_array,
            estimator=None,
            errorbar=None,
            color=color,
            marker=marker,
            markerfacecolor=color if filled else WHITE,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linewidth=1.55,
            legend=False,
            ax=ax,
            zorder=3,
        )
    ax.axhline(0, color=MUTED, linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlim(0.78, 4.22)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(-0.13, 1.09)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Effective exit $K$")
    clean_axis(ax)


def draw_training(ax: plt.Axes, grouped: dict[tuple[str, str], list[dict]]) -> None:
    # Steps 1, 500, and 1,000 were logged for every confirmation seed.
    steps = np.array([1, 500, 1000], dtype=float)
    for tying, supervision, _, color, marker, linestyle, filled in CONDITIONS:
        runs = grouped[(tying, supervision)]
        matrix = np.asarray([
            [
                next(
                    row["batch_accuracy_K1"]
                    for row in run["training_history"]
                    if row["step"] == int(step)
                )
                for step in steps
            ]
            for run in runs
        ], dtype=float)
        mean = matrix.mean(axis=0)
        low = matrix.min(axis=0)
        high = matrix.max(axis=0)
        ax.fill_between(steps, low, high, color=color, alpha=0.065, linewidth=0)
        sns.lineplot(
            x=steps,
            y=mean,
            estimator=None,
            errorbar=None,
            color=color,
            marker=marker,
            markerfacecolor=color if filled else WHITE,
            markeredgecolor=color,
            markeredgewidth=0.85,
            linestyle=linestyle,
            linewidth=1.45,
            legend=False,
            ax=ax,
            zorder=3,
        )
    ax.axhline(0.25, color=MUTED, linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlim(-25, 1025)
    ax.set_xticks([0, 500, 1000])
    ax.set_ylim(0.16, 0.94)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_xlabel("Training update")
    clean_axis(ax)


def main() -> None:
    grouped = load_runs()
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.20))

    draw_held_out(axes[0], grouped, "single")
    axes[0].set_title("Final-only loss", loc="left", pad=4)
    axes[0].set_ylabel("Held-out choice effect  $C_K$")

    draw_held_out(axes[1], grouped, "multi")
    axes[1].set_title("Every-exit loss", loc="left", pad=4)
    axes[1].set_ylabel("")

    draw_training(axes[2], grouped)
    axes[2].set_title("When K1 becomes readable", loc="left", pad=4)
    axes[2].set_ylabel("Logged K1 accuracy")

    legend_handles = [
        Line2D(
            [0], [0],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=color if filled else WHITE,
            markeredgecolor=color,
            markeredgewidth=0.85,
            linewidth=1.45,
            label=label,
        )
        for _, _, label, color, marker, linestyle, filled in CONDITIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.7,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.235, top=0.73, wspace=0.31)

    output_dir = Path(__file__).resolve().parents[1] / "reproduced/figures/appendix"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "anonymous reproducible appendix plotter"}
    for extension in ("pdf", "svg"):
        fig.savefig(
            output_dir / f"appendix_k1_factorial_trajectories.{extension}",
            metadata=metadata if extension == "pdf" else None,
            bbox_inches=None,
        )
    fig.savefig(
        output_dir / "appendix_k1_factorial_trajectories.png",
        dpi=360,
        bbox_inches=None,
    )
    plt.close(fig)
    mpl.rcdefaults()


if __name__ == "__main__":
    main()
