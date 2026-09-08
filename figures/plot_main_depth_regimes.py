#!/usr/bin/env python3
"""Render Version 3 Figure 2 directly from the frozen result artifacts."""

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
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
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
            "lines.markersize": 5.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "iclr2027-v3-figure2",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        },
    )


def clean_axis(ax: plt.Axes, *, xgrid: bool, ygrid: bool) -> None:
    ax.grid(False)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.55)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(length=2.5, width=0.65, pad=1.8)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_natural(ax: plt.Axes, payload: dict) -> None:
    order = ["ouro26_2wiki", "ouro26_musique", "loopus_2wiki", "loopus_musique"]
    labels = ["Ouro  |  2Wiki", "Ouro  |  MuSiQue", "LoopUS  |  2Wiki", "LoopUS  |  MuSiQue"]
    y_positions = np.array([3.25, 2.25, 0.75, -0.25])

    for key, y in zip(order, y_positions, strict=True):
        row = payload["results"][key]
        stat = row["exact_first_divergence_token"]["choice_control_gain"]
        estimate = stat["estimate"]
        low, high = stat["ci95"]
        is_ouro = key.startswith("ouro")
        color = BLUE if is_ouro else ORANGE
        marker = "o" if is_ouro else "s"
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt=marker,
            markersize=5.7,
            markerfacecolor=color,
            markeredgecolor=WHITE,
            markeredgewidth=0.65,
            color=color,
            ecolor=color,
            elinewidth=1.05,
            capsize=2.0,
            capthick=0.9,
            zorder=3,
        )

    ax.axvline(0, color=MUTED, linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(1.5, color=GRID, linewidth=0.65, zorder=0)
    ax.set_title("Natural questions", loc="left", pad=5)
    ax.set_xlabel(r"Exact-choice depth gain  $\Delta C$")
    ax.set_yticks(y_positions, labels=labels)
    ax.set_xlim(-0.045, 0.52)
    ax.set_xticks([0.0, 0.2, 0.4])
    ax.set_ylim(-0.8, 3.8)
    clean_axis(ax, xgrid=True, ygrid=False)


def plot_trajectory(ax: plt.Axes, payload: dict, *, title: str, color: str, marker: str) -> None:
    rows = []
    for depth, values in payload["result"]["by_K"].items():
        stat = values["choice_control_C"]
        rows.append((int(depth), stat["estimate"], stat["ci95"][0], stat["ci95"][1]))
    rows.sort()
    x = np.array([row[0] for row in rows], dtype=float)
    y = np.array([row[1] for row in rows], dtype=float)
    low = np.array([row[2] for row in rows], dtype=float)
    high = np.array([row[3] for row in rows], dtype=float)

    ax.errorbar(
        x,
        y,
        yerr=[y - low, high - y],
        fmt="none",
        ecolor=color,
        elinewidth=0.95,
        capsize=2.0,
        capthick=0.9,
        alpha=0.85,
        zorder=2,
    )
    sns.lineplot(
        x=x,
        y=y,
        estimator=None,
        errorbar=None,
        color=color,
        marker=marker,
        markeredgecolor=WHITE,
        markeredgewidth=0.65,
        linewidth=1.55,
        legend=False,
        ax=ax,
        zorder=3,
    )
    ax.axhline(0, color=MUTED, linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.set_title(title, loc="left", pad=5)
    ax.set_xlabel("Native exit $K$")
    ax.set_xticks(x.astype(int))
    ax.set_ylim(-0.055, 1.02)
    ax.set_yticks([0.0, 0.4, 0.8])
    clean_axis(ax, xgrid=False, ygrid=True)


def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    revision = repo_root / "reproduced" / "results"
    output_dir = repo_root / "reproduced" / "figures" / "main"

    natural = load_json(revision / "scale_invariant_path_control.json")
    ouro = load_json(revision / "fictional_ouro" / "nonce_path_control_ouro26.json")
    loopus = load_json(revision / "fictional_loopus8.json")

    configure_style()
    fig = plt.figure(figsize=(6.55, 1.82))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 1.0], wspace=0.49)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    plot_natural(axes[0], natural)
    plot_trajectory(axes[1], ouro, title="Fictional Ouro-2.6B", color=BLUE, marker="o")
    plot_trajectory(axes[2], loopus, title="Fictional LoopUS-8B", color=ORANGE, marker="s")
    axes[1].set_ylabel("Exact-choice path effect  $C_K$")
    axes[2].set_ylabel("")
    axes[2].tick_params(labelleft=False)
    axes[2].spines["left"].set_visible(False)

    fig.subplots_adjust(left=0.172, right=0.995, bottom=0.255, top=0.865)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg"):
        fig.savefig(output_dir / f"fig2_depth_regimes.{extension}", bbox_inches=None)
    fig.savefig(output_dir / "fig2_depth_regimes.png", dpi=360, bbox_inches=None)
    plt.close(fig)
    mpl.rcdefaults()


if __name__ == "__main__":
    main()
