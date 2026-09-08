#!/usr/bin/env python3
"""Render the standalone statistical components for the ICLR 2027 appendix."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1788652800")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/plot_ready/appendix"
OUT_DIR = ROOT / "reproduced/figures/appendix"

INK = "#182230"
MUTED = "#667085"
GRID = "#D9DEE7"
BLUE = "#2F6FB0"
VERMILION = "#D95F2B"
GREEN = "#3F8F5A"
PURPLE = "#7A5AA6"
RED = "#C83E36"
NEUTRAL = "#9AA4B2"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class PlotSpec:
    name: str
    width: float
    height: float
    input_names: tuple[str, ...]
    draw: Callable[[dict[str, pd.DataFrame]], Figure]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
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
            "mathtext.sf": "Avenir Next",
            "mathtext.fallback": None,
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "lines.linewidth": 1.45,
            "lines.markersize": 5.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "iclr2027-rebuilt-appendix-v1",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        },
    )


def finish_axis(ax: Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.grid(False)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.65)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(length=3.0, width=0.75, pad=2.5)


def horizontal_interval(
    ax: Axes,
    x: float,
    low: float,
    high: float,
    y: float,
    color: str,
    marker: str,
    *,
    filled: bool = True,
    size: float = 42,
) -> None:
    ax.errorbar(
        x,
        y,
        xerr=[[x - low], [high - x]],
        fmt="none",
        ecolor=color,
        elinewidth=1.15,
        capsize=2.6,
        capthick=1.0,
        zorder=2,
    )
    ax.scatter(
        [x],
        [y],
        s=size,
        marker=marker,
        facecolor=color if filled else WHITE,
        edgecolor=color,
        linewidth=1.05,
        zorder=3,
    )


def vertical_intervals(ax: Axes, frame: pd.DataFrame, color: str) -> None:
    x = frame["depth"].to_numpy(float)
    y = frame["estimate"].to_numpy(float)
    low = frame["ci_low"].to_numpy(float)
    high = frame["ci_high"].to_numpy(float)
    ax.errorbar(
        x,
        y,
        yerr=[y - low, high - y],
        fmt="none",
        ecolor=color,
        elinewidth=1.05,
        capsize=2.4,
        capthick=1.0,
        alpha=0.9,
        zorder=2,
    )


def draw_ecdf(
    ax: Axes,
    values: np.ndarray,
    color: str,
    *,
    marker: str = "o",
    linestyle: str | tuple = "-",
    filled: bool = True,
    label: str | None = None,
) -> None:
    ordered = np.sort(values.astype(float))
    y = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
    ax.step(ordered, y, where="post", color=color, linestyle=linestyle, linewidth=1.35, label=label)
    ax.scatter(
        ordered,
        y,
        s=13,
        marker=marker,
        facecolor=color if filled else WHITE,
        edgecolor=color,
        linewidth=0.65,
        alpha=0.9,
        zorder=3,
    )


def c1_backbone(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_c1_backbone_heterogeneity.csv"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.45), sharex=True, sharey=True)
    specs = (
        ("Ouro-2.6B", "2Wiki", BLUE, "o", "K1 to K3"),
        ("Ouro-2.6B", "MuSiQue", VERMILION, "s", "K1 to K3"),
        ("LoopUS", "2Wiki", GREEN, "D", "K1 to K8"),
        ("LoopUS", "MuSiQue", PURPLE, "^", "K1 to K8"),
    )
    for ax, (model, dataset, color, marker, depths) in zip(axes.flat, specs, strict=True):
        subset = frame[(frame["model"] == model) & (frame["dataset"] == dataset)]
        draw_ecdf(ax, subset["depth_gain"].to_numpy(), color, marker=marker)
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
        ax.set_title(f"{model} · {dataset} · {depths}", loc="left", color=color, pad=4)
        ax.set_xlim(-1.15, 8.95)
        ax.set_ylim(0, 1.035)
        ax.set_yticks([0, 0.5, 1.0])
        finish_axis(ax, xgrid=True, ygrid=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("")
    axes[1, 0].set_xlabel("Cluster mean depth gain (nats)")
    axes[1, 1].set_xlabel("Cluster mean depth gain (nats)")
    fig.supylabel("Empirical cumulative fraction", x=0.018, fontsize=9)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.915, wspace=0.17, hspace=0.38)
    return fig


def c1_specificity(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_c1_specificity_heterogeneity.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.70), sharex=True, sharey=True)
    specs = (("Ouro-2.6B", 3), ("LoopUS", 8))
    for ax, (model, depth) in zip(axes, specs, strict=True):
        subset = frame[frame["model"] == model]
        deep = subset[subset["metric"] == "deep specificity"]
        gain = subset[subset["metric"] == "specificity gain"]
        draw_ecdf(ax, deep["value"].to_numpy(), BLUE, marker="o", label="Deep specificity")
        draw_ecdf(
            ax,
            gain["value"].to_numpy(),
            VERMILION,
            marker="s",
            linestyle=(0, (4, 2)),
            filled=False,
            label="Specificity gain",
        )
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
        ax.set_title(f"{model} · deep K{depth}", loc="left", pad=4)
        ax.set_xlim(-0.5, 9.05)
        ax.set_ylim(0, 1.035)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("")
        finish_axis(ax, xgrid=True, ygrid=False)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.64, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.15,
        columnspacing=1.5,
    )
    fig.supxlabel("Relation-path mean (nats)", y=0.045, fontsize=9)
    fig.supylabel("Empirical cumulative fraction", x=0.018, fontsize=9)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.80, wspace=0.15)
    return fig


def e1_residual(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_e1_residual_depth.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(5.35, 2.18), sharey=True)
    specs = (("2Wiki", BLUE, "o", "-"), ("MuSiQue", VERMILION, "s", (0, (4, 2))))
    for index, (ax, (dataset, color, marker, linestyle)) in enumerate(zip(axes, specs, strict=True)):
        subset = frame[frame["dataset"] == dataset].sort_values("depth")
        vertical_intervals(ax, subset, color)
        sns.lineplot(
            data=subset,
            x="depth",
            y="estimate",
            estimator=None,
            errorbar=None,
            color=color,
            marker=marker,
            markersize=5.6,
            markeredgecolor=WHITE,
            markeredgewidth=0.65,
            linestyle=linestyle,
            linewidth=1.5,
            legend=False,
            ax=ax,
            zorder=3,
        )
        ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
        ax.set_title(dataset, loc="left", color=color, pad=4)
        ax.set_xlabel("Exit $K$")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.82, 4.18)
        ax.set_ylim(-0.14, 1.48)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        finish_axis(ax, xgrid=False, ygrid=True)
        if index == 0:
            ax.set_ylabel("Original minus repair (nats)")
        else:
            ax.set_ylabel("")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.25, top=0.90, wspace=0.15)
    return fig


def e2_stage1(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_e2_stage1_pairs.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    specs = (("2Wiki", BLUE), ("MuSiQue", VERMILION))
    for index, (ax, (dataset, color)) in enumerate(zip(axes, specs, strict=True)):
        subset = frame[frame["dataset"] == dataset]
        sns.scatterplot(
            data=subset,
            x="isolated_fact_preference",
            y="frozen_k3_residual",
            color=color,
            s=19,
            alpha=0.50,
            edgecolor=WHITE,
            linewidth=0.25,
            legend=False,
            ax=ax,
        )
        sns.regplot(
            data=subset,
            x="isolated_fact_preference",
            y="frozen_k3_residual",
            scatter=False,
            ci=None,
            truncate=True,
            color=color,
            line_kws={"linewidth": 1.25, "alpha": 0.9},
            ax=ax,
        )
        ax.axhline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        ax.set_title(dataset, loc="left", color=color, pad=4)
        ax.set_xlabel("Isolated terminal-fact preference (nats)")
        ax.set_xlim(-4.2, 7.4)
        ax.set_ylim(-3.4, 5.6)
        finish_axis(ax, xgrid=True, ygrid=True)
        if index == 0:
            ax.set_ylabel("Frozen K3 residual (nats)")
        else:
            ax.set_ylabel("")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.225, top=0.91, wspace=0.15)
    return fig


def e2_stage2(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_e2_stage2_effects.csv"]
    order = ["2Wiki", "MuSiQue", "Combined"]
    fig, ax = plt.subplots(figsize=(3.75, 1.95))
    colors = {"2Wiki": BLUE, "MuSiQue": VERMILION, "Combined": MUTED}
    markers = {"2Wiki": "o", "MuSiQue": "s", "Combined": "D"}
    for y, label in enumerate(reversed(order)):
        row = frame[frame["dataset"] == label].iloc[0]
        horizontal_interval(
            ax,
            float(row["estimate"]),
            float(row["ci_low"]),
            float(row["ci_high"]),
            y,
            colors[label],
            markers[label],
            filled=label != "Combined",
        )
    ax.axvline(0, color=INK, linewidth=0.85, linestyle=(0, (3, 2)), zorder=0)
    ax.set_title("Fact-only instruction: residual reduction", loc="left", pad=6)
    ax.set_xlabel("Registered reduction estimand (nats)")
    ax.set_yticks(range(3), labels=list(reversed(order)))
    ax.set_xlim(-0.082, 0.082)
    ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
    ax.set_ylim(-0.48, 2.48)
    finish_axis(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.28, top=0.82)
    return fig


def f1_retention(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_f1_list_retention.csv"]
    fig, ax = plt.subplots(figsize=(4.25, 2.35))
    metrics = ["G31", "D3"]
    for base_y, metric in enumerate(metrics):
        for interface, offset, color, marker, filled in (
            ("LIST", 0.10, BLUE, "o", True),
            ("NO_LIST", -0.10, VERMILION, "s", False),
        ):
            row = frame[(frame["metric"] == metric) & (frame["interface"] == interface)].iloc[0]
            horizontal_interval(
                ax,
                float(row["estimate"]),
                float(row["ci_low"]),
                float(row["ci_high"]),
                base_y + offset,
                color,
                marker,
                filled=filled,
            )
    ax.scatter([], [], s=42, marker="o", facecolor=BLUE, edgecolor=BLUE, label="LIST")
    ax.scatter([], [], s=42, marker="s", facecolor=WHITE, edgecolor=VERMILION, label="NO_LIST")
    ax.legend(loc="upper center", bbox_to_anchor=(0.56, 1.42), ncol=2, frameon=False)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.set_title("Likelihood effect survives answer-list removal", loc="left", pad=6)
    ax.set_xlabel("Graph-control effect (nats)")
    ax.set_yticks([0, 1], labels=["Depth gain $G_{31}$", "Deep effect $D_3$"])
    ax.set_xlim(-0.03, 0.94)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_ylim(-0.42, 1.42)
    finish_axis(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.24, top=0.70)
    return fig


def g1_emitted(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_g1_emitted_choice.csv"].copy()
    frame["label"] = frame.apply(
        lambda row: f"{row['model']} · {row['dataset']} · K{int(row['depth'])}", axis=1
    )
    order = ["Ouro-2.6B · 2Wiki · K3", "Ouro-2.6B · MuSiQue · K3", "LoopUS · 2Wiki · K8"]
    fig, ax = plt.subplots(figsize=(5.25, 2.52))
    for base_y, label in enumerate(reversed(order)):
        subset = frame[frame["label"] == label]
        for metric, offset, color, marker, filled in (
            ("deep effect", 0.10, BLUE, "o", True),
            ("depth gain", -0.10, VERMILION, "s", False),
        ):
            row = subset[subset["metric"] == metric].iloc[0]
            horizontal_interval(
                ax,
                float(row["estimate"]),
                float(row["ci_low"]),
                float(row["ci_high"]),
                base_y + offset,
                color,
                marker,
                filled=filled,
            )
    ax.scatter([], [], s=42, marker="o", facecolor=BLUE, edgecolor=BLUE, label="Deep effect")
    ax.scatter([], [], s=42, marker="s", facecolor=WHITE, edgecolor=VERMILION, label="Depth gain")
    ax.legend(loc="upper center", bbox_to_anchor=(0.58, 1.42), ncol=2, frameon=False)
    ax.axvline(0, color=INK, linewidth=0.85, linestyle=(0, (3, 2)), zorder=0)
    ax.set_title("Emitted-choice effects", loc="left", pad=6)
    ax.set_xlabel("Signed probability difference")
    ax.set_yticks(range(3), labels=list(reversed(order)))
    ax.set_xlim(-0.18, 0.33)
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.set_ylim(-0.46, 2.46)
    finish_axis(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(left=0.38, right=0.97, bottom=0.22, top=0.72)
    return fig


def g1_accuracy(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_g1_state_accuracy.csv"].copy()
    frame["label"] = frame.apply(
        lambda row: (
            f"{row['model']} · {row['dataset']} · K{int(row['depth'])}\n"
            f"{row['parser']} parser"
        ),
        axis=1,
    )
    order = list(frame["label"])
    fig, ax = plt.subplots(figsize=(5.25, 2.35))
    colors = [BLUE, VERMILION, GREEN]
    markers = ["o", "s", "D"]
    for y, (label, color, marker) in enumerate(zip(reversed(order), reversed(colors), reversed(markers), strict=True)):
        row = frame[frame["label"] == label].iloc[0]
        horizontal_interval(
            ax,
            float(row["estimate"]),
            float(row["ci_low"]),
            float(row["ci_high"]),
            y,
            color,
            marker,
        )
    ax.axvline(0.5, color=INK, linewidth=0.9, linestyle=(0, (3, 2)), zorder=0)
    ax.set_title("Competent free answering is not established", loc="left", pad=6)
    ax.set_xlabel("State-correct accuracy")
    ax.set_yticks(range(3), labels=list(reversed(order)))
    ax.set_xlim(0, 0.65)
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_ylim(-0.48, 2.48)
    finish_axis(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(left=0.39, right=0.97, bottom=0.24, top=0.86)
    return fig


def h1_access(frames: dict[str, pd.DataFrame]) -> Figure:
    frame = frames["appendix_h1_causal_access.csv"]
    order = list(frame["label"])
    palette = {
        "donor directed": BLUE,
        "mirrored": RED,
        "control": NEUTRAL,
        "positive control": GREEN,
    }
    markers = {
        "donor directed": "o",
        "mirrored": "s",
        "control": "D",
        "positive control": "^",
    }
    fig, ax = plt.subplots(figsize=(5.35, 3.05))
    for y, label in enumerate(reversed(order)):
        row = frame[frame["label"] == label].iloc[0]
        role = str(row["role"])
        horizontal_interval(
            ax,
            float(row["estimate"]),
            float(row["ci_low"]),
            float(row["ci_high"]),
            y,
            palette[role],
            markers[role],
            filled=role != "control",
            size=39,
        )
    ax.axvline(0, color=INK, linewidth=0.85, linestyle=(0, (3, 2)), zorder=0)
    ax.set_title("Candidate states provide causal access to preference", loc="left", pad=6)
    ax.set_xlabel(r"Normalized donor-preference recovery, $\rho$")
    ax.set_yticks(range(len(order)), labels=list(reversed(order)))
    ax.set_xlim(-0.30, 1.12)
    ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.55, len(order) - 0.45)
    finish_axis(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(left=0.38, right=0.97, bottom=0.19, top=0.90)
    return fig


def load_frames(manifest: dict, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    records = {Path(item["path"]).name: item for item in manifest["files"]}
    result: dict[str, pd.DataFrame] = {}
    for name in names:
        path = DATA_DIR / name
        if name not in records:
            raise ValueError(f"{name} is absent from data manifest")
        if sha256(path) != records[name]["sha256"]:
            raise ValueError(f"data drift for {name}")
        result[name] = pd.read_csv(path)
    return result


def render(spec: PlotSpec, frames: dict[str, pd.DataFrame]) -> list[Path]:
    figure = spec.draw(frames)
    observed = tuple(round(value, 2) for value in figure.get_size_inches())
    expected = (round(spec.width, 2), round(spec.height, 2))
    if observed != expected:
        raise ValueError(f"size mismatch for {spec.name}: {observed} != {expected}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    metadata = {"Creator": "anonymous reproducible appendix plotter"}
    for suffix in ("svg", "pdf", "png"):
        output = OUT_DIR / f"{spec.name}.{suffix}"
        kwargs: dict = {"bbox_inches": None, "facecolor": WHITE}
        if suffix == "png":
            kwargs["dpi"] = 300
            kwargs["metadata"] = {"Software": metadata["Creator"]}
        else:
            kwargs["metadata"] = metadata
        figure.savefig(output, **kwargs)
        outputs.append(output)
    plt.close(figure)
    return outputs


def main() -> None:
    configure_style()
    data_manifest_path = DATA_DIR / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    specs = (
        PlotSpec("appendix_c1_backbone_heterogeneity", 7.2, 3.45, ("appendix_c1_backbone_heterogeneity.csv",), c1_backbone),
        PlotSpec("appendix_c1_specificity_heterogeneity", 7.2, 2.70, ("appendix_c1_specificity_heterogeneity.csv",), c1_specificity),
        PlotSpec("appendix_e1_residual_depth", 5.35, 2.18, ("appendix_e1_residual_depth.csv",), e1_residual),
        PlotSpec("appendix_e2_stage1_compatibility", 7.2, 2.65, ("appendix_e2_stage1_pairs.csv",), e2_stage1),
        PlotSpec("appendix_e2_stage2_intervention", 3.75, 1.95, ("appendix_e2_stage2_effects.csv",), e2_stage2),
        PlotSpec("appendix_f1_list_retention", 4.25, 2.35, ("appendix_f1_list_retention.csv",), f1_retention),
        PlotSpec("appendix_g1_emitted_choice", 5.25, 2.52, ("appendix_g1_emitted_choice.csv",), g1_emitted),
        PlotSpec("appendix_g1_state_accuracy", 5.25, 2.35, ("appendix_g1_state_accuracy.csv",), g1_accuracy),
        PlotSpec("appendix_h1_causal_access", 5.35, 3.05, ("appendix_h1_causal_access.csv",), h1_access),
    )
    all_outputs: list[Path] = []
    rendered: list[dict] = []
    for spec in specs:
        frames = load_frames(data_manifest, spec.input_names)
        outputs = render(spec, frames)
        all_outputs.extend(outputs)
        rendered.append(
            {
                "name": spec.name,
                "width_inches": spec.width,
                "height_inches": spec.height,
                "inputs": list(spec.input_names),
                "outputs": [
                    {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in outputs
                ],
            }
        )
        print(f"{spec.name}: svg, pdf, png")
    generator = Path(__file__).resolve()
    manifest = {
        "schema": "iclr2027-rebuilt-appendix-statistical-v1",
        "generator": {"path": str(generator.relative_to(ROOT)), "sha256": sha256(generator)},
        "data_manifest": {
            "path": str(data_manifest_path.relative_to(ROOT)),
            "sha256": sha256(data_manifest_path),
        },
        "style": {
            "font": "Avenir Next with Helvetica Neue and Arial fallbacks",
            "minimum_font_points": 9,
            "embedded_numeric_ci_labels": False,
            "embedded_figure_numbers": False,
            "embedded_captions": False,
            "png_dpi": 300,
        },
        "versions": {
            "matplotlib": mpl.__version__,
            "pandas": pd.__version__,
            "seaborn": sns.__version__,
        },
        "figures": rendered,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
