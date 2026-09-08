#!/usr/bin/env python3
"""Analyze paired seeds from the tied/untied graph-composition factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RUN_SCHEMA = "iclr2027.tied_untied_graph_factorial.run.v1"
RESULT_SCHEMA = "iclr2027.tied_untied_graph_factorial.result.v1"
CONDITIONS = (
    ("tied", "single"),
    ("tied", "multi"),
    ("untied", "single"),
    ("untied", "multi"),
)
OUTCOMES = (
    "raw_gain_K4_minus_K1",
    "choice_gain_K4_minus_K1",
    "raw_effect_K4",
    "choice_effect_K4",
    "state_accuracy_K4",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SALT = "iclr2027-tied-untied-factorial-v1"


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SALT}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def condition_key(tying: str, supervision: str) -> str:
    return f"{tying}_{supervision}"


def load(paths: list[Path]) -> dict[int, dict[str, dict[str, Any]]]:
    runs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    comparison_configuration: dict[str, Any] | None = None
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("schema") != RUN_SCHEMA:
            raise RuntimeError(f"wrong schema: {path}")
        tying = artifact["condition"]["tying"]
        supervision = artifact["condition"]["supervision"]
        if (tying, supervision) not in CONDITIONS:
            raise RuntimeError(f"unexpected condition: {path}")
        key = condition_key(tying, supervision)
        seed = int(artifact["seed"])
        if key in runs[seed]:
            raise RuntimeError(f"duplicate seed-condition: {seed} {key}")
        config = dict(artifact["configuration"])
        for ignored in ("tying", "supervision", "seed", "log_every"):
            config.pop(ignored, None)
        if comparison_configuration is None:
            comparison_configuration = config
        elif config != comparison_configuration:
            raise RuntimeError("run hyperparameters differ across factorial cells")
        rows = artifact["evaluation"]["per_world"]
        if not rows or len({int(row["world_id"]) for row in rows}) != len(rows):
            raise RuntimeError(f"invalid evaluation worlds: {path}")
        runs[seed][key] = artifact
    expected = {condition_key(*condition) for condition in CONDITIONS}
    if not runs:
        raise RuntimeError("no runs supplied")
    for seed, cells in runs.items():
        if set(cells) != expected:
            raise RuntimeError(f"seed {seed} lacks a complete 2x2 factorial")
        world_ids = [
            [int(row["world_id"]) for row in cells[key]["evaluation"]["per_world"]]
            for key in sorted(cells)
        ]
        if any(ids != world_ids[0] for ids in world_ids[1:]):
            raise RuntimeError(f"seed {seed} is not paired over evaluation worlds")
    return dict(runs)


def vectors(
    runs: dict[int, dict[str, dict[str, Any]]], outcome: str
) -> dict[int, dict[str, np.ndarray]]:
    output: dict[int, dict[str, np.ndarray]] = {}
    for seed, cells in runs.items():
        output[seed] = {
            key: np.asarray(
                [float(row[outcome]) for row in artifact["evaluation"]["per_world"]]
            )
            for key, artifact in cells.items()
        }
    return output


def contrasts(cell: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ts = cell["tied_single"]
    tm = cell["tied_multi"]
    us = cell["untied_single"]
    um = cell["untied_multi"]
    return {
        "tied_single": ts,
        "tied_multi": tm,
        "untied_single": us,
        "untied_multi": um,
        "tying_effect_single": ts - us,
        "tying_effect_multi": tm - um,
        "supervision_effect_tied": tm - ts,
        "supervision_effect_untied": um - us,
        "factorial_interaction": (tm - ts) - (um - us),
    }


def estimate_hierarchical(
    by_seed: dict[int, dict[str, np.ndarray]], label: str
) -> dict[str, Any]:
    seeds = sorted(by_seed)
    observed_by_seed = {
        seed: float(np.mean(contrasts(by_seed[seed])[label])) for seed in seeds
    }
    point = float(np.mean(list(observed_by_seed.values())))
    if len(seeds) == 1:
        vector = contrasts(by_seed[seeds[0]])[label]
        rng = np.random.default_rng(stable_seed(label))
        bootstrap = np.asarray(
            [float(np.mean(rng.choice(vector, size=len(vector), replace=True)))
             for _ in range(BOOTSTRAP_REPLICATES)]
        )
        resampling = "development only: evaluation worlds within one training seed"
    else:
        rng = np.random.default_rng(stable_seed(label))
        bootstrap_values: list[float] = []
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            replicate: list[float] = []
            for seed in sampled_seeds:
                vector = contrasts(by_seed[int(seed)])[label]
                indices = rng.integers(0, len(vector), size=len(vector))
                replicate.append(float(np.mean(vector[indices])))
            bootstrap_values.append(float(np.mean(replicate)))
        bootstrap = np.asarray(bootstrap_values)
        resampling = "training seeds, then paired evaluation worlds within seed"
    lower, upper = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "estimate": point,
        "ci95": [float(lower), float(upper)],
        "per_seed": {str(seed): value for seed, value in observed_by_seed.items()},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "resampling": resampling,
    }


def fmt(value: dict[str, Any]) -> str:
    return f"{value['estimate']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    runs = load(args.inputs)
    results: dict[str, Any] = {}
    labels = tuple(contrasts(vectors(runs, OUTCOMES[0])[next(iter(runs))]).keys())
    for outcome in OUTCOMES:
        by_seed = vectors(runs, outcome)
        results[outcome] = {
            label: estimate_hierarchical(by_seed, label) for label in labels
        }
    minimum_k4_accuracy = min(
        float(artifact["evaluation"]["summary"]["state_accuracy_K4"])
        for cells in runs.values()
        for artifact in cells.values()
    )
    choice_gain = results["choice_gain_K4_minus_K1"]
    decisions = {
        "all_cells_pass_k4_competence": minimum_k4_accuracy >= 0.95,
        "minimum_seed_cell_k4_accuracy": minimum_k4_accuracy,
        "final_only_depth_gain_tied": choice_gain["tied_single"]["ci95"][0] > 0,
        "final_only_depth_gain_untied": choice_gain["untied_single"]["ci95"][0] > 0,
        "unique_tying_effect_under_final_only_supported": (
            choice_gain["tying_effect_single"]["ci95"][0] > 0
        ),
        "all_exit_supervision_moves_control_earlier": (
            choice_gain["supervision_effect_tied"]["ci95"][1] < 0
            and choice_gain["supervision_effect_untied"]["ci95"][1] < 0
        ),
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "training_seeds": sorted(runs),
        "seed_count": len(runs),
        "inference": (
            "Hierarchical paired bootstrap over training seeds and held-out worlds. "
            "A one-seed development analysis reflects world uncertainty only."
        ),
        "results": results,
        "decisions": decisions,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Compute-matched tied versus untied graph factorial",
        "",
        f"Training seeds: {len(runs)} ({', '.join(map(str, sorted(runs)))})",
        "",
        "| Condition or paired contrast | K4-K1 choice gain | K4 choice effect | K4 accuracy |",
        "|---|---:|---:|---:|",
    ]
    for label in labels:
        lines.append(
            f"| {label.replace('_', ' ')} | "
            f"{fmt(results['choice_gain_K4_minus_K1'][label])} | "
            f"{fmt(results['choice_effect_K4'][label])} | "
            f"{fmt(results['state_accuracy_K4'][label])} |"
        )
    lines.extend(
        [
            "",
            "Raw-margin outcomes are retained in the JSON artifact.",
            (
                "This is a one-seed development diagnostic, not training-seed inference."
                if len(runs) == 1
                else "Intervals include both training-seed and held-out-world variation."
            ),
            "",
            f"All cells pass K4 competence: `{decisions['all_cells_pass_k4_competence']}`.",
            f"Final-only depth gain appears in tied and untied models: "
            f"`{decisions['final_only_depth_gain_tied'] and decisions['final_only_depth_gain_untied']}`.",
            f"A unique tying effect under final-only training is supported: "
            f"`{decisions['unique_tying_effect_under_final_only_supported']}`.",
            f"All-exit supervision moves control earlier: "
            f"`{decisions['all_exit_supervision_moves_control_earlier']}`.",
        ]
    )
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.markdown_output)


if __name__ == "__main__":
    main()
