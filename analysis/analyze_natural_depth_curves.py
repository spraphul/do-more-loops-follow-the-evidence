#!/usr/bin/env python3
"""Recompute natural-panel depth curves and coherent-repair fractions.

The distributed inputs contain no benchmark prose or entity strings.  They do
retain the complete paired score grid, relation strata, and graph-state labels
needed for the paper's estimands and registered bootstrap procedures.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True)
class Configuration:
    key: str
    filename: str
    dataset: str
    shallow: int
    deep: int
    expected_rows: int
    expected_pairs: int
    expected_clusters: int
    bootstrap: str
    seed: int
    percentile_method: str


CONFIGURATIONS = (
    Configuration(
        "ouro26_2wiki", "ouro26_2wiki.json", "2Wiki", 1, 3,
        5_952, 124, 31, "cluster_batch_then_sorted_pair", 987535634705159947,
        "linear",
    ),
    Configuration(
        "ouro26_musique", "ouro26_musique.json", "MuSiQue", 1, 3,
        4_560, 95, 30, "pair", 17924897406598346847, "linear",
    ),
    Configuration(
        "loopus_2wiki", "loopus_2wiki.json", "2Wiki", 1, 8,
        1_860, 93, 31, "cluster_then_insertion_order_pair", 9151286686978729323,
        "nearest",
    ),
    Configuration(
        "loopus_musique", "loopus_musique.json", "MuSiQue", 1, 8,
        1_900, 95, 30, "pair", 9276142855844983147, "nearest",
    ),
)


def percentile(values: list[float], probability: float, method: str) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    if method == "nearest":
        index = max(0, min(len(ordered) - 1, round(position)))
        return float(ordered[index])
    if method != "linear":
        raise ValueError(method)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def interval(values: list[float], method: str) -> list[float]:
    return [percentile(values, 0.025, method), percentile(values, 0.975, method)]


def cluster(row: dict[str, Any], dataset: str) -> str:
    if dataset == "2Wiki":
        return " -> ".join(map(str, row["relation_path"]))
    return str(row["terminal_relation"])


def load_rows(path: Path) -> list[dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"missing score rows in {path}")
    return rows


def aggregate(
    rows: list[dict[str, Any]], configuration: Configuration
) -> tuple[list[str], dict[str, str], dict[tuple[str, str, int], dict[str, float]]]:
    if len(rows) != configuration.expected_rows:
        raise RuntimeError(
            f"{configuration.key}: {len(rows)} rows, expected {configuration.expected_rows}"
        )
    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError(f"{configuration.key}: duplicate row IDs")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    pair_cluster: dict[str, str] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        current_cluster = cluster(row, configuration.dataset)
        previous = pair_cluster.setdefault(pair_id, current_cluster)
        if previous != current_cluster:
            raise RuntimeError(f"{configuration.key}: pair changes inference cluster")
        grouped[(pair_id, str(row["evidence_state"]), int(row["K"]))].append(row)

    pair_ids = sorted(pair_cluster)
    if len(pair_ids) != configuration.expected_pairs:
        raise RuntimeError(f"{configuration.key}: pair count changed")
    if len(set(pair_cluster.values())) != configuration.expected_clusters:
        raise RuntimeError(f"{configuration.key}: cluster count changed")

    cells: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, members in grouped.items():
        if len(members) != 4:
            raise RuntimeError(f"{configuration.key}: incomplete arm/order cell {key}")
        margins = [float(row["reference_log_odds"]) for row in members]
        choices = [float(row["reference_argmax_credit"]) for row in members]
        state_accuracy = [float(row["state_correct_argmax_credit"]) for row in members]
        if not all(math.isfinite(value) for value in margins):
            raise RuntimeError(f"{configuration.key}: non-finite margin")
        cells[key] = {
            "margin": mean(margins),
            "reference_choice": mean(choices),
            "state_accuracy": mean(state_accuracy),
        }
    return pair_ids, pair_cluster, cells


def bootstrap_draws(
    pair_ids: list[str], pair_cluster: dict[str, str], configuration: Configuration
) -> list[list[str]]:
    rng = random.Random(configuration.seed)
    if configuration.bootstrap == "pair":
        return [
            [rng.choice(pair_ids) for _ in pair_ids]
            for _ in range(BOOTSTRAP_REPLICATES)
        ]
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for pair_id, label in pair_cluster.items():
        by_cluster[label].append(pair_id)
    if configuration.bootstrap in {
        "cluster_batch_then_sorted_pair",
        "cluster_then_sorted_pair",
    }:
        for members in by_cluster.values():
            members.sort()
    clusters = sorted(by_cluster)
    member_counts = {len(members) for members in by_cluster.values()}
    if len(member_counts) != 1:
        raise RuntimeError(f"{configuration.key}: unequal cluster sizes")
    members_per_cluster = next(iter(member_counts))
    draws: list[list[str]] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[str] = []
        chosen_clusters = (
            [rng.choice(clusters) for _cluster_draw in clusters]
            if configuration.bootstrap == "cluster_batch_then_sorted_pair"
            else None
        )
        for cluster_index, _cluster_draw in enumerate(clusters):
            chosen_cluster = (
                chosen_clusters[cluster_index]
                if chosen_clusters is not None
                else rng.choice(clusters)
            )
            candidates = by_cluster[chosen_cluster]
            sampled.extend(
                rng.choice(candidates) for _ in range(members_per_cluster)
            )
        draws.append(sampled)
    return draws


def summarize(configuration: Configuration, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_ids, pair_cluster, cells = aggregate(rows, configuration)
    depths = sorted({int(row["K"]) for row in rows})
    vectors: dict[str, dict[int, dict[str, float]]] = {
        "D": {},
        "C": {},
        "repair_increment": {},
    }
    original_accuracy: dict[int, dict[str, float]] = {}
    for depth in depths:
        if not all(
            (pair_id, state, depth) in cells
            for pair_id in pair_ids
            for state in ("original", "bridge_swap")
        ):
            raise RuntimeError(f"{configuration.key}: incomplete depth {depth}")
        vectors["D"][depth] = {
            pair_id: cells[(pair_id, "original", depth)]["margin"]
            - cells[(pair_id, "bridge_swap", depth)]["margin"]
            for pair_id in pair_ids
        }
        vectors["C"][depth] = {
            pair_id: cells[(pair_id, "original", depth)]["reference_choice"]
            - cells[(pair_id, "bridge_swap", depth)]["reference_choice"]
            for pair_id in pair_ids
        }
        original_accuracy[depth] = {
            pair_id: cells[(pair_id, "original", depth)]["state_accuracy"]
            for pair_id in pair_ids
        }
        if all((pair_id, "topology_rescue", depth) in cells for pair_id in pair_ids):
            vectors["repair_increment"][depth] = {
                pair_id: cells[(pair_id, "topology_rescue", depth)]["margin"]
                - cells[(pair_id, "bridge_swap", depth)]["margin"]
                for pair_id in pair_ids
            }

    draws = bootstrap_draws(pair_ids, pair_cluster, configuration)

    def estimate(values: dict[str, float]) -> dict[str, Any]:
        sampled = [mean(values[pair_id] for pair_id in draw) for draw in draws]
        return {
            "estimate": mean(values.values()),
            "ci95": interval(sampled, configuration.percentile_method),
            "pair_units": len(values),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }

    by_depth: dict[str, Any] = {}
    for depth in depths:
        by_depth[str(depth)] = {
            "path_margin_D": estimate(vectors["D"][depth]),
            "choice_control_C": estimate(vectors["C"][depth]),
            "original_accuracy": estimate(original_accuracy[depth]),
        }
        if depth in vectors["repair_increment"]:
            numerator = vectors["repair_increment"][depth]
            denominator = vectors["D"][depth]
            ratio_draws: list[float] = []
            invalid = 0
            for draw in draws:
                den = mean(denominator[pair_id] for pair_id in draw)
                if den <= 0.0:
                    invalid += 1
                else:
                    ratio_draws.append(
                        mean(numerator[pair_id] for pair_id in draw) / den
                    )
            point_denominator = mean(denominator.values())
            point_numerator = mean(numerator.values())
            by_depth[str(depth)]["repair_fraction"] = {
                "estimate": point_numerator / point_denominator,
                "ci95": interval(ratio_draws, configuration.percentile_method),
                "invalid_nonpositive_denominators": invalid,
                "pair_units": len(pair_ids),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }

    shallow = configuration.shallow
    deep = configuration.deep
    margin_gain = {
        pair_id: vectors["D"][deep][pair_id] - vectors["D"][shallow][pair_id]
        for pair_id in pair_ids
    }
    choice_gain = {
        pair_id: vectors["C"][deep][pair_id] - vectors["C"][shallow][pair_id]
        for pair_id in pair_ids
    }
    return {
        "dataset": configuration.dataset,
        "rows": len(rows),
        "pairs": len(pair_ids),
        "relation_clusters": len(set(pair_cluster.values())),
        "depths": depths,
        "bootstrap": {
            "method": configuration.bootstrap,
            "seed": configuration.seed,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "by_K": by_depth,
        "depth_gain": {
            "shallow_K": shallow,
            "deep_K": deep,
            "margin": estimate(margin_gain),
            "choice": estimate(choice_gain),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo / "data/analysis_ready/natural",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "reproduced/results/natural_depth_curves.json",
    )
    args = parser.parse_args()
    results = {
        configuration.key: summarize(
            configuration, load_rows(args.data_dir / configuration.filename)
        )
        for configuration in CONFIGURATIONS
    }
    payload = {
        "schema": "evidence-loops.natural-depth-curves.v1",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
