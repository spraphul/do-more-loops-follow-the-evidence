#!/usr/bin/env python3
"""Reproduce the text-free Ouro answer-boundary diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Hashable
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ROW_SCHEMA = "evidence-loops.decoding-boundary.rows.v1"
RESULT_SCHEMA = "evidence-loops.decoding-boundary.result.v1"
SEED = 20260914
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_ROWS = {"2Wiki": 992, "MuSiQue": 760}
VARIANTS = ("registered", "bare", "two_surface")
BOUNDARY_CATEGORIES = (
    "correct_candidate_prefix",
    "wrong_candidate_prefix",
    "shared_candidate_prefix",
    "outside_candidate_prefixes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shards(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for artifact, path in zip(artifacts, paths, strict=True):
        if artifact.get("schema") != ROW_SCHEMA:
            raise RuntimeError(f"unexpected decoding-boundary schema: {path}")
    shard_counts = {int(artifact["shard"]["count"]) for artifact in artifacts}
    shard_indices = sorted(int(artifact["shard"]["index"]) for artifact in artifacts)
    runner_hashes = {
        artifact["source_bindings"]["runner_sha256"] for artifact in artifacts
    }
    inventory_hashes = {
        artifact["source_bindings"]["full_inventory_sha256"]
        for artifact in artifacts
    }
    if (
        shard_counts != {4}
        or shard_indices != list(range(4))
        or len(runner_hashes) != 1
        or len(inventory_hashes) != 1
    ):
        raise RuntimeError("decoding-boundary shard bindings disagree")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    keys = [(row["dataset"], row["row_id"]) for row in rows]
    counts = Counter(row["dataset"] for row in rows)
    if len(keys) != len(set(keys)) or dict(counts) != EXPECTED_ROWS:
        raise RuntimeError("decoding-boundary row union changed")
    return rows, {
        "runner_source_sha256": next(iter(runner_hashes)),
        "full_inventory_sha256": next(iter(inventory_hashes)),
        "shards": [
            {"path": path.name, "sha256": sha256_file(path)} for path in paths
        ],
    }


def reference_choice_credit(margin: float) -> float:
    if margin > 0.0:
        return 1.0
    if margin < 0.0:
        return 0.0
    return 0.5


def graph_response(rows: list[dict[str, Any]], variant: str, depth: int) -> float:
    selected = [row for row in rows if int(row["K"]) == depth]
    original = [
        float(row["reference_margin_by_variant"][variant])
        for row in selected
        if row["cell"] == "XV"
    ]
    swapped = [
        float(row["reference_margin_by_variant"][variant])
        for row in selected
        if row["cell"] == "XN"
    ]
    if not original or len(original) != len(swapped):
        raise RuntimeError("unbalanced XV/XN graph-response rows")
    return 0.5 * (mean(original) - mean(swapped))


def choice_control(rows: list[dict[str, Any]], variant: str, depth: int) -> float:
    selected = [row for row in rows if int(row["K"]) == depth]
    original = [
        reference_choice_credit(float(row["reference_margin_by_variant"][variant]))
        for row in selected
        if row["cell"] == "XV"
    ]
    swapped = [
        reference_choice_credit(float(row["reference_margin_by_variant"][variant]))
        for row in selected
        if row["cell"] == "XN"
    ]
    if not original or len(original) != len(swapped):
        raise RuntimeError("unbalanced XV/XN choice-control rows")
    return mean(original) - mean(swapped)


def dataset_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    k3 = [row for row in rows if int(row["K"]) == 3]
    for variant in VARIANTS:
        output[f"{variant}_pairwise_state_accuracy"] = mean(
            0.5
            if row["predicted_identity_by_variant"][variant] is None
            else float(
                row["predicted_identity_by_variant"][variant]
                == row["expected_candidate_identity"]
            )
            for row in k3
        )
        comparable = [
            row
            for row in k3
            if row["generated_p0_identity"] is not None
            and row["predicted_identity_by_variant"][variant] is not None
        ]
        output[f"{variant}_greedy_identity_agreement_given_adherence"] = mean(
            row["predicted_identity_by_variant"][variant]
            == row["generated_p0_identity"]
            for row in comparable
        )
        d1 = graph_response(rows, variant, 1)
        d3 = graph_response(rows, variant, 3)
        output[f"{variant}_graph_response_D1"] = d1
        output[f"{variant}_graph_response_D3"] = d3
        output[f"{variant}_graph_response_G31"] = d3 - d1
        c1 = choice_control(rows, variant, 1)
        c3 = choice_control(rows, variant, 3)
        output[f"{variant}_choice_control_C1"] = c1
        output[f"{variant}_choice_control_C3"] = c3
        output[f"{variant}_choice_control_H31"] = c3 - c1
    for depth in (1, 3):
        selected = [row for row in rows if int(row["K"]) == depth]
        for category in BOUNDARY_CATEGORIES:
            output[f"K{depth}_boundary_{category}"] = mean(
                row["boundary"]["category"] == category for row in selected
            )
        output[f"K{depth}_boundary_correct_candidate_rank_mean"] = mean(
            row["boundary"]["best_candidate_first_token_rank_by_identity"][
                str(row["expected_candidate_identity"])
            ]
            for row in selected
        )
        output[f"K{depth}_boundary_correct_candidate_rank_at_most_10"] = mean(
            row["boundary"]["best_candidate_first_token_rank_by_identity"][
                str(row["expected_candidate_identity"])
            ]
            <= 10
            for row in selected
        )
        output[f"K{depth}_boundary_any_candidate_rank_at_most_10"] = mean(
            min(
                row["boundary"][
                    "best_candidate_first_token_rank_by_identity"
                ].values()
            )
            <= 10
            for row in selected
        )
        output[f"K{depth}_boundary_replay_top1_matches_stored"] = mean(
            row["boundary"]["replay_top1_matches_stored"] for row in selected
        )
    output["registered_to_bare_prediction_status_change_including_ties"] = mean(
        row["predicted_identity_by_variant"]["registered"]
        != row["predicted_identity_by_variant"]["bare"]
        for row in k3
    )
    non_tied = [
        row
        for row in k3
        if row["predicted_identity_by_variant"]["registered"] is not None
        and row["predicted_identity_by_variant"]["bare"] is not None
    ]
    output["registered_to_bare_prediction_reversal_given_non_ties"] = mean(
        row["predicted_identity_by_variant"]["registered"]
        != row["predicted_identity_by_variant"]["bare"]
        for row in non_tied
    )
    output["correct_prefix_but_final_not_correct"] = mean(
        row["boundary"]["category"] == "correct_candidate_prefix"
        and not row["generated_p0_state_correct"]
        for row in k3
    )
    output["boundary_replay_top1_matches_stored"] = mean(
        row["boundary"]["replay_top1_matches_stored"] for row in rows
    )
    return output


def pair_clusters(
    rows: list[dict[str, Any]], dataset: str
) -> tuple[list[Hashable], dict[Hashable, list[dict[str, Any]]]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    if dataset == "2Wiki":
        by_path: dict[str, list[str]] = defaultdict(list)
        for pair_id, pair_rows in by_pair.items():
            path_id = pair_rows[0].get("relation_path_id")
            if not path_id:
                raise RuntimeError("2Wiki relation-path stratum missing")
            by_path[str(path_id)].append(pair_id)
        paths = sorted(by_path)
        if len(paths) != 31 or {len(ids) for ids in by_path.values()} != {4}:
            raise RuntimeError("2Wiki relation-path cluster structure changed")
        return paths, {
            path: [
                row
                for pair_id in sorted(by_path[path])
                for row in by_pair[pair_id]
            ]
            for path in paths
        }
    pair_ids = sorted(by_pair)
    return pair_ids, dict(by_pair)


def bootstrap_metrics(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    observed = dataset_metrics(rows)
    units, unit_rows = pair_clusters(rows, dataset)
    rng = np.random.default_rng(SEED + (0 if dataset == "2Wiki" else 1))
    draws = {name: [] for name in observed}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[dict[str, Any]] = []
        for unit_index in rng.integers(0, len(units), size=len(units)):
            unit = units[int(unit_index)]
            if dataset == "2Wiki":
                pair_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in unit_rows[unit]:
                    pair_map[row["pair_id"]].append(row)
                pair_ids = sorted(pair_map)
                for pair_index in rng.integers(0, len(pair_ids), size=len(pair_ids)):
                    sampled.extend(pair_map[pair_ids[int(pair_index)]])
            else:
                sampled.extend(unit_rows[unit])
        values = dataset_metrics(sampled)
        for name, value in values.items():
            draws[name].append(value)
    return {
        name: {
            "estimate": float(value),
            "ci95": [
                float(np.quantile(draws[name], 0.025)),
                float(np.quantile(draws[name], 0.975)),
            ],
        }
        for name, value in observed.items()
    }


def descriptive_details(rows: list[dict[str, Any]]) -> dict[str, Any]:
    k3 = [row for row in rows if int(row["K"]) == 3]
    ranks_by_depth = {
        depth: [
            row["boundary"]["best_candidate_first_token_rank_by_identity"][
                str(row["expected_candidate_identity"])
            ]
            for row in rows
            if int(row["K"]) == depth
        ]
        for depth in (1, 3)
    }
    transitions = Counter(
        (
            row["boundary"]["category"],
            "final_correct"
            if row["generated_p0_state_correct"]
            else "final_wrong_candidate"
            if row["generated_p0_adherent"]
            else "final_invalid",
        )
        for row in k3
    )
    return {
        "rows_K1_K3": len(rows),
        "rows_K3": len(k3),
        "pairs": len({row["pair_id"] for row in rows}),
        "correct_candidate_first_token_rank_by_K": {
            str(depth): {
                "median": float(median(ranks)),
                "p25": float(np.quantile(ranks, 0.25)),
                "p75": float(np.quantile(ranks, 0.75)),
                "maximum": int(max(ranks)),
            }
            for depth, ranks in ranks_by_depth.items()
        },
        "boundary_to_final_counts": {
            f"{boundary}__{final}": count
            for (boundary, final), count in sorted(transitions.items())
        },
        "K3_greedy_agreement_denominators": {
            variant: sum(
                row["generated_p0_identity"] is not None
                and row["predicted_identity_by_variant"][variant] is not None
                for row in k3
            )
            for variant in VARIANTS
        },
        "K3_registered_to_bare_non_tied_rows": sum(
            row["predicted_identity_by_variant"]["registered"] is not None
            and row["predicted_identity_by_variant"]["bare"] is not None
            for row in k3
        ),
        "privacy_note": (
            "Decoded first-token strings and tokenizer IDs are excluded from the "
            "anonymous release; all paper-facing categories and ranks are retained."
        ),
    }


def format_metric(metric: dict[str, Any]) -> str:
    return (
        f"{metric['estimate']:.3f} "
        f"[{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    rows, bindings = load_shards([path.resolve() for path in args.inputs])
    datasets: dict[str, Any] = {}
    for dataset in ("2Wiki", "MuSiQue"):
        selected = [row for row in rows if row["dataset"] == dataset]
        datasets[dataset] = {
            "metrics": bootstrap_metrics(selected, dataset),
            "details": descriptive_details(selected),
        }
    result = {
        "schema": RESULT_SCHEMA,
        "status": "post_outcome_diagnostic",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "2Wiki": "relation path then whole pair",
            "MuSiQue": "whole pair",
        },
        "bindings": bindings,
        "datasets": datasets,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Ouro-2.6B answer-boundary audit",
        "",
        "Status: post-outcome diagnostic; no registered endpoint is changed.",
    ]
    for dataset in ("2Wiki", "MuSiQue"):
        metrics = datasets[dataset]["metrics"]
        details = datasets[dataset]["details"]
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                f"Population: {details['rows_K3']} K3 rows from {details['pairs']} pairs.",
                f"- Bare K3 graph response: {format_metric(metrics['bare_graph_response_D3'])}",
                f"- Bare K1-to-K3 choice gain: {format_metric(metrics['bare_choice_control_H31'])}",
                "- K3 prompt-only replay parity: "
                f"{format_metric(metrics['K3_boundary_replay_top1_matches_stored'])}",
                "- K3 outside-prefix starts: "
                f"{format_metric(metrics['K3_boundary_outside_candidate_prefixes'])}",
            ]
        )
    lines.extend(
        [
            "",
            "The diagnostic identifies a score-to-decoding boundary. It does not "
            "change the registered generation result or establish a neural mechanism.",
        ]
    )
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.json_output)
    print(args.markdown_output)


if __name__ == "__main__":
    main()
