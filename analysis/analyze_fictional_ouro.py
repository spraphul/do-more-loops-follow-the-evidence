#!/usr/bin/env python3
"""Merge and analyze completed fictional-graph path-control shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np


RUN_SCHEMA = "iclr2027.nonce_path_control.run.v1"
RESULT_SCHEMA = "iclr2027.nonce_path_control.result.v1"
PANEL_SHA256 = "7880c02c64af541629b9c23dce15bf4a6df98c4f99a8aed45b4a310df2e543e5"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SALT = "iclr2027-nonce-path-control-v1:full-analysis"
MODEL_DEPTHS = {
    "ouro26": (1, 2, 3, 4),
    "ouro14": (1, 2, 3, 4),
    "loopus": tuple(range(1, 9)),
    "tfqwen": (1, 2, 4, 6),
}
MODEL_DEEP = {"ouro26": 4, "ouro14": 4, "loopus": 8, "tfqwen": 6}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SALT}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def interval(values: np.ndarray, probability: float = 0.95) -> list[float]:
    tail = (1.0 - probability) / 2.0
    return [
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1.0 - tail)),
    ]


def load_artifact(path: Path, mode: str) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != RUN_SCHEMA
        or artifact.get("mode") != mode
        or artifact.get("panel", {}).get("canonical_sha256") != PANEL_SHA256
    ):
        raise RuntimeError(f"invalid {mode} artifact: {path}")
    return artifact


def merge_shards(paths: list[Path]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    artifacts = [load_artifact(path, "full") for path in paths]
    models = {artifact["model_key"] for artifact in artifacts}
    runners = {artifact["runner"]["sha256"] for artifact in artifacts}
    shard_counts = {int(artifact["shard"]["count"]) for artifact in artifacts}
    mock_flags = {bool(artifact.get("mock")) for artifact in artifacts}
    if (
        len(models) != 1
        or len(runners) != 1
        or len(shard_counts) != 1
        or len(mock_flags) != 1
    ):
        raise RuntimeError("full shards disagree on model, runner, shard count, or mock status")
    shard_count = next(iter(shard_counts))
    indices = [int(artifact["shard"]["index"]) for artifact in artifacts]
    if len(artifacts) != shard_count or sorted(indices) != list(range(shard_count)):
        raise RuntimeError("full shard set is incomplete")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("full shards contain duplicate row IDs")
    return next(iter(models)), rows, {
        "runner_sha256": next(iter(runners)),
        "mock": next(iter(mock_flags)),
        "shard_count": shard_count,
        "inputs": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in paths
        ],
    }


def validate_grid(rows: list[dict[str, Any]], model: str) -> None:
    depths = MODEL_DEPTHS[model]
    deep = MODEL_DEEP[model]
    repair_depths = depths if model in {"ouro26", "ouro14"} else (deep,)
    expected_rows = 288 * (8 * len(depths) + 4 * len(repair_depths))
    if len(rows) != expected_rows:
        raise RuntimeError(f"full row count changed: {len(rows)} != {expected_rows}")
    worlds = {row["world_id"] for row in rows}
    if len(worlds) != 288:
        raise RuntimeError("full result does not contain 288 worlds")
    by_cell = Counter(
        (row["world_id"], row["evidence_state"], int(row["K"])) for row in rows
    )
    expected: dict[tuple[str, str, int], int] = {}
    for world in worlds:
        for state in ("original", "bridge_swap"):
            for k in depths:
                expected[(world, state, k)] = 4
        for k in repair_depths:
            expected[(world, "topology_repair", k)] = 4
    if dict(by_cell) != expected:
        missing = set(expected) - set(by_cell)
        extra = set(by_cell) - set(expected)
        wrong = {
            key: (by_cell[key], expected[key])
            for key in set(by_cell) & set(expected)
            if by_cell[key] != expected[key]
        }
        raise RuntimeError(
            f"full factorial grid changed: missing={len(missing)}, "
            f"extra={len(extra)}, wrong={len(wrong)}"
        )
    for row in rows:
        score = row["score"]
        if set(score["raw_logits"]) != {"A", "B"}:
            raise RuntimeError("scored row does not retain both A/B logits")
        if score["expected_label"] != row["expected_label"]:
            raise RuntimeError("expected-label orientation changed")
        if not math.isfinite(float(score["reference_logit_margin"])):
            raise RuntimeError("non-finite reference margin")


def aggregate_worlds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_cell[(row["world_id"], row["evidence_state"], int(row["K"]))].append(row)
        observed = {
            "world_id": row["world_id"],
            "relation_family_index": int(row["relation_family_index"]),
            "ordered_code_pair": tuple(row["focal_terminal_codes"]),
        }
        previous = metadata.setdefault(row["world_id"], observed)
        if previous != observed:
            raise RuntimeError("world metadata changes across rows")
    units: list[dict[str, Any]] = []
    for key, cell in sorted(by_cell.items()):
        world, state, k = key
        if len(cell) != 4:
            raise RuntimeError("arm/order cell does not contain four rows")
        units.append(
            {
                **metadata[world],
                "state": state,
                "K": k,
                "reference_margin": mean(
                    float(row["score"]["reference_logit_margin"]) for row in cell
                ),
                "reference_choice_credit": mean(
                    float(row["score"]["reference_argmax_credit"]) for row in cell
                ),
                "state_correct_accuracy": mean(
                    float(row["score"]["state_correct_argmax_credit"]) for row in cell
                ),
            }
        )
    return units


def world_vectors(
    units: list[dict[str, Any]], model: str
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    depths = MODEL_DEPTHS[model]
    deep = MODEL_DEEP[model]
    lookup = {
        (unit["world_id"], unit["state"], int(unit["K"])): unit for unit in units
    }
    worlds = sorted({unit["world_id"] for unit in units})
    relation = np.asarray(
        [int(lookup[(world, "original", depths[0])]["relation_family_index"]) for world in worlds],
        dtype=int,
    )
    vectors: dict[str, np.ndarray] = {}
    for k in depths:
        original_margin = np.asarray(
            [lookup[(world, "original", k)]["reference_margin"] for world in worlds]
        )
        swap_margin = np.asarray(
            [lookup[(world, "bridge_swap", k)]["reference_margin"] for world in worlds]
        )
        original_choice = np.asarray(
            [lookup[(world, "original", k)]["reference_choice_credit"] for world in worlds]
        )
        swap_choice = np.asarray(
            [lookup[(world, "bridge_swap", k)]["reference_choice_credit"] for world in worlds]
        )
        vectors[f"D_K{k}"] = original_margin - swap_margin
        vectors[f"C_K{k}"] = original_choice - swap_choice
        vectors[f"original_accuracy_K{k}"] = np.asarray(
            [lookup[(world, "original", k)]["state_correct_accuracy"] for world in worlds]
        )
        vectors[f"swap_accuracy_K{k}"] = np.asarray(
            [lookup[(world, "bridge_swap", k)]["state_correct_accuracy"] for world in worlds]
        )
    vectors[f"repair_margin_K{deep}"] = np.asarray(
        [lookup[(world, "topology_repair", deep)]["reference_margin"] for world in worlds]
    )
    vectors[f"repair_accuracy_K{deep}"] = np.asarray(
        [lookup[(world, "topology_repair", deep)]["state_correct_accuracy"] for world in worlds]
    )
    if model in {"ouro26", "ouro14"}:
        for k in depths:
            vectors[f"repair_margin_K{k}"] = np.asarray(
                [
                    lookup[(world, "topology_repair", k)]["reference_margin"]
                    for world in worlds
                ]
            )
            vectors[f"repair_accuracy_K{k}"] = np.asarray(
                [
                    lookup[(world, "topology_repair", k)]["state_correct_accuracy"]
                    for world in worlds
                ]
            )
    return worlds, vectors, relation


def bootstrap_indices(relation: np.ndarray, model: str) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed(model))
    groups = {value: np.flatnonzero(relation == value) for value in sorted(set(relation))}
    if len(groups) != 6 or set(map(len, groups.values())) != {48}:
        raise RuntimeError("full panel relation-family balance changed")
    draws: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()]
        )
        draws.append(draw)
    return draws


def unstratified_bootstrap_indices(world_count: int, model: str) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed(f"{model}:unstratified"))
    return [
        rng.choice(world_count, size=world_count, replace=True)
        for _ in range(BOOTSTRAP_REPLICATES)
    ]


def metric(
    indices: list[np.ndarray], statistic: Callable[[np.ndarray], float],
    probability: float = 0.95,
) -> dict[str, Any]:
    full = np.arange(len(indices[0]), dtype=int)
    point = float(statistic(full))
    values = np.asarray([float(statistic(draw)) for draw in indices], dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 0.99 * len(values):
        raise RuntimeError("too many undefined bootstrap replicates")
    return {
        "estimate": point,
        f"ci{int(probability * 100)}": interval(finite, probability),
        "bootstrap_replicates": len(values),
        "finite_replicates": len(finite),
    }


def ratio_metric(
    indices: list[np.ndarray],
    numerator: Callable[[np.ndarray], float],
    denominator: Callable[[np.ndarray], float],
    probability: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap a ratio while treating a near-zero denominator as non-estimable."""
    full = np.arange(len(indices[0]), dtype=int)
    point_denominator = float(denominator(full))
    point = (
        float(numerator(full)) / point_denominator
        if not math.isclose(point_denominator, 0.0, abs_tol=1e-12)
        else None
    )
    values: list[float] = []
    for draw in indices:
        draw_denominator = float(denominator(draw))
        if math.isclose(draw_denominator, 0.0, abs_tol=1e-12):
            continue
        value = float(numerator(draw)) / draw_denominator
        if math.isfinite(value):
            values.append(value)
    minimum_finite = math.ceil(0.99 * len(indices))
    estimable = point is not None and len(values) >= minimum_finite
    key = f"ci{int(probability * 100)}"
    return {
        "estimate": point if estimable else None,
        key: interval(np.asarray(values), probability) if estimable else None,
        "point_denominator": point_denominator,
        "bootstrap_replicates": len(indices),
        "finite_replicates": len(values),
        "estimable": estimable,
    }


def analyze(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    units = aggregate_worlds(rows)
    worlds, vectors, relation = world_vectors(units, model)
    draws = bootstrap_indices(relation, model)
    unstratified_draws = unstratified_bootstrap_indices(len(worlds), model)
    depths = MODEL_DEPTHS[model]
    deep = MODEL_DEEP[model]

    by_k: dict[str, Any] = {}
    for k in depths:
        by_k[str(k)] = {
            "path_margin_D": metric(
                draws, lambda index, k=k: float(np.mean(vectors[f"D_K{k}"][index]))
            ),
            "choice_control_C": metric(
                draws, lambda index, k=k: float(np.mean(vectors[f"C_K{k}"][index]))
            ),
            "original_accuracy": metric(
                draws,
                lambda index, k=k: float(
                    np.mean(vectors[f"original_accuracy_K{k}"][index])
                ),
            ),
            "swap_accuracy": metric(
                draws,
                lambda index, k=k: float(
                    np.mean(vectors[f"swap_accuracy_K{k}"][index])
                ),
            ),
        }

    raw_gain = metric(
        draws,
        lambda index: float(
            np.mean(vectors[f"D_K{deep}"][index] - vectors["D_K1"][index])
        ),
    )
    choice_gain = metric(
        draws,
        lambda index: float(
            np.mean(vectors[f"C_K{deep}"][index] - vectors["C_K1"][index])
        ),
    )
    repair = vectors[f"repair_margin_K{deep}"]
    # Recovery uses means of margins, not a mean of per-world ratios.
    original_deep = np.asarray(
        [
            unit["reference_margin"]
            for unit in sorted(
                (
                    unit
                    for unit in units
                    if unit["state"] == "original" and int(unit["K"]) == deep
                ),
                key=lambda unit: unit["world_id"],
            )
        ]
    )
    swap_deep = np.asarray(
        [
            unit["reference_margin"]
            for unit in sorted(
                (
                    unit
                    for unit in units
                    if unit["state"] == "bridge_swap" and int(unit["K"]) == deep
                ),
                key=lambda unit: unit["world_id"],
            )
        ]
    )
    recovery = ratio_metric(
        draws,
        numerator=lambda index: float(np.mean(repair[index] - swap_deep[index])),
        denominator=lambda index: float(
            np.mean(original_deep[index] - swap_deep[index])
        ),
    )
    repair_increment = metric(
        draws,
        lambda index: float(np.mean(repair[index] - swap_deep[index])),
    )
    repair_by_k: dict[str, Any] = {}
    if model in {"ouro26", "ouro14"}:
        for k in depths:
            repair_margin = vectors[f"repair_margin_K{k}"]
            original_margin = np.asarray(
                [
                    unit["reference_margin"]
                    for unit in sorted(
                        (
                            unit
                            for unit in units
                            if unit["state"] == "original" and int(unit["K"]) == k
                        ),
                        key=lambda unit: unit["world_id"],
                    )
                ]
            )
            swap_margin = np.asarray(
                [
                    unit["reference_margin"]
                    for unit in sorted(
                        (
                            unit
                            for unit in units
                            if unit["state"] == "bridge_swap" and int(unit["K"]) == k
                        ),
                        key=lambda unit: unit["world_id"],
                    )
                ]
            )
            repair_by_k[str(k)] = {
                "repair_increment_over_swap": metric(
                    draws,
                    lambda index, repair_margin=repair_margin, swap_margin=swap_margin: float(
                        np.mean(repair_margin[index] - swap_margin[index])
                    ),
                ),
                "recovery_fraction": ratio_metric(
                    draws,
                    numerator=lambda index, repair_margin=repair_margin, swap_margin=swap_margin: float(
                        np.mean(repair_margin[index] - swap_margin[index])
                    ),
                    denominator=lambda index, original_margin=original_margin, swap_margin=swap_margin: float(
                        np.mean(original_margin[index] - swap_margin[index])
                    ),
                ),
                "repair_accuracy": metric(
                    draws,
                    lambda index, k=k: float(
                        np.mean(vectors[f"repair_accuracy_K{k}"][index])
                    ),
                ),
            }

    deep_ci = by_k[str(deep)]["path_margin_D"]["ci95"]
    choice_ci = choice_gain["ci95"]
    raw_gain_ci = raw_gain["ci95"]
    supports = bool(
        deep_ci[0] > 0.0 and raw_gain_ci[0] > 0.0 and choice_ci[0] > 0.0
    )
    timing: dict[str, Any] = {}
    if model == "loopus":
        fraction = ratio_metric(
            draws,
            numerator=lambda index: float(
                np.mean(vectors["C_K2"][index] - vectors["C_K1"][index])
            ),
            denominator=lambda index: float(
                np.mean(vectors["C_K8"][index] - vectors["C_K1"][index])
            ),
            probability=0.90,
        )
        denominator = choice_gain["estimate"]
        timing = {
            "early_gain_fraction_f2": fraction,
            "denominator_choice_gain": denominator,
            "confirms_K2_localization": bool(
                denominator > 0.05
                and choice_ci[0] > 0.0
                and fraction["estimable"]
                and fraction["ci90"][0] >= 0.8
                and fraction["ci90"][1] <= 1.2
            ),
        }
    elif model in {"ouro26", "ouro14"}:
        timing = {
            "choice_K4_minus_K3": metric(
                draws,
                lambda index: float(
                    np.mean(vectors["C_K4"][index] - vectors["C_K3"][index])
                ),
            ),
            "raw_margin_K4_minus_K3": metric(
                draws,
                lambda index: float(
                    np.mean(vectors["D_K4"][index] - vectors["D_K3"][index])
                ),
            ),
        }

    leave_one_relation_out: dict[str, Any] = {}
    for omitted in sorted(set(relation)):
        kept = relation != omitted
        leave_one_relation_out[str(int(omitted))] = {
            "choice_gain": float(
                np.mean(
                    vectors[f"C_K{deep}"][kept] - vectors["C_K1"][kept]
                )
            ),
            "raw_gain": float(
                np.mean(
                    vectors[f"D_K{deep}"][kept] - vectors["D_K1"][kept]
                )
            ),
        }

    code_pair: dict[str, list[int]] = defaultdict(list)
    metadata_by_world = {
        unit["world_id"]: unit for unit in units if unit["state"] == "original" and unit["K"] == 1
    }
    for index, world in enumerate(worlds):
        key = "__".join(metadata_by_world[world]["ordered_code_pair"])
        code_pair[key].append(index)
    by_code_pair = {
        key: {
            "worlds": len(index),
            "choice_gain": float(
                np.mean(vectors[f"C_K{deep}"][index] - vectors["C_K1"][index])
            ),
            "raw_gain": float(
                np.mean(vectors[f"D_K{deep}"][index] - vectors["D_K1"][index])
            ),
        }
        for key, index in sorted(code_pair.items())
    }
    return {
        "worlds": len(worlds),
        "world_cells": len(units),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": stable_seed(model),
            "resampling": "48 whole worlds within each of six relation families",
        },
        "by_K": by_k,
        "primary_choice_gain_H": choice_gain,
        "secondary_raw_margin_gain_G": raw_gain,
        "deep_repair_accuracy": metric(
            draws,
            lambda index: float(
                np.mean(vectors[f"repair_accuracy_K{deep}"][index])
            ),
        ),
        "deep_repair_increment_R": repair_increment,
        "deep_topology_recovery_fraction_F": recovery,
        "deep_topology_recovery_interpretable": bool(deep_ci[0] > 0.0),
        "repair_curve": repair_by_k,
        "timing": timing,
        "robustness": {
            "unstratified_whole_world_bootstrap": {
                "choice_gain": metric(
                    unstratified_draws,
                    lambda index: float(
                        np.mean(
                            vectors[f"C_K{deep}"][index]
                            - vectors["C_K1"][index]
                        )
                    ),
                ),
                "raw_gain": metric(
                    unstratified_draws,
                    lambda index: float(
                        np.mean(
                            vectors[f"D_K{deep}"][index]
                            - vectors["D_K1"][index]
                        )
                    ),
                ),
            },
            "leave_one_relation_family_out": leave_one_relation_out,
            "by_ordered_terminal_code_pair": by_code_pair,
        },
        "treatment_conjunction_passes_given_competence": supports,
        "treatment_decision": (
            "SUPPORT_TRUTH_BALANCED_DEPTH_DEPENDENT_PATH_CONTROL"
            if supports
            else "DO_NOT_SUPPORT_FULL_TREATMENT_CONJUNCTION"
        ),
    }


def format_metric(value: dict[str, Any], probability: int = 95) -> str:
    if value.get("estimate") is None or value.get(f"ci{probability}") is None:
        return "not estimable"
    lower, upper = value[f"ci{probability}"]
    return f"{value['estimate']:.3f} [{lower:.3f}, {upper:.3f}]"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    result = payload["result"]
    deep = MODEL_DEEP[payload["model_key"]]
    lines = [
        f"# Nonce path-control result: {payload['model_key']}",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        f"- Worlds: {result['worlds']}",
        f"- Deep K: {deep}",
        f"- Scale-invariant choice gain H: {format_metric(result['primary_choice_gain_H'])}",
        f"- Raw margin gain G: {format_metric(result['secondary_raw_margin_gain_G'])}",
        f"- Deep path contrast D: {format_metric(result['by_K'][str(deep)]['path_margin_D'])}",
        f"- Deep topology recovery F: {format_metric(result['deep_topology_recovery_fraction_F'])}",
        f"- Deep repair increment R: {format_metric(result['deep_repair_increment_R'])}",
        f"- Deep repair accuracy: {format_metric(result['deep_repair_accuracy'])}",
        "",
        "The result is prospective for this frozen fictional-world panel. The A/B",
        "comparison is an exact same-context token-logit decision.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competence-artifact", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    competence = load_artifact(args.competence_artifact.resolve(), "competence")
    model, rows, merge = merge_shards([path.resolve() for path in args.shards])
    if competence.get("model_key") != model:
        raise RuntimeError("competence and treatment model keys differ")
    if competence.get("runner", {}).get("sha256") != merge["runner_sha256"]:
        raise RuntimeError("competence and treatment runner hashes differ")
    if bool(competence.get("mock")) != bool(merge["mock"]):
        raise RuntimeError("competence and treatment mock status differs")
    if competence.get("summary", {}).get("passes_competence_gate") is not True:
        raise RuntimeError("full treatment cannot be interpreted after competence failure")
    validate_grid(rows, model)
    result = analyze(rows, model)
    conjunction = bool(
        competence["summary"]["passes_competence_gate"]
        and result["treatment_conjunction_passes_given_competence"]
    )
    decision = (
        "SUPPORT_TRUTH_BALANCED_DEPTH_DEPENDENT_PATH_CONTROL"
        if conjunction
        else "DO_NOT_SUPPORT_FULL_TREATMENT_CONJUNCTION"
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "model_key": model,
        "decision": decision,
        "competence": {
            "path": args.competence_artifact.name,
            "sha256": sha256_file(args.competence_artifact),
            "summary": competence["summary"],
        },
        "merge": merge,
        "result": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"nonce_path_control_{model}.json"
    markdown_path = args.output_dir / f"nonce_path_control_{model}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, markdown_path)
    print(json_path)
    print(markdown_path)
    print(json.dumps({"decision": decision, "result": result}, indent=2))


if __name__ == "__main__":
    main()
