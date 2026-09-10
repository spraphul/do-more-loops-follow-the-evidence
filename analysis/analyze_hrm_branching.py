#!/usr/bin/env python3
"""Analyze the HRM-Text-1B relation-conditioned branching confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


RUN_SCHEMA = "iclr2027.hrm_text_branching_xor.confirmation_run.v1"
PANEL_SCHEMA = "iclr2027.hrm_text_branching_xor.confirmation_panel.v1"
MODEL_KEY = "hrm_text_1b"
MODEL_REPOSITORY = "sapientinc/HRM-Text-1B"
MODEL_REVISION = "1f82ac2b71222f0c100a224a33f24b44a3000b6d"
PANEL_SHA256 = "e76bf583223dba4f6b46e970b592009b6b7f29bd9db0e30cd8b97c3217314f31"
DEPTHS = (1, 2)
STATES = ("original", "bridge_swap", "topology_repair")
ARMS = ("X", "Y")
GATE_WORLDS = 48
CONFIRMATION_WORLDS = 144
SHARDS = 12
BOOTSTRAP_REPLICATES = 10_000
GATE_SALT = "iclr2027-hrm-text-branching-xor-confirmation-v1:gate:accuracy"
ANALYSIS_SALT = "iclr2027-hrm-text-branching-xor-confirmation-v1:analysis"
SLICE_METRICS = (
    "primary_repair_xor_choice_H2",
    "primary_repair_xor_raw_H2",
    "primary_repair_xor_choice_H2_minus_H1",
    "primary_repair_xor_raw_H2_minus_H1",
    "supporting_bridge_xor_choice_H2",
    "supporting_bridge_xor_raw_H2",
    "primary_nearest_crossed_repair_choice_H2",
    "state_accuracy_original_H2",
    "state_accuracy_bridge_swap_H2",
    "state_accuracy_topology_repair_H2",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    value = {key: item for key, item in payload.items() if key != "panel_sha256"}
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(salt: str) -> int:
    return int.from_bytes(hashlib.sha256(salt.encode("utf-8")).digest()[:8], "big")


def credit(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return 0.0
    return 0.5


def orient(row: dict[str, Any]) -> dict[str, Any]:
    logits = row.get("score", {}).get("raw_logits", {})
    if (
        set(logits) != {"A", "B"}
        or {row.get("reference_label"), row.get("other_label")} != {"A", "B"}
        or row.get("expected_label") not in {"A", "B"}
    ):
        raise RuntimeError(f"invalid branching score row: {row.get('row_id')}")
    numeric = {label: float(logits[label]) for label in ("A", "B")}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise RuntimeError(f"non-finite branching logits: {row.get('row_id')}")
    reference = str(row["reference_label"])
    margin = numeric[reference] - numeric[str(row["other_label"])]
    reference_credit = credit(margin)
    state_credit = (
        reference_credit
        if row["expected_label"] == reference
        else 1.0 - reference_credit
    )
    return {
        **row,
        "score": {
            "raw_logits": numeric,
            "reference_logit_margin": margin,
            "reference_argmax_credit": reference_credit,
            "state_correct_argmax_credit": state_credit,
        },
    }


def load_panel(path: Path) -> dict[str, Any]:
    panel = json.loads(path.read_text(encoding="utf-8"))
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("panel_sha256") != PANEL_SHA256
        or canonical_hash(panel) != PANEL_SHA256
        or len(panel.get("worlds", [])) != 192
        or len(panel.get("cells", [])) != 4_608
    ):
        raise RuntimeError("HRM branching panel identity or dimensions changed")
    worlds = panel["worlds"]
    if Counter(world["split"] for world in worlds) != Counter(
        {"gate": GATE_WORLDS, "confirmation": CONFIRMATION_WORLDS}
    ):
        raise RuntimeError("HRM branching gate/confirmation split changed")
    return panel


def load_run(path: Path, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != RUN_SCHEMA
        or artifact.get("mode") != mode
        or artifact.get("model_key") != MODEL_KEY
        or artifact.get("model_revision") != MODEL_REVISION
        or artifact.get("mock") is not False
        or artifact.get("panel", {}).get("canonical_sha256") != PANEL_SHA256
    ):
        raise RuntimeError(f"HRM branching {mode} artifact identity changed: {path}")
    return artifact, [orient(row) for row in artifact.get("rows", [])]


def accuracy(rows: list[dict[str, Any]]) -> float:
    return mean(float(row["score"]["state_correct_argmax_credit"]) for row in rows)


def gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grid = Counter(
        (
            row["world_id"],
            int(row["source_arm"]),
            row["second_relation_arm"],
            row["candidate_order"],
        )
        for row in rows
    )
    if (
        len(rows) != 384
        or len(grid) != 384
        or set(grid.values()) != {1}
        or {row["split"] for row in rows} != {"gate"}
        or {row["evidence_state"] for row in rows} != {"original"}
        or {int(row["K"]) for row in rows} != {2}
    ):
        raise RuntimeError("HRM branching gate grid is incomplete")
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_by_world: dict[str, int] = {}
    for row in rows:
        by_world[row["world_id"]].append(row)
        relation_by_world[row["world_id"]] = int(row["relation_family_index"])
    if len(by_world) != GATE_WORLDS or set(map(len, by_world.values())) != {8}:
        raise RuntimeError("HRM branching gate worlds are incomplete")
    groups: dict[int, list[str]] = defaultdict(list)
    for world, relation in relation_by_world.items():
        groups[relation].append(world)
    if len(groups) != 6 or set(map(len, groups.values())) != {8}:
        raise RuntimeError("HRM branching gate relation balance changed")
    world_accuracy = {world: accuracy(members) for world, members in by_world.items()}
    rng = np.random.default_rng(stable_seed(GATE_SALT))
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[float] = []
        for relation in sorted(groups):
            members = groups[relation]
            indices = rng.integers(0, len(members), size=len(members))
            sampled.extend(world_accuracy[members[int(index)]] for index in indices)
        draws.append(float(np.mean(sampled)))
    point = float(np.mean(list(world_accuracy.values())))
    lower, upper = np.quantile(np.asarray(draws), [0.025, 0.975]).tolist()
    relation_arm = {
        arm: accuracy([row for row in rows if row["second_relation_arm"] == arm])
        for arm in ARMS
    }
    source_arm = {
        str(arm): accuracy([row for row in rows if int(row["source_arm"]) == arm])
        for arm in (0, 1)
    }
    candidate_order = {
        order: accuracy([row for row in rows if row["candidate_order"] == order])
        for order in ("original", "reversed")
    }
    relation_family = {
        str(index): accuracy(
            [row for row in rows if int(row["relation_family_index"]) == index]
        )
        for index in range(6)
    }
    passes = bool(
        point >= 0.65
        and lower > 0.50
        and min(relation_arm.values()) >= 0.55
        and min(source_arm.values()) >= 0.55
        and min(candidate_order.values()) >= 0.55
        and min(relation_family.values()) >= 0.50
    )
    return {
        "decision": "PROMOTE_TO_CONFIRMATION" if passes else "STOP_FRESH_PANEL_INCOMPETENT_SUBSTRATE",
        "passes_competence_gate": passes,
        "authorizes_real_confirmation": passes,
        "mock": False,
        "worlds": GATE_WORLDS,
        "prompt_cells": 384,
        "scored_rows": 384,
        "gate_exit": "H2",
        "deep_original_accuracy": {
            "estimate": point,
            "ci95": [float(lower), float(upper)],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "resampling": "8 whole worlds within each of six relation families",
        },
        "second_relation_arm_accuracy_H2": relation_arm,
        "source_arm_accuracy_H2": source_arm,
        "candidate_order_accuracy_H2": candidate_order,
        "relation_family_accuracy_H2": relation_family,
        "requirements": {
            "H2_point_at_least": 0.65,
            "H2_lower95_strictly_above": 0.5,
            "each_second_relation_arm_H2_at_least": 0.55,
            "each_source_arm_H2_at_least": 0.55,
            "each_candidate_order_H2_at_least": 0.55,
            "each_relation_family_H2_at_least": 0.5,
        },
        "seal": (
            "only original-state H2 outcomes from 48 gate worlds were decoded; "
            "confirmation worlds remained untouched"
        ),
    }


def load_confirmation(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(paths) != SHARDS:
        raise RuntimeError("expected 12 HRM branching confirmation shards")
    loaded = [load_run(path, "confirmation") for path in paths]
    indices = sorted(int(artifact.get("shard", {}).get("index", -1)) for artifact, _ in loaded)
    if indices != list(range(SHARDS)) or any(
        int(artifact.get("shard", {}).get("count", -1)) != SHARDS
        for artifact, _ in loaded
    ):
        raise RuntimeError("HRM branching confirmation shard set is incomplete")
    rows = [row for _artifact, shard_rows in loaded for row in shard_rows]
    if len(rows) != 6_912 or len({row["row_id"] for row in rows}) != 6_912:
        raise RuntimeError("HRM branching confirmation row union changed")
    return rows, {
        "shard_count": SHARDS,
        "inputs": [
            {"path": path.name, "sha256": file_sha256(path)} for path in paths
        ],
    }


def validate_confirmation(rows: list[dict[str, Any]], panel: dict[str, Any]) -> list[str]:
    grid = Counter(
        (
            row["world_id"],
            row["evidence_state"],
            int(row["source_arm"]),
            row["second_relation_arm"],
            row["candidate_order"],
            int(row["K"]),
        )
        for row in rows
    )
    if len(grid) != 6_912 or set(grid.values()) != {1}:
        raise RuntimeError("HRM branching confirmation factorial is incomplete")
    panel_worlds = {
        world["world_id"]: world
        for world in panel["worlds"]
        if world["split"] == "confirmation"
    }
    if set(panel_worlds) != {row["world_id"] for row in rows}:
        raise RuntimeError("HRM branching score worlds differ from the released panel")
    for row in rows:
        world = panel_worlds[row["world_id"]]
        if (
            row["split"] != "confirmation"
            or int(row["world_index"]) != int(world["world_index"])
            or int(row["relation_family_index"]) != int(world["relation_family_index"])
            or int(row["ordered_focal_code_pair_index"])
            != int(world["ordered_focal_code_pair_index"])
        ):
            raise RuntimeError(f"HRM branching row-to-panel binding changed: {row['row_id']}")
    return [
        world_id
        for world_id, world in sorted(
            panel_worlds.items(), key=lambda item: int(item[1]["split_world_index"])
        )
    ]


def aggregate(
    rows: list[dict[str, Any]], panel: dict[str, Any], worlds: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    panel_worlds = {world["world_id"]: world for world in panel["worlds"]}
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world_id"], row["evidence_state"], row["second_relation_arm"], int(row["K"]))].append(row)
    if len(grouped) != CONFIRMATION_WORLDS * 3 * 2 * 2 or set(map(len, grouped.values())) != {4}:
        raise RuntimeError("HRM branching source/order aggregation grid changed")
    summary = {
        key: (
            mean(float(row["score"]["reference_logit_margin"]) for row in members),
            mean(2.0 * float(row["score"]["reference_argmax_credit"]) - 1.0 for row in members),
            mean(float(row["score"]["state_correct_argmax_credit"]) for row in members),
        )
        for key, members in grouped.items()
    }

    def values(state: str, arm: str, depth: int, item: int) -> np.ndarray:
        return np.asarray(
            [summary[(world, state, arm, depth)][item] for world in worlds], dtype=float
        )

    vectors: dict[str, np.ndarray] = {}
    near_arm = {world: panel_worlds[world]["near_relation_arm"] for world in worlds}
    for depth in DEPTHS:
        raw = {(state, arm): values(state, arm, depth, 0) for state in STATES for arm in ARMS}
        signed = {(state, arm): values(state, arm, depth, 1) for state in STATES for arm in ARMS}
        state_accuracy = {
            (state, arm): values(state, arm, depth, 2) for state in STATES for arm in ARMS
        }
        task_raw = {
            (state, arm): raw[(state, arm)] if arm == "X" else -raw[(state, arm)]
            for state in STATES for arm in ARMS
        }
        task_choice = {
            (state, arm): (
                (signed[(state, arm)] + 1.0) / 2.0
                if arm == "X"
                else (1.0 - signed[(state, arm)]) / 2.0
            )
            for state in STATES for arm in ARMS
        }
        bridge_raw: dict[str, np.ndarray] = {}
        bridge_choice: dict[str, np.ndarray] = {}
        repair_raw: dict[str, np.ndarray] = {}
        repair_choice: dict[str, np.ndarray] = {}
        for arm in ARMS:
            bridge_raw[arm] = task_raw[("original", arm)] - task_raw[("bridge_swap", arm)]
            bridge_choice[arm] = task_choice[("original", arm)] - task_choice[("bridge_swap", arm)]
            repair_raw[arm] = task_raw[("topology_repair", arm)] - task_raw[("bridge_swap", arm)]
            repair_choice[arm] = task_choice[("topology_repair", arm)] - task_choice[("bridge_swap", arm)]
            vectors[f"supporting_bridge_effect_raw_{arm}_H{depth}"] = bridge_raw[arm]
            vectors[f"supporting_bridge_effect_choice_{arm}_H{depth}"] = bridge_choice[arm]
            vectors[f"primary_repair_effect_raw_{arm}_H{depth}"] = repair_raw[arm]
            vectors[f"primary_repair_effect_choice_{arm}_H{depth}"] = repair_choice[arm]
        vectors[f"supporting_bridge_xor_raw_H{depth}"] = (bridge_raw["X"] + bridge_raw["Y"]) / 4.0
        vectors[f"supporting_bridge_xor_choice_H{depth}"] = (bridge_choice["X"] + bridge_choice["Y"]) / 2.0
        vectors[f"primary_repair_xor_raw_H{depth}"] = (repair_raw["X"] + repair_raw["Y"]) / 4.0
        vectors[f"primary_repair_xor_choice_H{depth}"] = (repair_choice["X"] + repair_choice["Y"]) / 2.0
        vectors[f"primary_nearest_matched_repair_choice_H{depth}"] = np.asarray(
            [repair_choice[near_arm[world]][index] for index, world in enumerate(worlds)]
        )
        vectors[f"primary_nearest_crossed_repair_choice_H{depth}"] = np.asarray(
            [
                repair_choice["Y" if near_arm[world] == "X" else "X"][index]
                for index, world in enumerate(worlds)
            ]
        )
        for state in STATES:
            vectors[f"state_accuracy_{state}_H{depth}"] = (
                state_accuracy[(state, "X")] + state_accuracy[(state, "Y")]
            ) / 2.0
    for kind in ("raw", "choice"):
        vectors[f"primary_repair_xor_{kind}_H2_minus_H1"] = (
            vectors[f"primary_repair_xor_{kind}_H2"] - vectors[f"primary_repair_xor_{kind}_H1"]
        )
        vectors[f"supporting_bridge_xor_{kind}_H2_minus_H1"] = (
            vectors[f"supporting_bridge_xor_{kind}_H2"] - vectors[f"supporting_bridge_xor_{kind}_H1"]
        )
    relations = np.asarray(
        [int(panel_worlds[world]["relation_family_index"]) for world in worlds], dtype=int
    )
    pairs = np.asarray(
        [int(panel_worlds[world]["ordered_focal_code_pair_index"]) for world in worlds], dtype=int
    )
    return relations, pairs, vectors


def bootstrap_indices(relations: np.ndarray) -> list[np.ndarray]:
    groups = {
        relation: np.flatnonzero(relations == relation)
        for relation in sorted(set(relations.tolist()))
    }
    if len(groups) != 6 or set(map(len, groups.values())) != {24}:
        raise RuntimeError("HRM branching confirmation relation balance changed")
    rng = np.random.default_rng(stable_seed(ANALYSIS_SALT))
    return [
        np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups.values()]
        )
        for _ in range(BOOTSTRAP_REPLICATES)
    ]


def estimate(vector: np.ndarray, draws: list[np.ndarray]) -> dict[str, Any]:
    sampled = np.asarray([float(np.mean(vector[index])) for index in draws])
    lower, upper = np.quantile(sampled, [0.025, 0.975]).tolist()
    return {
        "estimate": float(np.mean(vector)),
        "ci95": [float(lower), float(upper)],
        "bootstrap_replicates": len(draws),
        "resampling_unit": "whole world",
    }


def ratio_estimate(
    numerator: np.ndarray, denominator: np.ndarray, draws: list[np.ndarray]
) -> dict[str, Any]:
    point_denominator = float(np.mean(denominator))
    sampled = [
        float(np.mean(numerator[index])) / value
        for index in draws
        if abs(value := float(np.mean(denominator[index]))) > 1e-12
    ]
    if abs(point_denominator) <= 1e-12 or not sampled:
        return {"estimate": None, "ci95": [None, None], "defined": False}
    lower, upper = np.quantile(np.asarray(sampled), [0.025, 0.975]).tolist()
    return {
        "estimate": float(np.mean(numerator)) / point_denominator,
        "ci95": [float(lower), float(upper)],
        "bootstrap_replicates": len(sampled),
        "resampling_unit": "whole world",
        "defined": True,
    }


def descriptive_point(vectors: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    metrics = {name: float(np.mean(vectors[name][mask])) for name in SLICE_METRICS}
    denominator = float(np.mean(vectors["supporting_bridge_xor_choice_H2"][mask]))
    metrics["repair_fraction_choice_H2"] = (
        None
        if abs(denominator) <= 1e-12
        else float(np.mean(vectors["primary_repair_xor_choice_H2"][mask])) / denominator
    )
    return {"worlds": int(np.sum(mask)), "metrics": metrics}


def diagnostics(
    panel: dict[str, Any], relations: np.ndarray, pairs: np.ndarray, vectors: dict[str, np.ndarray]
) -> dict[str, Any]:
    relation_family = {}
    leave_one_out = {}
    for relation in range(6):
        relation_family[str(relation)] = {
            **descriptive_point(vectors, relations == relation),
            "relations": panel["relation_families"][relation],
        }
        leave_one_out[str(relation)] = {
            **descriptive_point(vectors, relations != relation),
            "omitted_relations": panel["relation_families"][relation],
        }
    pair_codes = {
        int(world["ordered_focal_code_pair_index"]): world["focal_terminal_codes"]
        for world in panel["worlds"] if world["split"] == "confirmation"
    }
    ordered_pairs = {
        str(pair): {
            **descriptive_point(vectors, pairs == pair),
            "ordered_focal_codes": pair_codes[pair],
        }
        for pair in range(12)
    }
    return {
        "status": "descriptive_only_not_additional_confirmatory_tests",
        "relation_family": relation_family,
        "ordered_focal_code_pair": ordered_pairs,
        "leave_one_relation_family_out": leave_one_out,
    }


def lower(metric: dict[str, Any]) -> float:
    value = metric.get("ci95", [None])[0]
    return -math.inf if value is None else float(value)


def decision(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    composition = {
        "primary_repair_choice_H2_point_at_least_0_10": results["primary_repair_xor_choice_H2"]["estimate"] >= 0.10,
        "primary_repair_choice_H2_lower95_above_0": lower(results["primary_repair_xor_choice_H2"]) > 0.0,
        "primary_repair_raw_H2_lower95_above_0": lower(results["primary_repair_xor_raw_H2"]) > 0.0,
        "repair_choice_X_H2_lower95_above_0": lower(results["primary_repair_effect_choice_X_H2"]) > 0.0,
        "repair_choice_Y_H2_lower95_above_0": lower(results["primary_repair_effect_choice_Y_H2"]) > 0.0,
        "crossed_locality_repair_choice_H2_lower95_above_0": lower(results["primary_nearest_crossed_repair_choice_H2"]) > 0.0,
        "original_accuracy_H2_point_at_least_0_65": results["state_accuracy_original_H2"]["estimate"] >= 0.65,
        "original_accuracy_H2_lower95_above_0_50": lower(results["state_accuracy_original_H2"]) > 0.50,
        "bridge_swap_accuracy_H2_point_at_least_0_60": results["state_accuracy_bridge_swap_H2"]["estimate"] >= 0.60,
        "bridge_swap_accuracy_H2_lower95_above_0_50": lower(results["state_accuracy_bridge_swap_H2"]) > 0.50,
        "topology_repair_accuracy_H2_point_at_least_0_60": results["state_accuracy_topology_repair_H2"]["estimate"] >= 0.60,
        "topology_repair_accuracy_H2_lower95_above_0_50": lower(results["state_accuracy_topology_repair_H2"]) > 0.50,
        "repair_fraction_choice_H2_defined": results["repair_fraction_choice_H2"].get("defined") is True,
        "repair_fraction_choice_H2_point_at_least_0_50": results["repair_fraction_choice_H2"]["estimate"] >= 0.50,
        "repair_fraction_choice_H2_lower95_at_least_0_50": lower(results["repair_fraction_choice_H2"]) >= 0.50,
    }
    composition_passes = all(composition.values())
    depth = {
        "primary_repair_choice_H2_minus_H1_point_at_least_0_05": results["primary_repair_xor_choice_H2_minus_H1"]["estimate"] >= 0.05,
        "primary_repair_choice_H2_minus_H1_lower95_above_0": lower(results["primary_repair_xor_choice_H2_minus_H1"]) > 0.0,
        "primary_repair_raw_H2_minus_H1_lower95_above_0": lower(results["primary_repair_xor_raw_H2_minus_H1"]) > 0.0,
    }
    depth_passes = composition_passes and all(depth.values())
    return {
        "decision": (
            "CONFIRM_HRM_DEPTH_DEPENDENT_BRANCHING_COMPOSITION"
            if depth_passes
            else "CONFIRM_HRM_BRANCHING_COMPOSITION_ONLY"
            if composition_passes
            else "DO_NOT_CONFIRM_HRM_BRANCHING_COMPOSITION"
        ),
        "confirmation_claim_eligible": composition_passes,
        "composition_confirmed": composition_passes,
        "depth_dependent_composition_confirmed": depth_passes,
        "composition_gates_pass": composition_passes,
        "composition_gates": composition,
        "fixed_sequence_depth_test_reached": composition_passes,
        "depth_dependent_composition_gates_pass": depth_passes,
        "depth_gates": depth,
    }


def fmt(metric: dict[str, Any]) -> str:
    return f"{metric['estimate']:.3f} [{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--gate-artifact", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    panel = load_panel(args.panel)
    gate_artifact, gate_rows = load_run(args.gate_artifact, "gate")
    gate = gate_summary(gate_rows)
    if gate != gate_artifact.get("summary") or gate["passes_competence_gate"] is not True:
        raise RuntimeError("released HRM branching gate summary does not match raw logits")
    rows, execution = load_confirmation([path.resolve() for path in args.inputs])
    worlds = validate_confirmation(rows, panel)
    relations, pairs, vectors = aggregate(rows, panel, worlds)
    draws = bootstrap_indices(relations)
    results = {name: estimate(vector, draws) for name, vector in vectors.items()}
    results["repair_fraction_choice_H2"] = ratio_estimate(
        vectors["primary_repair_xor_choice_H2"],
        vectors["supporting_bridge_xor_choice_H2"],
        draws,
    )
    confirmation = decision(results)
    payload = {
        "schema": "evidence-loops.hrm-branching-analysis.v1",
        "status": "fresh_confirmation",
        "model_key": MODEL_KEY,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "worlds": len(worlds),
        "scored_rows": len(rows),
        "panel_sha256": PANEL_SHA256,
        "execution": execution,
        "gate": gate,
        "inference": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "shared_draws_across_all_metrics": True,
            "resampling_unit": "whole world",
            "stratification": "six relation families, 24 worlds per family",
            "source_and_candidate_order_aggregation": "within world before resampling",
            "raw_score_contract": "margins and exact-choice credits recomputed from A/B logits",
        },
        "results": results,
        "descriptive_diagnostics": diagnostics(panel, relations, pairs, vectors),
        "confirmation": confirmation,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(
        "\n".join(
            (
                "# HRM-Text-1B branching confirmation",
                "",
                f"Decision: `{confirmation['decision']}`",
                "",
                f"- H2 gate accuracy: {fmt(gate['deep_original_accuracy'])}",
                f"- H2 repair choice interaction: {fmt(results['primary_repair_xor_choice_H2'])}",
                f"- H2 minus H1 choice gain: {fmt(results['primary_repair_xor_choice_H2_minus_H1'])}",
                f"- H2 repair raw interaction: {fmt(results['primary_repair_xor_raw_H2'])}",
                f"- H2 choice-repair fraction: {fmt(results['repair_fraction_choice_H2'])}",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(args.json_output)


if __name__ == "__main__":
    main()
