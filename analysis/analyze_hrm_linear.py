#!/usr/bin/env python3
"""Analyze the HRM-Text-1B linear path-control transfer."""

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


COMPETENCE_SCHEMA = "iclr2027.hrm_text_nonce_competence.run.v1"
TREATMENT_SCHEMA = "iclr2027.hrm_text_nonce_treatment.run.v1"
MODEL_KEY = "hrm_text_1b"
MODEL_REPOSITORY = "sapientinc/HRM-Text-1B"
MODEL_REVISION = "1f82ac2b71222f0c100a224a33f24b44a3000b6d"
PANEL_SHA256 = "7880c02c64af541629b9c23dce15bf4a6df98c4f99a8aed45b4a310df2e543e5"
PRIMARY_WORLD_IDS_SHA256 = "480dfe02403f544654e73b89524aefd7881e624bd5dca94cb0f7b3a6d505b123"
DEPTHS = (1, 2)
DEEP = 2
BOOTSTRAP_REPLICATES = 10_000
GATE_SALT = "iclr2027-nonce-path-control-v1:competence:hrm_text_1b"
TREATMENT_SALT = "iclr2027-hrm-text-nonce-treatment-v1:untouched-240"


def stable_seed(salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def credit(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return 0.0
    return 0.5


def orient(row: dict[str, Any]) -> dict[str, Any]:
    labels = {row.get("reference_label"), row.get("other_label")}
    logits = row.get("score", {}).get("raw_logits", {})
    if labels != {"A", "B"} or set(logits) != {"A", "B"}:
        raise RuntimeError(f"invalid A/B score row: {row.get('row_id')}")
    numeric = {label: float(logits[label]) for label in ("A", "B")}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise RuntimeError(f"non-finite HRM logits: {row.get('row_id')}")
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
            "expected_label": row["expected_label"],
            "reference_logit_margin": margin,
            "reference_argmax_credit": reference_credit,
            "state_correct_argmax_credit": state_credit,
        },
    }


def load_competence(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != COMPETENCE_SCHEMA
        or artifact.get("mode") != "competence"
        or artifact.get("model_key") != MODEL_KEY
        or artifact.get("model_revision") != MODEL_REVISION
        or artifact.get("mock") is not False
        or artifact.get("panel", {}).get("canonical_sha256") != PANEL_SHA256
    ):
        raise RuntimeError("HRM linear competence artifact identity changed")
    rows = [orient(row) for row in artifact.get("rows", [])]
    if len(rows) != 384 or len({row["row_id"] for row in rows}) != 384:
        raise RuntimeError("HRM linear competence row grid changed")
    return rows, artifact.get("summary", {})


def competence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        (
            row["world_id"],
            row["query_arm"],
            row["candidate_order"],
            int(row["K"]),
        )
        for row in rows
    )
    if (
        len(counts) != 384
        or set(counts.values()) != {1}
        or len({row["world_id"] for row in rows}) != 48
        or {int(row["K"]) for row in rows} != {1, 2}
    ):
        raise RuntimeError("HRM linear competence factorial is incomplete")
    deep = [row for row in rows if int(row["K"]) == DEEP]
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_by_world: dict[str, int] = {}
    for row in deep:
        by_world[row["world_id"]].append(row)
        relation_by_world[row["world_id"]] = int(row["relation_family_index"])
    if len(by_world) != 48 or set(map(len, by_world.values())) != {4}:
        raise RuntimeError("HRM linear competence worlds are incomplete")
    relation_groups: dict[int, list[str]] = defaultdict(list)
    for world, relation in relation_by_world.items():
        relation_groups[relation].append(world)
    if len(relation_groups) != 6 or set(map(len, relation_groups.values())) != {8}:
        raise RuntimeError("HRM linear competence relation balance changed")
    world_accuracy = {
        world: mean(
            float(row["score"]["state_correct_argmax_credit"]) for row in members
        )
        for world, members in by_world.items()
    }
    rng = np.random.default_rng(stable_seed(GATE_SALT))
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[float] = []
        for relation in sorted(relation_groups):
            members = relation_groups[relation]
            indices = rng.integers(0, len(members), size=len(members))
            sampled.extend(world_accuracy[members[int(index)]] for index in indices)
        draws.append(float(np.mean(sampled)))
    point = float(np.mean(list(world_accuracy.values())))
    lower, upper = np.quantile(np.asarray(draws), [0.025, 0.975]).tolist()

    def accuracy(selected: list[dict[str, Any]]) -> float:
        return mean(
            float(row["score"]["state_correct_argmax_credit"]) for row in selected
        )

    query = {
        arm: accuracy([row for row in deep if row["query_arm"] == arm])
        for arm in ("A", "B")
    }
    order = {
        value: accuracy(
            [row for row in deep if row["candidate_order"] == value]
        )
        for value in ("original", "reversed")
    }
    passes = bool(
        point >= 0.65
        and lower > 0.50
        and min(query.values()) >= 0.55
        and min(order.values()) >= 0.55
    )
    return {
        "decision": "PROMOTE_TO_TREATMENT" if passes else "STOP_INCOMPETENT_SUBSTRATE",
        "passes_competence_gate": passes,
        "gate_depth": DEEP,
        "worlds": 48,
        "rows": len(deep),
        "graph_correct_accuracy": {
            "estimate": point,
            "ci95": [float(lower), float(upper)],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "resampling": "48 worlds stratified within six relation families",
        },
        "query_arm_accuracy": query,
        "candidate_order_accuracy": order,
        "requirements": {
            "lower95_strictly_above": 0.5,
            "point_at_least": 0.65,
            "each_query_arm_point_at_least": 0.55,
            "each_candidate_order_point_at_least": 0.55,
        },
        "descriptive_original_accuracy_by_H_exit": {
            f"H{depth}": accuracy([row for row in rows if int(row["K"]) == depth])
            for depth in DEPTHS
        },
        "H1_is_not_a_promotion_endpoint": True,
    }


def load_treatment(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(paths) != 8:
        raise RuntimeError("expected eight HRM linear treatment shards")
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    indices = []
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        if (
            artifact.get("schema") != TREATMENT_SCHEMA
            or artifact.get("mode") != "treatment"
            or artifact.get("model_key") != MODEL_KEY
            or artifact.get("model_revision") != MODEL_REVISION
            or artifact.get("mock") is not False
            or artifact.get("panel", {}).get("canonical_sha256") != PANEL_SHA256
            or int(artifact.get("shard", {}).get("count", -1)) != 8
        ):
            raise RuntimeError("HRM linear treatment shard identity changed")
        indices.append(int(artifact["shard"]["index"]))
        rows.extend(orient(row) for row in artifact.get("rows", []))
    if sorted(indices) != list(range(8)):
        raise RuntimeError("HRM linear treatment shard set is incomplete")
    if len(rows) != 4_800 or len({row["row_id"] for row in rows}) != 4_800:
        raise RuntimeError("HRM linear treatment row union changed")
    return rows, {
        "shard_count": 8,
        "inputs": [
            {"path": path.name, "sha256": file_sha256(path)} for path in paths
        ],
    }


def analyze_treatment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import analyze_fictional_ouro as base

    cell_counts = Counter(
        (row["world_id"], row["evidence_state"], int(row["K"])) for row in rows
    )
    worlds = sorted({row["world_id"] for row in rows})
    world_hash = hashlib.sha256(
        json.dumps(worlds, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected: dict[tuple[str, str, int], int] = {}
    for world in worlds:
        for state in ("original", "bridge_swap"):
            for depth in DEPTHS:
                expected[(world, state, depth)] = 4
        expected[(world, "topology_repair", DEEP)] = 4
    if (
        len(worlds) != 240
        or world_hash != PRIMARY_WORLD_IDS_SHA256
        or dict(cell_counts) != expected
    ):
        raise RuntimeError("HRM linear treatment factorial is incomplete")

    base.MODEL_DEPTHS[MODEL_KEY] = DEPTHS
    base.MODEL_DEEP[MODEL_KEY] = DEEP
    units = base.aggregate_worlds(rows)
    ordered_worlds, vectors, relations = base.world_vectors(units, MODEL_KEY)
    groups = {
        relation: np.flatnonzero(relations == relation)
        for relation in sorted(set(relations.tolist()))
    }
    if len(groups) != 6 or set(map(len, groups.values())) != {40}:
        raise RuntimeError("HRM linear treatment relation balance changed")
    rng = np.random.default_rng(stable_seed(TREATMENT_SALT))
    draws = [
        np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups.values()]
        )
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    by_h: dict[str, Any] = {}
    for depth in DEPTHS:
        by_h[str(depth)] = {
            "path_margin_D": base.metric(
                draws, lambda index, depth=depth: float(np.mean(vectors[f"D_K{depth}"][index]))
            ),
            "choice_control_C": base.metric(
                draws, lambda index, depth=depth: float(np.mean(vectors[f"C_K{depth}"][index]))
            ),
            "original_accuracy": base.metric(
                draws,
                lambda index, depth=depth: float(
                    np.mean(vectors[f"original_accuracy_K{depth}"][index])
                ),
            ),
            "swap_accuracy": base.metric(
                draws,
                lambda index, depth=depth: float(
                    np.mean(vectors[f"swap_accuracy_K{depth}"][index])
                ),
            ),
        }
    choice_gain = base.metric(
        draws,
        lambda index: float(np.mean(vectors["C_K2"][index] - vectors["C_K1"][index])),
    )
    raw_gain = base.metric(
        draws,
        lambda index: float(np.mean(vectors["D_K2"][index] - vectors["D_K1"][index])),
    )
    lookup = {
        (unit["world_id"], unit["state"], int(unit["K"])): unit for unit in units
    }
    original = np.asarray(
        [lookup[(world, "original", DEEP)]["reference_margin"] for world in ordered_worlds]
    )
    swap = np.asarray(
        [lookup[(world, "bridge_swap", DEEP)]["reference_margin"] for world in ordered_worlds]
    )
    repair = np.asarray(
        [lookup[(world, "topology_repair", DEEP)]["reference_margin"] for world in ordered_worlds]
    )
    repair_increment = base.metric(
        draws, lambda index: float(np.mean(repair[index] - swap[index]))
    )
    recovery = base.ratio_metric(
        draws,
        numerator=lambda index: float(np.mean(repair[index] - swap[index])),
        denominator=lambda index: float(np.mean(original[index] - swap[index])),
    )
    repair_accuracy = base.metric(
        draws, lambda index: float(np.mean(vectors["repair_accuracy_K2"][index]))
    )
    bridge_passes = bool(
        by_h["2"]["path_margin_D"]["ci95"][0] > 0.0
        and raw_gain["ci95"][0] > 0.0
        and choice_gain["ci95"][0] > 0.0
    )
    topology_passes = bool(bridge_passes and repair_increment["ci95"][0] > 0.0)
    decision = (
        "SUPPORT_HRM_TOPOLOGY_SENSITIVE_PATH_CONTROL"
        if topology_passes
        else "SUPPORT_HRM_DEPTH_DEPENDENT_BRIDGE_SWAP_SENSITIVITY_ONLY"
        if bridge_passes
        else "DO_NOT_SUPPORT_HRM_DEPTH_DEPENDENT_BRIDGE_SWAP_SENSITIVITY"
    )
    leave_one_out = {}
    for omitted in sorted(set(relations.tolist())):
        keep = relations != omitted
        leave_one_out[str(int(omitted))] = {
            "choice_gain": float(np.mean(vectors["C_K2"][keep] - vectors["C_K1"][keep])),
            "raw_gain": float(np.mean(vectors["D_K2"][keep] - vectors["D_K1"][keep])),
            "repair_increment": float(np.mean((repair - swap)[keep])),
        }
    return {
        "decision": decision,
        "primary_worlds": len(ordered_worlds),
        "primary_split": "240 worlds untouched by the HRM competence gate",
        "primary_world_ids_sha256": PRIMARY_WORLD_IDS_SHA256,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": stable_seed(TREATMENT_SALT),
            "resampling": "40 whole worlds within each of six relation families",
        },
        "by_H_exit": by_h,
        "H2_minus_H1_choice_gain": choice_gain,
        "H2_minus_H1_raw_margin_gain": raw_gain,
        "H2_repair_increment": repair_increment,
        "H2_topology_recovery_fraction": recovery,
        "H2_repair_accuracy": repair_accuracy,
        "depth_dependent_bridge_swap_sensitivity_passes": bridge_passes,
        "topology_sensitive_path_control_passes": topology_passes,
        "leave_one_relation_family_out": leave_one_out,
    }


def fmt(metric: dict[str, Any]) -> str:
    return (
        f"{metric['estimate']:.3f} "
        f"[{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competence-artifact", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    competence_rows, stored_summary = load_competence(args.competence_artifact)
    gate = competence_summary(competence_rows)
    if gate != stored_summary or gate["passes_competence_gate"] is not True:
        raise RuntimeError("released HRM linear competence summary does not match raw logits")
    treatment_rows, execution = load_treatment([path.resolve() for path in args.inputs])
    result = analyze_treatment(treatment_rows)
    payload = {
        "schema": "evidence-loops.hrm-linear-analysis.v1",
        "status": "prespecified_model_transfer",
        "model_key": MODEL_KEY,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "decision": result["decision"],
        "competence": gate,
        "execution": execution,
        "result": result,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(
        "\n".join(
            (
                "# HRM-Text-1B linear path-control transfer",
                "",
                f"Decision: `{result['decision']}`",
                "",
                f"- H2 gate accuracy: {fmt(gate['graph_correct_accuracy'])}",
                f"- H2 path-margin contrast: {fmt(result['by_H_exit']['2']['path_margin_D'])}",
                f"- H2 minus H1 exact-choice gain: {fmt(result['H2_minus_H1_choice_gain'])}",
                f"- H2 minus H1 raw-margin gain: {fmt(result['H2_minus_H1_raw_margin_gain'])}",
                f"- H2 topology-recovery fraction: {fmt(result['H2_topology_recovery_fraction'])}",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(args.json_output)


if __name__ == "__main__":
    main()
