#!/usr/bin/env python3
"""Analyze fictional on/off-path and adverse-locality result shards."""

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


RUN_SCHEMA = "iclr2027.nonce_structural_falsifiers.run.v1"
RESULT_SCHEMA = "iclr2027.nonce_structural_falsifiers.result.v1"
DEPTHS = (1, 2, 3, 4)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SALT = "iclr2027-nonce-structural-falsifiers-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def panel_sha256(path: Path) -> str:
    panel = json.loads(path.read_text(encoding="utf-8"))
    stored = panel.pop("panel_sha256", None)
    observed = canonical_json_sha256(panel)
    if not isinstance(stored, str) or observed != stored:
        raise RuntimeError("structural panel canonical hash changed")
    return stored


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SALT}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_and_merge(
    paths: list[Path], mode: str, expected_panel_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for artifact, path in zip(artifacts, paths, strict=True):
        if (
            artifact.get("schema") != RUN_SCHEMA
            or artifact.get("mode") != mode
            or artifact.get("model_key") != "ouro26"
            or artifact.get("panel", {}).get("canonical_sha256")
            != expected_panel_sha256
        ):
            raise RuntimeError(f"invalid result artifact: {path}")
    shard_counts = {int(artifact["shard"]["count"]) for artifact in artifacts}
    runner_hashes = {artifact["runner"]["sha256"] for artifact in artifacts}
    mocks = {bool(artifact["mock"]) for artifact in artifacts}
    if len(shard_counts) != 1 or len(runner_hashes) != 1 or len(mocks) != 1:
        raise RuntimeError("shards disagree on execution identity")
    shard_count = next(iter(shard_counts))
    indices = sorted(int(artifact["shard"]["index"]) for artifact in artifacts)
    if len(artifacts) != shard_count or indices != list(range(shard_count)):
        raise RuntimeError("shard set is incomplete")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate rows across shards")
    return rows, {
        "runner_sha256": next(iter(runner_hashes)),
        "mock": next(iter(mocks)),
        "shards": [
            {"path": path.name, "sha256": sha256_file(path)} for path in paths
        ],
    }


def validate(rows: list[dict[str, Any]], mode: str) -> None:
    expected_worlds = 144 if mode == "confirm" else 24
    worlds = {row["world_id"] for row in rows}
    if len(worlds) != expected_worlds:
        raise RuntimeError(f"{mode} world count changed: {len(worlds)}")
    expected_rows = expected_worlds * 32 * len(DEPTHS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"row count changed: {len(rows)} != {expected_rows}")
    per_world = Counter((row["world_id"], int(row["K"])) for row in rows)
    if set(per_world.values()) != {32}:
        raise RuntimeError("per-world factorial is incomplete")
    for row in rows:
        score = row["score"]
        if set(score["raw_logits"]) != {"A", "B"}:
            raise RuntimeError("row does not retain both label logits")
        if not math.isfinite(float(score["reference_logit_margin"])):
            raise RuntimeError("non-finite reference margin")


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    worlds = sorted({row["world_id"] for row in rows})
    relation_by_world: dict[str, int] = {}
    cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        relation_by_world[row["world_id"]] = int(row["relation_family_index"])
        if row["experiment"] == "path_specificity":
            key = (
                row["world_id"], "specificity", row["query_role"],
                row["edit_state"], int(row["K"]),
            )
        else:
            key = (
                row["world_id"], "locality", row["physical_locality"],
                row["terminal_binding"], int(row["K"]),
            )
        cell[key].append(row)

    summary: dict[tuple[Any, ...], tuple[float, float, float]] = {}
    for key, members in cell.items():
        if len(members) != 4:
            raise RuntimeError(f"arm/order cell is incomplete: {key}")
        summary[key] = (
            mean(float(row["score"]["reference_logit_margin"]) for row in members),
            mean(float(row["score"]["reference_argmax_credit"]) for row in members),
            mean(float(row["score"]["state_correct_argmax_credit"]) for row in members),
        )

    vectors: dict[str, np.ndarray] = {}
    for k in DEPTHS:
        on_base = np.asarray([summary[(w, "specificity", "on_path", "base", k)][0] for w in worlds])
        on_edit = np.asarray([summary[(w, "specificity", "on_path", "bridge_transposition", k)][0] for w in worlds])
        off_base = np.asarray([summary[(w, "specificity", "off_path", "base", k)][0] for w in worlds])
        off_edit = np.asarray([summary[(w, "specificity", "off_path", "bridge_transposition", k)][0] for w in worlds])
        on_base_c = np.asarray([summary[(w, "specificity", "on_path", "base", k)][1] for w in worlds])
        on_edit_c = np.asarray([summary[(w, "specificity", "on_path", "bridge_transposition", k)][1] for w in worlds])
        off_base_c = np.asarray([summary[(w, "specificity", "off_path", "base", k)][1] for w in worlds])
        off_edit_c = np.asarray([summary[(w, "specificity", "off_path", "bridge_transposition", k)][1] for w in worlds])
        vectors[f"on_effect_raw_K{k}"] = on_base - on_edit
        vectors[f"off_effect_raw_K{k}"] = off_base - off_edit
        vectors[f"specificity_raw_K{k}"] = (on_base - on_edit) - (off_base - off_edit)
        vectors[f"on_effect_choice_K{k}"] = on_base_c - on_edit_c
        vectors[f"off_effect_choice_K{k}"] = off_base_c - off_edit_c
        vectors[f"specificity_choice_K{k}"] = (on_base_c - on_edit_c) - (off_base_c - off_edit_c)

        for locality in ("matched", "crossed"):
            normal = np.asarray([summary[(w, "locality", locality, "normal", k)][0] for w in worlds])
            transposed = np.asarray([summary[(w, "locality", locality, "transposed", k)][0] for w in worlds])
            normal_c = np.asarray([summary[(w, "locality", locality, "normal", k)][1] for w in worlds])
            transposed_c = np.asarray([summary[(w, "locality", locality, "transposed", k)][1] for w in worlds])
            accuracy = np.asarray(
                [
                    mean(
                        (
                            summary[(w, "locality", locality, "normal", k)][2],
                            summary[(w, "locality", locality, "transposed", k)][2],
                        )
                    )
                    for w in worlds
                ]
            )
            vectors[f"graph_effect_raw_{locality}_K{k}"] = transposed - normal
            vectors[f"graph_effect_choice_{locality}_K{k}"] = transposed_c - normal_c
            vectors[f"state_accuracy_{locality}_K{k}"] = accuracy
        vectors[f"locality_amplification_raw_K{k}"] = (
            vectors[f"graph_effect_raw_matched_K{k}"]
            - vectors[f"graph_effect_raw_crossed_K{k}"]
        )
        vectors[f"locality_amplification_choice_K{k}"] = (
            vectors[f"graph_effect_choice_matched_K{k}"]
            - vectors[f"graph_effect_choice_crossed_K{k}"]
        )

    for kind in ("raw", "choice"):
        vectors[f"specificity_{kind}_K4_minus_K1"] = (
            vectors[f"specificity_{kind}_K4"] - vectors[f"specificity_{kind}_K1"]
        )
        vectors[f"crossed_graph_{kind}_K4_minus_K1"] = (
            vectors[f"graph_effect_{kind}_crossed_K4"]
            - vectors[f"graph_effect_{kind}_crossed_K1"]
        )
    relations = np.asarray([relation_by_world[world] for world in worlds], dtype=int)
    return worlds, relations, vectors


def bootstrap_indices(relations: np.ndarray, mode: str) -> list[np.ndarray]:
    groups = {value: np.flatnonzero(relations == value) for value in sorted(set(relations))}
    expected_per_group = 24 if mode == "confirm" else 4
    if len(groups) != 6 or set(map(len, groups.values())) != {expected_per_group}:
        raise RuntimeError("relation-family split balance changed")
    rng = np.random.default_rng(stable_seed(mode))
    return [
        np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups.values()]
        )
        for _ in range(BOOTSTRAP_REPLICATES)
    ]


def estimate(vector: np.ndarray, draws: list[np.ndarray]) -> dict[str, Any]:
    bootstrap = np.asarray([float(np.mean(vector[draw])) for draw in draws])
    lower, upper = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "estimate": float(np.mean(vector)),
        "ci95": [float(lower), float(upper)],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def fmt(result: dict[str, Any]) -> str:
    return (
        f"{result['estimate']:.3f} "
        f"[{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirm"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    expected_panel_sha256 = panel_sha256(args.panel)
    rows, execution = load_and_merge(args.inputs, args.mode, expected_panel_sha256)
    validate(rows, args.mode)
    worlds, relations, vectors = aggregate(rows)
    draws = bootstrap_indices(relations, args.mode)
    results = {name: estimate(vector, draws) for name, vector in vectors.items()}

    path_support = all(
        results[name]["ci95"][0] > 0
        for name in (
            "specificity_raw_K4",
            "specificity_choice_K4",
            "specificity_raw_K4_minus_K1",
            "specificity_choice_K4_minus_K1",
        )
    )
    locality_support = all(
        results[name]["ci95"][0] > 0
        for name in (
            "graph_effect_raw_crossed_K4",
            "graph_effect_choice_crossed_K4",
            "crossed_graph_raw_K4_minus_K1",
            "crossed_graph_choice_K4_minus_K1",
        )
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "mode": args.mode,
        "model": "ByteDance/Ouro-2.6B",
        "worlds": len(worlds),
        "inference": (
            "10,000 shared world bootstrap draws stratified within six relation families"
        ),
        "execution": execution,
        "results": results,
        "decisions": {
            "answer_path_relevance_supported": path_support,
            "graph_control_survives_adverse_locality": locality_support,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fictional structural falsifiers",
        "",
        f"Mode: `{args.mode}`. Worlds: {len(worlds)}.",
        "",
        "| Quantity | Raw margin | Exact A/B choice |",
        "|---|---:|---:|",
        f"| On-path effect, K4 | {fmt(results['on_effect_raw_K4'])} | {fmt(results['on_effect_choice_K4'])} |",
        f"| Off-path effect, K4 | {fmt(results['off_effect_raw_K4'])} | {fmt(results['off_effect_choice_K4'])} |",
        f"| On-minus-off specificity, K4 | {fmt(results['specificity_raw_K4'])} | {fmt(results['specificity_choice_K4'])} |",
        f"| Specificity, K4 minus K1 | {fmt(results['specificity_raw_K4_minus_K1'])} | {fmt(results['specificity_choice_K4_minus_K1'])} |",
        f"| Matched-locality graph effect, K4 | {fmt(results['graph_effect_raw_matched_K4'])} | {fmt(results['graph_effect_choice_matched_K4'])} |",
        f"| Adverse-locality graph effect, K4 | {fmt(results['graph_effect_raw_crossed_K4'])} | {fmt(results['graph_effect_choice_crossed_K4'])} |",
        f"| Adverse graph effect, K4 minus K1 | {fmt(results['crossed_graph_raw_K4_minus_K1'])} | {fmt(results['crossed_graph_choice_K4_minus_K1'])} |",
        f"| Locality amplification, K4 | {fmt(results['locality_amplification_raw_K4'])} | {fmt(results['locality_amplification_choice_K4'])} |",
        "",
        f"Answer-path relevance supported: `{path_support}`.",
        f"Graph control survives adverse locality: `{locality_support}`.",
        "",
        "Development estimates are feasibility diagnostics only. Confirmation estimates use",
        "a disjoint, perfectly balanced relation-by-ordered-terminal-pair sample.",
    ]
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.markdown_output)
    print(json.dumps(payload["decisions"], indent=2))


if __name__ == "__main__":
    main()
