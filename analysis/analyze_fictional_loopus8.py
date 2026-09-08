#!/usr/bin/env python3
"""Merge and analyze the LoopUS-8B fictional path-control extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


RUN_SCHEMA = "iclr2027.loopus8_nonce_path_control.run.v1"
RESULT_SCHEMA = "iclr2027.loopus8_nonce_path_control.result.v1"
COMPETENCE_SCHEMA = "iclr2027.loopus8_nonce_competence.run.v1"
PANEL_SHA256 = "7880c02c64af541629b9c23dce15bf4a6df98c4f99a8aed45b4a310df2e543e5"
DEPTHS = (1, 2, 4, 8)
DEEP = 8
BOOTSTRAP_REPLICATES = 10_000


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


def load_panel(path: Path) -> dict[str, Any]:
    panel = json.loads(path.read_text(encoding="utf-8"))
    stored = panel.pop("panel_sha256", None)
    observed = canonical_json_sha256(panel)
    panel["panel_sha256"] = stored
    if stored != PANEL_SHA256 or observed != stored:
        raise RuntimeError("fictional panel binding changed")
    return panel


def merge_shards(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for artifact, path in zip(artifacts, paths, strict=True):
        if (
            artifact.get("schema") != RUN_SCHEMA
            or artifact.get("mode") != "treatment"
            or artifact.get("model_key") != "loopus8"
            or artifact.get("panel", {}).get("canonical_sha256") != PANEL_SHA256
        ):
            raise RuntimeError(f"invalid treatment artifact: {path}")
    shard_counts = {int(a["shard"]["count"]) for a in artifacts}
    indices = sorted(int(a["shard"]["index"]) for a in artifacts)
    runners = {a["runner"]["sha256"] for a in artifacts}
    backends = {a["backend_runner"]["sha256"] for a in artifacts}
    unlocks = {a["competence_unlock"]["sha256"] for a in artifacts}
    mock_flags = {bool(a.get("mock")) for a in artifacts}
    if any(len(values) != 1 for values in (shard_counts, runners, backends, unlocks, mock_flags)):
        raise RuntimeError("treatment shards disagree on execution bindings")
    shard_count = next(iter(shard_counts))
    if len(artifacts) != shard_count or indices != list(range(shard_count)):
        raise RuntimeError("treatment shard set is incomplete")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("treatment shards contain duplicate rows")
    return rows, {
        "runner_sha256": next(iter(runners)),
        "backend_runner_sha256": next(iter(backends)),
        "competence_artifact_sha256": next(iter(unlocks)),
        "mock": next(iter(mock_flags)),
        "shards": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in paths
        ],
    }


def subset_sensitivity(
    units: list[dict[str, Any]], excluded_worlds: set[str]
) -> dict[str, Any]:
    lookup = {
        (unit["world_id"], unit["state"], int(unit["K"])): unit for unit in units
    }
    worlds = sorted({unit["world_id"] for unit in units} - excluded_worlds)
    relation = np.asarray(
        [int(lookup[(world, "original", 1)]["relation_family_index"]) for world in worlds]
    )

    def vector(state: str, depth: int, field: str) -> np.ndarray:
        return np.asarray([lookup[(world, state, depth)][field] for world in worlds])

    d1 = vector("original", 1, "reference_margin") - vector(
        "bridge_swap", 1, "reference_margin"
    )
    d8 = vector("original", 8, "reference_margin") - vector(
        "bridge_swap", 8, "reference_margin"
    )
    c1 = vector("original", 1, "reference_choice_credit") - vector(
        "bridge_swap", 1, "reference_choice_credit"
    )
    c8 = vector("original", 8, "reference_choice_credit") - vector(
        "bridge_swap", 8, "reference_choice_credit"
    )

    seed = int.from_bytes(
        hashlib.sha256(b"loopus8-gate-excluded-sensitivity-v1").digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(relation == value) for value in sorted(set(relation))]
    if len(groups) != 6 or {len(group) for group in groups} != {40}:
        raise RuntimeError("gate-excluded relation balance changed")
    draws = [
        np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for _ in range(BOOTSTRAP_REPLICATES)
    ]

    def summarize(values: np.ndarray) -> dict[str, Any]:
        sampled = np.asarray([float(np.mean(values[index])) for index in draws])
        return {
            "estimate": float(np.mean(values)),
            "ci95": [
                float(np.quantile(sampled, 0.025)),
                float(np.quantile(sampled, 0.975)),
            ],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }

    return {
        "worlds": len(worlds),
        "relation_worlds": [len(group) for group in groups],
        "deep_path_effect_D8": summarize(d8),
        "raw_margin_gain_G81": summarize(d8 - d1),
        "choice_gain_H81": summarize(c8 - c1),
    }


def fmt(metric: dict[str, Any]) -> str:
    lower, upper = metric["ci95"]
    return f"{metric['estimate']:.3f} [{lower:.3f}, {upper:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--competence-artifact", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    import analyze_fictional_ouro as base

    panel = load_panel(args.panel.resolve())
    if panel["panel_sha256"] != PANEL_SHA256:
        raise RuntimeError("panel binding changed")
    competence = json.loads(args.competence_artifact.read_text(encoding="utf-8"))
    if (
        competence.get("schema") != COMPETENCE_SCHEMA
        or competence.get("mode") != "competence"
        or competence.get("model_key") != "loopus8"
        or competence.get("summary", {}).get("passes_competence_gate") is not True
        or competence.get("panel", {}).get("canonical_sha256") != PANEL_SHA256
    ):
        raise RuntimeError("invalid competence unlock artifact")

    rows, execution = merge_shards([path.resolve() for path in args.inputs])
    if execution["mock"] != bool(competence.get("mock")):
        raise RuntimeError("competence and treatment mock status differ")
    base.MODEL_DEPTHS["loopus8"] = DEPTHS
    base.MODEL_DEEP["loopus8"] = DEEP
    base.validate_grid(rows, "loopus8")
    result = base.analyze(rows, "loopus8")
    units = base.aggregate_worlds(rows)
    excluded = set(panel["competence_world_ids"])
    sensitivity = subset_sensitivity(units, excluded)

    decision = result["treatment_decision"]
    payload = {
        "schema": RESULT_SCHEMA,
        "model_key": "loopus8",
        "model_repository": competence["model_repository"],
        "model_revision": competence["model_revision"],
        "decision": decision,
        "competence": competence["summary"],
        "execution": execution,
        "result": result,
        "gate_excluded_sensitivity": sensitivity,
        "analysis": {
            "path": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    deep = result["by_K"]["8"]
    lines = [
        "# LoopUS-8B fictional path-control extension",
        "",
        f"Decision: `{decision}`",
        "",
        f"- Competence accuracy: {fmt(competence['summary']['graph_correct_accuracy'])}",
        f"- Deep path effect D8: {fmt(deep['path_margin_D'])}",
        f"- Raw depth gain G81: {fmt(result['secondary_raw_margin_gain_G'])}",
        f"- Exact-choice depth gain H81: {fmt(result['primary_choice_gain_H'])}",
        f"- K8 topology recovery F: {fmt(result['deep_topology_recovery_fraction_F'])}",
        "",
        "## Gate-excluded sensitivity",
        "",
        f"- Worlds: {sensitivity['worlds']}",
        f"- Deep path effect D8: {fmt(sensitivity['deep_path_effect_D8'])}",
        f"- Raw depth gain G81: {fmt(sensitivity['raw_margin_gain_G81'])}",
        f"- Exact-choice depth gain H81: {fmt(sensitivity['choice_gain_H81'])}",
    ]
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.json_output)
    print(args.markdown_output)
    print(json.dumps({"decision": decision}, indent=2))


if __name__ == "__main__":
    main()
