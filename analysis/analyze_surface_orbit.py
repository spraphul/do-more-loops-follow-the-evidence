#!/usr/bin/env python3
"""Reproduce the fresh four-rendering Ouro surface-orbit confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np


PANEL_SCHEMA = "iclr2027.ouro26_surface_orbit_confirmation.panel.v3"
RUN_SCHEMA = "iclr2027.ouro26_surface_orbit_confirmation.run.v3"
RESULT_SCHEMA = "iclr2027.ouro26_surface_orbit_confirmation.result.v3"
PANEL_CANONICAL_SHA256 = (
    "448bae9f9ed496a5a52d300d8c64c722ed4f4429933a648856ffdef6926d34f2"
)
SEED = 20260912
BOOTSTRAP_REPLICATES = 10_000
DEPTHS = (1, 4)
SERIALIZATIONS = ("arrows", "tuples", "json", "sentences")
ROOT = Path(__file__).resolve().parents[1]
OURO_REPOSITORY = "ByteDance/Ouro-2.6B"
OURO_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"


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


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(
        f"surface-orbit-confirmation-v3:{SEED}:{label}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_panel(path: Path) -> dict[str, Any]:
    panel = load(path)
    stored = panel.pop("panel_sha256", None)
    observed = canonical_json_sha256(panel)
    panel["panel_sha256"] = stored
    if (
        panel.get("schema") != PANEL_SCHEMA
        or stored != PANEL_CANONICAL_SHA256
        or observed != stored
    ):
        raise RuntimeError("surface-orbit panel binding changed")
    expected_dimensions = {
        "worlds": 72,
        "gate_worlds": 24,
        "confirmation_worlds": 48,
        "surface_realizations_per_world": 4,
        "prompt_cells": 1920,
        "globally_unique_visible_symbols": 4032,
    }
    if panel.get("dimensions") != expected_dimensions:
        raise RuntimeError("surface-orbit panel dimensions changed")
    return panel


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def competence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 384 or len({row["row_id"] for row in rows}) != 384:
        raise RuntimeError("surface competence row grid changed")
    world_ids = sorted({row["world_id"] for row in rows})
    if len(world_ids) != 24:
        raise RuntimeError("surface competence world count changed")
    overall = mean(float(row["score"]["state_correct_argmax_credit"]) for row in rows)
    world_values = np.asarray(
        [
            mean(
                float(row["score"]["state_correct_argmax_credit"])
                for row in rows
                if row["world_id"] == world_id
            )
            for world_id in world_ids
        ]
    )
    rng = np.random.default_rng(stable_seed("competence-bootstrap"))
    bootstrap = np.asarray(
        [
            float(np.mean(rng.choice(world_values, size=len(world_values), replace=True)))
            for _ in range(BOOTSTRAP_REPLICATES)
        ]
    )
    ci95 = interval(bootstrap)
    by_serialization = {
        style: mean(
            float(row["score"]["state_correct_argmax_credit"])
            for row in rows
            if row["serialization"] == style
        )
        for style in SERIALIZATIONS
    }
    by_query_arm = {
        arm: mean(
            float(row["score"]["state_correct_argmax_credit"])
            for row in rows
            if row["query_arm"] == arm
        )
        for arm in ("A", "B")
    }
    by_candidate_order = {
        order: mean(
            float(row["score"]["state_correct_argmax_credit"])
            for row in rows
            if row["candidate_order"] == order
        )
        for order in ("original", "reversed")
    }
    minimum_slice = min(
        *by_serialization.values(),
        *by_query_arm.values(),
        *by_candidate_order.values(),
    )
    passes = overall >= 0.65 and ci95[0] > 0.50 and minimum_slice >= 0.55
    return {
        "decision": "PROMOTE_TO_TREATMENT" if passes else "STOP_INCOMPETENT_SURFACES",
        "passes_competence_gate": passes,
        "worlds": len(world_ids),
        "graph_correct_accuracy": overall,
        "world_bootstrap_ci95": ci95,
        "accuracy_by_serialization": by_serialization,
        "accuracy_by_query_arm": by_query_arm,
        "accuracy_by_candidate_order": by_candidate_order,
        "requirements": {
            "overall_at_least": 0.65,
            "world_bootstrap_lower_strictly_above": 0.50,
            "each_serialization_query_arm_and_candidate_order_at_least": 0.55,
        },
    }


def validate_competence_execution(value: dict[str, Any]) -> dict[str, Any]:
    tokenization = value.get("tokenization", {})
    parity = value.get("native_readout_parity", {})
    checkpoint = value.get("checkpoint", {})
    per_depth = parity.get("per_K", {})
    if (
        tokenization.get("prompts_checked") != 384
        or tokenization.get("labels") != ["A", "B"]
        or tokenization.get("all_are_exact_one_token_extensions") is not True
        or parity.get("passes") is not True
        or set(per_depth) != {"1", "2", "3", "4"}
        or checkpoint.get("repository") != OURO_REPOSITORY
        or checkpoint.get("revision") != OURO_REVISION
        or checkpoint.get("native_K_values") != [1, 2, 3, 4]
        or checkpoint.get("readout_selector") != "exit_at_step=K-1"
        or checkpoint.get("runtime_truncation_readout_parity_required") is not True
    ):
        raise RuntimeError("surface competence execution audit changed")
    for depth in ("1", "2", "3", "4"):
        item = per_depth[depth]
        if (
            item.get("full_vocabulary_max_abs_logit_difference") != 0.0
            or item.get("answer_token_max_abs_logit_difference") != 0.0
            or item.get("torch_allclose") is not True
            or item.get("exact_equal") is not True
        ):
            raise RuntimeError(f"surface native-readout parity failed at K={depth}")
    return value


def validate_competence_crosswalk(
    provenance_path: Path,
    competence_path: Path,
    source_sha256: str,
) -> None:
    provenance = load(provenance_path)
    matches = [
        item
        for item in provenance.get("artifacts", [])
        if item.get("artifact_id") == "surface_orbit_competence"
    ]
    if len(matches) != 1:
        raise RuntimeError("surface competence provenance entry is missing or ambiguous")
    entry = matches[0]
    if (
        entry.get("source_sha256") != source_sha256
        or entry.get("release_sha256") != sha256_file(competence_path)
        or entry.get("release_path")
        != "data/analysis_ready/surface_orbit/competence.json"
    ):
        raise RuntimeError("surface competence source-to-release crosswalk changed")


def merge_treatment(
    paths: list[Path], panel_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = [load(path) for path in paths]
    for artifact, path in zip(artifacts, paths, strict=True):
        if (
            artifact.get("schema") != RUN_SCHEMA
            or artifact.get("mode") != "treatment"
            or artifact.get("model_key") != "ouro26"
            or artifact.get("panel", {}).get("canonical_sha256") != panel_sha256
        ):
            raise RuntimeError(f"invalid surface treatment artifact: {path}")
    shard_counts = {int(artifact["shard"]["count"]) for artifact in artifacts}
    shard_indices = sorted(int(artifact["shard"]["index"]) for artifact in artifacts)
    runner_hashes = {artifact.get("runner", {}).get("sha256") for artifact in artifacts}
    unlock_hashes = {
        artifact.get("competence_unlock", {}).get("sha256") for artifact in artifacts
    }
    if (
        shard_counts != {8}
        or shard_indices != list(range(8))
        or len(runner_hashes) != 1
        or len(unlock_hashes) != 1
    ):
        raise RuntimeError("surface treatment shard bindings disagree")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    if len(rows) != 3072 or len({row["row_id"] for row in rows}) != 3072:
        raise RuntimeError("surface treatment row grid changed")
    return rows, {
        "runner_source_sha256": next(iter(runner_hashes)),
        "competence_source_sha256": next(iter(unlock_hashes)),
        "shards": [
            {"path": path.name, "sha256": sha256_file(path)} for path in paths
        ],
    }


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    design_block: dict[str, int] = {}
    surface_metadata: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (
            row["world_id"],
            row["serialization"],
            row["evidence_state"],
            int(row["K"]),
        )
        grouped[key].append(row)
        design_block[row["world_id"]] = int(row["design_block"])
        surface_key = (row["world_id"], row["serialization"])
        metadata = {
            "morphology_index": int(row["morphology_index"]),
            "wording_index": int(row["wording_index"]),
            "order_index": int(row["order_index"]),
        }
        previous = surface_metadata.setdefault(surface_key, metadata)
        if previous != metadata:
            raise RuntimeError("surface metadata changes across paired cells")
    worlds = sorted(design_block)
    if len(worlds) != 48:
        raise RuntimeError("surface confirmation world count changed")

    def cell(world: str, style: str, state: str, depth: int, field: str) -> float:
        values = grouped[(world, style, state, depth)]
        if len(values) != 4:
            raise RuntimeError("surface arm/order crossing changed")
        return mean(float(row["score"][field]) for row in values)

    def vectors(
        selector: Callable[[str, dict[str, int]], bool]
    ) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for depth in DEPTHS:
            styles_by_world = {
                world: tuple(
                    style
                    for style in SERIALIZATIONS
                    if selector(style, surface_metadata[(world, style)])
                )
                for world in worlds
            }
            if any(not styles for styles in styles_by_world.values()):
                raise RuntimeError("requested surface slice is empty within a world")
            original_margin = np.asarray(
                [
                    mean(
                        cell(world, style, "original", depth, "reference_logit_margin")
                        for style in styles_by_world[world]
                    )
                    for world in worlds
                ]
            )
            swap_margin = np.asarray(
                [
                    mean(
                        cell(
                            world,
                            style,
                            "bridge_swap",
                            depth,
                            "reference_logit_margin",
                        )
                        for style in styles_by_world[world]
                    )
                    for world in worlds
                ]
            )
            original_choice = np.asarray(
                [
                    mean(
                        cell(
                            world,
                            style,
                            "original",
                            depth,
                            "reference_argmax_credit",
                        )
                        for style in styles_by_world[world]
                    )
                    for world in worlds
                ]
            )
            swap_choice = np.asarray(
                [
                    mean(
                        cell(
                            world,
                            style,
                            "bridge_swap",
                            depth,
                            "reference_argmax_credit",
                        )
                        for style in styles_by_world[world]
                    )
                    for world in worlds
                ]
            )
            output[f"D{depth}"] = original_margin - swap_margin
            output[f"C{depth}"] = original_choice - swap_choice
        output["G41"] = output["D4"] - output["D1"]
        output["H41"] = output["C4"] - output["C1"]
        return output

    block_array = np.asarray([design_block[world] for world in worlds])
    strata = {
        value: np.flatnonzero(block_array == value)
        for value in sorted(set(block_array))
    }
    if len(strata) != 16 or {len(indices) for indices in strata.values()} != {3}:
        raise RuntimeError("surface confirmation design-block balance changed")
    rng = np.random.default_rng(stable_seed("bootstrap"))
    draws = [
        np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in strata.values()
            ]
        )
        for _ in range(BOOTSTRAP_REPLICATES)
    ]

    def summarize(values: np.ndarray) -> dict[str, Any]:
        bootstrap = np.asarray([float(np.mean(values[draw])) for draw in draws])
        return {"estimate": float(np.mean(values)), "ci95": interval(bootstrap)}

    def summarize_slice(
        selector: Callable[[str, dict[str, int]], bool]
    ) -> dict[str, Any]:
        return {name: summarize(values) for name, values in vectors(selector).items()}

    overall = summarize_slice(lambda _style, _metadata: True)
    by_serialization = {
        style: summarize_slice(
            lambda candidate, _metadata, selected=style: candidate == selected
        )
        for style in SERIALIZATIONS
    }
    by_morphology = {
        str(index): summarize_slice(
            lambda _style, metadata, selected=index: metadata["morphology_index"]
            == selected
        )
        for index in range(4)
    }
    by_question_wording = {
        str(index): summarize_slice(
            lambda _style, metadata, selected=index: metadata["wording_index"]
            == selected
        )
        for index in range(4)
    }
    primary = all(overall[name]["ci95"][0] > 0 for name in ("D4", "G41", "H41"))
    all_serialization_points_positive = all(
        by_serialization[style][name]["estimate"] > 0
        for style in SERIALIZATIONS
        for name in ("D4", "G41", "H41")
    )
    return {
        "decision": (
            "CONFIRM_SURFACE_ROBUST_ACQUISITION"
            if primary
            else "DO_NOT_CONFIRM_SURFACE_ROBUST_ACQUISITION"
        ),
        "primary_confirmation_passes": primary,
        "all_serialization_point_estimates_positive": all_serialization_points_positive,
        "worlds": len(worlds),
        "surface_realizations_per_world": 4,
        "overall": overall,
        "by_serialization": by_serialization,
        "by_morphology": by_morphology,
        "by_question_wording": by_question_wording,
    }


def format_metric(metric: dict[str, Any]) -> str:
    return (
        f"{metric['estimate']:.3f} "
        f"[{metric['ci95'][0]:.3f}, {metric['ci95'][1]:.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--competence-artifact", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT / "data/PROVENANCE.json",
    )
    args = parser.parse_args()

    panel = validate_panel(args.panel.resolve())
    competence_artifact = load(args.competence_artifact.resolve())
    if (
        competence_artifact.get("schema") != RUN_SCHEMA
        or competence_artifact.get("mode") != "competence"
        or competence_artifact.get("model_key") != "ouro26"
        or competence_artifact.get("panel", {}).get("canonical_sha256")
        != panel["panel_sha256"]
    ):
        raise RuntimeError("invalid surface competence artifact")
    competence = competence_summary(competence_artifact["rows"])
    competence_execution = validate_competence_execution(
        competence_artifact.get("backend_audit", {})
    )
    if not competence["passes_competence_gate"]:
        raise RuntimeError("surface treatment was not unlocked")
    rows, execution = merge_treatment(
        [path.resolve() for path in args.inputs], panel["panel_sha256"]
    )
    validate_competence_crosswalk(
        args.provenance.resolve(),
        args.competence_artifact.resolve(),
        execution["competence_source_sha256"],
    )
    result = analyze_rows(rows)
    payload = {
        "schema": RESULT_SCHEMA,
        **result,
        "competence": competence,
        "resampling": (
            "10000 percentile bootstrap draws over worlds, stratified by the 16 "
            "frozen morphology-by-wording offset blocks; all repeated conditions "
            "remain coupled within world"
        ),
        "execution": {
            "panel": {
                "path": args.panel.name,
                "sha256": sha256_file(args.panel.resolve()),
                "canonical_sha256": panel["panel_sha256"],
            },
            "competence": {
                "path": args.competence_artifact.name,
                "sha256": sha256_file(args.competence_artifact.resolve()),
                "execution_audit": competence_execution,
            },
            **execution,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Ouro-2.6B surface-orbit confirmation",
        "",
        f"Decision: `{result['decision']}`",
        "",
        f"- Intact-graph competence: {competence['graph_correct_accuracy']:.3f} "
        f"[{competence['world_bootstrap_ci95'][0]:.3f}, "
        f"{competence['world_bootstrap_ci95'][1]:.3f}]",
        f"- K4 path contrast D4: {format_metric(result['overall']['D4'])}",
        f"- Raw K1-to-K4 gain G41: {format_metric(result['overall']['G41'])}",
        f"- Choice K1-to-K4 gain H41: {format_metric(result['overall']['H41'])}",
        "- All serialization-specific point estimates positive: "
        f"{'yes' if result['all_serialization_point_estimates_positive'] else 'no'}",
    ]
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.json_output)
    print(args.markdown_output)


if __name__ == "__main__":
    main()
