#!/usr/bin/env python3
"""Build the public, anonymous artifact from the private research workspace.

This maintainer-only script is not needed by reviewers.  It deliberately
exports model scores and generated worlds while excluding licensed benchmark
text, model weights, credentials, machine paths, timestamps, and user names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


RELEASE_SCHEMA = "evidence-loops.anonymous-release.v1"
SECRET_PATTERN = re.compile(r"(?:hf|sk)-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,}")


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def clean_string(value: str) -> str:
    if SECRET_PATTERN.search(value):
        return "<credential-removed>"
    if value.startswith("/"):
        return "<local-path-removed>"
    return value


DROP_KEYS = {
    "created_at",
    "cwd",
    "dataset_path",
    "device",
    "hostname",
    "log_path",
    "output_path",
    "rows_jsonl",
    "script_path",
    "snapshot_dir",
    "tokenizer_dir",
}


def scrub(value: Any, *, drop_rows: bool = False) -> Any:
    if isinstance(value, str):
        return clean_string(value)
    if isinstance(value, list):
        return [scrub(item, drop_rows=drop_rows) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key in DROP_KEYS:
            continue
        if drop_rows and key in {
            "cells",
            "per_world",
            "prompts",
            "rows",
            "world_cells",
        }:
            continue
        if key == "path" and isinstance(item, str):
            output[key] = Path(item).name if item else item
            continue
        output[key] = scrub(item, drop_rows=drop_rows)
    return output


def pseudonymize_token_traces(
    reference_ids: Iterable[int], other_ids: Iterable[int]
) -> tuple[list[int], list[int]]:
    mapping: dict[int, int] = {}

    def encode(values: Iterable[int]) -> list[int]:
        output: list[int] = []
        for raw in map(int, values):
            if raw not in mapping:
                mapping[raw] = len(mapping) + 1
            output.append(mapping[raw])
        return output

    return encode(reference_ids), encode(other_ids)


def natural_row(row: dict[str, Any], pair_id: str) -> dict[str, Any]:
    state = str(row["evidence_state"])
    query = str(row.get("query_arm", "A"))
    order = str(row.get("candidate_order", "original"))
    depth = int(row["K"])
    score = row["score"]
    reference_log_odds = (
        row["reference_log_odds"]
        if "reference_log_odds" in row
        else score["reference_log_odds"]
    )
    reference_argmax_credit = (
        row["reference_argmax_credit"]
        if "reference_argmax_credit" in row
        else score["reference_argmax_credit"]
    )
    output: dict[str, Any] = {
        "row_id": f"{pair_id}__q{query}__{state}__{order}__K{depth}",
        "pair_id": pair_id,
        "K": depth,
        "evidence_state": state,
        "query_arm": query,
        "candidate_order": order,
        "expected_candidate_identity": int(row["expected_candidate_identity"]),
        "reference_candidate_identity": int(row["reference_candidate_identity"]),
        "reference_log_odds": float(reference_log_odds),
        "reference_argmax_credit": float(reference_argmax_credit),
    }
    reference_accuracy = output["reference_argmax_credit"]
    expected_is_reference = (
        output["expected_candidate_identity"]
        == output["reference_candidate_identity"]
    )
    derived_state_accuracy = (
        reference_accuracy if expected_is_reference else 1.0 - reference_accuracy
    )
    output["state_correct_argmax_credit"] = float(
        row.get("state_correct_argmax_credit", derived_state_accuracy)
    )
    if "relation_path" in row:
        output["relation_path"] = list(map(str, row["relation_path"]))
    if "terminal_relation" in row:
        output["terminal_relation"] = str(row["terminal_relation"])

    if "canonical" in score:
        reference_child = score["canonical"]
        other_child = score["foil"]
        reference_ids = row.get(
            "reference_token_ids", reference_child["continuation_token_ids"]
        )
        other_ids = row.get("other_token_ids", other_child["continuation_token_ids"])
        safe_reference, safe_other = pseudonymize_token_traces(reference_ids, other_ids)
        output["reference_token_ids"] = safe_reference
        output["other_token_ids"] = safe_other
        output["score"] = {
            "canonical": {
                "continuation_token_ids": safe_reference,
                "token_log_probabilities": list(
                    map(float, reference_child["token_log_probabilities"])
                ),
                "sequence_log_probability": float(
                    reference_child["sequence_log_probability"]
                ),
            },
            "foil": {
                "continuation_token_ids": safe_other,
                "token_log_probabilities": list(
                    map(float, other_child["token_log_probabilities"])
                ),
                "sequence_log_probability": float(other_child["sequence_log_probability"]),
            },
            "reference_log_odds": output["reference_log_odds"],
            "reference_argmax_credit": output["reference_argmax_credit"],
        }
    else:
        reference_child = score["reference"]
        other_child = score["other"]
        safe_reference, safe_other = pseudonymize_token_traces(
            reference_child["continuation_token_ids"],
            other_child["continuation_token_ids"],
        )

        def release_child(child: dict[str, Any], ids: list[int]) -> dict[str, Any]:
            released = {
                "continuation_token_ids": ids,
                "token_log_probabilities": list(
                    map(float, child["token_log_probabilities"])
                ),
                "sequence_log_probability": float(child["sequence_log_probability"]),
            }
            if "executed_reasoning_steps" in child:
                released["executed_reasoning_steps"] = float(
                    child["executed_reasoning_steps"]
                )
            if "prompt_tokens" in child:
                released["prompt_tokens"] = int(child["prompt_tokens"])
            return released

        output["score"] = {
            "reference": release_child(reference_child, safe_reference),
            "other": release_child(other_child, safe_other),
            "reference_log_odds": output["reference_log_odds"],
            "reference_argmax_credit": output["reference_argmax_credit"],
        }
    return output


def model_metadata(value: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "adaptive_halting_disabled_with_q_threshold",
        "architecture",
        "config_sha256",
        "dtype",
        "fixed_depths",
        "frozen",
        "repository",
        "revision",
        "tokenizer_files_sha256",
        "torch_version",
        "transformers_version",
        "weight_bytes",
        "weight_sha256",
    }
    return scrub({key: value[key] for key in keep if key in value})


def export_natural(
    source_root: Path,
    release_root: Path,
    artifact_id: str,
    relative_sources: list[str],
    output_name: str,
    provenance: list[dict[str, Any]],
) -> None:
    source_paths = [source_root / relative for relative in relative_sources]
    artifacts = [load_json(path) for path in source_paths]
    source_rows = [row for artifact in artifacts for row in artifact["rows"]]
    original_pairs = sorted({str(row["pair_id"]) for row in source_rows})
    pair_map = {
        pair: f"pair_{index:04d}" for index, pair in enumerate(original_pairs, start=1)
    }
    rows = [natural_row(row, pair_map[str(row["pair_id"])]) for row in source_rows]
    payload = {
        "schema": RELEASE_SCHEMA,
        "artifact_id": artifact_id,
        "model": model_metadata(artifacts[0].get("model", {})),
        "row_count": len(rows),
        "pair_count": len(original_pairs),
        "privacy_boundary": (
            "Benchmark questions, entity strings, prompts, continuations, raw token IDs, "
            "source item IDs, timestamps, and machine paths are excluded."
        ),
        "rows": rows,
    }
    output_path = release_root / "data" / "analysis_ready" / "natural" / output_name
    released_sha = write_json(output_path, payload)
    provenance.append(
        {
            "artifact_id": artifact_id,
            "source_sha256": [sha256_file(path) for path in source_paths],
            "release_path": output_path.relative_to(release_root).as_posix(),
            "release_sha256": released_sha,
            "transformation": "licensed-text-free row-score projection with opaque pair and token IDs",
        }
    )
    for index, artifact in enumerate(artifacts):
        if "summary" not in artifact:
            continue
        suffix = "" if len(artifacts) == 1 else f"_part{index + 1}"
        summary_path = (
            release_root
            / "expected"
            / "execution_summaries"
            / f"{artifact_id}{suffix}_runner_summary.json"
        )
        write_json(summary_path, scrub(artifact["summary"], drop_rows=True))


def fictional_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "K",
        "candidate_order",
        "edit_state",
        "evidence_state",
        "expected_candidate_identity",
        "expected_label",
        "focal_terminal_codes",
        "experiment",
        "physical_locality",
        "query_arm",
        "query_role",
        "reference_candidate_identity",
        "reference_label",
        "relation_family_index",
        "relations",
        "row_id",
        "split",
        "terminal_binding",
        "world_id",
    )
    output = {key: row[key] for key in fields if key in row}
    score = row["score"]
    output["score"] = {
        key: score[key]
        for key in (
            "executed_recurrent_steps",
            "expected_label",
            "other_label",
            "raw_logits",
            "readout_exit_K",
            "reference_argmax_credit",
            "reference_label",
            "reference_logit_margin",
            "reference_pair_probability",
            "state_correct_argmax_credit",
        )
        if key in score
    }
    return output


def export_fictional_artifact(
    source: Path,
    destination: Path,
    artifact_id: str,
    provenance: list[dict[str, Any]],
    *,
    structural_panel_sha256: str | None = None,
) -> None:
    artifact = load_json(source)
    keep = (
        "backend_runner",
        "competence_unlock",
        "mock",
        "mode",
        "model_key",
        "model_repository",
        "model_revision",
        "panel",
        "runner",
        "schema",
        "shard",
        "summary",
    )
    payload = {key: scrub(artifact[key]) for key in keep if key in artifact}
    if structural_panel_sha256 is not None:
        payload.setdefault("panel", {})["canonical_sha256"] = structural_panel_sha256
    if "rows" in artifact:
        payload["rows"] = [fictional_row(row) for row in artifact["rows"]]
    released_sha = write_json(destination, payload)
    provenance.append(
        {
            "artifact_id": artifact_id,
            "source_sha256": sha256_file(source),
            "release_path": destination.relative_to(destination.parents[3]).as_posix(),
            "release_sha256": released_sha,
            "transformation": "score-only projection; prompts are recoverable from the released generated panel",
        }
    )


def export_factorial(
    source_root: Path, release_root: Path, provenance: list[dict[str, Any]]
) -> None:
    source_dir = (
        source_root
        / "paper/iclr2027/revision_strong_accept/results/"
        "tied_untied_graph_factorial_confirmation/runs"
    )
    destination_dir = release_root / "data/analysis_ready/factorial/runs"
    for source in sorted(source_dir.glob("seed*.json")):
        artifact = load_json(source)
        payload = {
            key: scrub(artifact[key])
            for key in (
                "condition",
                "configuration",
                "depths",
                "evaluation",
                "matched_compute_contract",
                "parameters",
                "schema",
                "seed",
                "task",
                "training_history",
            )
            if key in artifact
        }
        destination = destination_dir / source.name
        released_sha = write_json(destination, payload)
        provenance.append(
            {
                "artifact_id": f"factorial_{source.stem}",
                "source_sha256": sha256_file(source),
                "release_path": destination.relative_to(release_root).as_posix(),
                "release_sha256": released_sha,
                "transformation": "removed runtime metadata; numeric training and evaluation records unchanged",
            }
        )


def export_frozen_summaries(
    source_root: Path, release_root: Path, provenance: list[dict[str, Any]]
) -> None:
    sources = {
        "ouro_candidate_list_ablation": "reports/discovery/ouro26_2wiki_candidate_list_ablation_v1_20260906T034900Z/model-merged.json",
        "ouro_candidate_state_transplant": "reports/discovery/ouro26_carrier_scope_confirmation_v1_20260906T025500Z/model.json",
        "ouro_musique_generation": "reports/discovery/ouro26_musique_answer_only_generation_v1_20260906T002500Z/merged.json",
        "loopus_2wiki_generation": "reports/discovery/loopus_2wiki_greedy_generation_v1_20260906/result.json",
        "ouro_terminal_compatibility": "reports/discovery/ouro26_terminal_fact_compatibility_v1_20260906/result.json",
        "ouro_instruction_intervention": "reports/discovery/ouro26_fact_only_instruction_intervention_v1_20260906/result.json",
        "ouro_natural_offpath": "reports/discovery/ouro26_2wiki_offpath_specificity_v2_20260905T191900Z/model_merged.json",
        "loopus_natural_offpath": "reports/discovery/loopus_2wiki_offpath_specificity_v1_20260906/result.json",
        "ouro_natural_locality": "reports/discovery/ouro26_2wiki_terminal_binding_crossover_v1_20260905T202300Z/model_merged.json",
        "ouro14_competence": "reports/discovery/ouro14_2wiki_matched_scale_competence_v1_20260906/result.json",
        "ouro14_postscreen": "reports/discovery/ouro14_2wiki_postscreen_k4_treatment_v1_20260906/result.json",
        "huginn_competence": "reports/discovery/huginn_2wiki_natural_competence_v1_20260906/result.json",
        "loopus_early_choice": "paper/iclr2027/revision_strong_accept/results/loopus_early_choice_localization.json",
        "ouro_k3_headroom": "paper/iclr2027/revision_strong_accept/results/ouro_k3_headroom.json",
        "ouro_k3_k4_change": "paper/iclr2027/revision_strong_accept/results/ouro_k3_k4_decision_change.json",
        "tfqwen_fictional": "paper/iclr2027/revision_strong_accept/results/nonce_path_control/tfqwen/nonce_path_control_tfqwen.json",
        "ouro_threehop_development": "paper/iclr2027/revision_strong_accept/results/ouro26_threehop_development/development.json",
        "loopus_longpath_3": "paper/iclr2027/revision_strong_accept/results/loopus8_long_path_development/length3.json",
        "loopus_longpath_4": "paper/iclr2027/revision_strong_accept/results/loopus8_long_path_development/length4.json",
        "loopus_longpath_5": "paper/iclr2027/revision_strong_accept/results/loopus8_long_path_development/length5.json",
    }
    index: list[dict[str, Any]] = []
    for artifact_id, relative in sources.items():
        source = source_root / relative
        if not source.exists():
            continue
        artifact = load_json(source)
        payload = scrub(artifact, drop_rows=True)
        destination = release_root / "data/frozen_summaries" / f"{artifact_id}.json"
        released_sha = write_json(destination, payload)
        entry = {
            "artifact_id": artifact_id,
            "source_sha256": sha256_file(source),
            "release_path": destination.relative_to(release_root).as_posix(),
            "release_sha256": released_sha,
            "transformation": "curated frozen summary; row-level text-bearing payload excluded",
        }
        provenance.append(entry)
        index.append(
            {
                "artifact_id": artifact_id,
                "path": entry["release_path"],
                "role": "appendix diagnostic or stopped-branch evidence",
            }
        )
    write_json(release_root / "data/frozen_summaries/INDEX.json", index)


def copy_generated_panels(
    source_root: Path, release_root: Path, provenance: list[dict[str, Any]]
) -> str:
    parent_source = (
        source_root
        / "paper/iclr2027/revision_strong_accept/experiment_specs/"
        "nonce_path_control_v1.panel.json"
    )
    parent_destination = release_root / "data/generated/nonce_path_control_v1.panel.json"
    parent_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent_source, parent_destination)
    provenance.append(
        {
            "artifact_id": "nonce_path_control_panel",
            "source_sha256": sha256_file(parent_source),
            "release_path": parent_destination.relative_to(release_root).as_posix(),
            "release_sha256": sha256_file(parent_destination),
            "transformation": "none; fully generated fictional panel",
        }
    )

    structural_source = (
        source_root
        / "paper/iclr2027/revision_strong_accept/experiment_specs/"
        "nonce_structural_falsifiers_v1.panel.json"
    )
    structural = load_json(structural_source)
    original_canonical = structural.pop("panel_sha256")
    structural["parent_panel"]["path"] = (
        "data/generated/nonce_path_control_v1.panel.json"
    )
    released_canonical = canonical_json_sha256(structural)
    structural["panel_sha256"] = released_canonical
    structural_destination = (
        release_root / "data/generated/nonce_structural_falsifiers_v1.panel.json"
    )
    released_sha = write_json(structural_destination, structural)
    provenance.append(
        {
            "artifact_id": "nonce_structural_falsifiers_panel",
            "source_sha256": sha256_file(structural_source),
            "source_canonical_sha256": original_canonical,
            "release_path": structural_destination.relative_to(release_root).as_posix(),
            "release_sha256": released_sha,
            "release_canonical_sha256": released_canonical,
            "transformation": "replaced one author-machine parent-panel path; scientific cells unchanged",
        }
    )
    return released_canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--release-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    release_root = args.release_root.resolve()
    provenance: list[dict[str, Any]] = []

    structural_hash = copy_generated_panels(source_root, release_root, provenance)

    natural_sources = [
        (
            "natural_ouro26_2wiki",
            ["reports/discovery/ouro26_2wiki_bridge_validation_v1_20260905T172500Z/merged.json"],
            "ouro26_2wiki.json",
        ),
        (
            "natural_ouro26_musique",
            ["reports/discovery/ouro26_musique_bridge_validation_v1_20260905T181200Z/merged.json"],
            "ouro26_musique.json",
        ),
        (
            "natural_loopus_2wiki",
            [
                "reports/discovery/loopus_2wiki_heldout_confirmation_v1_20260906/raw/panel1.json",
                "reports/discovery/loopus_2wiki_heldout_confirmation_v1_20260906/raw/panel2_retry.json",
                "reports/discovery/loopus_2wiki_heldout_confirmation_v1_20260906/raw/panel3.json",
            ],
            "loopus_2wiki.json",
        ),
        (
            "natural_loopus_musique",
            [
                f"reports/discovery/loopus_musique_matched_confirmation_v1_20260906/raw/shard{index}.json"
                for index in range(4)
            ],
            "loopus_musique.json",
        ),
        (
            "natural_loopus_full_depth_2wiki",
            [
                f"paper/iclr2027/revision_strong_accept/results/source_artifacts/loopus_full_depth_diagnostic_20260906/2wiki/panel{index}/artifact.json"
                for index in (1, 2, 3)
            ],
            "loopus_full_depth_2wiki.json",
        ),
        (
            "natural_loopus_full_depth_musique",
            [
                f"paper/iclr2027/revision_strong_accept/results/source_artifacts/loopus_full_depth_diagnostic_20260906/musique/shard{index}/artifact.json"
                for index in range(4)
            ],
            "loopus_full_depth_musique.json",
        ),
    ]
    for artifact_id, sources, output in natural_sources:
        export_natural(
            source_root, release_root, artifact_id, sources, output, provenance
        )

    ouro_root = (
        source_root
        / "paper/iclr2027/revision_strong_accept/results/nonce_path_control/source_artifacts/ouro26"
    )
    export_fictional_artifact(
        ouro_root / "competence_final_runner.json",
        release_root / "data/analysis_ready/fictional_ouro/competence.json",
        "fictional_ouro_competence",
        provenance,
    )
    for index in range(8):
        export_fictional_artifact(
            ouro_root / f"full/shard_{index}.json",
            release_root / f"data/analysis_ready/fictional_ouro/shard_{index}.json",
            f"fictional_ouro_shard_{index}",
            provenance,
        )

    loopus_root = (
        source_root
        / "paper/iclr2027/revision_strong_accept/results/loopus8_nonce_path_control"
    )
    export_fictional_artifact(
        loopus_root / "competence.json",
        release_root / "data/analysis_ready/fictional_loopus8/competence.json",
        "fictional_loopus8_competence",
        provenance,
    )
    for index in range(4):
        export_fictional_artifact(
            loopus_root / f"treatment_shard{index}.json",
            release_root / f"data/analysis_ready/fictional_loopus8/shard_{index}.json",
            f"fictional_loopus8_shard_{index}",
            provenance,
        )

    structural_root = (
        source_root
        / "paper/iclr2027/revision_strong_accept/results/"
        "nonce_structural_falsifiers_confirmation"
    )
    for index in range(4):
        export_fictional_artifact(
            structural_root / f"confirm_shard{index}.json",
            release_root / f"data/analysis_ready/structural/shard_{index}.json",
            f"structural_shard_{index}",
            provenance,
            structural_panel_sha256=structural_hash,
        )

    export_factorial(source_root, release_root, provenance)
    export_frozen_summaries(source_root, release_root, provenance)
    write_json(
        release_root / "data/PROVENANCE.json",
        {
            "schema": RELEASE_SCHEMA,
            "release_boundary": (
                "Original source hashes are retained without private source paths. "
                "Release hashes bind the anonymous projections distributed here."
            ),
            "artifacts": provenance,
        },
    )
    print(f"Materialized {len(provenance)} anonymous artifacts in {release_root}")


if __name__ == "__main__":
    main()
