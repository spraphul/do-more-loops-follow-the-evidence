#!/usr/bin/env python3
"""Build prospective fictional on/off-path and adverse-locality falsifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "iclr2027.nonce_structural_falsifiers.panel.v1"
PARENT_FILE_SHA256 = "f176ce9f06b841feeb33d819536a3bf68a47aaa3bb11a46a9a7e9724af201a16"
PARENT_CANONICAL_SHA256 = "7880c02c64af541629b9c23dce15bf4a6df98c4f99a8aed45b4a310df2e543e5"
SYSTEM_PROMPT = (
    "Use only the supplied fictional directed graph. Follow the named relations "
    "exactly. Your entire answer must be one label: A or B."
)
DEVELOPMENT_WORLDS_PER_RELATION = 4
CONFIRM_REPLICATES_PER_STRATUM = 2


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


def load_parent(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PARENT_FILE_SHA256:
        raise RuntimeError("parent fictional panel file hash changed")
    panel = json.loads(path.read_text(encoding="utf-8"))
    stored = panel.pop("panel_sha256", None)
    observed = canonical_json_sha256(panel)
    panel["panel_sha256"] = stored
    if stored != PARENT_CANONICAL_SHA256 or observed != stored:
        raise RuntimeError("parent fictional panel canonical hash changed")
    return panel


def stable_rank(label: str) -> str:
    return hashlib.sha256(
        f"nonce-structural-falsifiers-v1:{label}".encode("utf-8")
    ).hexdigest()


def assign_splits(worlds: list[dict[str, Any]]) -> dict[str, str]:
    """Select two confirmatory replicates in every relation-by-code stratum."""
    strata: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for world in worlds:
        key = (
            int(world["relation_family_index"]),
            int(world["ordered_focal_code_pair_index"]),
        )
        strata[key].append(world)
    split: dict[str, str] = {}
    reserve: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key, members in sorted(strata.items()):
        ranked = sorted(members, key=lambda world: stable_rank(world["world_id"]))
        if len(ranked) != 4:
            raise RuntimeError(f"parent stratum {key} no longer contains four worlds")
        for world in ranked[:CONFIRM_REPLICATES_PER_STRATUM]:
            split[world["world_id"]] = "confirm"
        reserve[key[0]].extend(ranked[CONFIRM_REPLICATES_PER_STRATUM:])
    for relation, members in sorted(reserve.items()):
        ranked = sorted(members, key=lambda world: stable_rank(f"dev:{world['world_id']}"))
        for world in ranked[:DEVELOPMENT_WORLDS_PER_RELATION]:
            split[world["world_id"]] = "development"
    return split


def edit_pair(world: dict[str, Any]) -> tuple[int, int]:
    parity = (
        int(world["relation_family_index"])
        + int(world["ordered_focal_code_pair_index"])
        + int(world["replicate_within_relation_and_code_pair"])
    ) % 2
    return (0, 1) if parity == 0 else (2, 3)


def solve(edges: list[list[str]], source: str, relations: list[str]) -> str:
    mapping: dict[tuple[str, str], str] = {}
    for subject, relation, object_value in edges:
        key = (subject, relation)
        if key in mapping:
            raise RuntimeError("duplicate graph edge key")
        mapping[key] = object_value
    bridge = mapping[(source, relations[0])]
    return mapping[(bridge, relations[1])]


def label_fields(
    candidate_codes: list[str], candidate_order: str, reference_identity: int,
    expected_code: str,
) -> dict[str, Any]:
    identity_order = [0, 1] if candidate_order == "original" else [1, 0]
    labels = {
        identity: label
        for identity, label in zip(identity_order, ("A", "B"), strict=True)
    }
    expected_identity = candidate_codes.index(expected_code)
    reference_label = labels[reference_identity]
    return {
        "candidate_identity_order": identity_order,
        "candidate_codes_in_display_order": [candidate_codes[i] for i in identity_order],
        "reference_candidate_identity": reference_identity,
        "expected_candidate_identity": expected_identity,
        "reference_label": reference_label,
        "other_label": "B" if reference_label == "A" else "A",
        "expected_label": labels[expected_identity],
    }


def render_prompt(
    edges: list[list[str]], source: str, relations: list[str], displayed: list[str]
) -> str:
    records = "\n".join(
        f"SUBJECT = {subject} | RELATION = {relation} | OBJECT = {object_value}"
        for subject, relation, object_value in edges
    )
    return (
        "This is a fictional directed graph. Every identifier and record is invented. "
        "Use only the records below.\n\n"
        f"RECORDS\n{records}\n\n"
        "QUESTION\n"
        f"Starting from {source}, follow {relations[0]} once and then "
        f"{relations[1]} once. Which candidate terminal code is reached?\n\n"
        f"CANDIDATES\nA. {displayed[0]}\nB. {displayed[1]}\n\n"
        "Return exactly A or B and nothing else.\nANSWER ="
    )


def specificity_edges(
    world: dict[str, Any], transposed: bool, edited: tuple[int, int]
) -> list[list[str]]:
    first_targets = list(world["bridges"])
    if transposed:
        left, right = edited
        first_targets[left], first_targets[right] = (
            first_targets[right], first_targets[left]
        )
    first = [
        [world["sources"][i], world["relations"][0], first_targets[i]]
        for i in range(4)
    ]
    second = [
        [world["bridges"][i], world["relations"][1], world["terminal_codes"][i]]
        for i in range(4)
    ]
    return [
        *(first[i] for i in world["first_hop_physical_order"]),
        *(second[i] for i in world["second_hop_physical_order"]),
    ]


def build_specificity_cells(world: dict[str, Any], split: str) -> list[dict[str, Any]]:
    edited = edit_pair(world)
    disconnected = (2, 3) if edited == (0, 1) else (0, 1)
    output: list[dict[str, Any]] = []
    for edit_state, transposed in (("base", False), ("bridge_transposition", True)):
        edges = specificity_edges(world, transposed, edited)
        for query_role, pair in (("on_path", edited), ("off_path", disconnected)):
            candidate_codes = [world["terminal_codes"][i] for i in pair]
            for query_arm, path_index in enumerate(pair):
                expected_code = solve(
                    edges, world["sources"][path_index], world["relations"]
                )
                if expected_code not in candidate_codes:
                    raise RuntimeError("specificity query escaped its candidate pair")
                for candidate_order in ("original", "reversed"):
                    labels = label_fields(
                        candidate_codes, candidate_order, query_arm, expected_code
                    )
                    prompt = render_prompt(
                        edges,
                        world["sources"][path_index],
                        world["relations"],
                        labels["candidate_codes_in_display_order"],
                    )
                    row_id = (
                        f"{world['world_id']}__specificity__edit{edited[0]}{edited[1]}__"
                        f"{query_role}__q{query_arm}__{edit_state}__{candidate_order}"
                    )
                    output.append(
                        {
                            "row_id": row_id,
                            "world_id": world["world_id"],
                            "split": split,
                            "experiment": "path_specificity",
                            "relation_family_index": world["relation_family_index"],
                            "ordered_focal_code_pair_index": world[
                                "ordered_focal_code_pair_index"
                            ],
                            "relations": world["relations"],
                            "edit_pair_indices": list(edited),
                            "query_role": query_role,
                            "query_arm": query_arm,
                            "query_path_index": path_index,
                            "edit_state": edit_state,
                            "candidate_order": candidate_order,
                            "candidate_codes": candidate_codes,
                            "evidence_triples": edges,
                            "system_prompt": SYSTEM_PROMPT,
                            "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            **labels,
                        }
                    )
    return output


def locality_edges(
    world: dict[str, Any], pair: tuple[int, int], binding: str, locality: str
) -> tuple[list[list[str]], dict[int, str]]:
    left, right = pair
    first_targets = list(world["bridges"])
    first_targets[left], first_targets[right] = first_targets[right], first_targets[left]
    second_targets = list(world["terminal_codes"])
    if binding == "transposed":
        second_targets[left], second_targets[right] = (
            second_targets[right], second_targets[left]
        )
    elif binding != "normal":
        raise ValueError(binding)

    first = {
        i: [world["sources"][i], world["relations"][0], first_targets[i]]
        for i in range(4)
    }
    second = {
        i: [world["bridges"][i], world["relations"][1], second_targets[i]]
        for i in range(4)
    }
    arm_order = [left, right]
    if int(world["world_index"]) % 2:
        arm_order.reverse()
    focal: list[list[str]] = []
    adjacent: dict[int, str] = {}
    for path_index in arm_order:
        own_target_index = right if path_index == left else left
        adjacent_index = (
            own_target_index
            if locality == "matched"
            else (left if own_target_index == right else right)
        )
        focal.extend((first[path_index], second[adjacent_index]))
        adjacent[path_index] = second_targets[adjacent_index]
    if locality not in {"matched", "crossed"}:
        raise ValueError(locality)

    distractor_indices = [i for i in range(4) if i not in pair]
    if (int(world["world_index"]) // 2) % 2:
        distractor_indices.reverse()
    distractors = [edge for i in distractor_indices for edge in (first[i], second[i])]
    edges = [*focal, *distractors]
    if (int(world["world_index"]) // 4) % 2:
        edges = [*distractors, *focal]
    return edges, adjacent


def build_locality_cells(world: dict[str, Any], split: str) -> list[dict[str, Any]]:
    pair = edit_pair(world)
    candidate_codes = [world["terminal_codes"][i] for i in pair]
    output: list[dict[str, Any]] = []
    for binding in ("normal", "transposed"):
        for locality in ("matched", "crossed"):
            edges, adjacent = locality_edges(world, pair, binding, locality)
            for query_arm, path_index in enumerate(pair):
                expected_code = solve(
                    edges, world["sources"][path_index], world["relations"]
                )
                if expected_code not in candidate_codes:
                    raise RuntimeError("locality query escaped its candidate pair")
                if locality == "matched" and adjacent[path_index] != expected_code:
                    raise RuntimeError("matched locality does not predict the graph answer")
                if locality == "crossed" and adjacent[path_index] == expected_code:
                    raise RuntimeError("crossed locality does not oppose the graph answer")
                for candidate_order in ("original", "reversed"):
                    labels = label_fields(
                        candidate_codes, candidate_order, query_arm, expected_code
                    )
                    prompt = render_prompt(
                        edges,
                        world["sources"][path_index],
                        world["relations"],
                        labels["candidate_codes_in_display_order"],
                    )
                    row_id = (
                        f"{world['world_id']}__locality__pair{pair[0]}{pair[1]}__"
                        f"{locality}__{binding}__q{query_arm}__{candidate_order}"
                    )
                    output.append(
                        {
                            "row_id": row_id,
                            "world_id": world["world_id"],
                            "split": split,
                            "experiment": "adverse_locality",
                            "relation_family_index": world["relation_family_index"],
                            "ordered_focal_code_pair_index": world[
                                "ordered_focal_code_pair_index"
                            ],
                            "relations": world["relations"],
                            "focal_pair_indices": list(pair),
                            "query_arm": query_arm,
                            "query_path_index": path_index,
                            "terminal_binding": binding,
                            "physical_locality": locality,
                            "adjacent_terminal_code": adjacent[path_index],
                            "candidate_order": candidate_order,
                            "candidate_codes": candidate_codes,
                            "evidence_triples": edges,
                            "system_prompt": SYSTEM_PROMPT,
                            "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            **labels,
                        }
                    )
    return output


def audit(
    worlds: list[dict[str, Any]], cells: list[dict[str, Any]], split_map: dict[str, str]
) -> dict[str, Any]:
    split_counts = Counter(split_map.values())
    relation_split = Counter(
        (split_map[world["world_id"]], int(world["relation_family_index"]))
        for world in worlds
        if world["world_id"] in split_map
    )
    confirm_strata = Counter(
        (
            int(world["relation_family_index"]),
            int(world["ordered_focal_code_pair_index"]),
        )
        for world in worlds
        if split_map.get(world["world_id"]) == "confirm"
    )
    selected_worlds = set(split_map)
    by_world_experiment = Counter((cell["world_id"], cell["experiment"]) for cell in cells)
    if split_counts != {"confirm": 144, "development": 24}:
        raise RuntimeError(f"unexpected split counts: {split_counts}")
    if set(confirm_strata.values()) != {2} or len(confirm_strata) != 72:
        raise RuntimeError("confirmatory relation-by-code strata are not exactly balanced")
    if any(by_world_experiment[(world, experiment)] != 16 for world in selected_worlds for experiment in ("path_specificity", "adverse_locality")):
        raise RuntimeError("new factorial is incomplete")
    if len(cells) != len(selected_worlds) * 32:
        raise RuntimeError("unexpected panel size")
    if len({cell["row_id"] for cell in cells}) != len(cells):
        raise RuntimeError("duplicate panel row IDs")

    specificity: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for cell in cells:
        if cell["experiment"] != "path_specificity":
            continue
        key = (cell["world_id"], cell["edit_state"])
        specificity[key][cell["query_role"]] = canonical_json_sha256(
            cell["evidence_triples"]
        )
    if any(len(set(role_hashes.values())) != 1 for role_hashes in specificity.values()):
        raise RuntimeError("on/off-path prompts do not share the exact evidence edit")

    return {
        "selected_worlds": len(selected_worlds),
        "split_world_counts": dict(sorted(split_counts.items())),
        "relation_family_counts_by_split": {
            f"{split}__relation{relation}": count
            for (split, relation), count in sorted(relation_split.items())
        },
        "confirmatory_worlds_per_relation_by_ordered_code_pair": 2,
        "prompt_cells": len(cells),
        "prompt_cells_per_world": 32,
        "path_specificity_cells_per_world": 16,
        "adverse_locality_cells_per_world": 16,
        "same_exact_evidence_edit_across_on_and_off_path_queries": True,
        "independent_graph_solver_passed_all_cells": True,
        "matched_adjacency_agrees_with_graph_all_cells": True,
        "crossed_adjacency_opposes_graph_all_cells": True,
        "development_worlds_excluded_from_confirmation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = load_parent(args.parent_panel.resolve())
    split_map = assign_splits(parent["worlds"])
    selected_worlds = [
        world for world in parent["worlds"] if world["world_id"] in split_map
    ]
    cells: list[dict[str, Any]] = []
    for world in selected_worlds:
        split = split_map[world["world_id"]]
        cells.extend(build_specificity_cells(world, split))
        cells.extend(build_locality_cells(world, split))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_before_new_intervention_inference",
        "parent_panel": {
            "path": "data/generated/nonce_path_control_v1.panel.json",
            "file_sha256": PARENT_FILE_SHA256,
            "canonical_sha256": PARENT_CANONICAL_SHA256,
        },
        "system_prompt": SYSTEM_PROMPT,
        "split_contract": {
            "development": (
                "24 hash-selected reserve worlds, four per relation family; "
                "excluded from confirmatory inference"
            ),
            "confirm": (
                "144 worlds, exactly two of four replicates in every one of the "
                "six-relation by twelve-ordered-terminal-pair strata"
            ),
            "unused_holdout_worlds": 120,
        },
        "primary_estimands": {
            "path_specificity": (
                "[base minus transposition reference margin] on-path minus off-path"
            ),
            "adverse_locality": (
                "transposed-binding minus normal-binding reference margin, "
                "evaluated under crossed physical locality"
            ),
        },
        "worlds": selected_worlds,
        "cells": cells,
    }
    payload["audit"] = audit(parent["worlds"], cells, split_map)
    payload["panel_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    print(sha256_file(args.output))
    print(payload["panel_sha256"])
    print(json.dumps(payload["audit"], indent=2))


if __name__ == "__main__":
    main()
