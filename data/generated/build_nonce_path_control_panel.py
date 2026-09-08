#!/usr/bin/env python3
"""Build the frozen truth-balanced fictional graph panel.

This script performs no model inference. It creates a deterministic panel in
which every graph fact is invented, every entity occurs in one world only, and
the same two focal terminal codes exchange graph support under a bridge swap.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "iclr2027.nonce_path_control.panel.v1"
SEED = 20260906
WORLDS = 288
WORLD_REPLICATES_PER_RELATION_AND_ORDERED_CODE_PAIR = 4
ANSWER_CODES = ("red", "blue", "green", "black")
QUERY_ARMS = ("A", "B")
GRAPH_STATES = ("original", "bridge_swap", "topology_repair")
CANDIDATE_ORDERS = ("original", "reversed")
RELATION_FAMILIES = (
    ("ROUTES_TO", "ENDS_AT"),
    ("LINKS_TO", "RETURNS"),
    ("POINTS_TO", "YIELDS"),
    ("CONNECTS_TO", "RESOLVES_TO"),
    ("FIRST_HOP", "SECOND_HOP"),
    ("MAPS_TO", "TERMINATES_AT"),
)
SYSTEM_PROMPT = (
    "Use only the supplied fictional directed graph. Follow the named relations "
    "exactly. Your entire answer must be one label: A or B."
)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def opaque_identifier(rng: random.Random, prefix: str, used: set[str]) -> str:
    while True:
        suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(6))
        value = f"{prefix}_{suffix}"
        if value not in used:
            used.add(value)
            return value


def physical_orders(index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    base = (0, 1, 2, 3)
    rotation = index % 4
    first = base[rotation:] + base[:rotation]
    if (index // 4) % 2:
        first = tuple(reversed(first))
    second_rotation = (rotation + 2) % 4
    second = base[second_rotation:] + base[:second_rotation]
    if (index // 8) % 2:
        second = tuple(reversed(second))
    return tuple(first), tuple(second)


def state_edges(world: dict[str, Any], state: str) -> list[list[str]]:
    relation_1, relation_2 = world["relations"]
    sources = world["sources"]
    bridges = world["bridges"]
    codes = world["terminal_codes"]
    first_targets = list(bridges)
    second_targets = list(codes)
    if state in {"bridge_swap", "topology_repair"}:
        first_targets[0], first_targets[1] = first_targets[1], first_targets[0]
    if state == "topology_repair":
        second_targets[0], second_targets[1] = second_targets[1], second_targets[0]
    if state not in GRAPH_STATES:
        raise ValueError(state)
    first_edges = [
        [sources[index], relation_1, first_targets[index]] for index in range(4)
    ]
    second_edges = [
        [bridges[index], relation_2, second_targets[index]] for index in range(4)
    ]
    return [
        *(first_edges[index] for index in world["first_hop_physical_order"]),
        *(second_edges[index] for index in world["second_hop_physical_order"]),
    ]


def solve_identity(world: dict[str, Any], state: str, query_arm: str) -> int:
    relation_1, relation_2 = world["relations"]
    edges: dict[tuple[str, str], str] = {}
    for subject, relation, object_value in state_edges(world, state):
        key = (subject, relation)
        if key in edges:
            raise RuntimeError("duplicate graph edge key")
        edges[key] = object_value
    source_index = 0 if query_arm == "A" else 1
    bridge = edges[(world["sources"][source_index], relation_1)]
    terminal = edges[(bridge, relation_2)]
    focal_codes = world["terminal_codes"][:2]
    if terminal not in focal_codes:
        raise RuntimeError("focal query solved to a distractor terminal")
    return focal_codes.index(terminal)


def render_prompt(
    world: dict[str, Any], state: str, query_arm: str, candidate_order: str
) -> tuple[str, list[list[str]], list[int]]:
    edges = state_edges(world, state)
    records = "\n".join(
        f"SUBJECT = {subject} | RELATION = {relation} | OBJECT = {object_value}"
        for subject, relation, object_value in edges
    )
    relation_1, relation_2 = world["relations"]
    source_index = 0 if query_arm == "A" else 1
    identity_order = [0, 1] if candidate_order == "original" else [1, 0]
    candidates = [world["terminal_codes"][identity] for identity in identity_order]
    prompt = (
        "This is a fictional directed graph. Every identifier and record is invented. "
        "Use only the records below.\n\n"
        f"RECORDS\n{records}\n\n"
        "QUESTION\n"
        f"Starting from {world['sources'][source_index]}, follow {relation_1} once and "
        f"then {relation_2} once. Which candidate terminal code is reached?\n\n"
        f"CANDIDATES\nA. {candidates[0]}\nB. {candidates[1]}\n\n"
        "Return exactly A or B and nothing else.\nANSWER ="
    )
    return prompt, edges, identity_order


def build_worlds() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    used: set[str] = set()
    worlds: list[dict[str, Any]] = []
    ordered_code_pairs = list(itertools.permutations(ANSWER_CODES, 2))
    for relation_index, relations in enumerate(RELATION_FAMILIES):
        for code_pair_index, focal_codes in enumerate(ordered_code_pairs):
            distractors = [code for code in ANSWER_CODES if code not in focal_codes]
            for replicate in range(
                WORLD_REPLICATES_PER_RELATION_AND_ORDERED_CODE_PAIR
            ):
                world_index = len(worlds)
                first_order, second_order = physical_orders(world_index)
                if (relation_index + code_pair_index + replicate) % 2:
                    distractors.reverse()
                sources = [opaque_identifier(rng, "SRC", used) for _ in range(4)]
                bridges = [opaque_identifier(rng, "MID", used) for _ in range(4)]
                worlds.append(
                    {
                        "world_id": f"NPC{world_index + 1:04d}",
                        "world_index": world_index,
                        "relation_family_index": relation_index,
                        "relations": list(relations),
                        "ordered_focal_code_pair_index": code_pair_index,
                        "replicate_within_relation_and_code_pair": replicate,
                        "sources": sources,
                        "bridges": bridges,
                        "terminal_codes": [*focal_codes, *distractors],
                        "first_hop_physical_order": list(first_order),
                        "second_hop_physical_order": list(second_order),
                    }
                )
    if len(worlds) != WORLDS:
        raise RuntimeError(f"world count changed: {len(worlds)}")
    return worlds


def build_cells(worlds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for world in worlds:
        for state in GRAPH_STATES:
            for query_arm in QUERY_ARMS:
                reference_identity = 0 if query_arm == "A" else 1
                expected_identity = solve_identity(world, state, query_arm)
                expected_from_design = (
                    1 - reference_identity if state == "bridge_swap" else reference_identity
                )
                if expected_identity != expected_from_design:
                    raise RuntimeError("independent graph solver disagrees with design")
                for candidate_order in CANDIDATE_ORDERS:
                    prompt, edges, identity_order = render_prompt(
                        world, state, query_arm, candidate_order
                    )
                    label_for_identity = {
                        identity: label
                        for identity, label in zip(identity_order, ("A", "B"), strict=True)
                    }
                    reference_label = label_for_identity[reference_identity]
                    other_label = "B" if reference_label == "A" else "A"
                    expected_label = label_for_identity[expected_identity]
                    row_id = (
                        f"{world['world_id']}__q{query_arm}__{state}__"
                        f"candidates_{candidate_order}"
                    )
                    cells.append(
                        {
                            "row_id": row_id,
                            "world_id": world["world_id"],
                            "relation_family_index": world["relation_family_index"],
                            "relations": world["relations"],
                            "evidence_state": state,
                            "query_arm": query_arm,
                            "candidate_order": candidate_order,
                            "candidate_identity_order": identity_order,
                            "focal_terminal_codes": world["terminal_codes"][:2],
                            "reference_candidate_identity": reference_identity,
                            "expected_candidate_identity": expected_identity,
                            "reference_label": reference_label,
                            "other_label": other_label,
                            "expected_label": expected_label,
                            "evidence_triples": edges,
                            "system_prompt": SYSTEM_PROMPT,
                            "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
    expected = WORLDS * len(GRAPH_STATES) * len(QUERY_ARMS) * len(CANDIDATE_ORDERS)
    if len(cells) != expected or len({cell["row_id"] for cell in cells}) != expected:
        raise RuntimeError("panel cell grid is incomplete or non-unique")
    return cells


def competence_world_ids(worlds: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for relation_index in range(len(RELATION_FAMILIES)):
        family = [
            world for world in worlds if world["relation_family_index"] == relation_index
        ]
        family.sort(
            key=lambda world: hashlib.sha256(
                f"nonce-path-control-v1:competence:{world['world_id']}".encode()
            ).hexdigest()
        )
        selected.extend(world["world_id"] for world in family[:8])
    return sorted(selected)


def audit(worlds: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    code_pair_counts = Counter(tuple(world["terminal_codes"][:2]) for world in worlds)
    relation_counts = Counter(world["relation_family_index"] for world in worlds)
    code_by_focal_slot = {
        str(slot): Counter(world["terminal_codes"][slot] for world in worlds)
        for slot in (0, 1)
    }
    state_expected_counts = Counter(
        (cell["evidence_state"], cell["expected_candidate_identity"])
        for cell in cells
    )
    all_entities = [
        entity
        for world in worlds
        for entity in (*world["sources"], *world["bridges"])
    ]
    if len(all_entities) != len(set(all_entities)):
        raise RuntimeError("nonce entity repeats across worlds")
    expected_pair_count = WORLD_REPLICATES_PER_RELATION_AND_ORDERED_CODE_PAIR * len(
        RELATION_FAMILIES
    )
    if set(code_pair_counts.values()) != {expected_pair_count}:
        raise RuntimeError("ordered focal code pairs are imbalanced")
    if set(relation_counts.values()) != {WORLDS // len(RELATION_FAMILIES)}:
        raise RuntimeError("relation families are imbalanced")
    if any(set(counter.values()) != {WORLDS // len(ANSWER_CODES)} for counter in code_by_focal_slot.values()):
        raise RuntimeError("answer codes are imbalanced across focal roles")
    return {
        "worlds": len(worlds),
        "prompt_cells": len(cells),
        "unique_nonce_entities": len(all_entities),
        "entity_reuse_across_worlds": False,
        "ordered_focal_code_pair_counts": {
            "__".join(pair): count for pair, count in sorted(code_pair_counts.items())
        },
        "relation_family_world_counts": {
            str(key): value for key, value in sorted(relation_counts.items())
        },
        "focal_code_counts_by_slot": {
            slot: dict(sorted(counter.items()))
            for slot, counter in code_by_focal_slot.items()
        },
        "state_expected_identity_counts": {
            f"{state}__identity{identity}": count
            for (state, identity), count in sorted(state_expected_counts.items())
        },
        "all_states_share_entity_relation_and_terminal_inventory_within_world": True,
        "both_query_arms_and_candidate_orders_crossed_within_world": True,
        "independent_graph_solver_passed_all_cells": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "nonce_path_control_v1.panel.json",
    )
    args = parser.parse_args()
    worlds = build_worlds()
    cells = build_cells(worlds)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_before_model_inference",
        "seed": SEED,
        "system_prompt": SYSTEM_PROMPT,
        "answer_codes": list(ANSWER_CODES),
        "relation_families": [list(pair) for pair in RELATION_FAMILIES],
        "competence_world_ids": competence_world_ids(worlds),
        "audit": audit(worlds, cells),
        "worlds": worlds,
        "cells": cells,
    }
    payload["panel_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(payload["panel_sha256"])
    print(json.dumps(payload["audit"], indent=2))


if __name__ == "__main__":
    main()
