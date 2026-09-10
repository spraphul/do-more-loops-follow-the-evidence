#!/usr/bin/env python3
"""Rebuild the frozen four-rendering surface-orbit confirmation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


PANEL_SCHEMA = "iclr2027.ouro26_surface_orbit_confirmation.panel.v3"
SEED = 20260912
GATE_WORLDS = 24
CONFIRMATION_WORLDS = 48
SURFACES_PER_WORLD = 4
SERIALIZATIONS = ("arrows", "tuples", "json", "sentences")
STATES = ("original", "bridge_swap")
QUERY_ARMS = ("A", "B")
CANDIDATE_ORDERS = ("original", "reversed")
SYSTEM_PROMPT = (
    "Use only the supplied fictional directed graph. Follow the named relations "
    "exactly. Your entire answer must be one label: A or B."
)
QUESTION_WORDINGS = (
    "Start at {source}. Follow {r1}, then {r2}. Which listed endpoint do you reach?",
    "From {source}, take relation {r1} once, followed by relation {r2} once. Select the endpoint.",
    (
        "Follow the two-edge path from {source}, first using {r1} and then {r2}. "
        "Which option is reached?"
    ),
    "Beginning at {source}, traverse {r1} and next {r2}. Choose the resulting endpoint.",
)
EVIDENCE_ORDERS = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (7, 6, 5, 4, 3, 2, 1, 0),
    (2, 3, 4, 5, 6, 7, 0, 1),
    (4, 5, 6, 7, 0, 1, 2, 3),
    (0, 4, 1, 5, 2, 6, 3, 7),
    (4, 0, 5, 1, 6, 2, 7, 3),
    (3, 7, 2, 6, 1, 5, 0, 4),
    (7, 3, 6, 2, 5, 1, 4, 0),
)
RESERVED_ROLE_SUBSTRINGS = (
    "source",
    "src",
    "bridge",
    "mid",
    "terminal",
    "end",
    "relation",
    "rel",
    "answer",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(
        f"surface-orbit-confirmation-v3:{SEED}:{label}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def nonce_word(rng: random.Random, syllables: int = 3) -> str:
    onsets = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z")
    vowels = ("a", "e", "i", "o", "u")
    return "".join(rng.choice(onsets) + rng.choice(vowels) for _ in range(syllables))


def unique_symbols(
    rng: random.Random,
    count: int,
    morphology: int,
    prefix: str,
    globally_used: set[str],
) -> list[str]:
    if prefix != "TOK":
        raise RuntimeError("v3 requires a neutral symbol prefix")
    output: list[str] = []
    while len(output) < count:
        nonce = nonce_word(rng).upper()
        if morphology == 0:
            candidate = f"TOK_{nonce}"
        elif morphology == 1:
            candidate = f"tok_{nonce.lower()}"
        elif morphology == 2:
            candidate = f"Tok{nonce.title()}"
        elif morphology == 3:
            candidate = f"TOK{nonce}"
        else:
            raise ValueError(morphology)
        lowered = candidate.lower()
        if candidate in globally_used or any(
            term in lowered for term in RESERVED_ROLE_SUBSTRINGS
        ):
            continue
        globally_used.add(candidate)
        output.append(candidate)
    return output


def build_surface_world(
    split: str,
    world_index: int,
    surface_index: int,
    globally_used: set[str],
) -> dict[str, Any]:
    if split not in {"gate", "confirmation"}:
        raise ValueError(split)
    prefix = "SORCG" if split == "gate" else "SORCT"
    world_id = f"{prefix}{world_index + 1:04d}"
    rng = random.Random(stable_seed(f"{world_id}:surface:{surface_index}"))
    if split == "confirmation":
        design_block = world_index % 16
        morphology_offset = design_block // 4
        wording_offset = design_block % 4
    else:
        design_block = None
        morphology_offset = world_index % 4
        wording_offset = (world_index + world_index // 4) % 4
    morphology = (surface_index + morphology_offset) % 4
    wording_index = (surface_index + wording_offset) % 4
    order_index = (world_index + 2 * surface_index) % len(EVIDENCE_ORDERS)
    symbols = unique_symbols(rng, 14, morphology, "TOK", globally_used)
    rng.shuffle(symbols)
    sources = symbols[:4]
    bridges = symbols[4:8]
    terminals = symbols[8:12]
    relations = symbols[12:14]
    symbols = [*sources, *bridges, *terminals, *relations]
    if len(symbols) != len(set(symbols)):
        raise RuntimeError("surface symbols collide within a realization")
    return {
        "surface_world_id": f"{world_id}__surface{surface_index + 1}",
        "world_id": world_id,
        "split": split,
        "world_index": world_index,
        "design_block": design_block,
        "morphology_offset": morphology_offset,
        "wording_offset": wording_offset,
        "surface_index": surface_index,
        "serialization": SERIALIZATIONS[surface_index],
        "morphology_index": morphology,
        "wording_index": wording_index,
        "order_index": order_index,
        "sources": sources,
        "bridges": bridges,
        "terminal_codes": terminals,
        "relations": relations,
        "record_order": list(EVIDENCE_ORDERS[order_index]),
    }


def state_edges(world: dict[str, Any], state: str) -> list[list[str]]:
    if state not in STATES:
        raise ValueError(state)
    first_targets = list(world["bridges"])
    if state == "bridge_swap":
        first_targets[0], first_targets[1] = first_targets[1], first_targets[0]
    r1, r2 = world["relations"]
    edges = [
        *[[world["sources"][index], r1, first_targets[index]] for index in range(4)],
        *[
            [world["bridges"][index], r2, world["terminal_codes"][index]]
            for index in range(4)
        ],
    ]
    return [edges[index] for index in world["record_order"]]


def solve_identity(world: dict[str, Any], state: str, query_arm: str) -> int:
    r1, r2 = world["relations"]
    lookup = {
        (source, relation): target
        for source, relation, target in state_edges(world, state)
    }
    source_index = 0 if query_arm == "A" else 1
    bridge = lookup[(world["sources"][source_index], r1)]
    terminal = lookup[(bridge, r2)]
    return world["terminal_codes"][:2].index(terminal)


def serialize_records(world: dict[str, Any], edges: list[list[str]]) -> str:
    style = world["serialization"]
    if style == "arrows":
        return "\n".join(f"{source} --[{relation}]--> {target}" for source, relation, target in edges)
    if style == "tuples":
        return "\n".join(
            f"(source: {source}; relation: {relation}; target: {target})"
            for source, relation, target in edges
        )
    if style == "json":
        return "\n".join(
            json.dumps(
                {"source": source, "relation": relation, "target": target},
                separators=(",", ":"),
            )
            for source, relation, target in edges
        )
    if style == "sentences":
        return "\n".join(
            f"Relation {relation} maps {source} to {target}."
            for source, relation, target in edges
        )
    raise ValueError(style)


def render_prompt(
    world: dict[str, Any], state: str, query_arm: str, candidate_order: str
) -> tuple[str, list[list[str]], list[int]]:
    edges = state_edges(world, state)
    records = serialize_records(world, edges)
    source_index = 0 if query_arm == "A" else 1
    r1, r2 = world["relations"]
    source = world["sources"][source_index]
    question = QUESTION_WORDINGS[int(world["wording_index"])].format(
        source=source, r1=r1, r2=r2
    )
    identity_order = [0, 1] if candidate_order == "original" else [1, 0]
    candidates = [world["terminal_codes"][identity] for identity in identity_order]
    prompt = (
        "All symbols below belong to a newly invented directed graph. Use no outside facts.\n\n"
        f"GRAPH RECORDS\n{records}\n\n"
        f"QUERY\n{question}\n\n"
        f"OPTIONS\nA. {candidates[0]}\nB. {candidates[1]}\n\n"
        "Return exactly A or B and nothing else.\nANSWER ="
    )
    return prompt, edges, identity_order


def build_panel() -> dict[str, Any]:
    globally_used: set[str] = set()
    gate_surfaces = [
        build_surface_world("gate", world_index, surface_index, globally_used)
        for world_index in range(GATE_WORLDS)
        for surface_index in range(SURFACES_PER_WORLD)
    ]
    confirmation_surfaces = [
        build_surface_world("confirmation", world_index, surface_index, globally_used)
        for world_index in range(CONFIRMATION_WORLDS)
        for surface_index in range(SURFACES_PER_WORLD)
    ]
    surface_worlds = [*gate_surfaces, *confirmation_surfaces]
    all_symbols = [
        symbol
        for world in surface_worlds
        for symbol in (
            *world["sources"],
            *world["bridges"],
            *world["terminal_codes"],
            *world["relations"],
        )
    ]
    if len(all_symbols) != len(set(all_symbols)):
        raise RuntimeError("visible symbols repeat across confirmation surfaces")

    cells: list[dict[str, Any]] = []
    for world in surface_worlds:
        states = ("original",) if world["split"] == "gate" else STATES
        for state in states:
            for query_arm in QUERY_ARMS:
                reference_identity = 0 if query_arm == "A" else 1
                expected_identity = solve_identity(world, state, query_arm)
                designed = (
                    1 - reference_identity
                    if state == "bridge_swap"
                    else reference_identity
                )
                if expected_identity != designed:
                    raise RuntimeError("symbolic solver disagrees with confirmation graph")
                for candidate_order in CANDIDATE_ORDERS:
                    prompt, edges, identity_order = render_prompt(
                        world, state, query_arm, candidate_order
                    )
                    labels = dict(zip(identity_order, ("A", "B"), strict=True))
                    reference_label = labels[reference_identity]
                    cells.append(
                        {
                            "row_id": (
                                f"{world['surface_world_id']}__q{query_arm}__{state}__"
                                f"candidates_{candidate_order}"
                            ),
                            "surface_world_id": world["surface_world_id"],
                            "world_id": world["world_id"],
                            "split": world["split"],
                            "world_index": world["world_index"],
                            "design_block": world["design_block"],
                            "morphology_offset": world["morphology_offset"],
                            "wording_offset": world["wording_offset"],
                            "surface_index": world["surface_index"],
                            "serialization": world["serialization"],
                            "morphology_index": world["morphology_index"],
                            "wording_index": world["wording_index"],
                            "order_index": world["order_index"],
                            "evidence_state": state,
                            "query_arm": query_arm,
                            "candidate_order": candidate_order,
                            "focal_terminal_codes": world["terminal_codes"][:2],
                            "reference_candidate_identity": reference_identity,
                            "expected_candidate_identity": expected_identity,
                            "reference_label": reference_label,
                            "other_label": "B" if reference_label == "A" else "A",
                            "expected_label": labels[expected_identity],
                            "evidence_triples": edges,
                            "system_prompt": SYSTEM_PROMPT,
                            "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        }
                    )

    expected = (
        GATE_WORLDS * SURFACES_PER_WORLD * 2 * 2
        + CONFIRMATION_WORLDS * SURFACES_PER_WORLD * 2 * 2 * 2
    )
    if len(cells) != expected or len({cell["row_id"] for cell in cells}) != expected:
        raise RuntimeError("confirmation cell grid is incomplete")

    morphology_balance = {
        style: {
            str(morphology): sum(
                world["serialization"] == style
                and int(world["morphology_index"]) == morphology
                for world in confirmation_surfaces
            )
            for morphology in range(4)
        }
        for style in SERIALIZATIONS
    }
    if {count for counts in morphology_balance.values() for count in counts.values()} != {12}:
        raise RuntimeError("serialization and morphology are not balanced")
    wording_balance = {
        style: {
            str(wording): sum(
                world["serialization"] == style
                and int(world["wording_index"]) == wording
                for world in confirmation_surfaces
            )
            for wording in range(4)
        }
        for style in SERIALIZATIONS
    }
    if {count for counts in wording_balance.values() for count in counts.values()} != {12}:
        raise RuntimeError("serialization and question wording are not balanced")
    order_balance = {
        style: {
            str(order): sum(
                world["serialization"] == style
                and int(world["order_index"]) == order
                for world in confirmation_surfaces
            )
            for order in range(8)
        }
        for style in SERIALIZATIONS
    }
    if {count for counts in order_balance.values() for count in counts.values()} != {6}:
        raise RuntimeError("serialization and evidence order are not balanced")
    design_block_counts = {
        str(block): len(
            {
                world["world_id"]
                for world in confirmation_surfaces
                if int(world["design_block"]) == block
            }
        )
        for block in range(16)
    }
    if set(design_block_counts.values()) != {3}:
        raise RuntimeError("confirmation design blocks are not balanced")

    payload: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "status": "frozen_before_model_inference",
        "seed": SEED,
        "system_prompt": SYSTEM_PROMPT,
        "dimensions": {
            "worlds": GATE_WORLDS + CONFIRMATION_WORLDS,
            "gate_worlds": GATE_WORLDS,
            "confirmation_worlds": CONFIRMATION_WORLDS,
            "surface_realizations_per_world": SURFACES_PER_WORLD,
            "prompt_cells": len(cells),
            "globally_unique_visible_symbols": len(all_symbols),
        },
        "confirmation_morphology_by_serialization": morphology_balance,
        "confirmation_wording_by_serialization": wording_balance,
        "confirmation_order_by_serialization": order_balance,
        "confirmation_design_block_counts": design_block_counts,
        "gate_world_ids": sorted({world["world_id"] for world in gate_surfaces}),
        "confirmation_world_ids": sorted(
            {world["world_id"] for world in confirmation_surfaces}
        ),
        "surface_worlds": surface_worlds,
        "cells": cells,
    }
    payload["panel_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_panel()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
