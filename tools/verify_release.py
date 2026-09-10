#!/usr/bin/env python3
"""Validate checksums, panel bindings, released grids, and anonymity."""

from __future__ import annotations

import hashlib
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("panel_sha256", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest() -> int:
    path = ROOT / "MANIFEST.sha256"
    if not path.exists():
        raise RuntimeError("MANIFEST.sha256 is missing")
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64:
            raise RuntimeError(f"malformed manifest line {line_number}")
        target = ROOT / relative
        if not target.is_file():
            raise RuntimeError(f"manifest target missing: {relative}")
        observed = sha256(target)
        if observed != expected:
            raise RuntimeError(f"manifest mismatch: {relative}")
        count += 1
    return count


def verify_panels() -> None:
    parent = json.loads(
        (ROOT / "data/generated/nonce_path_control_v1.panel.json").read_text(
            encoding="utf-8"
        )
    )
    structural = json.loads(
        (ROOT / "data/generated/nonce_structural_falsifiers_v1.panel.json").read_text(
            encoding="utf-8"
        )
    )
    surface = json.loads(
        (ROOT / "data/generated/surface_orbit_confirmation_v3.panel.json").read_text(
            encoding="utf-8"
        )
    )
    hrm_branching = json.loads(
        (ROOT / "data/generated/hrm_branching_confirmation_v1.panel.json").read_text(
            encoding="utf-8"
        )
    )
    for label, payload in (("fictional", parent), ("structural", structural)):
        if payload.get("panel_sha256") != canonical_hash(payload):
            raise RuntimeError(f"{label} panel canonical hash mismatch")
    if len(parent.get("worlds", [])) != 288 or len(parent.get("cells", [])) != 3456:
        raise RuntimeError("fictional panel dimensions changed")
    if len(structural.get("worlds", [])) != 168:
        raise RuntimeError("structural panel world pool changed")
    if structural.get("parent_panel", {}).get("canonical_sha256") != parent["panel_sha256"]:
        raise RuntimeError("structural parent binding changed")
    if surface.get("panel_sha256") != canonical_hash(surface):
        raise RuntimeError("surface-orbit panel canonical hash mismatch")
    if surface.get("dimensions") != {
        "worlds": 72,
        "gate_worlds": 24,
        "confirmation_worlds": 48,
        "surface_realizations_per_world": 4,
        "prompt_cells": 1920,
        "globally_unique_visible_symbols": 4032,
    }:
        raise RuntimeError("surface-orbit panel dimensions changed")
    visible_symbols = [
        symbol
        for world in surface.get("surface_worlds", [])
        for symbol in (
            *world["sources"],
            *world["bridges"],
            *world["terminal_codes"],
            *world["relations"],
        )
    ]
    if len(visible_symbols) != 4032 or len(set(visible_symbols)) != 4032:
        raise RuntimeError("surface-orbit visible-symbol uniqueness changed")
    if (
        hrm_branching.get("panel_sha256") != canonical_hash(hrm_branching)
        or hrm_branching.get("panel_sha256")
        != "e76bf583223dba4f6b46e970b592009b6b7f29bd9db0e30cd8b97c3217314f31"
        or len(hrm_branching.get("worlds", [])) != 192
        or len(hrm_branching.get("cells", [])) != 4_608
        or Counter(world["split"] for world in hrm_branching["worlds"])
        != Counter({"gate": 48, "confirmation": 144})
    ):
        raise RuntimeError("HRM branching panel binding or split changed")


def verify_natural() -> None:
    expected = {
        "ouro26_2wiki.json": (5952, 124),
        "ouro26_musique.json": (4560, 95),
        "loopus_2wiki.json": (1860, 93),
        "loopus_musique.json": (1900, 95),
        "loopus_full_depth_2wiki.json": (6324, 93),
        "loopus_full_depth_musique.json": (6460, 95),
    }
    for filename, (row_count, pair_count) in expected.items():
        payload = json.loads(
            (ROOT / "data/analysis_ready/natural" / filename).read_text(encoding="utf-8")
        )
        rows = payload.get("rows", [])
        if len(rows) != row_count or payload.get("row_count") != row_count:
            raise RuntimeError(f"{filename}: row count changed")
        pairs = {row["pair_id"] for row in rows}
        if len(pairs) != pair_count or payload.get("pair_count") != pair_count:
            raise RuntimeError(f"{filename}: pair count changed")
        row_ids = [row["row_id"] for row in rows]
        if len(row_ids) != len(set(row_ids)):
            raise RuntimeError(f"{filename}: duplicate row IDs")


def verify_model_rows() -> None:
    expected = {
        "fictional_ouro": {"competence.json": 192, **{f"shard_{i}.json": 1728 for i in range(8)}},
        "fictional_loopus8": {"competence.json": 192, **{f"shard_{i}.json": 2592 for i in range(4)}},
        "structural": {
            "shard_0.json": 4992,
            "shard_1.json": 3840,
            "shard_2.json": 4736,
            "shard_3.json": 4864,
        },
    }
    for directory, files in expected.items():
        treatment_identifiers: list[str] = []
        for filename, count in files.items():
            payload = json.loads(
                (ROOT / "data/analysis_ready" / directory / filename).read_text(
                    encoding="utf-8"
                )
            )
            rows = payload.get("rows", [])
            if len(rows) != count:
                raise RuntimeError(f"{directory}/{filename}: row count changed")
            identifiers = [str(row["row_id"]) for row in rows]
            if len(identifiers) != len(set(identifiers)):
                raise RuntimeError(f"{directory}/{filename}: duplicate row identifiers")
            if filename != "competence.json":
                treatment_identifiers.extend(identifiers)
        duplicates = [
            key for key, value in Counter(treatment_identifiers).items() if value > 1
        ]
        if duplicates:
            raise RuntimeError(f"{directory}: duplicate row identifiers across files")

    surface_expected = {
        "competence.json": 384,
        **{f"shard_{index}.json": 384 for index in range(8)},
    }
    surface_rows: list[dict[str, Any]] = []
    surface_artifacts: dict[str, dict[str, Any]] = {}
    for filename, count in surface_expected.items():
        payload = json.loads(
            (ROOT / "data/analysis_ready/surface_orbit" / filename).read_text(
                encoding="utf-8"
            )
        )
        surface_artifacts[filename] = payload
        rows = payload.get("rows", [])
        if len(rows) != count or len({row["row_id"] for row in rows}) != count:
            raise RuntimeError(f"surface_orbit/{filename}: row grid changed")
        if filename != "competence.json":
            surface_rows.extend(rows)
    if len(surface_rows) != 3072 or len({row["row_id"] for row in surface_rows}) != 3072:
        raise RuntimeError("surface-orbit treatment union changed")
    audit = surface_artifacts["competence.json"].get("backend_audit", {})
    tokenization = audit.get("tokenization", {})
    parity = audit.get("native_readout_parity", {})
    checkpoint = audit.get("checkpoint", {})
    if (
        tokenization.get("prompts_checked") != 384
        or tokenization.get("labels") != ["A", "B"]
        or tokenization.get("all_are_exact_one_token_extensions") is not True
        or parity.get("passes") is not True
        or set(parity.get("per_K", {})) != {"1", "2", "3", "4"}
        or checkpoint.get("repository") != "ByteDance/Ouro-2.6B"
        or checkpoint.get("revision")
        != "1ed04250da1a9936042725d302e81c8fa2ab5abd"
        or checkpoint.get("native_K_values") != [1, 2, 3, 4]
        or checkpoint.get("readout_selector") != "exit_at_step=K-1"
    ):
        raise RuntimeError("surface-orbit competence execution audit changed")
    for depth, item in parity["per_K"].items():
        if (
            item.get("full_vocabulary_max_abs_logit_difference") != 0.0
            or item.get("answer_token_max_abs_logit_difference") != 0.0
            or item.get("torch_allclose") is not True
            or item.get("exact_equal") is not True
        ):
            raise RuntimeError(f"surface-orbit native-readout parity failed at K={depth}")
    unlock_hashes = {
        artifact.get("competence_unlock", {}).get("sha256")
        for filename, artifact in surface_artifacts.items()
        if filename != "competence.json"
    }
    if unlock_hashes != {
        "e6d566adf718b7f504bd39a34d28ea70a5e142777966771275eadc86ce1a7e7d"
    }:
        raise RuntimeError("surface-orbit competence unlock binding changed")

    hrm_linear_dir = ROOT / "data/analysis_ready/hrm_linear"
    linear_competence = json.loads(
        (hrm_linear_dir / "competence.json").read_text(encoding="utf-8")
    )
    if (
        len(linear_competence.get("rows", [])) != 384
        or linear_competence.get("summary", {}).get("decision")
        != "PROMOTE_TO_TREATMENT"
        or linear_competence.get("summary", {}).get("passes_competence_gate")
        is not True
    ):
        raise RuntimeError("HRM linear competence release changed")
    linear_rows: list[dict[str, Any]] = []
    linear_unlocks = set()
    for index in range(8):
        artifact = json.loads(
            (hrm_linear_dir / f"shard_{index}.json").read_text(encoding="utf-8")
        )
        rows = artifact.get("rows", [])
        if (
            artifact.get("shard") != {"count": 8, "index": index}
            or len(rows) != 600
        ):
            raise RuntimeError(f"HRM linear shard {index} changed")
        linear_rows.extend(rows)
        linear_unlocks.add(artifact.get("competence_unlock", {}).get("sha256"))
    if (
        len(linear_rows) != 4_800
        or len({row["row_id"] for row in linear_rows}) != 4_800
        or len({row["world_id"] for row in linear_rows}) != 240
        or linear_unlocks
        != {"fecebbb0d0b2302d61270b20b5d6ecb13a898a187757ef06deac94d98b1d52e3"}
    ):
        raise RuntimeError("HRM linear treatment union or gate binding changed")

    hrm_branching_dir = ROOT / "data/analysis_ready/hrm_branching"
    branching_gate = json.loads(
        (hrm_branching_dir / "gate.json").read_text(encoding="utf-8")
    )
    if (
        len(branching_gate.get("rows", [])) != 384
        or branching_gate.get("summary", {}).get("decision")
        != "PROMOTE_TO_CONFIRMATION"
        or branching_gate.get("summary", {}).get("authorizes_real_confirmation")
        is not True
    ):
        raise RuntimeError("HRM branching gate release changed")
    branching_rows: list[dict[str, Any]] = []
    branching_unlocks = set()
    for index in range(12):
        artifact = json.loads(
            (hrm_branching_dir / f"shard_{index}.json").read_text(encoding="utf-8")
        )
        rows = artifact.get("rows", [])
        if (
            artifact.get("shard") != {"count": 12, "index": index}
            or len(rows) != 576
        ):
            raise RuntimeError(f"HRM branching shard {index} changed")
        branching_rows.extend(rows)
        branching_unlocks.add(artifact.get("gate_unlock", {}).get("sha256"))
    if (
        len(branching_rows) != 6_912
        or len({row["row_id"] for row in branching_rows}) != 6_912
        or len({row["world_id"] for row in branching_rows}) != 144
        or branching_unlocks
        != {"197fe8c3a02f2df82dbb0a221296dd486a333964315bbf63c09a0c0e6877a1bb"}
    ):
        raise RuntimeError("HRM branching confirmation union or gate binding changed")
    for row in (*linear_competence["rows"], *linear_rows, *branching_gate["rows"], *branching_rows):
        logits = row.get("score", {}).get("raw_logits", {})
        if set(logits) != {"A", "B"} or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in logits.values()
        ):
            raise RuntimeError("HRM released row lacks numeric A/B logits")

    boundary_rows: list[dict[str, Any]] = []
    boundary_counts = (440, 440, 440, 432)
    for index, count in enumerate(boundary_counts):
        payload = json.loads(
            (ROOT / "data/analysis_ready/decoding_boundary" / f"shard_{index}.json").read_text(
                encoding="utf-8"
            )
        )
        rows = payload.get("rows", [])
        if len(rows) != count:
            raise RuntimeError(f"decoding_boundary/shard_{index}.json: row count changed")
        boundary_rows.extend(rows)
    if len(boundary_rows) != 1752 or len({row["row_id"] for row in boundary_rows}) != 1752:
        raise RuntimeError("decoding-boundary row union changed")
    if Counter(row["dataset"] for row in boundary_rows) != Counter(
        {"2Wiki": 992, "MuSiQue": 760}
    ):
        raise RuntimeError("decoding-boundary dataset counts changed")
    pair_counts = {
        dataset: len(
            {row["pair_id"] for row in boundary_rows if row["dataset"] == dataset}
        )
        for dataset in ("2Wiki", "MuSiQue")
    }
    if pair_counts != {"2Wiki": 124, "MuSiQue": 95}:
        raise RuntimeError("decoding-boundary independent-unit counts changed")
    path_counts = Counter(
        row["relation_path_id"]
        for row in boundary_rows
        if row["dataset"] == "2Wiki" and int(row["K"]) == 1
    )
    pair_path = {
        (row["pair_id"], row["relation_path_id"])
        for row in boundary_rows
        if row["dataset"] == "2Wiki"
    }
    if len(path_counts) != 31 or len(pair_path) != 124:
        raise RuntimeError("decoding-boundary path strata changed")

    factorial = sorted((ROOT / "data/analysis_ready/factorial/runs").glob("seed*.json"))
    if len(factorial) != 32:
        raise RuntimeError("factorial run count changed")
    cells: Counter[tuple[int, str, str]] = Counter()
    for path in factorial:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells[(int(payload["seed"]), payload["condition"]["tying"], payload["condition"]["supervision"])] += 1
    if set(cells.values()) != {1} or len({key[0] for key in cells}) != 8:
        raise RuntimeError("factorial seed-condition grid changed")


def verify_provenance() -> None:
    payload = json.loads((ROOT / "data/PROVENANCE.json").read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", [])
    if len(artifacts) < 90:
        raise RuntimeError("provenance inventory is unexpectedly small")
    for artifact in artifacts:
        relative = artifact["release_path"]
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"provenance target missing: {relative}")
        if sha256(path) != artifact["release_sha256"]:
            raise RuntimeError(f"provenance digest mismatch: {relative}")
    surface_competence = [
        artifact
        for artifact in artifacts
        if artifact.get("artifact_id") == "surface_orbit_competence"
    ]
    if len(surface_competence) != 1:
        raise RuntimeError("surface-orbit competence provenance entry changed")
    competence = surface_competence[0]
    if (
        competence.get("source_sha256")
        != "e6d566adf718b7f504bd39a34d28ea70a5e142777966771275eadc86ce1a7e7d"
        or competence.get("release_path")
        != "data/analysis_ready/surface_orbit/competence.json"
    ):
        raise RuntimeError("surface-orbit competence provenance crosswalk changed")
    hrm_sources = {
        item["artifact_id"]: item
        for item in artifacts
        if item.get("artifact_id")
        in {"hrm_linear_competence", "hrm_branching_gate"}
    }
    if (
        set(hrm_sources) != {"hrm_linear_competence", "hrm_branching_gate"}
        or hrm_sources["hrm_linear_competence"].get("source_sha256")
        != "fecebbb0d0b2302d61270b20b5d6ecb13a898a187757ef06deac94d98b1d52e3"
        or hrm_sources["hrm_branching_gate"].get("source_sha256")
        != "197fe8c3a02f2df82dbb0a221296dd486a333964315bbf63c09a0c0e6877a1bb"
    ):
        raise RuntimeError("HRM gate provenance crosswalk changed")


def verify_expected() -> None:
    required = {
        "scale_invariant_path_control.json",
        "natural_depth_curves.json",
        "loopus_full_depth.json",
        "fictional_ouro/nonce_path_control_ouro26.json",
        "fictional_loopus8.json",
        "hrm_linear.json",
        "hrm_branching.json",
        "structural_falsifiers.json",
        "surface_orbit_confirmation.json",
        "decoding_boundary_audit.json",
        "training_factorial.json",
        "headline_results.json",
    }
    missing = sorted(
        relative for relative in required if not (ROOT / "expected/analysis" / relative).is_file()
    )
    if missing:
        raise RuntimeError(f"expected analysis outputs missing: {missing}")
    figure_root = ROOT / "expected/figures"
    for suffix in ("pdf", "png", "svg"):
        count = len(list(figure_root.rglob(f"*.{suffix}")))
        if count != 12:
            raise RuntimeError(f"expected {suffix} figure count changed: {count}")


def verify_plot_ready() -> None:
    directory = ROOT / "data/plot_ready/appendix"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        path = directory / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"plot-ready binding changed: {item['path']}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = sum(1 for _row in csv.DictReader(handle))
        if rows != int(item["rows"]):
            raise RuntimeError(f"plot-ready row count changed: {item['path']}")


def main() -> None:
    manifest_count = verify_manifest()
    verify_panels()
    verify_natural()
    verify_model_rows()
    verify_provenance()
    verify_expected()
    verify_plot_ready()
    subprocess.run([sys.executable, "tools/audit_anonymity.py"], cwd=ROOT, check=True)
    print(f"Release verification passed: {manifest_count} manifest entries")


if __name__ == "__main__":
    main()
