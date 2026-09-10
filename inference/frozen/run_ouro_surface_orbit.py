#!/usr/bin/env python3
"""Rescore the frozen Ouro surface-orbit panel or audit its schedule in mock mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
for directory in (FROZEN, ANALYSIS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import analyze_surface_orbit as surface_analysis
import run_nonce_path_control as base


RUN_SCHEMA = "iclr2027.ouro26_surface_orbit_confirmation.run.v3"
PANEL_SCHEMA = "iclr2027.ouro26_surface_orbit_confirmation.panel.v3"
PANEL_FILE_SHA256 = "d2a88663190a6b4cb0aead5652dcace13a247a065988da1839e17cc850a155fe"
PANEL_CANONICAL_SHA256 = (
    "448bae9f9ed496a5a52d300d8c64c722ed4f4429933a648856ffdef6926d34f2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PANEL_FILE_SHA256:
        raise RuntimeError("surface-orbit panel file hash changed")
    panel = json.loads(path.read_text(encoding="utf-8"))
    if panel.get("schema") != PANEL_SCHEMA:
        raise RuntimeError("unexpected surface-orbit panel schema")
    stored = panel.pop("panel_sha256", None)
    observed = base.canonical_json_sha256(panel)
    panel["panel_sha256"] = stored
    if stored != PANEL_CANONICAL_SHA256 or observed != stored:
        raise RuntimeError("surface-orbit panel canonical hash changed")
    return panel


def selected_cells(
    panel: dict[str, Any],
    mode: str,
    shard_index: int,
    num_shards: int,
    smoke_worlds: int,
) -> list[tuple[dict[str, Any], tuple[int, ...]]]:
    if mode == "smoke":
        allowed = set(panel["gate_world_ids"][:smoke_worlds])
        split = "gate"
        states = {"original"}
        depths = (4,)
    elif mode == "competence":
        allowed = set(panel["gate_world_ids"])
        split = "gate"
        states = {"original"}
        depths = (4,)
    elif mode == "treatment":
        allowed = set(panel["confirmation_world_ids"])
        split = "confirmation"
        states = {"original", "bridge_swap"}
        depths = (1, 4)
    else:
        raise ValueError(mode)
    selected = []
    for cell in panel["cells"]:
        if (
            cell["split"] != split
            or cell["world_id"] not in allowed
            or cell["evidence_state"] not in states
            or int(cell["world_index"]) % num_shards != shard_index
        ):
            continue
        selected.append((cell, depths))
    return sorted(selected, key=lambda item: item[0]["row_id"])


def competence_unlock(path: Path, panel: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != RUN_SCHEMA
        or artifact.get("mode") != "competence"
        or artifact.get("model_key") != "ouro26"
        or artifact.get("panel", {}).get("canonical_sha256")
        != panel["panel_sha256"]
        or artifact.get("summary", {}).get("passes_competence_gate") is not True
    ):
        raise RuntimeError("competence artifact does not unlock treatment")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "decision": artifact["summary"]["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "competence", "treatment"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--competence-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--smoke-worlds", type=int, default=2)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard coordinates")
    if args.mode == "competence" and (args.shard_index, args.num_shards) != (0, 1):
        raise SystemExit("competence must run as one complete artifact")
    if args.mode == "treatment" and args.competence_artifact is None:
        raise SystemExit("treatment requires --competence-artifact")

    panel_path = args.panel.resolve()
    panel = load_panel(panel_path)
    unlock = (
        competence_unlock(args.competence_artifact.resolve(), panel)
        if args.mode == "treatment"
        else None
    )
    scheduled = selected_cells(
        panel, args.mode, args.shard_index, args.num_shards, args.smoke_worlds
    )
    if not scheduled:
        raise RuntimeError("surface-orbit schedule is empty")
    backend = None
    if not args.mock:
        if args.snapshot_dir is None:
            raise SystemExit("real scoring requires --snapshot-dir")
        backend = base.OuroBackend(
            "ouro26",
            args.snapshot_dir.resolve(),
            args.device,
            args.offline,
            panel["system_prompt"],
        )
    prompts = [cell["prompt"] for cell, _depths in scheduled]
    backend_audit = (
        {"mock": True, "tokenization": {"all_labels_exact_one_token_extensions": True}}
        if backend is None
        else backend.prepare(prompts)
    )
    rows: list[dict[str, Any]] = []
    for index, (cell, depths) in enumerate(scheduled, start=1):
        scores = (
            {depth: base.mock_raw_score(cell, depth) for depth in depths}
            if backend is None
            else backend.score(cell["prompt"], depths)
        )
        for depth in depths:
            rows.append(
                {
                    **cell,
                    "K": depth,
                    "row_id": f"{cell['row_id']}__K{depth}",
                    "score": base.orient_score(cell, scores[depth]),
                }
            )
        if index % 50 == 0 or index == len(scheduled):
            print(f"[{index}/{len(scheduled)} prompts; {len(rows)} rows]", flush=True)
    summary = (
        surface_analysis.competence_summary(rows)
        if args.mode == "competence"
        else {
            "decision": "SMOKE_COMPLETE" if args.mode == "smoke" else "SHARD_COMPLETE",
            "worlds": len({row["world_id"] for row in rows}),
            "prompt_cells": len(scheduled),
            "scored_rows": len(rows),
        }
    )
    runner = Path(__file__).resolve()
    artifact = {
        "schema": RUN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model_key": "ouro26",
        "mock": args.mock,
        "runner": {"path": runner.name, "sha256": sha256_file(runner)},
        "panel": {
            "path": args.panel.name,
            "file_sha256": PANEL_FILE_SHA256,
            "canonical_sha256": panel["panel_sha256"],
        },
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "competence_unlock": unlock,
        "backend_audit": backend_audit,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
