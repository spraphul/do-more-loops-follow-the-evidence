#!/usr/bin/env python3
"""Run the frozen LoopUS-8B fictional path-control treatment extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "iclr2027.loopus8_nonce_path_control.run.v1"
COMPETENCE_SCHEMA = "iclr2027.loopus8_nonce_competence.run.v1"
PANEL_SHA256 = "7880c02c64af541629b9c23dce15bf4a6df98c4f99a8aed45b4a310df2e543e5"
DEPTHS = (1, 2, 4, 8)
DEEP = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_competence(path: Path, panel_sha256: str, mock: bool) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != COMPETENCE_SCHEMA
        or artifact.get("mode") != "competence"
        or artifact.get("model_key") != "loopus8"
        or artifact.get("panel", {}).get("canonical_sha256") != panel_sha256
        or artifact.get("summary", {}).get("passes_competence_gate") is not True
        or bool(artifact.get("mock")) != mock
    ):
        raise RuntimeError("competence artifact does not unlock treatment")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--competence-artifact", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index")

    analysis_dir = Path(__file__).resolve().parent
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    import run_loopus8_nonce_competence as competence_runner
    import run_nonce_path_control as parent

    panel_path = args.panel.resolve()
    panel = parent.load_panel(panel_path)
    if panel["panel_sha256"] != PANEL_SHA256:
        raise RuntimeError("parent panel binding changed")
    competence = load_competence(
        args.competence_artifact.resolve(), panel["panel_sha256"], args.mock
    )

    world_index = {
        world["world_id"]: int(world["world_index"]) for world in panel["worlds"]
    }
    scheduled: list[tuple[dict[str, Any], tuple[int, ...]]] = []
    for cell in panel["cells"]:
        if world_index[cell["world_id"]] % args.num_shards != args.shard_index:
            continue
        state = cell["evidence_state"]
        if state in {"original", "bridge_swap"}:
            depths = DEPTHS
        elif state == "topology_repair":
            depths = (DEEP,)
        else:
            raise RuntimeError(f"unexpected evidence state: {state}")
        scheduled.append((cell, depths))
    scheduled.sort(key=lambda item: item[0]["row_id"])

    expected_worlds = sum(
        index % args.num_shards == args.shard_index
        for index in world_index.values()
    )
    if len(scheduled) != expected_worlds * 12:
        raise RuntimeError("sharded prompt schedule is incomplete")

    if args.mock:
        backend = None
        audit: dict[str, Any] = {"mock": True}
    else:
        if args.snapshot_dir is None or args.source_dir is None:
            raise SystemExit("--snapshot-dir and --source-dir are required")
        backend = competence_runner.LoopUS8Backend(
            args.snapshot_dir.resolve(),
            args.source_dir.resolve(),
            args.device,
            panel["system_prompt"],
        )
        audit = backend.prepare(cell["prompt"] for cell, _ in scheduled)

    rows: list[dict[str, Any]] = []
    for prompt_index, (cell, depths) in enumerate(scheduled, start=1):
        for depth in depths:
            raw = (
                parent.mock_raw_score(cell, depth)
                if backend is None
                else backend.score(cell["prompt"], depth)
            )
            rows.append(
                {
                    **cell,
                    "K": depth,
                    "row_id": f"{cell['row_id']}__K{depth}",
                    "score": parent.orient_score(cell, raw),
                }
            )
        if prompt_index % 25 == 0 or prompt_index == len(scheduled):
            print(
                f"[{prompt_index}/{len(scheduled)} prompts; {len(rows)} scored rows]",
                flush=True,
            )

    if len(rows) != expected_worlds * 36:
        raise RuntimeError("sharded treatment row count changed")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate treatment row identifier")

    runner = Path(__file__).resolve()
    artifact = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "treatment",
        "model_key": "loopus8",
        "mock": args.mock,
        "depths": list(DEPTHS),
        "deep_depth": DEEP,
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "backend_runner": {
            "path": str(Path(competence_runner.__file__).resolve()),
            "sha256": sha256_file(Path(competence_runner.__file__).resolve()),
        },
        "panel": {
            "path": str(panel_path),
            "canonical_sha256": panel["panel_sha256"],
        },
        "competence_unlock": {
            "path": str(args.competence_artifact.resolve()),
            "sha256": sha256_file(args.competence_artifact),
            "summary": competence["summary"],
        },
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "backend_audit": audit,
        "summary": {
            "decision": "SHARD_COMPLETE",
            "worlds": expected_worlds,
            "prompt_cells": len(scheduled),
            "scored_rows": len(rows),
            "outcome_analysis_deferred_until_merge": True,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
