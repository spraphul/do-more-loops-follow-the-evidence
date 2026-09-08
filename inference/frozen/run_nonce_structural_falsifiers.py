#!/usr/bin/env python3
"""Score the frozen fictional structural-falsifier panel with native Ouro exits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "iclr2027.nonce_structural_falsifiers.run.v1"
PANEL_SCHEMA = "iclr2027.nonce_structural_falsifiers.panel.v1"
PANEL_FILE_SHA256 = "1ec125a0231c774ea7f1ba3a7c8e9299e0f3aa69b9f21999a675730f3f7a4ae7"
PANEL_CANONICAL_SHA256 = "2f82a391f842b6576cc5dffb274eaddab04784c8af52e6b98ef7e5f49bbcfc8f"
DEPTHS = (1, 2, 3, 4)


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
    if sha256_file(path) != PANEL_FILE_SHA256:
        raise RuntimeError("frozen structural-falsifier panel file hash changed")
    panel = json.loads(path.read_text(encoding="utf-8"))
    if panel.get("schema") != PANEL_SCHEMA:
        raise RuntimeError("unexpected structural-falsifier panel schema")
    stored = panel.pop("panel_sha256", None)
    observed = canonical_json_sha256(panel)
    panel["panel_sha256"] = stored
    if stored != PANEL_CANONICAL_SHA256 or observed != stored:
        raise RuntimeError("frozen structural-falsifier canonical hash changed")
    return panel


def credit(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return 0.0
    return 0.5


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def mock_score(cell: dict[str, Any], k: int) -> dict[str, Any]:
    expected = cell["expected_label"]
    strength = 0.50 * (k - 1)
    logits = {"A": -strength, "B": -strength}
    logits[expected] = strength
    jitter = int(hashlib.sha256(f"{cell['row_id']}:{k}".encode()).hexdigest()[:2], 16)
    if k > 1:
        logits[cell["reference_label"]] += (jitter - 127.5) / 5000.0
    return {
        "raw_logits": logits,
        "target_token_ids": {"A": 1, "B": 2},
        "prompt_tokens": len(cell["prompt"].split()),
        "executed_recurrent_steps": 4,
        "readout_exit_K": k,
        "trajectory_contract": "deterministic structural-falsifier mock",
    }


def orient(cell: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    reference = cell["reference_label"]
    other = cell["other_label"]
    margin = float(raw["raw_logits"][reference]) - float(raw["raw_logits"][other])
    reference_credit = credit(margin)
    state_credit = (
        reference_credit
        if cell["expected_label"] == reference
        else 1.0 - reference_credit
    )
    return {
        **raw,
        "reference_label": reference,
        "other_label": other,
        "expected_label": cell["expected_label"],
        "reference_logit_margin": margin,
        "reference_pair_probability": sigmoid(margin),
        "reference_argmax_credit": reference_credit,
        "state_correct_argmax_credit": state_credit,
    }


def make_backend(args: argparse.Namespace, panel: dict[str, Any]) -> Any:
    if args.mock:
        return None
    if args.snapshot_dir is None:
        raise SystemExit("--snapshot-dir is required for model inference")
    analysis_dir = Path(__file__).resolve().parent
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    import run_nonce_path_control as parent_runner

    return parent_runner.OuroBackend(
        "ouro26",
        args.snapshot_dir.resolve(),
        args.device,
        args.offline,
        panel["system_prompt"],
    )


def selected_cells(
    panel: dict[str, Any], split: str, shard_index: int, num_shards: int,
    limit_worlds: int,
) -> list[dict[str, Any]]:
    world_index = {
        world["world_id"]: int(world["world_index"]) for world in panel["worlds"]
    }
    allowed = sorted(
        {
            cell["world_id"]
            for cell in panel["cells"]
            if cell["split"] == split
        }
    )
    if limit_worlds:
        if split != "development":
            raise SystemExit("--limit-worlds is allowed only for development")
        allowed = allowed[:limit_worlds]
    allowed_set = set(allowed)
    selected = [
        cell
        for cell in panel["cells"]
        if cell["world_id"] in allowed_set
        and world_index[cell["world_id"]] % num_shards == shard_index
    ]
    return sorted(selected, key=lambda cell: cell["row_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirm"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-worlds", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard coordinates")
    panel = load_panel(args.panel.resolve())
    cells = selected_cells(
        panel, args.mode, args.shard_index, args.num_shards, args.limit_worlds
    )
    if not cells:
        raise RuntimeError("selected schedule is empty")
    backend = make_backend(args, panel)
    backend_audit = (
        {"mock": True, "tokenization": {"all_labels_exact_one_token_extensions": True}}
        if backend is None
        else backend.prepare(cell["prompt"] for cell in cells)
    )

    rows: list[dict[str, Any]] = []
    for prompt_index, cell in enumerate(cells, start=1):
        by_k = (
            {k: mock_score(cell, k) for k in DEPTHS}
            if backend is None
            else backend.score(cell["prompt"], DEPTHS)
        )
        for k in DEPTHS:
            rows.append(
                {
                    **cell,
                    "K": k,
                    "row_id": f"{cell['row_id']}__K{k}",
                    "score": orient(cell, by_k[k]),
                }
            )
        if prompt_index % 25 == 0 or prompt_index == len(cells):
            print(
                f"[{prompt_index}/{len(cells)} prompts; {len(rows)} scored rows]",
                flush=True,
            )

    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate scored row IDs")
    artifact = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "model_key": "ouro26",
        "mock": args.mock,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "panel": {
            "path": str(args.panel.resolve()),
            "file_sha256": PANEL_FILE_SHA256,
            "canonical_sha256": PANEL_CANONICAL_SHA256,
        },
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "selection": {
            "worlds": len({cell["world_id"] for cell in cells}),
            "prompt_cells": len(cells),
            "limit_worlds": args.limit_worlds,
        },
        "backend_audit": backend_audit,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
