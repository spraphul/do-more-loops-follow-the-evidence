#!/usr/bin/env python3
"""Exercise the frozen generated-panel schedulers with deterministic mock logits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduced/mock_inference"


def run(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent_panel = "data/generated/nonce_path_control_v1.panel.json"
    structural_panel = "data/generated/nonce_structural_falsifiers_v1.panel.json"
    parent_output = OUT / "ouro_schedule_smoke.json"
    loopus_output = OUT / "loopus_schedule_smoke.json"
    structural_output = OUT / "structural_schedule_smoke.json"

    run(
        [
            "inference/frozen/run_nonce_path_control.py",
            "--model",
            "ouro26",
            "--mode",
            "smoke",
            "--panel",
            parent_panel,
            "--output",
            str(parent_output),
            "--smoke-worlds",
            "2",
            "--mock",
        ]
    )
    run(
        [
            "inference/frozen/run_loopus8_nonce_competence.py",
            "--panel",
            parent_panel,
            "--output",
            str(loopus_output),
            "--max-worlds",
            "2",
            "--mock",
        ]
    )
    run(
        [
            "inference/frozen/run_nonce_structural_falsifiers.py",
            "--mode",
            "development",
            "--panel",
            structural_panel,
            "--output",
            str(structural_output),
            "--limit-worlds",
            "2",
            "--mock",
        ]
    )

    expected_minimum_rows = {
        parent_output: 1,
        loopus_output: 8,
        structural_output: 1,
    }
    for path, minimum in expected_minimum_rows.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mock") is not True or len(payload.get("rows", [])) < minimum:
            raise RuntimeError(f"invalid mock output: {path}")
    print("Frozen scheduler smoke tests passed; mock outputs are not scientific evidence")


if __name__ == "__main__":
    main()

