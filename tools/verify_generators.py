#!/usr/bin/env python3
"""Rebuild both fictional panels and require byte-identical output."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evidence-loops-panels-") as directory:
        temporary = Path(directory)
        parent = temporary / "parent.json"
        structural = temporary / "structural.json"
        run(
            [
                sys.executable,
                "data/generated/build_nonce_path_control_panel.py",
                "--output",
                str(parent),
            ]
        )
        run(
            [
                sys.executable,
                "data/generated/build_nonce_structural_falsifiers.py",
                "--parent-panel",
                str(parent),
                "--output",
                str(structural),
            ]
        )
        expected_parent = ROOT / "data/generated/nonce_path_control_v1.panel.json"
        expected_structural = (
            ROOT / "data/generated/nonce_structural_falsifiers_v1.panel.json"
        )
        if parent.read_bytes() != expected_parent.read_bytes():
            raise RuntimeError("fictional parent generator is not byte-identical")
        if structural.read_bytes() != expected_structural.read_bytes():
            raise RuntimeError("structural generator is not byte-identical")
    print("Generated panels rebuild byte-identically")


if __name__ == "__main__":
    main()

