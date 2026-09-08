#!/usr/bin/env python3
"""Compare fresh analysis JSON with the frozen reviewer reference outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAPPING = {
    "natural-scale": "scale_invariant_path_control.json",
    "natural-curves": "natural_depth_curves.json",
    "loopus-full-depth": "loopus_full_depth.json",
    "fictional-ouro": "fictional_ouro/nonce_path_control_ouro26.json",
    "fictional-loopus": "fictional_loopus8.json",
    "structural": "structural_falsifiers.json",
    "factorial": "training_factorial.json",
}


def compare(expected: Any, observed: Any, path: str = "$") -> None:
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected is not observed:
            raise AssertionError(f"{path}: {observed!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(float(expected), float(observed), rel_tol=5e-12, abs_tol=5e-12):
            raise AssertionError(f"{path}: {observed!r} != {expected!r}")
        return
    if type(expected) is not type(observed):
        raise AssertionError(
            f"{path}: type {type(observed).__name__} != {type(expected).__name__}"
        )
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise AssertionError(f"{path}: key mismatch; missing={missing}, extra={extra}")
        for key in expected:
            compare(expected[key], observed[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(expected) != len(observed):
            raise AssertionError(f"{path}: length {len(observed)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            compare(left, right, f"{path}[{index}]")
        return
    if expected != observed:
        raise AssertionError(f"{path}: {observed!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", choices=tuple(MAPPING), default=list(MAPPING))
    args = parser.parse_args()
    for name in args.names:
        relative = MAPPING[name]
        expected_path = ROOT / "expected/analysis" / relative
        observed_path = ROOT / "reproduced/results" / relative
        if not expected_path.exists():
            raise FileNotFoundError(expected_path)
        if not observed_path.exists():
            raise FileNotFoundError(observed_path)
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        observed = json.loads(observed_path.read_text(encoding="utf-8"))
        compare(expected, observed)
        print(f"[match] {name}: {relative}")


if __name__ == "__main__":
    main()

