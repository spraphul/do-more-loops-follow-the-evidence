#!/usr/bin/env python3
"""Recompute the diagnostic K1-through-K8 LoopUS natural trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_natural_depth_curves import Configuration, load_rows, summarize


CONFIGURATIONS = (
    Configuration(
        "loopus_2wiki",
        "loopus_full_depth_2wiki.json",
        "2Wiki",
        1,
        8,
        6_324,
        93,
        31,
        "cluster_then_sorted_pair",
        12709350839368180215,
        "nearest",
    ),
    Configuration(
        "loopus_musique",
        "loopus_full_depth_musique.json",
        "MuSiQue",
        1,
        8,
        6_460,
        95,
        30,
        "pair",
        1614396649522333385,
        "nearest",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo / "data/analysis_ready/natural",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "reproduced/results/loopus_full_depth.json",
    )
    args = parser.parse_args()
    results = {
        configuration.key: summarize(
            configuration, load_rows(args.data_dir / configuration.filename)
        )
        for configuration in CONFIGURATIONS
    }
    payload = {
        "schema": "evidence-loops.loopus-full-depth-diagnostic.v1",
        "evidence_status": "diagnostic_full_exit_trajectory",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
