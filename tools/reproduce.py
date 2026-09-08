#!/usr/bin/env python3
"""Run the frozen CPU analyses and compare them with expected outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def paths(pattern: str) -> list[str]:
    values = sorted(ROOT.glob(pattern))
    if not values:
        raise FileNotFoundError(pattern)
    return [str(path) for path in values]


def commands() -> dict[str, list[str]]:
    results = ROOT / "reproduced/results"
    panel = ROOT / "data/generated/nonce_path_control_v1.panel.json"
    structural_panel = ROOT / "data/generated/nonce_structural_falsifiers_v1.panel.json"
    return {
        "natural-scale": [PYTHON, "analysis/analyze_scale_invariant_natural.py"],
        "natural-curves": [PYTHON, "analysis/analyze_natural_depth_curves.py"],
        "loopus-full-depth": [PYTHON, "analysis/analyze_loopus_full_depth.py"],
        "fictional-ouro": [
            PYTHON,
            "analysis/analyze_fictional_ouro.py",
            "--competence-artifact",
            "data/analysis_ready/fictional_ouro/competence.json",
            "--shards",
            *paths("data/analysis_ready/fictional_ouro/shard_*.json"),
            "--output-dir",
            str(results / "fictional_ouro"),
        ],
        "fictional-loopus": [
            PYTHON,
            "analysis/analyze_fictional_loopus8.py",
            "--inputs",
            *paths("data/analysis_ready/fictional_loopus8/shard_*.json"),
            "--competence-artifact",
            "data/analysis_ready/fictional_loopus8/competence.json",
            "--panel",
            str(panel),
            "--json-output",
            str(results / "fictional_loopus8.json"),
            "--markdown-output",
            str(results / "fictional_loopus8.md"),
        ],
        "structural": [
            PYTHON,
            "analysis/analyze_structural_falsifiers.py",
            "--mode",
            "confirm",
            "--panel",
            str(structural_panel),
            "--inputs",
            *paths("data/analysis_ready/structural/shard_*.json"),
            "--json-output",
            str(results / "structural_falsifiers.json"),
            "--markdown-output",
            str(results / "structural_falsifiers.md"),
        ],
        "factorial": [
            PYTHON,
            "analysis/analyze_training_factorial.py",
            "--inputs",
            *paths("data/analysis_ready/factorial/runs/seed*.json"),
            "--json-output",
            str(results / "training_factorial.json"),
            "--markdown-output",
            str(results / "training_factorial.md"),
        ],
    }


def run_one(name: str, command: list[str]) -> tuple[str, float, str]:
    started = time.monotonic()
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(
            f"{name} failed after {elapsed:.1f}s\n{completed.stdout.rstrip()}"
        )
    return name, elapsed, completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(commands()),
        help="run one named analysis; repeat to select several",
    )
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    selected = commands()
    if args.only:
        selected = {name: selected[name] for name in args.only}
    (ROOT / "reproduced/results").mkdir(parents=True, exist_ok=True)
    print(f"Running {len(selected)} frozen analyses with {args.jobs} worker(s)")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_one, name, command): name
            for name, command in selected.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, elapsed, output = future.result()
                tail = output.strip().splitlines()[-1] if output.strip() else "complete"
                print(f"[ok] {name}: {elapsed:.1f}s ({tail})")
            except Exception as error:
                failures.append(str(error))
                print(f"[failed] {name}", file=sys.stderr)
    if failures:
        raise SystemExit("\n\n".join(failures))

    all_names = set(commands())
    if set(selected) == all_names:
        subprocess.run([PYTHON, "tools/build_claim_table.py"], cwd=ROOT, check=True)
    if not args.skip_compare:
        subprocess.run(
            [PYTHON, "tools/verify_expected.py", "--names", *selected],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()

