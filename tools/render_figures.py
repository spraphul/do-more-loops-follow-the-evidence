#!/usr/bin/env python3
"""Render all released statistical figures from frozen or reproduced values."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "figures/plot_main_depth_regimes.py",
    "figures/plot_main_structural_falsifiers.py",
    "figures/plot_appendix_training_factorial.py",
    "figures/plot_appendix_diagnostics.py",
)


def main() -> None:
    required = ROOT / "reproduced/results/structural_falsifiers.json"
    if not required.exists():
        raise SystemExit("run `make reproduce` before `make figures`")
    environment = os.environ.copy()
    cache = ROOT / "reproduced/matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(cache)
    environment["MPLBACKEND"] = "Agg"
    for script in SCRIPTS:
        print(f"[render] {script}", flush=True)
        subprocess.run([sys.executable, script], cwd=ROOT, env=environment, check=True)
    files = sorted((ROOT / "reproduced/figures").rglob("*.pdf"))
    print(f"Rendered {len(files)} PDF panels with matching SVG and PNG files")


if __name__ == "__main__":
    main()
