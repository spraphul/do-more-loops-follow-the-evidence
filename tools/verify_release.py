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
    for label, payload in (("fictional", parent), ("structural", structural)):
        if payload.get("panel_sha256") != canonical_hash(payload):
            raise RuntimeError(f"{label} panel canonical hash mismatch")
    if len(parent.get("worlds", [])) != 288 or len(parent.get("cells", [])) != 3456:
        raise RuntimeError("fictional panel dimensions changed")
    if len(structural.get("worlds", [])) != 168:
        raise RuntimeError("structural panel world pool changed")
    if structural.get("parent_panel", {}).get("canonical_sha256") != parent["panel_sha256"]:
        raise RuntimeError("structural parent binding changed")


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
    if len(artifacts) < 50:
        raise RuntimeError("provenance inventory is unexpectedly small")
    for artifact in artifacts:
        relative = artifact["release_path"]
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"provenance target missing: {relative}")
        if sha256(path) != artifact["release_sha256"]:
            raise RuntimeError(f"provenance digest mismatch: {relative}")


def verify_expected() -> None:
    required = {
        "scale_invariant_path_control.json",
        "natural_depth_curves.json",
        "loopus_full_depth.json",
        "fictional_ouro/nonce_path_control_ouro26.json",
        "fictional_loopus8.json",
        "structural_falsifiers.json",
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
