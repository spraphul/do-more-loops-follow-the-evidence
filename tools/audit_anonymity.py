#!/usr/bin/env python3
"""Fail when the release contains likely identity, credential, or private-data leaks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".py", ".sh", ".tex", ".toml", ".txt", ".yaml", ".yml"
}
SKIP_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "reproduced"}


def joined(*parts: str) -> str:
    return "".join(parts)


SENSITIVE_PATTERNS = {
    "personal user name": re.compile(joined("prap", "sing"), re.IGNORECASE),
    "private project path": re.compile(joined("score-state", "-feedback"), re.IGNORECASE),
    "macOS user path": re.compile(re.escape(joined("/", "Users", "/"))),
    "Linux user path": re.compile(re.escape(joined("/", "home", "/"))),
    "cluster path": re.compile(re.escape(joined("/", "sds", "/"))),
    "mount path": re.compile(re.escape(joined("/", "mount", "/"))),
    "environment secret file": re.compile(re.escape(joined(".", "ord-gpu", ".env"))),
    "credential-shaped token": re.compile(
        joined("(?<![A-Za-z0-9])(?:", "hf_", "|", "sk-", ")[A-Za-z0-9_-]{16,}")
    ),
}
NATURAL_FORBIDDEN_KEYS = {
    "answer",
    "continuation",
    "dataset_path",
    "evidence_triples",
    "natural_question",
    "prompt",
    "question",
    "source_item_id",
    "source_item_ids",
}


def release_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        yield path


def scan_text(path: Path, failures: list[str]) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile":
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        failures.append(f"non-UTF-8 text file: {path.relative_to(ROOT)}")
        return
    for label, pattern in SENSITIVE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            failures.append(f"{label}: {path.relative_to(ROOT)}:{line}")


def walk_keys(value: object, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), f"{path}.{key}"
            yield from walk_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_keys(child, f"{path}[{index}]")


def scan_natural_schema(failures: list[str]) -> None:
    for path in sorted((ROOT / "data/analysis_ready/natural").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, location in walk_keys(payload.get("rows", [])):
            if key.lower() in NATURAL_FORBIDDEN_KEYS:
                failures.append(
                    f"natural text-bearing field {key!r}: {path.relative_to(ROOT)} {location}"
                )


def scan_pdf_metadata(failures: list[str]) -> None:
    executable = shutil.which("pdfinfo")
    if executable is None:
        return
    for path in release_files():
        if path.suffix.lower() != ".pdf":
            continue
        result = subprocess.run(
            [executable, str(path)], text=True, capture_output=True, check=False
        )
        metadata = result.stdout
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(metadata):
                failures.append(f"{label} in PDF metadata: {path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for path in release_files():
        scan_text(path, failures)
    scan_natural_schema(failures)
    scan_pdf_metadata(failures)
    if failures:
        raise SystemExit("Anonymity audit failed:\n- " + "\n- ".join(sorted(set(failures))))
    print("Anonymity audit passed")


if __name__ == "__main__":
    main()

