#!/usr/bin/env python3
"""Create a stable SHA-256 manifest for the public release."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "reproduced"}
EXCLUDED_FILES = {"MANIFEST.sha256", ".DS_Store"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path.name not in EXCLUDED_FILES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
    )


def main() -> None:
    paths = sorted(path for path in ROOT.rglob("*") if included(path))
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Bound {len(paths)} files in MANIFEST.sha256")


if __name__ == "__main__":
    main()

