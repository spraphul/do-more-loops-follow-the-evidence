from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def canonical_hash(payload: dict) -> str:
    value = dict(payload)
    value.pop("panel_sha256", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReleaseTests(unittest.TestCase):
    def test_generated_panel_binding_and_balance(self) -> None:
        panel = load("data/generated/nonce_path_control_v1.panel.json")
        self.assertEqual(panel["panel_sha256"], canonical_hash(panel))
        self.assertEqual(len(panel["worlds"]), 288)
        self.assertEqual(len(panel["cells"]), 3456)
        families = [world["relation_family_index"] for world in panel["worlds"]]
        self.assertEqual(set(families), set(range(6)))
        self.assertEqual({families.count(index) for index in range(6)}, {48})

    def test_structural_panel_binding(self) -> None:
        panel = load("data/generated/nonce_structural_falsifiers_v1.panel.json")
        parent = load("data/generated/nonce_path_control_v1.panel.json")
        self.assertEqual(panel["panel_sha256"], canonical_hash(panel))
        self.assertEqual(panel["parent_panel"]["canonical_sha256"], parent["panel_sha256"])

    def test_natural_rows_are_finite_and_unique(self) -> None:
        for path in sorted((ROOT / "data/analysis_ready/natural").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["rows"]
            self.assertEqual(len(rows), payload["row_count"])
            self.assertEqual(len({row["row_id"] for row in rows}), len(rows))
            for row in rows:
                self.assertTrue(math.isfinite(float(row["reference_log_odds"])))
                self.assertIn(float(row["reference_argmax_credit"]), (0.0, 0.5, 1.0))

    def test_headline_claim_table_has_intervals(self) -> None:
        payload = load("expected/analysis/headline_results.json")
        self.assertGreaterEqual(len(payload["claims"]), 20)
        identifiers = [claim["claim_id"] for claim in payload["claims"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for claim in payload["claims"]:
            low, high = claim["ci95"]
            self.assertLessEqual(low, claim["estimate"])
            self.assertLessEqual(claim["estimate"], high)

    def test_frozen_summary_index_resolves(self) -> None:
        index = load("data/frozen_summaries/INDEX.json")
        self.assertGreaterEqual(len(index), 15)
        for item in index:
            self.assertTrue((ROOT / item["path"]).is_file())


if __name__ == "__main__":
    unittest.main()

