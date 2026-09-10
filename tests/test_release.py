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

    def test_surface_orbit_panel_binding_and_split(self) -> None:
        panel = load("data/generated/surface_orbit_confirmation_v3.panel.json")
        self.assertEqual(panel["panel_sha256"], canonical_hash(panel))
        self.assertEqual(panel["dimensions"]["gate_worlds"], 24)
        self.assertEqual(panel["dimensions"]["confirmation_worlds"], 48)
        self.assertEqual(panel["dimensions"]["surface_realizations_per_world"], 4)
        self.assertEqual(
            {world["serialization"] for world in panel["surface_worlds"]},
            {"arrows", "tuples", "json", "sentences"},
        )

    def test_hrm_branching_panel_binding_and_split(self) -> None:
        panel = load("data/generated/hrm_branching_confirmation_v1.panel.json")
        self.assertEqual(panel["panel_sha256"], canonical_hash(panel))
        self.assertEqual(len(panel["worlds"]), 192)
        self.assertEqual(len(panel["cells"]), 4608)
        self.assertEqual(
            {split: sum(world["split"] == split for world in panel["worlds"])
             for split in ("gate", "confirmation")},
            {"gate": 48, "confirmation": 144},
        )

    def test_natural_rows_are_finite_and_unique(self) -> None:
        for path in sorted((ROOT / "data/analysis_ready/natural").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["rows"]
            self.assertEqual(len(rows), payload["row_count"])
            self.assertEqual(len({row["row_id"] for row in rows}), len(rows))
            for row in rows:
                self.assertTrue(math.isfinite(float(row["reference_log_odds"])))
                self.assertIn(float(row["reference_argmax_credit"]), (0.0, 0.5, 1.0))

    def test_surface_orbit_rows_are_complete(self) -> None:
        directory = ROOT / "data/analysis_ready/surface_orbit"
        competence = json.loads((directory / "competence.json").read_text(encoding="utf-8"))
        self.assertEqual(len(competence["rows"]), 384)
        rows = []
        for index in range(8):
            payload = json.loads(
                (directory / f"shard_{index}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["shard"], {"count": 8, "index": index})
            self.assertEqual(len(payload["rows"]), 384)
            rows.extend(payload["rows"])
        self.assertEqual(len(rows), 3072)
        self.assertEqual(len({row["row_id"] for row in rows}), 3072)

    def test_headline_claim_table_has_intervals(self) -> None:
        payload = load("expected/analysis/headline_results.json")
        self.assertGreaterEqual(len(payload["claims"]), 20)
        identifiers = [claim["claim_id"] for claim in payload["claims"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for claim in payload["claims"]:
            low, high = claim["ci95"]
            self.assertLessEqual(low, claim["estimate"])
            self.assertLessEqual(claim["estimate"], high)

    def test_new_expected_outputs_have_intervals(self) -> None:
        surface = load("expected/analysis/surface_orbit_confirmation.json")
        self.assertTrue(surface["primary_confirmation_passes"])
        for key in ("D4", "G41", "H41"):
            low, high = surface["overall"][key]["ci95"]
            self.assertLess(low, surface["overall"][key]["estimate"])
            self.assertLess(surface["overall"][key]["estimate"], high)
        boundary = load("expected/analysis/decoding_boundary_audit.json")
        self.assertEqual(boundary["status"], "post_outcome_diagnostic")
        for dataset in ("2Wiki", "MuSiQue"):
            metric = boundary["datasets"][dataset]["metrics"][
                "K3_boundary_replay_top1_matches_stored"
            ]
            self.assertLessEqual(metric["ci95"][0], metric["estimate"])
            self.assertLessEqual(metric["estimate"], metric["ci95"][1])

    def test_hrm_expected_outputs_match_manuscript_values(self) -> None:
        linear = load("expected/analysis/hrm_linear.json")
        self.assertEqual(
            linear["decision"], "SUPPORT_HRM_TOPOLOGY_SENSITIVE_PATH_CONTROL"
        )
        self.assertAlmostEqual(
            linear["result"]["H2_minus_H1_choice_gain"]["estimate"],
            0.8958333333333334,
        )
        self.assertAlmostEqual(
            linear["result"]["H2_topology_recovery_fraction"]["estimate"],
            1.0445373505631954,
        )
        branching = load("expected/analysis/hrm_branching.json")
        self.assertEqual(
            branching["confirmation"]["decision"],
            "CONFIRM_HRM_DEPTH_DEPENDENT_BRANCHING_COMPOSITION",
        )
        self.assertAlmostEqual(
            branching["results"]["primary_repair_xor_choice_H2_minus_H1"]["estimate"],
            0.5620659722222222,
        )
        self.assertAlmostEqual(
            branching["results"]["repair_fraction_choice_H2"]["estimate"],
            1.009353078721746,
        )

    def test_frozen_summary_index_resolves(self) -> None:
        index = load("data/frozen_summaries/INDEX.json")
        self.assertGreaterEqual(len(index), 15)
        for item in index:
            self.assertTrue((ROOT / item["path"]).is_file())

    def test_boundary_release_is_text_free(self) -> None:
        forbidden = {
            "generated_answer_span_raw",
            "prompt",
            "question",
            "reference_candidate",
            "alternate_candidate",
            "stored_top_token_text",
            "replay_top_token_text",
            "token_id",
            "token_ids",
        }

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(map(str, value)) | set().union(
                    *(keys(child) for child in value.values())
                )
            if isinstance(value, list):
                return set().union(*(keys(child) for child in value))
            return set()

        rows = []
        for path in sorted((ROOT / "data/analysis_ready/decoding_boundary").glob("*.json")):
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        self.assertEqual(len(rows), 1752)
        self.assertFalse(forbidden & keys(rows))


if __name__ == "__main__":
    unittest.main()
