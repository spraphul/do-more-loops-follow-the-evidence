#!/usr/bin/env python3
"""Extract the paper-facing results without changing any reported number."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproduced" / "results"


def load(relative: str) -> dict[str, Any]:
    path = RESULTS / relative
    if not path.exists():
        raise FileNotFoundError(f"missing reproduced result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def record(
    claim_id: str,
    status: str,
    system: str,
    panel: str,
    estimand: str,
    contrast: str,
    metric: dict[str, Any],
    unit: str,
    source: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "evidence_status": status,
        "system": system,
        "panel": panel,
        "estimand": estimand,
        "contrast": contrast,
        "estimate": metric["estimate"],
        "ci95": metric["ci95"],
        "unit": unit,
        "source": source,
    }


def build() -> dict[str, Any]:
    natural = load("scale_invariant_path_control.json")
    curves = load("natural_depth_curves.json")
    ouro = load("fictional_ouro/nonce_path_control_ouro26.json")
    loopus = load("fictional_loopus8.json")
    structural = load("structural_falsifiers.json")
    factorial = load("training_factorial.json")
    claims: list[dict[str, Any]] = []

    natural_specs = {
        "ouro26_2wiki": ("Ouro-2.6B", "2Wiki", "K1 to K3", "held-out confirmation"),
        "ouro26_musique": ("Ouro-2.6B", "MuSiQue", "K1 to K3", "independent confirmation"),
        "loopus_2wiki": ("LoopUS-1.7B", "2Wiki", "K1 to K8", "registered transfer"),
        "loopus_musique": ("LoopUS-1.7B", "MuSiQue", "K1 to K8", "registered transfer"),
    }
    for key, (system, panel, contrast, status) in natural_specs.items():
        values = natural["results"][key]
        claims.append(
            record(
                f"natural-{key}-margin-gain",
                status,
                system,
                panel,
                "full-answer path-margin depth gain",
                contrast,
                values["raw_margin"]["depth_gain"],
                "nats",
                "scale_invariant_path_control.json",
            )
        )
        claims.append(
            record(
                f"natural-{key}-choice-gain",
                status,
                system,
                panel,
                "first-divergence exact-choice depth gain",
                contrast,
                values["exact_first_divergence_token"]["choice_control_gain"],
                "proportion",
                "scale_invariant_path_control.json",
            )
        )
        deep = str(values["deep_K"])
        claims.append(
            record(
                f"natural-{key}-repair",
                status,
                system,
                panel,
                "coherent-repair fraction",
                f"K{deep}",
                curves["results"][key]["by_K"][deep]["repair_fraction"],
                "fraction",
                "natural_depth_curves.json",
            )
        )

    fictional_specs = (
        (
            "fictional-ouro-choice-acquisition",
            "prospective confirmation",
            "Ouro-2.6B",
            "K1 to K4",
            ouro["result"]["primary_choice_gain_H"],
            "fictional_ouro/nonce_path_control_ouro26.json",
        ),
        (
            "fictional-ouro-margin-acquisition",
            "prospective confirmation",
            "Ouro-2.6B",
            "K1 to K4",
            ouro["result"]["secondary_raw_margin_gain_G"],
            "fictional_ouro/nonce_path_control_ouro26.json",
        ),
        (
            "fictional-loopus-choice-change",
            "frozen extension",
            "LoopUS-8B",
            "K1 to K8",
            loopus["result"]["primary_choice_gain_H"],
            "fictional_loopus8.json",
        ),
        (
            "fictional-loopus-margin-sharpening",
            "frozen extension",
            "LoopUS-8B",
            "K1 to K8",
            loopus["result"]["secondary_raw_margin_gain_G"],
            "fictional_loopus8.json",
        ),
    )
    for claim_id, status, system, contrast, metric, source in fictional_specs:
        claims.append(
            record(
                claim_id,
                status,
                system,
                "arbitrary fictional two-hop worlds",
                "exact-choice gain" if "choice" in claim_id else "path-margin gain",
                contrast,
                metric,
                "proportion" if "choice" in claim_id else "label-logit units",
                source,
            )
        )

    for claim_id, estimand, key in (
        ("structural-on-off-specificity", "on-path minus off-path choice effect", "specificity_choice_K4"),
        ("structural-crossed-locality", "graph-directed choice under adverse locality", "graph_effect_choice_crossed_K4"),
        ("structural-locality-amplification", "matched minus crossed locality contribution", "locality_amplification_choice_K4"),
    ):
        claims.append(
            record(
                claim_id,
                "prospective confirmation",
                "Ouro-2.6B",
                "held-out structural worlds",
                estimand,
                "K4",
                structural["results"][key],
                "proportion",
                "structural_falsifiers.json",
            )
        )

    factorial_choice = factorial["results"]["choice_gain_K4_minus_K1"]
    for claim_id, estimand, key in (
        ("factorial-tying-final-only", "tied minus untied depth gain under final-only supervision", "tying_effect_single"),
        ("factorial-tying-all-exit", "tied minus untied depth gain under all-exit supervision", "tying_effect_multi"),
        ("factorial-supervision-tied", "all-exit minus final-only depth gain in tied model", "supervision_effect_tied"),
        ("factorial-supervision-untied", "all-exit minus final-only depth gain in untied model", "supervision_effect_untied"),
        ("factorial-interaction", "tying by supervision interaction", "factorial_interaction"),
    ):
        claims.append(
            record(
                claim_id,
                "confirmatory factorial",
                "controlled graph-memory model",
                "eight confirmation seeds",
                estimand,
                "K1 to K4 choice gain",
                factorial_choice[key],
                "proportion",
                "training_factorial.json",
            )
        )

    return {
        "schema": "evidence-loops.headline-claim-table.v1",
        "note": "Values are copied from reproduced outputs without recomputation or rerounding.",
        "claims": claims,
    }


def main() -> None:
    payload = build()
    output_dir = RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "headline_results.json"
    csv_path = output_dir / "headline_results.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = (
        "claim_id",
        "evidence_status",
        "system",
        "panel",
        "estimand",
        "contrast",
        "estimate",
        "ci95_low",
        "ci95_high",
        "unit",
        "source",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in payload["claims"]:
            row = dict(item)
            low, high = row.pop("ci95")
            row["ci95_low"] = low
            row["ci95_high"] = high
            writer.writerow(row)
    print(json_path.relative_to(ROOT))
    print(csv_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

