#!/usr/bin/env python3
"""Reviewer-driven analysis of confidence scaling versus path control.

This analysis is retrospective. It uses only already-scored rows and does not
alter the registered primary analysis. Its purpose is to determine whether the
depth interaction survives measurements that are invariant, or substantially
less sensitive, to a common exit-level rescaling of candidate margins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import numpy as np


BOOTSTRAP_REPLICATES = 10_000
SEED_SALT = "iclr2027-strong-accept:scale-invariant-path-control:v1"


@dataclass(frozen=True)
class Source:
    key: str
    model: str
    dataset: str
    paths: tuple[Path, ...]
    shallow_k: int
    deep_k: int
    primary_bootstrap: str
    expected_rows: int
    expected_pairs: int
    expected_clusters: int


@dataclass(frozen=True)
class PairSummary:
    pair_id: str
    cluster: str
    original_shallow: float
    swap_shallow: float
    original_deep: float
    swap_deep: float
    original_accuracy_shallow: float
    swap_accuracy_shallow: float
    original_accuracy_deep: float
    swap_accuracy_deep: float
    original_first_divergence_shallow: float
    swap_first_divergence_shallow: float
    original_first_divergence_deep: float
    swap_first_divergence_deep: float
    original_first_divergence_accuracy_shallow: float
    swap_first_divergence_accuracy_shallow: float
    original_first_divergence_accuracy_deep: float
    swap_first_divergence_accuracy_deep: float
    original_row_squared_margin_shallow: float
    swap_row_squared_margin_shallow: float
    original_row_squared_margin_deep: float
    swap_row_squared_margin_deep: float

    @property
    def d_shallow(self) -> float:
        return self.original_shallow - self.swap_shallow

    @property
    def d_deep(self) -> float:
        return self.original_deep - self.swap_deep

    @property
    def state_accuracy_shallow(self) -> float:
        return 0.5 * (
            self.original_accuracy_shallow + self.swap_accuracy_shallow
        )

    @property
    def state_accuracy_deep(self) -> float:
        return 0.5 * (self.original_accuracy_deep + self.swap_accuracy_deep)

    @property
    def choice_control_shallow(self) -> float:
        return 2.0 * self.state_accuracy_shallow - 1.0

    @property
    def choice_control_deep(self) -> float:
        return 2.0 * self.state_accuracy_deep - 1.0

    @property
    def joint_follow_shallow(self) -> float:
        return float(self.original_shallow > 0.0 and self.swap_shallow < 0.0)

    @property
    def joint_follow_deep(self) -> float:
        return float(self.original_deep > 0.0 and self.swap_deep < 0.0)

    @property
    def first_divergence_d_shallow(self) -> float:
        return (
            self.original_first_divergence_shallow
            - self.swap_first_divergence_shallow
        )

    @property
    def first_divergence_d_deep(self) -> float:
        return self.original_first_divergence_deep - self.swap_first_divergence_deep

    @property
    def first_divergence_state_accuracy_shallow(self) -> float:
        return 0.5 * (
            self.original_first_divergence_accuracy_shallow
            + self.swap_first_divergence_accuracy_shallow
        )

    @property
    def first_divergence_state_accuracy_deep(self) -> float:
        return 0.5 * (
            self.original_first_divergence_accuracy_deep
            + self.swap_first_divergence_accuracy_deep
        )

    @property
    def first_divergence_choice_control_shallow(self) -> float:
        return 2.0 * self.first_divergence_state_accuracy_shallow - 1.0

    @property
    def first_divergence_choice_control_deep(self) -> float:
        return 2.0 * self.first_divergence_state_accuracy_deep - 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    value = hashlib.sha256(f"{SEED_SALT}:{label}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big", signed=False)


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
        artifact_rows = artifact.get("rows")
        if not isinstance(artifact_rows, list) or not artifact_rows:
            raise RuntimeError(f"{path} does not contain scored rows")
        rows.extend(artifact_rows)
    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("duplicate row identifiers in source artifacts")
    return rows


def row_margin(row: dict[str, Any]) -> float:
    if "reference_log_odds" in row:
        value = float(row["reference_log_odds"])
    else:
        value = float(row["score"]["reference_log_odds"])
    if not math.isfinite(value):
        raise RuntimeError("non-finite candidate margin")
    return value


def row_reference_accuracy(row: dict[str, Any]) -> float:
    if "reference_argmax_credit" in row:
        value = float(row["reference_argmax_credit"])
    else:
        value = float(row["score"]["reference_argmax_credit"])
    if value not in {0.0, 0.5, 1.0}:
        raise RuntimeError("invalid reference argmax credit")
    return value


def row_state_accuracy(row: dict[str, Any]) -> float:
    reference_accuracy = row_reference_accuracy(row)
    expected = int(row["expected_candidate_identity"])
    reference = int(row["reference_candidate_identity"])
    value = reference_accuracy if expected == reference else 1.0 - reference_accuracy
    if "state_correct_argmax_credit" in row:
        recorded = float(row["state_correct_argmax_credit"])
        if not math.isclose(value, recorded, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("recorded state accuracy disagrees with candidate identity")
    return value


def sign_credit(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return 0.0
    return 0.5


def candidate_token_trace(
    row: dict[str, Any], role: str
) -> tuple[list[int], list[float]]:
    if role not in {"reference", "other"}:
        raise ValueError(role)
    score = row["score"]
    if "canonical" in score:
        child = score["canonical" if role == "reference" else "foil"]
        top_level_ids = row[
            "reference_token_ids" if role == "reference" else "other_token_ids"
        ]
        if list(map(int, child["continuation_token_ids"])) != list(
            map(int, top_level_ids)
        ):
            raise RuntimeError("Ouro candidate token IDs disagree across saved fields")
    else:
        child = score[role]
    token_ids = list(map(int, child["continuation_token_ids"]))
    token_log_probabilities = list(map(float, child["token_log_probabilities"]))
    if not token_ids or len(token_ids) != len(token_log_probabilities):
        raise RuntimeError("invalid saved candidate token trace")
    if any(not math.isfinite(value) for value in token_log_probabilities):
        raise RuntimeError("non-finite saved token log probability")
    return token_ids, token_log_probabilities


def first_divergence_margin(row: dict[str, Any]) -> tuple[float, int]:
    """Return an exact same-context reference-minus-other token-logit margin.

    At the first position where candidate token IDs differ, both candidates have
    the same prompt and token prefix. Their log-softmax normalizers therefore
    cancel exactly, making the saved log-probability difference equal to the
    corresponding token-logit difference.
    """

    reference_ids, reference_logps = candidate_token_trace(row, "reference")
    other_ids, other_logps = candidate_token_trace(row, "other")
    for index, (reference_id, other_id) in enumerate(
        zip(reference_ids, other_ids, strict=False)
    ):
        if reference_id != other_id:
            return reference_logps[index] - other_logps[index], index
    raise RuntimeError("candidate pair has no token divergence before one sequence ends")


def first_divergence_state_accuracy(row: dict[str, Any]) -> float:
    margin, _ = first_divergence_margin(row)
    reference_accuracy = sign_credit(margin)
    expected = int(row["expected_candidate_identity"])
    reference = int(row["reference_candidate_identity"])
    return reference_accuracy if expected == reference else 1.0 - reference_accuracy


def pair_cluster(row: dict[str, Any], dataset: str) -> str:
    if dataset == "2Wiki":
        relation_path = row.get("relation_path")
        if not isinstance(relation_path, list) or len(relation_path) != 2:
            raise RuntimeError("2Wiki row lacks a two-relation path")
        return " -> ".join(map(str, relation_path))
    terminal_relation = row.get("terminal_relation")
    if not isinstance(terminal_relation, str) or not terminal_relation:
        raise RuntimeError("MuSiQue row lacks a terminal relation")
    return terminal_relation


def summarize_pairs(source: Source, rows: list[dict[str, Any]]) -> tuple[list[PairSummary], dict[str, Any]]:
    if len(rows) != source.expected_rows:
        raise RuntimeError(
            f"{source.key} row count changed: {len(rows)} != {source.expected_rows}"
        )
    retained = [
        row
        for row in rows
        if int(row["K"]) in {source.shallow_k, source.deep_k}
        and str(row["evidence_state"]) in {"original", "bridge_swap"}
    ]
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    clusters: dict[str, str] = {}
    for row in retained:
        pair_id = str(row["pair_id"])
        state = str(row["evidence_state"])
        k = int(row["K"])
        by_cell[(pair_id, state, k)].append(row)
        cluster = pair_cluster(row, source.dataset)
        previous = clusters.setdefault(pair_id, cluster)
        if previous != cluster:
            raise RuntimeError(f"pair {pair_id} changes bootstrap cluster")

    pair_ids = sorted(clusters)
    summaries: list[PairSummary] = []
    cell_sizes: set[int] = set()
    divergence_positions: Counter[int] = Counter()
    divergence_ties: Counter[str] = Counter()
    for row in rows:
        divergence_margin, divergence_index = first_divergence_margin(row)
        divergence_positions[divergence_index] += 1
        if divergence_margin == 0.0:
            divergence_ties[
                f"{row['evidence_state']}__K{int(row['K'])}"
            ] += 1
    for pair_id in pair_ids:
        cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for state in ("original", "bridge_swap"):
            for k in (source.shallow_k, source.deep_k):
                cell = by_cell.get((pair_id, state, k), [])
                if not cell:
                    raise RuntimeError(f"missing cell for {pair_id}, {state}, K={k}")
                cells[(state, k)] = cell
                cell_sizes.add(len(cell))

        def margin(state: str, k: int) -> float:
            return mean(row_margin(row) for row in cells[(state, k)])

        def accuracy(state: str, k: int) -> float:
            return mean(row_state_accuracy(row) for row in cells[(state, k)])

        def divergence_margin(state: str, k: int) -> float:
            return mean(
                first_divergence_margin(row)[0] for row in cells[(state, k)]
            )

        def divergence_accuracy(state: str, k: int) -> float:
            return mean(
                first_divergence_state_accuracy(row) for row in cells[(state, k)]
            )

        def squared_margin(state: str, k: int) -> float:
            return mean(row_margin(row) ** 2 for row in cells[(state, k)])

        summaries.append(
            PairSummary(
                pair_id=pair_id,
                cluster=clusters[pair_id],
                original_shallow=margin("original", source.shallow_k),
                swap_shallow=margin("bridge_swap", source.shallow_k),
                original_deep=margin("original", source.deep_k),
                swap_deep=margin("bridge_swap", source.deep_k),
                original_accuracy_shallow=accuracy("original", source.shallow_k),
                swap_accuracy_shallow=accuracy("bridge_swap", source.shallow_k),
                original_accuracy_deep=accuracy("original", source.deep_k),
                swap_accuracy_deep=accuracy("bridge_swap", source.deep_k),
                original_first_divergence_shallow=divergence_margin(
                    "original", source.shallow_k
                ),
                swap_first_divergence_shallow=divergence_margin(
                    "bridge_swap", source.shallow_k
                ),
                original_first_divergence_deep=divergence_margin(
                    "original", source.deep_k
                ),
                swap_first_divergence_deep=divergence_margin(
                    "bridge_swap", source.deep_k
                ),
                original_first_divergence_accuracy_shallow=divergence_accuracy(
                    "original", source.shallow_k
                ),
                swap_first_divergence_accuracy_shallow=divergence_accuracy(
                    "bridge_swap", source.shallow_k
                ),
                original_first_divergence_accuracy_deep=divergence_accuracy(
                    "original", source.deep_k
                ),
                swap_first_divergence_accuracy_deep=divergence_accuracy(
                    "bridge_swap", source.deep_k
                ),
                original_row_squared_margin_shallow=squared_margin(
                    "original", source.shallow_k
                ),
                swap_row_squared_margin_shallow=squared_margin(
                    "bridge_swap", source.shallow_k
                ),
                original_row_squared_margin_deep=squared_margin(
                    "original", source.deep_k
                ),
                swap_row_squared_margin_deep=squared_margin(
                    "bridge_swap", source.deep_k
                ),
            )
        )

    if len(summaries) != source.expected_pairs:
        raise RuntimeError(
            f"{source.key} pair count changed: {len(summaries)} != {source.expected_pairs}"
        )
    observed_clusters = len(set(clusters.values()))
    if observed_clusters != source.expected_clusters:
        raise RuntimeError(
            f"{source.key} cluster count changed: "
            f"{observed_clusters} != {source.expected_clusters}"
        )

    metadata = {
        "rows_loaded": len(rows),
        "rows_used": len(retained),
        "pairs": len(summaries),
        "clusters": observed_clusters,
        "counterbalanced_rows_per_pair_state_exit": sorted(cell_sizes),
        "first_divergence_position_counts_zero_based": {
            str(key): int(value) for key, value in sorted(divergence_positions.items())
        },
        "first_divergence_tie_counts_all_saved_states_and_exits": {
            key: int(value) for key, value in sorted(divergence_ties.items())
        },
    }
    return summaries, metadata


def percentile_interval(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    return [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]


def bootstrap_draws(
    pairs: list[PairSummary], method: str, label: str
) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed(label))
    n = len(pairs)
    if method == "pair":
        return [rng.integers(0, n, size=n) for _ in range(BOOTSTRAP_REPLICATES)]
    if method != "cluster_then_pair":
        raise ValueError(f"unknown bootstrap method: {method}")

    by_cluster: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        by_cluster[pair.cluster].append(index)
    cluster_names = sorted(by_cluster)
    draws: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_clusters = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        indices: list[int] = []
        for cluster in sampled_clusters:
            members = by_cluster[str(cluster)]
            indices.extend(rng.choice(members, size=len(members), replace=True).tolist())
        draws.append(np.asarray(indices, dtype=int))
    return draws


def metric(
    pairs: list[PairSummary],
    draws: list[np.ndarray],
    statistic: Callable[[list[PairSummary]], float],
) -> dict[str, Any]:
    point = float(statistic(pairs))
    estimates = [float(statistic([pairs[index] for index in draw])) for draw in draws]
    return {
        "estimate": point,
        "ci95": percentile_interval(estimates),
        "bootstrap_replicates": len(draws),
    }


def array_mean(pairs: list[PairSummary], attribute: str) -> float:
    return float(np.mean([float(getattr(pair, attribute)) for pair in pairs]))


def row_rms_normalized_d(
    pairs: list[PairSummary], depth: str, denominator_states: str
) -> float:
    numerator = array_mean(pairs, f"d_{depth}")
    original_squares = np.asarray(
        [
            float(getattr(pair, f"original_row_squared_margin_{depth}"))
            for pair in pairs
        ],
        dtype=float,
    )
    if denominator_states == "original":
        squared_margins = original_squares
    elif denominator_states == "pooled":
        swap_squares = np.asarray(
            [
                float(getattr(pair, f"swap_row_squared_margin_{depth}"))
                for pair in pairs
            ],
            dtype=float,
        )
        squared_margins = 0.5 * (original_squares + swap_squares)
    else:
        raise ValueError(denominator_states)
    denominator = float(np.sqrt(np.mean(squared_margins)))
    if denominator <= 0.0:
        raise RuntimeError("zero RMS margin scale")
    return float(numerator / denominator)


def paired_standardized_d(pairs: list[PairSummary], depth: str) -> float:
    values = np.asarray([float(getattr(pair, f"d_{depth}")) for pair in pairs])
    spread = float(np.std(values, ddof=1))
    if spread <= 0.0:
        raise RuntimeError("zero paired-effect spread")
    return float(np.mean(values) / spread)


def positive_scale_fit(deep: np.ndarray, shallow: np.ndarray) -> float:
    """Fit shallow = scale * deep with a nonnegative, zero-intercept slope."""

    if len(deep) < 3:
        raise RuntimeError("too few calibration observations")
    denominator = float(np.dot(deep, deep))
    if denominator <= 1e-15:
        return 0.0
    return max(0.0, float(np.dot(deep, shallow) / denominator))


def fold_for_cluster(cluster: str, folds: int = 5) -> int:
    digest = hashlib.sha256(f"fold:{cluster}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds


def crossfit_positive_scale_residuals(
    pairs: list[PairSummary], include_fits: bool = True
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Scale deep margins onto the shallow scale using intact controls only."""

    residuals = np.empty(len(pairs), dtype=float)
    fits: list[dict[str, float]] = []
    folds = np.asarray([fold_for_cluster(pair.cluster) for pair in pairs], dtype=int)
    for fold in sorted(set(folds.tolist())):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        deep = np.asarray([pairs[i].original_deep for i in train], dtype=float)
        shallow = np.asarray([pairs[i].original_shallow for i in train], dtype=float)
        scale = positive_scale_fit(deep, shallow)
        if include_fits:
            fits.append(
                {
                    "fold": float(fold),
                    "training_observations": float(len(deep)),
                    "deep_to_shallow_positive_scale": scale,
                }
            )
        for index in test:
            pair = pairs[int(index)]
            residuals[index] = scale * pair.d_deep - pair.d_shallow
    return residuals, fits


def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    sorted_y = y[order]
    unique_x, inverse = np.unique(sorted_x, return_inverse=True)
    sums = np.bincount(inverse, weights=sorted_y)
    weights = np.bincount(inverse).astype(float)
    levels = sums / weights

    blocks: list[list[float]] = []
    for index, (level, weight) in enumerate(zip(levels, weights, strict=True)):
        blocks.append([float(index), float(index), float(weight), float(level)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[2] + right[2]
            merged_level = (left[2] * left[3] + right[2] * right[3]) / total_weight
            blocks.append([left[0], right[1], total_weight, merged_level])

    fitted = np.empty(len(unique_x), dtype=float)
    for start, end, _weight, level in blocks:
        fitted[int(start) : int(end) + 1] = level
    return unique_x, fitted


def isotonic_predict(model: tuple[np.ndarray, np.ndarray], x: float) -> float:
    support, fitted = model
    return float(np.interp(float(x), support, fitted, left=fitted[0], right=fitted[-1]))


def crossfit_isotonic_residuals(pairs: list[PairSummary]) -> np.ndarray:
    """Map deep margins to the shallow scale using intact controls only."""

    residuals = np.empty(len(pairs), dtype=float)
    folds = np.asarray([fold_for_cluster(pair.cluster) for pair in pairs], dtype=int)
    for fold in sorted(set(folds.tolist())):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        deep = np.asarray([pairs[i].original_deep for i in train], dtype=float)
        shallow = np.asarray([pairs[i].original_shallow for i in train], dtype=float)
        model = isotonic_fit(deep, shallow)
        for index in test:
            pair = pairs[int(index)]
            scaled_original = isotonic_predict(model, pair.original_deep)
            scaled_swap = isotonic_predict(model, pair.swap_deep)
            residuals[index] = (
                scaled_original - scaled_swap - pair.d_shallow
            )
    return residuals


def crossfit_metric(
    pairs: list[PairSummary],
    draws: list[np.ndarray],
    residual_function: Callable[[list[PairSummary]], np.ndarray],
) -> dict[str, Any]:
    point = float(np.mean(residual_function(pairs)))
    estimates = [
        float(np.mean(residual_function([pairs[index] for index in draw])))
        for draw in draws
    ]
    return {
        "estimate": point,
        "ci95": percentile_interval(estimates),
        "bootstrap_replicates": len(draws),
        "calibration_refit_within_each_bootstrap_draw": True,
    }


def analyze_source(source: Source) -> dict[str, Any]:
    rows = load_rows(source.paths)
    pairs, metadata = summarize_pairs(source, rows)
    primary_draws = bootstrap_draws(pairs, source.primary_bootstrap, source.key)
    sensitivity_draws = None
    if source.dataset == "MuSiQue":
        sensitivity_draws = bootstrap_draws(
            pairs, "cluster_then_pair", f"{source.key}:terminal-relation"
        )

    def m(attribute: str) -> dict[str, Any]:
        return metric(pairs, primary_draws, lambda sample: array_mean(sample, attribute))

    _positive_scale_residuals, positive_scale_fits = (
        crossfit_positive_scale_residuals(pairs)
    )

    def positive_scale_values(sample: list[PairSummary]) -> np.ndarray:
        return crossfit_positive_scale_residuals(sample, include_fits=False)[0]

    def isotonic_values(sample: list[PairSummary]) -> np.ndarray:
        return crossfit_isotonic_residuals(sample)

    output: dict[str, Any] = {
        "model": source.model,
        "dataset": source.dataset,
        "shallow_K": source.shallow_k,
        "deep_K": source.deep_k,
        "inputs": [
            {"path": path.name, "sha256": sha256_file(path)} for path in source.paths
        ],
        "data": metadata,
        "bootstrap": {
            "primary": source.primary_bootstrap,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_salt": SEED_SALT,
            "MuSiQue_sensitivity": "terminal_relation_then_pair"
            if sensitivity_draws is not None
            else None,
        },
        "raw_margin": {
            "original_shallow": m("original_shallow"),
            "swap_shallow": m("swap_shallow"),
            "original_deep": m("original_deep"),
            "swap_deep": m("swap_deep"),
            "D_shallow": m("d_shallow"),
            "D_deep": m("d_deep"),
            "depth_gain": metric(
                pairs,
                primary_draws,
                lambda sample: mean(pair.d_deep - pair.d_shallow for pair in sample),
            ),
        },
        "exact_first_divergence_token": {
            "definition": (
                "Reference-minus-other log probability at the first candidate token "
                "where token IDs diverge. The context is identical at that position, "
                "so the value is exactly the corresponding token-logit difference."
            ),
            "original_shallow": m("original_first_divergence_shallow"),
            "swap_shallow": m("swap_first_divergence_shallow"),
            "original_deep": m("original_first_divergence_deep"),
            "swap_deep": m("swap_first_divergence_deep"),
            "D_shallow": m("first_divergence_d_shallow"),
            "D_deep": m("first_divergence_d_deep"),
            "continuous_depth_gain": metric(
                pairs,
                primary_draws,
                lambda sample: mean(
                    pair.first_divergence_d_deep
                    - pair.first_divergence_d_shallow
                    for pair in sample
                ),
            ),
            "graph_directed_state_accuracy_shallow": m(
                "first_divergence_state_accuracy_shallow"
            ),
            "graph_directed_state_accuracy_deep": m(
                "first_divergence_state_accuracy_deep"
            ),
            "choice_control_shallow": m(
                "first_divergence_choice_control_shallow"
            ),
            "choice_control_deep": m("first_divergence_choice_control_deep"),
            "choice_control_gain": metric(
                pairs,
                primary_draws,
                lambda sample: mean(
                    pair.first_divergence_choice_control_deep
                    - pair.first_divergence_choice_control_shallow
                    for pair in sample
                ),
            ),
        },
        "scale_invariant_choice": {
            "scope_note": (
                "Sign of the saved complete-string pairwise margin. This is invariant "
                "to positive rescaling of that margin, but it is not an exact token-level "
                "temperature intervention when candidate token lengths differ."
            ),
            "graph_directed_state_accuracy_shallow": m("state_accuracy_shallow"),
            "graph_directed_state_accuracy_deep": m("state_accuracy_deep"),
            "state_accuracy_gain": metric(
                pairs,
                primary_draws,
                lambda sample: mean(
                    pair.state_accuracy_deep - pair.state_accuracy_shallow
                    for pair in sample
                ),
            ),
            "choice_control_shallow": m("choice_control_shallow"),
            "choice_control_deep": m("choice_control_deep"),
            "choice_control_gain": metric(
                pairs,
                primary_draws,
                lambda sample: mean(
                    pair.choice_control_deep - pair.choice_control_shallow
                    for pair in sample
                ),
            ),
            "joint_follow_rate_shallow": m("joint_follow_shallow"),
            "joint_follow_rate_deep": m("joint_follow_deep"),
            "joint_follow_rate_gain": metric(
                pairs,
                primary_draws,
                lambda sample: mean(
                    pair.joint_follow_deep - pair.joint_follow_shallow
                    for pair in sample
                ),
            ),
        },
        "scale_normalized_margin": {
            "original_state_row_RMS_normalized_D_shallow": metric(
                pairs,
                primary_draws,
                lambda sample: row_rms_normalized_d(sample, "shallow", "original"),
            ),
            "original_state_row_RMS_normalized_D_deep": metric(
                pairs,
                primary_draws,
                lambda sample: row_rms_normalized_d(sample, "deep", "original"),
            ),
            "original_state_row_RMS_normalized_gain": metric(
                pairs,
                primary_draws,
                lambda sample: row_rms_normalized_d(sample, "deep", "original")
                - row_rms_normalized_d(sample, "shallow", "original"),
            ),
            "pooled_state_row_RMS_normalized_gain": metric(
                pairs,
                primary_draws,
                lambda sample: row_rms_normalized_d(sample, "deep", "pooled")
                - row_rms_normalized_d(sample, "shallow", "pooled"),
            ),
            "paired_standardized_D_shallow": metric(
                pairs,
                primary_draws,
                lambda sample: paired_standardized_d(sample, "shallow"),
            ),
            "paired_standardized_D_deep": metric(
                pairs,
                primary_draws,
                lambda sample: paired_standardized_d(sample, "deep"),
            ),
            "paired_standardized_gain": metric(
                pairs,
                primary_draws,
                lambda sample: paired_standardized_d(sample, "deep")
                - paired_standardized_d(sample, "shallow"),
            ),
        },
        "crossfit_positive_scale_control": {
            "interpretation": (
                "Deep path contrast after mapping deep margins to the shallow scale "
                "with one nonnegative, zero-intercept scale fit only on held-out intact "
                "controls, minus the shallow path contrast."
            ),
            "residual_depth_gain": crossfit_metric(
                pairs, primary_draws, positive_scale_values
            ),
            "point_estimate_fold_fits": positive_scale_fits,
        },
        "crossfit_monotone_calibration_sensitivity": {
            "interpretation": (
                "Sensitivity only: deep margins are mapped to the shallow scale by a "
                "five-fold relation-cluster-held-out isotonic map fit on intact controls."
            ),
            "residual_depth_gain": crossfit_metric(
                pairs, primary_draws, isotonic_values
            ),
        },
    }

    if sensitivity_draws is not None:
        output["MuSiQue_terminal_relation_cluster_sensitivity"] = {
            "choice_control_gain": metric(
                pairs,
                sensitivity_draws,
                lambda sample: mean(
                    pair.choice_control_deep - pair.choice_control_shallow
                    for pair in sample
                ),
            ),
            "first_divergence_choice_control_gain": metric(
                pairs,
                sensitivity_draws,
                lambda sample: mean(
                    pair.first_divergence_choice_control_deep
                    - pair.first_divergence_choice_control_shallow
                    for pair in sample
                ),
            ),
            "original_state_row_RMS_normalized_gain": metric(
                pairs,
                sensitivity_draws,
                lambda sample: row_rms_normalized_d(sample, "deep", "original")
                - row_rms_normalized_d(sample, "shallow", "original"),
            ),
            "positive_scale_residual_depth_gain": crossfit_metric(
                pairs, sensitivity_draws, positive_scale_values
            ),
        }
    return output


def format_metric(value: dict[str, Any], digits: int = 3) -> str:
    estimate = float(value["estimate"])
    lower, upper = map(float, value["ci95"])
    return f"{estimate:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def write_markdown(results: dict[str, Any], path: Path) -> None:
    lines = [
        "# Scale-invariant path-control analysis",
        "",
        "Status: retrospective reviewer-driven analysis of previously scored rows.",
        "",
        "The registered raw-margin analysis is unchanged. This analysis tests whether its",
        "interpretation survives exact same-context token decisions, complete-answer choices,",
        "scale-normalized margins, and held-out common calibration maps.",
        "",
        "| Checkpoint | Dataset | Raw gain | First-divergence choice gain | Complete-answer choice gain | Original-RMS gain | Positive-scale residual |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, result in results["results"].items():
        raw = result["raw_margin"]["depth_gain"]
        first_divergence = result["exact_first_divergence_token"][
            "choice_control_gain"
        ]
        choice = result["scale_invariant_choice"]["choice_control_gain"]
        normalized = result["scale_normalized_margin"][
            "original_state_row_RMS_normalized_gain"
        ]
        scale_residual = result["crossfit_positive_scale_control"][
            "residual_depth_gain"
        ]
        lines.append(
            f"| {result['model']} | {result['dataset']} | {format_metric(raw)} | "
            f"{format_metric(first_divergence)} | {format_metric(choice)} | "
            f"{format_metric(normalized)} | {format_metric(scale_residual)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- The first-divergence decision compares two token logits under exactly the same context. Its sign is invariant even to row-specific positive temperature scaling.",
            "- A positive complete-answer choice-control gain cannot be produced by positive rescaling of the saved pairwise margin alone.",
            "- A positive RMS-normalized gain indicates stronger path separation relative to intact-control margin scale.",
            "- A positive held-out scale residual indicates that one common positive scaling fit on intact controls cannot explain the graph-state interaction.",
            "- These are retrospective robustness analyses. They strengthen or narrow interpretation, but do not become preregistered outcomes after the fact.",
            "- The exact token result concerns the first candidate-identifying decision. It does not re-temperature later tokens after candidate prefixes diverge.",
            "",
            "## Dataset-specific details",
            "",
        ]
    )
    for key, result in results["results"].items():
        lines.extend(
            [
                f"### {result['model']}, {result['dataset']}",
                "",
                f"- Pairs: {result['data']['pairs']}",
                f"- Relation clusters: {result['data']['clusters']}",
                f"- Original margin, shallow: {format_metric(result['raw_margin']['original_shallow'])}",
                f"- Swap margin, shallow: {format_metric(result['raw_margin']['swap_shallow'])}",
                f"- Original margin, deep: {format_metric(result['raw_margin']['original_deep'])}",
                f"- Swap margin, deep: {format_metric(result['raw_margin']['swap_deep'])}",
                f"- Graph-directed state accuracy gain: {format_metric(result['scale_invariant_choice']['state_accuracy_gain'])}",
                f"- First-divergence choice-control gain: {format_metric(result['exact_first_divergence_token']['choice_control_gain'])}",
                f"- Joint-following gain: {format_metric(result['scale_invariant_choice']['joint_follow_rate_gain'])}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sources(repo: Path) -> list[Source]:
    data = repo / "data" / "analysis_ready" / "natural"
    return [
        Source(
            key="ouro26_2wiki",
            model="Ouro-2.6B",
            dataset="2Wiki",
            paths=(data / "ouro26_2wiki.json",),
            shallow_k=1,
            deep_k=3,
            primary_bootstrap="cluster_then_pair",
            expected_rows=5_952,
            expected_pairs=124,
            expected_clusters=31,
        ),
        Source(
            key="ouro26_musique",
            model="Ouro-2.6B",
            dataset="MuSiQue",
            paths=(data / "ouro26_musique.json",),
            shallow_k=1,
            deep_k=3,
            primary_bootstrap="pair",
            expected_rows=4_560,
            expected_pairs=95,
            expected_clusters=30,
        ),
        Source(
            key="loopus_2wiki",
            model="LoopUS Qwen3-1.7B",
            dataset="2Wiki",
            paths=(data / "loopus_2wiki.json",),
            shallow_k=1,
            deep_k=8,
            primary_bootstrap="cluster_then_pair",
            expected_rows=1_860,
            expected_pairs=93,
            expected_clusters=31,
        ),
        Source(
            key="loopus_musique",
            model="LoopUS Qwen3-1.7B",
            dataset="MuSiQue",
            paths=(data / "loopus_musique.json",),
            shallow_k=1,
            deep_k=8,
            primary_bootstrap="pair",
            expected_rows=1_900,
            expected_pairs=95,
            expected_clusters=30,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduced" / "results",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_results: dict[str, Any] = {}
    for source in sources(repo):
        if not source.paths or any(not path.exists() for path in source.paths):
            raise FileNotFoundError(f"missing source artifacts for {source.key}")
        source_results[source.key] = analyze_source(source)

    payload = {
        "schema": "iclr2027.scale_invariant_path_control.v2",
        "status": "retrospective_reviewer_driven_analysis",
        "claim_boundary": (
            "Tests robustness of the depth-by-path interaction to confidence scaling. "
            "The strongest exact control concerns the first candidate-identifying token; "
            "the analysis does not retroactively alter registered endpoints or establish "
            "recurrence specificity."
        ),
        "results": source_results,
    }
    json_path = args.output_dir / "scale_invariant_path_control.json"
    markdown_path = args.output_dir / "scale_invariant_path_control.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, markdown_path)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
