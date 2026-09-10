#!/usr/bin/env python3
"""Run the prospective fictional-graph path-control panel.

Artifacts are written to the explicitly supplied output path and analyzed only
after all requested shards finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np


SCHEMA = "iclr2027.nonce_path_control.run.v1"
PANEL_FILE_SHA256 = "f176ce9f06b841feeb33d819536a3bf68a47aaa3bb11a46a9a7e9724af201a16"
PANEL_CANONICAL_SHA256 = "7880c02c64af541629b9c23dce15bf4a6df98c4f99a8aed45b4a310df2e543e5"
PANEL_SCHEMA = "iclr2027.nonce_path_control.panel.v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SALT = "iclr2027-nonce-path-control-v1:competence"

MODEL_DEPTHS = {
    "ouro26": (1, 2, 3, 4),
    "ouro14": (1, 2, 3, 4),
    "loopus": tuple(range(1, 9)),
    "tfqwen": (1, 2, 4, 6),
}
MODEL_GATE_DEPTH = {"ouro26": 4, "ouro14": 4, "loopus": 8, "tfqwen": 1}
MODEL_DEEP_DEPTH = {"ouro26": 4, "ouro14": 4, "loopus": 8, "tfqwen": 6}
TFQWEN_LOOP_REPOSITORY = "https://github.com/L-z-Chen/Training-Free-Looped-Transformer"
TFQWEN_LOOP_REVISION = "2b20a873f05b3db6ba1d306ba6fe22ce8f34baf0"
TFQWEN_LOOP_WRAPPER_SHA256 = (
    "47b648fde769d983a74311178b87f07da0f81c1b26f7bdbe95359f53cc9cc7d7"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SALT}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_panel(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PANEL_FILE_SHA256:
        raise RuntimeError("frozen nonce panel file hash changed")
    panel = json.loads(path.read_text(encoding="utf-8"))
    if panel.get("schema") != PANEL_SCHEMA:
        raise RuntimeError("unexpected nonce panel schema")
    stored = panel.pop("panel_sha256", None)
    observed = canonical_json_sha256(panel)
    panel["panel_sha256"] = stored
    if stored != PANEL_CANONICAL_SHA256 or observed != stored:
        raise RuntimeError("frozen nonce panel canonical hash changed")
    if len(panel.get("worlds", [])) != 288 or len(panel.get("cells", [])) != 3456:
        raise RuntimeError("frozen nonce panel dimensions changed")
    return panel


def import_scripts() -> Path:
    configured = os.environ.get("EVIDENCE_LOOPS_ADAPTER_DIR")
    if not configured:
        raise RuntimeError(
            "real model execution requires EVIDENCE_LOOPS_ADAPTER_DIR to point "
            "to the separately obtained model-adapter source directory"
        )
    scripts = Path(configured).resolve()
    if not scripts.is_dir():
        raise RuntimeError("EVIDENCE_LOOPS_ADAPTER_DIR is not a directory")
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return scripts


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def credit(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return 0.0
    return 0.5


class OuroBackend:
    def __init__(
        self,
        model_key: str,
        snapshot_dir: Path,
        device: str,
        offline: bool,
        system_prompt: str,
    ) -> None:
        import_scripts()
        import run_ouro14_2wiki_counterfactual_scale_transfer as ouro14_binding
        import run_ouro_semantic_time_pilot as ouro
        import run_ouro_twohop_difficulty_ladder as ladder

        if model_key == "ouro26":
            import ouro26_likelihood_adapter as ouro26_binding

            self.snapshot_binding = ouro26_binding.validate_snapshot_dir(snapshot_dir)
        elif model_key == "ouro14":
            self.snapshot_binding = ouro14_binding.validate_snapshot_dir(snapshot_dir)
        else:
            raise ValueError(model_key)
        ouro.design.SYSTEM_PROMPT = system_prompt
        ouro.design.ANSWER_LABELS = ("A", "B")
        self.scorer = ouro.OuroScorer(snapshot_dir, device, offline)
        self.actual_shared_logits = ladder.actual_shared_logits
        self.model_key = model_key
        self.audit: dict[str, Any] = {}

    def prepare(self, prompts: Iterable[str]) -> dict[str, Any]:
        prompt_list = sorted(set(prompts))
        tokenization = self.scorer.validate_label_continuations(prompt_list)
        parity = self.scorer.validate_intermediate_readouts(prompt_list[0])
        if not parity.get("passes"):
            raise RuntimeError("Ouro native readout parity failed")
        self.audit = {
            "tokenization": tokenization,
            "native_readout_parity": parity,
            "snapshot": self.snapshot_binding,
        }
        return self.audit

    def score(self, prompt: str, depths: tuple[int, ...]) -> dict[int, dict[str, Any]]:
        logits_by_k, prompt_tokens = self.actual_shared_logits(self.scorer, prompt)
        token_ids = self.scorer.label_token_ids
        output: dict[int, dict[str, Any]] = {}
        for k in depths:
            logits = logits_by_k[k].float()
            log_probs = self.scorer.torch.log_softmax(logits, dim=-1)
            output[k] = {
                "raw_logits": {
                    label: float(logits[token_ids[label]].item()) for label in ("A", "B")
                },
                "full_vocabulary_log_probabilities": {
                    label: float(log_probs[token_ids[label]].item())
                    for label in ("A", "B")
                },
                "target_token_ids": dict(token_ids),
                "prompt_tokens": prompt_tokens,
                "executed_recurrent_steps": 4,
                "readout_exit_K": k,
                "trajectory_contract": (
                    "one native four-step trajectory with all intermediate exits retained"
                ),
            }
        return output


class LoopUSBackend:
    def __init__(
        self,
        snapshot_dir: Path,
        source_dir: Path,
        device: str,
        system_prompt: str,
    ) -> None:
        import_scripts()
        import run_loopus_2wiki_bridge_screen as loopus

        self.binding = {
            "snapshot": loopus.validate_snapshot(snapshot_dir),
            "source": loopus.validate_loopus_source(source_dir),
        }
        self.scorer = loopus.LoopUSScorer(snapshot_dir, source_dir, device)
        self.system_prompt = system_prompt
        self.token_ids: dict[str, int] = {}
        self.audit: dict[str, Any] = {}

    def render(self, prompt: str) -> str:
        return self.scorer.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    def prepare(self, prompts: Iterable[str]) -> dict[str, Any]:
        observed: dict[str, set[int]] = {"A": set(), "B": set()}
        unique_prompts = sorted(set(prompts))
        token_counts: list[int] = []
        for prompt in unique_prompts:
            rendered = self.render(prompt)
            prefix = self.scorer.tokenizer.encode(rendered, add_special_tokens=False)
            direct = self.scorer.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
                return_dict=False,
            )
            if list(direct) != prefix:
                raise RuntimeError("LoopUS chat renderings disagree")
            for label in ("A", "B"):
                extended = self.scorer.tokenizer.encode(
                    rendered + label, add_special_tokens=False
                )
                if extended[:-1] != prefix or len(extended) != len(prefix) + 1:
                    raise RuntimeError(f"LoopUS label {label} is not a one-token extension")
                observed[label].add(int(extended[-1]))
            token_counts.append(len(prefix))
        if any(len(ids) != 1 for ids in observed.values()):
            raise RuntimeError("LoopUS answer-label token IDs vary by prompt")
        self.token_ids = {label: next(iter(ids)) for label, ids in observed.items()}
        if self.token_ids["A"] == self.token_ids["B"]:
            raise RuntimeError("LoopUS answer labels share a token ID")
        self.audit = {
            **self.binding,
            "tokenization": {
                "prompts_checked": len(unique_prompts),
                "all_labels_exact_one_token_extensions": True,
                "target_token_ids": dict(self.token_ids),
                "minimum_prompt_tokens": min(token_counts),
                "maximum_prompt_tokens": max(token_counts),
            },
        }
        return self.audit

    def score(self, prompt: str, depths: tuple[int, ...]) -> dict[int, dict[str, Any]]:
        torch = self.scorer.torch
        rendered = self.render(prompt)
        prefix = self.scorer.tokenizer.encode(rendered, add_special_tokens=False)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.scorer.device)
        attention_mask = torch.ones_like(input_ids)
        output: dict[int, dict[str, Any]] = {}
        for k in depths:
            self.scorer.model.N = k
            self.scorer.model.config.N = k
            self.scorer.model.reset_runtime_stats()
            with torch.inference_mode():
                result = self.scorer.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            runtime = self.scorer.model.get_runtime_stats()
            calls = int(runtime.get("forward_calls", 0))
            mean_steps = float(runtime.get("avg_reasoning_steps", float("nan")))
            if calls != 1 or mean_steps != float(k):
                raise RuntimeError(
                    f"LoopUS fixed-depth execution failed at K{k}: {calls}, {mean_steps}"
                )
            logits = result.logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            output[k] = {
                "raw_logits": {
                    label: float(logits[self.token_ids[label]].item())
                    for label in ("A", "B")
                },
                "full_vocabulary_log_probabilities": {
                    label: float(log_probs[self.token_ids[label]].item())
                    for label in ("A", "B")
                },
                "target_token_ids": dict(self.token_ids),
                "prompt_tokens": len(prefix),
                "executed_recurrent_steps": mean_steps,
                "readout_exit_K": k,
                "trajectory_contract": "one fixed-depth LoopUS forward per prompt and K",
            }
        return output


class TrainingFreeQwenBackend:
    def __init__(
        self,
        snapshot_dir: Path,
        loop_module: Path,
        device: str,
        dtype: str,
        offline: bool,
        system_prompt: str,
    ) -> None:
        import_scripts()
        import run_looped_evidence_conservation_pilot as tfqwen

        if loop_module.name != "loop_qwen3.py":
            raise RuntimeError("training-free loop module must be loop_qwen3.py")
        if sha256_file(loop_module) != TFQWEN_LOOP_WRAPPER_SHA256:
            raise RuntimeError("training-free loop wrapper hash changed")
        if snapshot_dir.name != "cdbee75f17c01a7cc42f958dc650907174af0554":
            raise RuntimeError("training-free Qwen snapshot revision changed")
        tfqwen.SYSTEM_PROMPT = system_prompt
        self.module = tfqwen
        self.scorer = tfqwen.LoopedQwenScorer(
            snapshot_dir, loop_module, device, dtype, offline
        )
        self.binding = {
            "repository": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": snapshot_dir.name,
            "snapshot_dir": str(snapshot_dir),
            "loop_repository": TFQWEN_LOOP_REPOSITORY,
            "loop_revision": TFQWEN_LOOP_REVISION,
            "loop_wrapper_sha256": TFQWEN_LOOP_WRAPPER_SHA256,
            "dtype": dtype,
        }
        self.audit: dict[str, Any] = {}

    def prepare(self, prompts: Iterable[str]) -> dict[str, Any]:
        unique_prompts = sorted(set(prompts))
        tokenization = self.scorer.validate_label_continuations(unique_prompts)
        parity = self.scorer.verify_k1_parity(unique_prompts[0])
        if not parity.get("torch_allclose"):
            raise RuntimeError("training-free Qwen K1 wrapper parity failed")
        self.audit = {
            "snapshot_and_wrapper": self.binding,
            "tokenization": tokenization,
            "K1_wrapper_parity": parity,
        }
        return self.audit

    def score(self, prompt: str, depths: tuple[int, ...]) -> dict[int, dict[str, Any]]:
        output: dict[int, dict[str, Any]] = {}
        for k in depths:
            description = self.scorer.set_loop_count(k)
            score = self.scorer.score(prompt)
            output[k] = {
                "raw_logits": score["raw_logits"],
                "target_token_ids": dict(self.scorer.label_token_ids),
                "prompt_tokens": int(score["prompt_tokens"]),
                "label_pair_total_vocabulary_probability": score[
                    "label_pair_total_vocabulary_probability"
                ],
                "executed_recurrent_steps": k,
                "readout_exit_K": k,
                "trajectory_contract": description,
            }
        return output


def make_backend(args: argparse.Namespace, panel: dict[str, Any]) -> Any:
    if args.mock:
        return None
    if args.snapshot_dir is None:
        raise SystemExit("--snapshot-dir is required for model inference")
    snapshot = args.snapshot_dir.resolve()
    if args.model in {"ouro26", "ouro14"}:
        return OuroBackend(
            args.model, snapshot, args.device, args.offline, panel["system_prompt"]
        )
    if args.model == "loopus":
        if args.loopus_source is None:
            raise SystemExit("--loopus-source is required for LoopUS")
        return LoopUSBackend(
            snapshot, args.loopus_source.resolve(), args.device, panel["system_prompt"]
        )
    if args.model == "tfqwen":
        if args.loop_module is None:
            raise SystemExit("--loop-module is required for training-free Qwen")
        return TrainingFreeQwenBackend(
            snapshot,
            args.loop_module.resolve(),
            args.device,
            args.dtype,
            args.offline,
            panel["system_prompt"],
        )
    raise ValueError(args.model)


def selected_cells(
    panel: dict[str, Any], mode: str, model: str, shard_index: int, num_shards: int,
    smoke_worlds: int,
) -> list[tuple[dict[str, Any], tuple[int, ...]]]:
    world_index = {world["world_id"]: int(world["world_index"]) for world in panel["worlds"]}
    if mode == "competence":
        allowed_worlds = set(panel["competence_world_ids"])
        states = {"original"}
    elif mode == "smoke":
        allowed_worlds = {
            world["world_id"] for world in panel["worlds"][:smoke_worlds]
        }
        states = {"original", "bridge_swap", "topology_repair"}
    elif mode == "full":
        allowed_worlds = {world["world_id"] for world in panel["worlds"]}
        states = {"original", "bridge_swap", "topology_repair"}
    else:
        raise ValueError(mode)

    selected: list[tuple[dict[str, Any], tuple[int, ...]]] = []
    for cell in panel["cells"]:
        if cell["world_id"] not in allowed_worlds or cell["evidence_state"] not in states:
            continue
        index = world_index[cell["world_id"]]
        if index % num_shards != shard_index:
            continue
        if mode == "competence":
            depths = (MODEL_GATE_DEPTH[model],)
        elif (
            cell["evidence_state"] == "topology_repair"
            and model not in {"ouro26", "ouro14"}
        ):
            depths = (MODEL_DEEP_DEPTH[model],)
        else:
            depths = MODEL_DEPTHS[model]
        selected.append((cell, depths))
    return sorted(selected, key=lambda item: item[0]["row_id"])


def validate_competence_unlock(
    path: Path, model: str, panel: dict[str, Any]
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("mode") != "competence"
        or artifact.get("model_key") != model
        or artifact.get("panel", {}).get("canonical_sha256") != panel["panel_sha256"]
        or artifact.get("summary", {}).get("passes_competence_gate") is not True
    ):
        raise RuntimeError("competence artifact does not unlock this model and panel")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "decision": artifact["summary"]["decision"],
    }


def mock_raw_score(cell: dict[str, Any], k: int) -> dict[str, Any]:
    expected = cell["expected_label"]
    reference = cell["reference_label"]
    state = cell["evidence_state"]
    depth_gain = 0.35 * (k - 1)
    strength = 1.5 + depth_gain
    if state == "topology_repair":
        strength *= 0.9
    logits = {"A": -strength, "B": -strength}
    logits[expected] = strength
    digest = hashlib.sha256(f"{cell['row_id']}:{k}".encode()).digest()
    perturbation = (digest[0] - 127.5) / 1000.0
    logits[reference] += perturbation
    return {
        "raw_logits": logits,
        "target_token_ids": {"A": 1, "B": 2},
        "prompt_tokens": len(cell["prompt"].split()),
        "executed_recurrent_steps": k,
        "readout_exit_K": k,
        "trajectory_contract": "deterministic mock",
    }


def orient_score(cell: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    reference_label = cell["reference_label"]
    other_label = cell["other_label"]
    logits = raw["raw_logits"]
    margin = float(logits[reference_label]) - float(logits[other_label])
    reference_credit = credit(margin)
    state_credit = (
        reference_credit
        if cell["expected_label"] == reference_label
        else 1.0 - reference_credit
    )
    return {
        **raw,
        "reference_label": reference_label,
        "other_label": other_label,
        "expected_label": cell["expected_label"],
        "reference_logit_margin": margin,
        "reference_pair_probability": sigmoid(margin),
        "reference_argmax_credit": reference_credit,
        "state_correct_argmax_credit": state_credit,
    }


def competence_summary(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    expected = 48 * 2 * 2
    if len(rows) != expected:
        raise RuntimeError(f"competence row count changed: {len(rows)} != {expected}")
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_world[row["world_id"]].append(row)
    if len(by_world) != 48 or any(len(values) != 4 for values in by_world.values()):
        raise RuntimeError("competence world grid is incomplete")
    world_accuracy = {
        world: mean(float(row["score"]["state_correct_argmax_credit"]) for row in values)
        for world, values in by_world.items()
    }
    by_relation: dict[int, list[str]] = defaultdict(list)
    for world, values in by_world.items():
        by_relation[int(values[0]["relation_family_index"])].append(world)
    if set(map(len, by_relation.values())) != {8} or len(by_relation) != 6:
        raise RuntimeError("competence relation-family balance changed")
    rng = np.random.default_rng(stable_seed(model))
    bootstrap: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[float] = []
        for relation in sorted(by_relation):
            worlds = by_relation[relation]
            indices = rng.integers(0, len(worlds), size=len(worlds))
            sampled.extend(world_accuracy[worlds[int(index)]] for index in indices)
        bootstrap.append(float(np.mean(sampled)))
    point = float(np.mean(list(world_accuracy.values())))
    lower, upper = np.quantile(np.asarray(bootstrap), [0.025, 0.975]).tolist()
    arm_accuracy = {
        arm: mean(
            float(row["score"]["state_correct_argmax_credit"])
            for row in rows
            if row["query_arm"] == arm
        )
        for arm in ("A", "B")
    }
    order_accuracy = {
        order: mean(
            float(row["score"]["state_correct_argmax_credit"])
            for row in rows
            if row["candidate_order"] == order
        )
        for order in ("original", "reversed")
    }
    passes = bool(
        lower > 0.50
        and point >= 0.65
        and min(arm_accuracy.values()) >= 0.55
        and min(order_accuracy.values()) >= 0.55
    )
    return {
        "decision": "PROMOTE_TO_TREATMENT" if passes else "STOP_INCOMPETENT_SUBSTRATE",
        "passes_competence_gate": passes,
        "gate_depth": MODEL_GATE_DEPTH[model],
        "worlds": len(by_world),
        "rows": len(rows),
        "graph_correct_accuracy": {
            "estimate": point,
            "ci95": [float(lower), float(upper)],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "resampling": "48 worlds stratified within six relation families",
        },
        "query_arm_accuracy": arm_accuracy,
        "candidate_order_accuracy": order_accuracy,
        "requirements": {
            "lower95_strictly_above": 0.50,
            "point_at_least": 0.65,
            "each_query_arm_point_at_least": 0.55,
            "each_candidate_order_point_at_least": 0.55,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_DEPTHS), required=True)
    parser.add_argument("--mode", choices=("smoke", "competence", "full"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--loopus-source", type=Path)
    parser.add_argument("--loop-module", type=Path)
    parser.add_argument("--competence-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--smoke-worlds", type=int, default=2)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard coordinates")
    if args.mode == "competence" and (args.num_shards, args.shard_index) != (1, 0):
        raise SystemExit("competence runs are one complete unsharded artifact")
    if args.mode == "full" and args.competence_artifact is None:
        raise SystemExit("--competence-artifact is required for full treatment")

    panel = load_panel(args.panel.resolve())
    unlock = None
    if args.mode == "full":
        unlock = validate_competence_unlock(
            args.competence_artifact.resolve(), args.model, panel
        )
    scheduled = selected_cells(
        panel,
        args.mode,
        args.model,
        args.shard_index,
        args.num_shards,
        args.smoke_worlds,
    )
    if not scheduled:
        raise RuntimeError("selected schedule is empty")
    backend = make_backend(args, panel)
    prompts = [cell["prompt"] for cell, _ in scheduled]
    backend_audit = (
        {
            "mock": True,
            "tokenization": {"all_labels_exact_one_token_extensions": True},
        }
        if backend is None
        else backend.prepare(prompts)
    )

    rows: list[dict[str, Any]] = []
    for prompt_index, (cell, depths) in enumerate(scheduled, start=1):
        by_k = (
            {k: mock_raw_score(cell, k) for k in depths}
            if backend is None
            else backend.score(cell["prompt"], depths)
        )
        if set(by_k) != set(depths):
            raise RuntimeError("backend did not return every scheduled depth")
        for k in depths:
            rows.append(
                {
                    **cell,
                    "K": k,
                    "row_id": f"{cell['row_id']}__K{k}",
                    "score": orient_score(cell, by_k[k]),
                }
            )
        if prompt_index % 25 == 0 or prompt_index == len(scheduled):
            print(
                f"[{prompt_index}/{len(scheduled)} prompts; {len(rows)} scored rows]",
                flush=True,
            )

    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("scored row identifiers are not unique")
    if any(
        row["score"]["expected_label"] != row["expected_label"] for row in rows
    ):
        raise RuntimeError("score orientation changed expected labels")
    summary = (
        competence_summary(rows, args.model)
        if args.mode == "competence"
        else {
            "decision": "SMOKE_COMPLETE" if args.mode == "smoke" else "SHARD_COMPLETE",
            "prompt_cells": len(scheduled),
            "scored_rows": len(rows),
            "worlds": len({row["world_id"] for row in rows}),
            "depths": sorted({int(row["K"]) for row in rows}),
            "states": sorted({row["evidence_state"] for row in rows}),
            "outcome_analysis_deferred_until_merge": True,
        }
    )
    runner = Path(__file__).resolve()
    artifact = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model_key": args.model,
        "mock": args.mock,
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "panel": {
            "path": str(args.panel.resolve()),
            "file_sha256": PANEL_FILE_SHA256,
            "canonical_sha256": panel["panel_sha256"],
        },
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "competence_unlock": unlock,
        "backend_audit": backend_audit,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
