#!/usr/bin/env python3
"""Competence-only screen of Qwen3-8B LoopUS on the frozen nonce grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "iclr2027.loopus8_nonce_competence.run.v1"
MODEL_REPOSITORY = "Thrillcrazyer/Qwen3-8B_LoopUS"
MODEL_REVISION = "4424cfd8e36c77fee337a04602e3ddf6faae533f"
SOURCE_REVISION = "dd655b77861d9edf8fb981e90763436b95b1f3ff"
GATE_DEPTH = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_binding(snapshot: Path, source: Path) -> dict[str, Any]:
    import subprocess

    snapshot = snapshot.resolve()
    source = source.resolve()
    if not snapshot.is_dir() or snapshot.name != MODEL_REVISION:
        raise RuntimeError("LoopUS-8B snapshot is not the pinned immutable revision")
    source_head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_head != SOURCE_REVISION:
        raise RuntimeError("LoopUS source revision changed")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "lds"
        or "Qwen3-8B" not in str(config.get("base_model_name_or_path"))
        or int(config.get("N", 0)) < GATE_DEPTH
        or len(config.get("encoder_layer_indices", [])) < 1
        or len(config.get("reasoning_layer_indices", [])) < 1
        or len(config.get("decoder_layer_indices", [])) < 1
    ):
        raise RuntimeError("LoopUS-8B architecture contract changed")
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("LoopUS-8B weights are missing")
    required_small_files = ("config.json", "tokenizer_config.json", "tokenizer.json")
    optional_small_files = ("chat_template.jinja", "generation_config.json")
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "snapshot_dir": str(snapshot),
        "source_dir": str(source),
        "source_revision": source_head,
        "architecture": {
            key: config.get(key)
            for key in (
                "N",
                "base_model_name_or_path",
                "encoder_layer_indices",
                "reasoning_layer_indices",
                "decoder_layer_indices",
            )
        },
        "weights_bytes": sum(path.stat().st_size for path in weights),
        "weight_blobs": [path.resolve().name for path in weights],
        "small_files_sha256": {
            name: sha256_file(snapshot / name)
            for name in (*required_small_files, *optional_small_files)
            if (snapshot / name).is_file()
        },
    }


class LoopUS8Backend:
    def __init__(
        self, snapshot: Path, source: Path, device: str, system_prompt: str
    ) -> None:
        analysis_dir = Path(__file__).resolve().parent
        repo_scripts = Path(__file__).resolve().parents[4] / "scripts"
        for path in (source.resolve(), analysis_dir, repo_scripts):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        # Transformers 5.3 calls the configuration class with no arguments
        # while formatting a loaded config for logging.  LoopUS's no-argument
        # default performs an unrelated Hub lookup for Qwen3-0.6B.  The saved
        # 8B checkpoint already contains its complete ``base_config_dict``, so
        # declaring that this custom class has no safe default constructor
        # avoids the lookup without changing the loaded model configuration.
        from models.configuration_lds import LDSConfig

        LDSConfig.has_no_defaults_at_init = True
        import run_loopus_2wiki_bridge_screen as loopus

        self.binding = snapshot_binding(snapshot, source)
        self.binding["loader_compatibility"] = {
            "lds_has_no_defaults_at_init": True,
            "purpose": "avoid unrelated default Qwen3-0.6B config lookup",
        }
        self.scorer = loopus.LoopUSScorer(snapshot, source, device)
        self.system_prompt = system_prompt
        self.token_ids: dict[str, int] = {}

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
        token_counts: list[int] = []
        unique = sorted(set(prompts))
        for prompt in unique:
            rendered = self.render(prompt)
            prefix = self.scorer.tokenizer.encode(rendered, add_special_tokens=False)
            for label in ("A", "B"):
                extended = self.scorer.tokenizer.encode(
                    rendered + label, add_special_tokens=False
                )
                if extended[:-1] != prefix or len(extended) != len(prefix) + 1:
                    raise RuntimeError(f"label {label} is not a one-token extension")
                observed[label].add(int(extended[-1]))
            token_counts.append(len(prefix))
        if any(len(ids) != 1 for ids in observed.values()):
            raise RuntimeError("answer token IDs vary across prompts")
        self.token_ids = {label: next(iter(ids)) for label, ids in observed.items()}
        if self.token_ids["A"] == self.token_ids["B"]:
            raise RuntimeError("answer labels share a token ID")
        return {
            "snapshot_and_source": self.binding,
            "tokenization": {
                "prompts_checked": len(unique),
                "all_labels_exact_one_token_extensions": True,
                "target_token_ids": self.token_ids,
                "minimum_prompt_tokens": min(token_counts),
                "maximum_prompt_tokens": max(token_counts),
            },
        }

    def score(self, prompt: str, depth: int = GATE_DEPTH) -> dict[str, Any]:
        if depth < 1 or depth > int(self.binding["architecture"]["N"]):
            raise ValueError(f"invalid recurrent depth: {depth}")
        torch = self.scorer.torch
        rendered = self.render(prompt)
        prefix = self.scorer.tokenizer.encode(rendered, add_special_tokens=False)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.scorer.device)
        attention_mask = torch.ones_like(input_ids)
        self.scorer.model.N = depth
        self.scorer.model.config.N = depth
        self.scorer.model.reset_runtime_stats()
        with torch.inference_mode():
            result = self.scorer.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        runtime = self.scorer.model.get_runtime_stats()
        calls = int(runtime.get("forward_calls", 0))
        mean_steps = float(runtime.get("avg_reasoning_steps", float("nan")))
        if calls != 1 or mean_steps != float(depth):
            raise RuntimeError("LoopUS fixed-depth execution contract failed")
        logits = result.logits[0, -1].float()
        return {
            "raw_logits": {
                label: float(logits[token_id].item())
                for label, token_id in self.token_ids.items()
            },
            "target_token_ids": self.token_ids,
            "prompt_tokens": len(prefix),
            "executed_recurrent_steps": mean_steps,
            "readout_exit_K": depth,
            "trajectory_contract": f"one fixed-depth LoopUS forward at K{depth}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-worlds", type=int)
    args = parser.parse_args()

    analysis_dir = Path(__file__).resolve().parent
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    import run_nonce_path_control as parent

    panel = parent.load_panel(args.panel.resolve())
    world_ids = list(panel["competence_world_ids"])
    if args.max_worlds is not None:
        world_ids = world_ids[: args.max_worlds]
    allowed = set(world_ids)
    cells = sorted(
        [
            cell
            for cell in panel["cells"]
            if cell["world_id"] in allowed and cell["evidence_state"] == "original"
        ],
        key=lambda cell: cell["row_id"],
    )
    if len(cells) != len(world_ids) * 4:
        raise RuntimeError("competence prompt grid is incomplete")
    if args.mock:
        backend = None
        audit: dict[str, Any] = {"mock": True}
    else:
        if args.snapshot_dir is None or args.source_dir is None:
            raise SystemExit("--snapshot-dir and --source-dir are required")
        backend = LoopUS8Backend(
            args.snapshot_dir.resolve(),
            args.source_dir.resolve(),
            args.device,
            panel["system_prompt"],
        )
        audit = backend.prepare(cell["prompt"] for cell in cells)

    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        if backend is None:
            raw = {
                "raw_logits": {
                    "A": 2.0 if cell["expected_label"] == "A" else -2.0,
                    "B": 2.0 if cell["expected_label"] == "B" else -2.0,
                },
                "target_token_ids": {"A": 1, "B": 2},
                "prompt_tokens": len(cell["prompt"].split()),
                "executed_recurrent_steps": GATE_DEPTH,
                "readout_exit_K": GATE_DEPTH,
                "trajectory_contract": "competence mock",
            }
        else:
            raw = backend.score(cell["prompt"])
        rows.append(
            {
                **cell,
                "K": GATE_DEPTH,
                "row_id": f"{cell['row_id']}__K{GATE_DEPTH}",
                "score": parent.orient_score(cell, raw),
            }
        )
        if index % 25 == 0 or index == len(cells):
            print(f"[{index}/{len(cells)} prompts]", flush=True)

    summary = (
        parent.competence_summary(rows, "loopus")
        if len(world_ids) == 48
        else {
            "decision": "TECHNICAL_SMOKE_COMPLETE",
            "worlds": len(world_ids),
            "rows": len(rows),
            "treatment_locked": True,
        }
    )
    artifact = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "model_key": "loopus8",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "mode": "competence" if len(world_ids) == 48 else "smoke",
        "gate_depth": GATE_DEPTH,
        "mock": args.mock,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "panel": {
            "path": str(args.panel.resolve()),
            "canonical_sha256": panel["panel_sha256"],
        },
        "backend_audit": audit,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
