#!/usr/bin/env python3
"""Train one cell of a compute-matched tied/untied graph-memory factorial."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


SCHEMA = "iclr2027.tied_untied_graph_factorial.run.v1"
DEPTHS = (1, 2, 3, 4)
PATHS = 4
FOCAL_PATHS = 2
SOURCE_POOL = 32
BRIDGE_POOL = 32
TERMINALS = 4
SOURCE_OFFSET = 0
BRIDGE_OFFSET = SOURCE_OFFSET + SOURCE_POOL
TERMINAL_OFFSET = BRIDGE_OFFSET + BRIDGE_POOL
ENTITY_VOCABULARY = TERMINAL_OFFSET + TERMINALS
MEMORY_EDGES = PATHS * 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class World:
    world_id: int
    sources: tuple[int, ...]
    bridges: tuple[int, ...]
    terminals: tuple[int, ...]
    memory_order: tuple[int, ...]


def make_world(rng: np.random.Generator, world_id: int) -> World:
    sources = tuple(int(x) for x in rng.choice(SOURCE_POOL, PATHS, replace=False))
    bridges = tuple(int(x) for x in rng.choice(BRIDGE_POOL, PATHS, replace=False))
    terminals = tuple(int(x) for x in rng.permutation(TERMINALS))
    memory_order = tuple(int(x) for x in rng.permutation(MEMORY_EDGES))
    return World(world_id, sources, bridges, terminals, memory_order)


def graph_memory(world: World, swapped: bool) -> tuple[list[int], list[int]]:
    first_targets = list(world.bridges)
    if swapped:
        first_targets[0], first_targets[1] = first_targets[1], first_targets[0]
    edges: list[tuple[int, int]] = []
    for index in range(PATHS):
        edges.append(
            (
                SOURCE_OFFSET + world.sources[index],
                BRIDGE_OFFSET + first_targets[index],
            )
        )
    for index in range(PATHS):
        edges.append(
            (
                BRIDGE_OFFSET + world.bridges[index],
                TERMINAL_OFFSET + world.terminals[index],
            )
        )
    for index in range(PATHS):
        terminal = TERMINAL_OFFSET + world.terminals[index]
        edges.append((terminal, terminal))
    ordered = [edges[index] for index in world.memory_order]
    return [edge[0] for edge in ordered], [edge[1] for edge in ordered]


def encode(world: World, swapped: bool, query_arm: int) -> tuple[list[int], list[int], int, int, int]:
    subjects, objects = graph_memory(world, swapped)
    query = SOURCE_OFFSET + world.sources[query_arm]
    reached_path = 1 - query_arm if swapped else query_arm
    expected = world.terminals[reached_path]
    reference = world.terminals[query_arm]
    return subjects, objects, query, expected, reference


def make_dataset(
    examples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    subjects = np.empty((examples, MEMORY_EDGES), dtype=np.int64)
    objects = np.empty((examples, MEMORY_EDGES), dtype=np.int64)
    queries = np.empty(examples, dtype=np.int64)
    labels = np.empty(examples, dtype=np.int64)
    for index in range(examples):
        world = make_world(rng, index)
        sub, obj, query, label, _ = encode(
            world,
            swapped=bool(rng.integers(0, 2)),
            query_arm=int(rng.integers(0, FOCAL_PATHS)),
        )
        subjects[index] = sub
        objects[index] = obj
        queries[index] = query
        labels[index] = label
    return tuple(
        torch.from_numpy(array) for array in (subjects, objects, queries, labels)
    )


def evaluation_worlds(count: int, seed: int) -> list[World]:
    rng = np.random.default_rng(seed)
    return [make_world(rng, index) for index in range(count)]


class GraphHopBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        routing_init: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.post_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            identity = torch.eye(d_model)
            target = torch.cat([identity, identity, identity])
            self.attention.in_proj_weight.mul_(1.0 - routing_init).add_(
                target, alpha=routing_init
            )
            self.attention.in_proj_bias.zero_()
            self.attention.out_proj.weight.mul_(1.0 - routing_init).add_(
                identity, alpha=routing_init
            )
            self.attention.out_proj.bias.zero_()
            self.feed_forward[-1].weight.zero_()
            self.feed_forward[-1].bias.zero_()

    def forward(
        self, state: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(state),
            self.memory_norm(keys),
            values,
            need_weights=False,
        )
        state = self.dropout(attended)
        state = state + self.dropout(
            self.feed_forward(self.post_attention_norm(state))
        )
        return state


class GraphMemoryTransformer(nn.Module):
    def __init__(
        self,
        tying: str,
        d_model: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        routing_init: float,
    ) -> None:
        super().__init__()
        self.tying = tying
        self.entity_embedding = nn.Embedding(ENTITY_VOCABULARY, d_model)
        base = GraphHopBlock(d_model, heads, ff_mult, dropout, routing_init)
        if tying == "tied":
            self.shared_block = base
            self.blocks = None
        elif tying == "untied":
            self.shared_block = None
            self.blocks = nn.ModuleList([copy.deepcopy(base) for _ in DEPTHS])
        else:
            raise ValueError(tying)
        self.final_norm = nn.LayerNorm(d_model)
        self.logit_scale = d_model**-0.5

    def forward(
        self, subjects: torch.Tensor, objects: torch.Tensor, queries: torch.Tensor
    ) -> list[torch.Tensor]:
        keys = self.entity_embedding(subjects)
        values = self.entity_embedding(objects)
        state = self.entity_embedding(queries)[:, None, :]
        terminal_embeddings = self.entity_embedding.weight[
            TERMINAL_OFFSET : TERMINAL_OFFSET + TERMINALS
        ]
        outputs: list[torch.Tensor] = []
        for depth_index in range(len(DEPTHS)):
            block = self.shared_block if self.tying == "tied" else self.blocks[depth_index]
            assert block is not None
            state = block(state, keys, values)
            readout = self.final_norm(state[:, 0])
            outputs.append(self.logit_scale * readout @ terminal_embeddings.T)
        return outputs


def parameter_counts(model: GraphMemoryTransformer) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


@torch.no_grad()
def evaluate(
    model: GraphMemoryTransformer,
    worlds: list[World],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    subjects: list[list[int]] = []
    objects: list[list[int]] = []
    queries: list[int] = []
    metadata: list[tuple[int, bool, int, int, int, int]] = []
    for world in worlds:
        for swapped in (False, True):
            for arm in range(FOCAL_PATHS):
                sub, obj, query, label, reference = encode(world, swapped, arm)
                subjects.append(sub)
                objects.append(obj)
                queries.append(query)
                metadata.append(
                    (
                        world.world_id,
                        swapped,
                        arm,
                        label,
                        reference,
                        world.terminals[1 - arm],
                    )
                )
    logits_by_depth: list[list[np.ndarray]] = [[] for _ in DEPTHS]
    for start in range(0, len(subjects), batch_size):
        sub = torch.tensor(subjects[start : start + batch_size], device=device)
        obj = torch.tensor(objects[start : start + batch_size], device=device)
        query = torch.tensor(queries[start : start + batch_size], device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(sub, obj, query)
        for depth_index, logits in enumerate(outputs):
            logits_by_depth[depth_index].extend(logits.float().cpu().numpy())

    lookup: dict[tuple[int, bool, int, int], tuple[float, float, float]] = {}
    for row_index, meta in enumerate(metadata):
        world_id, swapped, arm, label, reference, other = meta
        for depth_index, k in enumerate(DEPTHS):
            logits = logits_by_depth[depth_index][row_index]
            margin = float(logits[reference] - logits[other])
            reference_credit = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            correct = 1.0 if int(np.argmax(logits)) == label else 0.0
            lookup[(world_id, swapped, arm, k)] = (
                margin,
                reference_credit,
                correct,
            )

    per_world: list[dict[str, Any]] = []
    for world in worlds:
        row: dict[str, Any] = {"world_id": world.world_id}
        for k in DEPTHS:
            original = [lookup[(world.world_id, False, arm, k)] for arm in range(2)]
            swapped = [lookup[(world.world_id, True, arm, k)] for arm in range(2)]
            row[f"raw_effect_K{k}"] = float(
                np.mean([value[0] for value in original])
                - np.mean([value[0] for value in swapped])
            )
            row[f"choice_effect_K{k}"] = float(
                np.mean([value[1] for value in original])
                - np.mean([value[1] for value in swapped])
            )
            row[f"state_accuracy_K{k}"] = float(
                np.mean([value[2] for value in (*original, *swapped)])
            )
        row["raw_gain_K4_minus_K1"] = row["raw_effect_K4"] - row["raw_effect_K1"]
        row["choice_gain_K4_minus_K1"] = (
            row["choice_effect_K4"] - row["choice_effect_K1"]
        )
        per_world.append(row)
    summary = {
        key: float(np.mean([row[key] for row in per_world]))
        for key in per_world[0]
        if key != "world_id"
    }
    return {"summary": summary, "per_world": per_world}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tying", choices=("tied", "untied"), required=True)
    parser.add_argument("--supervision", choices=("single", "multi"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-examples", type=int, default=100000)
    parser.add_argument("--eval-worlds", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--routing-init", type=float, default=0.60)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    if not 0.0 <= args.routing_init <= 1.0:
        raise SystemExit("--routing-init must lie in [0, 1]")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    train_sub, train_obj, train_query, train_y = make_dataset(
        args.train_examples, 10_000 + args.seed
    )
    eval_worlds = evaluation_worlds(args.eval_worlds, 90_000 + args.seed)
    model = GraphMemoryTransformer(
        args.tying,
        args.d_model,
        args.heads,
        args.ff_mult,
        args.dropout,
        args.routing_init,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(20_000 + args.seed)
    order = torch.randperm(args.train_examples, generator=generator)
    cursor = 0
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(args.train_examples, generator=generator)
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch_sub = train_sub[indices].to(device, non_blocking=True)
        batch_obj = train_obj[indices].to(device, non_blocking=True)
        batch_query = train_query[indices].to(device, non_blocking=True)
        batch_y = train_y[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(batch_sub, batch_obj, batch_query)
            losses = [nn.functional.cross_entropy(logits, batch_y) for logits in outputs]
            loss = losses[-1] if args.supervision == "single" else torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record: dict[str, float | int] = {
                "step": step,
                "loss": float(loss.detach().cpu()),
            }
            record.update(
                {
                    f"batch_accuracy_K{k}": float(
                        (outputs[index].argmax(dim=-1) == batch_y)
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                    for index, k in enumerate(DEPTHS)
                }
            )
            history.append(record)
            print(json.dumps(record), flush=True)

    evaluation = evaluate(model, eval_worlds, device, args.eval_batch_size)
    script = Path(__file__).resolve()
    artifact = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "condition": {"tying": args.tying, "supervision": args.supervision},
        "seed": args.seed,
        "script": {"path": str(script), "sha256": sha256_file(script)},
        "configuration": {
            key: value for key, value in vars(args).items() if key not in {"output", "device"}
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "task": {
            "paths": PATHS,
            "focal_paths": FOCAL_PATHS,
            "memory_edges": MEMORY_EDGES,
            "terminal_classes": TERMINALS,
            "terminal_self_loops_preserve_resolved_state": True,
        },
        "depths": list(DEPTHS),
        "matched_compute_contract": {
            "cross_attention_and_feedforward_applications": 4,
            "identical_tensor_shapes": True,
            "untied_blocks_initialized_as_exact_copies": True,
            "difference": "shared parameter object versus four independently updated copies",
        },
        "parameters": parameter_counts(model),
        "training_history": history,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluation["summary"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
