# Frozen inference runners

These files expose the generated-panel schedule, prompt rendering, competence
locks, score orientation, sharding, and execution checks used to create the
released model-score rows.

## Safe local smoke test

Run `make mock-smoke` from the repository root. The command uses deterministic
mock logits and writes only to `reproduced/mock_inference/`. It validates that
the frozen panels and schedulers still agree.

## Real generated-panel scoring

Real runs require checkpoint files and the corresponding upstream model source
obtained under provider terms. The key bindings are:

- Ouro-2.6B: `ByteDance/Ouro-2.6B`, revision
  `1ed04250da1a9936042725d302e81c8fa2ab5abd`.
- LoopUS-8B: `Thrillcrazyer/Qwen3-8B_LoopUS`, revision
  `4424cfd8e36c77fee337a04602e3ddf6faae533f`.

All checkpoint, source, output, shard, and device locations are explicit CLI
arguments. The release never looks for credentials or a cache automatically.
Use `--help` on each runner for the complete interface.

The controlled training runner is self-contained once PyTorch is installed.
The released confirmation records use seeds 101, 211, 307, 401, 503, 601,
701, and 809, four tying-by-supervision conditions, 1,000 optimization steps,
width 64, four heads, 600 evaluation worlds, and zero dropout.

Mock outputs are wiring tests and must never be analyzed as scientific model
results.

