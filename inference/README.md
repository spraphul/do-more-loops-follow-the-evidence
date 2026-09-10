# Frozen inference runners

These files expose the generated-panel schedule, prompt rendering, competence
locks, score orientation, sharding, and execution checks used to create the
released model-score rows.

## Mock schedule check

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
- HRM-Text-1B: `sapientinc/HRM-Text-1B`, revision
  `1f82ac2b71222f0c100a224a33f24b44a3000b6d`.

All checkpoint, source, output, shard, and device locations are explicit CLI
arguments. The release never looks for credentials or a cache automatically.
Each runner documents its complete interface under `--help`.

The HRM release provides the generated panels, raw A/B logits, checkpoint and
runner bindings, and CPU analyses. Its GPU runner depends on the upstream HRM
implementation and is not included in the mock schedule target.

`run_ouro_surface_orbit.py` exposes the disjoint gate and eight-shard treatment
schedule for the fresh four-rendering confirmation. The released panel already
contains every generated prompt. Run its `smoke` mode without a checkpoint, or
its `competence` and `treatment` modes after supplying the pinned Ouro snapshot
and the external adapter directory described above.

The answer-boundary GPU replay is not presented as self-contained because its
prompts come from licensed natural-question panels. The anonymous release
instead provides a text-free row-level projection and the complete CPU analysis
needed to reproduce every reported boundary quantity.

The controlled training runner is self-contained once PyTorch is installed.
The released confirmation records use seeds 101, 211, 307, 401, 503, 601,
701, and 809, four tying-by-supervision conditions, 1,000 optimization steps,
width 64, four heads, 600 evaluation worlds, and zero dropout.

Mock outputs test runner wiring only and are not scientific results.
