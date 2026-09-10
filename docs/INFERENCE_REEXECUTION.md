# Optional model re-execution

The central statistical claims can be reproduced from the released row scores
without downloading any checkpoint. This document describes the separate,
optional route for rescoring generated prompts.

## Why it is separate

Ouro and LoopUS require different upstream implementations and several large
checkpoint files. Their licenses and access requirements remain with the
original providers. No weights, authentication files, caches, or vendored
third-party model code are included here.

## Frozen runners

`inference/frozen/` contains the released Ouro and LoopUS panel schedulers,
competence gates, score orientation, sharding, and execution audits. The
runners accept explicit checkpoint and upstream-source paths; they never read
a token from this repository.

For HRM, the artifact releases both generated panels, raw A/B logits, gate and
runner bindings, full-vocabulary H2 parity checks, and the complete CPU
analyses. The source runner and result digests are recorded in
`data/PROVENANCE.json`. The optional GPU runner is not presented as a
self-contained replay because it depends on the upstream HRM implementation
and checkpoint.

The inexpensive schedule check is:

```bash
make mock-smoke
```

It runs deterministic mock scores through the real prompt selection,
orientation, row identifiers, and output schemas. It is a wiring check only,
not scientific evidence.

For real execution, create the optional environment:

```bash
python3.11 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install -r requirements-inference.txt
```

Then obtain the pinned checkpoint and upstream implementation directly from
the provider. Example commands and required revisions are listed in
`inference/README.md`. Hardware assignments are explicit command-line
arguments, so no cluster-specific path or device map is embedded in the
release.

Natural model prompts cannot be distributed because they contain benchmark
content. Their row-level scores, execution summaries, checkpoint revisions,
and inference units are included for exact statistical replay.

The same boundary applies to the post-outcome answer-boundary diagnostic. Its
GPU replay required the licensed natural prompts, so the anonymous release
ships the text-free row projection and complete CPU analysis and does not
provide a self-contained model replay. The fresh four-format task is generated
and is therefore distributed in full with its frozen panel.
