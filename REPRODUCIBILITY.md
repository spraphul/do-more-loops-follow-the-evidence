# Reproducing the results

## Reproduction levels

The artifact separates statistical reproduction from optional model
re-execution.

1. **Integrity audit:** `make verify` checks the frozen release without GPU
   access.
2. **Statistical reproduction:** `make reproduce` starts from row-level model
   scores and recomputes all central estimands and 95% percentile bootstrap
   intervals.
3. **Figure reproduction:** `make figures` renders the statistical panels from
   the freshly reproduced results.
4. **Model re-execution:** the released runners in `inference/` rescore their
   corresponding generated panels when the pinned checkpoints and upstream
   implementations are available. See `docs/INFERENCE_REEXECUTION.md`.

Levels 1 through 3 are sufficient to audit every central numerical claim in
the submission. Level 4 is intentionally separate because checkpoint access,
GPU hardware, and model-specific source code are external dependencies.

## Tested environment

- Python 3.11
- NumPy 1.26.4
- Linux or macOS
- CPU-only for analysis

Create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run:

```bash
make verify
make reproduce
```

The analysis orchestrator runs independent jobs concurrently. Reduce resource
use with `python tools/reproduce.py --jobs 1` or increase it with `--jobs N`.

## Outputs

Fresh files are written only beneath `reproduced/`, which is ignored by Git.
The main products are:

- `reproduced/results/headline_results.json`
- `reproduced/results/headline_results.csv`
- `reproduced/results/scale_invariant_path_control.json`
- `reproduced/results/natural_depth_curves.json`
- `reproduced/results/fictional_ouro/nonce_path_control_ouro26.json`
- `reproduced/results/fictional_loopus8.json`
- `reproduced/results/hrm_linear.json`
- `reproduced/results/hrm_branching.json`
- `reproduced/results/surface_orbit_confirmation.json`
- `reproduced/results/structural_falsifiers.json`
- `reproduced/results/training_factorial.json`
- `reproduced/results/decoding_boundary_audit.json`

Each fresh JSON product is compared recursively with the corresponding file in
`expected/analysis/`. Numbers use a tight floating-point tolerance; strings,
keys, decisions, and array structure must match exactly.

## Statistical contract

All central intervals use 10,000 nonparametric percentile draws. Natural
2Wiki analyses resample relation paths and then paired examples. Natural
MuSiQue analyses resample whole paired examples. Fictional analyses resample
whole worlds within each relation family. The controlled training experiment
resamples confirmation seeds and then held-out worlds within seed. All states,
query versions, candidate orders, prompt rows, and readouts from a sampled unit
are resampled together.

The HRM linear transfer resamples 40 inference worlds within each of six
relation families. Its 48 gate worlds are excluded. The branching confirmation
resamples 24 worlds within each of six relation families after a separate
48-world gate. Source arms, requested relations, graph states, answer orders,
and H1/H2 readouts remain paired within world.

The fresh prompt-format confirmation averages its four renderings within each
of 48 inference worlds, then resamples whole worlds within 16 frozen
morphology-by-wording design blocks. Its 24 gate worlds are disjoint and never
enter treatment inference. The answer-boundary diagnostic preserves the source
natural-panel schemes: 2Wiki resamples relation-path strata and then pairs,
while MuSiQue resamples whole pairs. The boundary analysis remains explicitly
post-outcome and does not replace any registered interval.

The scripts preserve the frozen seeds and recompute nonlinear statistics, such
as repair ratios, inside every draw. Prompt rows are never treated as
independent observations.

## Data and licensing boundary

Natural benchmark prose is deliberately absent. The released natural files
contain oriented scores, paired-unit identifiers, relation strata, depth,
graph state, query arm, candidate order, and pseudonymous token equality
classes. They support exact recomputation of both sequence-margin and
first-divergence choice analyses.

Generated fictional prompts and worlds are included in full. Checkpoint and
dataset acquisition is governed by the original providers. See `DATA_LICENSE.md`
and `docs/MODELS_AND_DATASETS.md`.

The boundary projection contains no benchmark questions, prompts, candidate or
answer strings, generated continuations, decoded token text, or raw tokenizer
IDs. It retains only pseudonymous units, categorical first-token outcomes,
full-vocabulary ranks, candidate identities, and score margins required for the
reported audit.

## Integrity

`MANIFEST.sha256` binds every tracked release file other than the manifest
itself. Verify it directly with:

```bash
python tools/verify_release.py
```

The same command also validates the canonical JSON hashes stored in the
generated panels and runs the anonymity audit.
