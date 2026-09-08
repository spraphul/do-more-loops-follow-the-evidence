# Reproducibility guide

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
4. **Model re-execution:** the frozen runners in `inference/` can rescore the
   generated panels when the pinned checkpoints and upstream model
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
- `reproduced/results/structural_falsifiers.json`
- `reproduced/results/training_factorial.json`

Each fresh JSON product is compared recursively with the corresponding file in
`expected/analysis/`. Numbers use a tight floating-point tolerance; strings,
keys, decisions, and array structure must match exactly.

## Statistical contract

All central intervals use 10,000 nonparametric percentile draws. Natural
2Wiki analyses resample relation paths and then paired examples. Natural
MuSiQue analyses resample whole paired examples. Fictional analyses resample
whole worlds within each relation family. The controlled training experiment
resamples confirmation seeds and then held-out worlds within seed. Every state,
query arm, candidate order, prompt row, and exit belonging to a sampled unit
travels with that unit.

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

## Integrity

`MANIFEST.sha256` binds every tracked release file other than the manifest
itself. Verify it directly with:

```bash
python tools/verify_release.py
```

The same command also validates the canonical JSON hashes stored in both
generated panels and runs the anonymity audit.

