# Do More Loops Follow the Evidence?

## Causal Tests of Relational Evidence Control in Recurrent Language Models

This anonymous artifact accompanies an ICLR submission. It contains the
generated experimental panels, text-free natural-question score rows, complete
fictional-world score rows, controlled-training records, analysis code,
expected outputs, and plotting code needed to audit the paper's central causal
claims.

The fastest reviewer path is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make verify
make reproduce
```

`make verify` checks release integrity, panel hashes, row grids, privacy
boundaries, expected-output coverage, and anonymity. `make reproduce` reruns
all central analyses with 10,000 bootstrap draws, writes outputs under
`reproduced/results/`, and compares them with the frozen expected outputs.
The full analysis is CPU-only. On an ordinary laptop it can take several
minutes because the registered bootstraps are intentionally rerun rather than
cached.

To rebuild the statistical figures after reproducing the results:

```bash
python -m pip install -r requirements-figures.txt
make figures
```

## What is included

- `data/generated/`: all 288 fictional worlds, frozen prompt cells, splits,
  balance audits, and exact symbolic validations.
- `data/analysis_ready/`: row-level scores for the central natural,
  fictional, structural-control, and training-factorial analyses.
- `data/frozen_summaries/`: compact machine-readable records for secondary
  diagnostics and stopped branches.
- `analysis/`: estimands, competence checks, registered resampling schemes,
  and sensitivity analyses.
- `inference/`: frozen model-facing runners and an executable mock mode for
  validating schedules without model weights.
- `expected/`: frozen analysis outputs and publication-ready statistical
  figures against which a fresh run is checked.
- `tools/`: one-command orchestration, integrity checking, anonymity auditing,
  and claim-table generation.
- `docs/`: the claim crosswalk, evidence status, data card, execution details,
  and artifact map.

## Release boundary

The natural-question files retain the complete paired score grid and the
independent inference units, but not licensed question text, prompts, entity
strings, source item identifiers, raw tokenizer identifiers, or answer
continuations. This is sufficient to reproduce every released natural-panel
estimand and interval without redistributing benchmark content. The fictional
worlds are generated for this study and are released in full, including their
exact prompts.

Model checkpoints and upstream benchmark files are not bundled. They must be
obtained from their original providers under the applicable terms. Frozen
score rows make the statistical reproduction independent of gated-model
access.

## Reviewer map

Start with [CLAIM_INDEX.md](CLAIM_INDEX.md) for the shortest route from each
paper claim to its inputs, command, and expected output. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment and execution details,
and [docs/RESULT_STATUS.md](docs/RESULT_STATUS.md) for the distinction between
prospective confirmation, registered transfer, diagnostic analysis, and a
failed competence gate.

No authors, affiliations, private repository history, tokens, host names, or
local machine paths are included in this release.

