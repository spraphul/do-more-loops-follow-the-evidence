# Artifact map

## Data flow

```text
generated panel or licensed benchmark source
                 |
                 v
       frozen model-facing runner
                 |
                 v
  anonymous row-level score projection
                 |
                 v
        estimand and bootstrap code
                 |
                 +--> headline claim table
                 +--> statistical figures
                 +--> expected-output comparison
```

For natural panels, the public release begins at the anonymous row-level score
projection. For fictional panels, the complete generated input and every
prompt cell are also public.

## Directory roles

| Path | Role | Needed for central replay |
|---|---|---:|
| `data/generated/` | Frozen fictional worlds, prompts, splits, and symbolic audits | yes |
| `data/analysis_ready/natural/` | Text-free paired natural score grids | yes |
| `data/analysis_ready/fictional_ouro/` | Ouro competence and treatment rows | yes |
| `data/analysis_ready/fictional_loopus8/` | LoopUS competence and treatment rows | yes |
| `data/analysis_ready/structural/` | On/off-path and locality-control rows | yes |
| `data/analysis_ready/factorial/` | Four conditions across eight confirmation seeds | yes |
| `data/plot_ready/appendix/` | Small frozen tables for secondary appendix plots | no |
| `data/frozen_summaries/` | Diagnostics and stopped-branch records | no |
| `analysis/` | Scientific estimands and registered resampling | yes |
| `figures/` | Statistical rendering only | no |
| `inference/` | Model re-execution and mock schedule audit | optional |
| `expected/analysis/` | Frozen reference outputs | yes |
| `expected/execution_summaries/` | Sanitized frozen runner checks | no |
| `expected/figures/` | Release-rendered statistical panels | no |

## Provenance

`data/PROVENANCE.json` records, for each exported artifact, the private-source
digest, public release path, public digest, and transformation class. It omits
private source paths. `MANIFEST.sha256` independently binds the release files
that a reviewer receives.
