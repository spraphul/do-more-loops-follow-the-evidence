# Data card

## Intended use

This release supports verification of a research claim about how supplied
relational evidence controls answer choice across recurrent exits. It is not a
general-purpose question-answering dataset and should not be used to infer
human reasoning ability or deployment readiness.

## Released central units

| Stream | Unit count | Repeated rows within unit |
|---|---:|---|
| Ouro 2Wiki natural | 124 pairs | state, arm, order, K1 to K4 |
| Ouro MuSiQue natural | 95 pairs | state, arm, order, K1 to K4 |
| LoopUS 2Wiki natural endpoint | 93 pairs | state, arm, order, K1 and K8 |
| LoopUS MuSiQue natural endpoint | 95 pairs | state, arm, order, K1 and K8 |
| LoopUS full-depth natural | 93 and 95 pairs | original/swap across K1 to K8 |
| Ouro fictional | 288 worlds | graph state, arm, order, K1 to K4 |
| LoopUS-8B fictional | 288 worlds | graph state, arm, order, K1, K2, K4, K8 |
| Structural falsifiers | 144 worlds | structural condition and K1 to K4 |
| Controlled factorial | 8 seeds | four conditions, 600 held-out worlds per seed |

## Privacy and content

The study contains no participants or private user data. Natural benchmark
text is removed as a licensing and double-blind precaution. Pair identifiers
are opaque and tokenizer identifiers are replaced by within-row equality
classes. Scores and inference strata are unchanged.

Fictional records contain generated opaque entities and relation terms. Their
full prompts are public so the graph intervention can be independently
inspected and rescored.

## Known limitations

The model families use different recurrence mechanisms and training regimes.
K is meaningful only within each family. Natural panels contain two-hop
questions from two sources and do not establish a universal trajectory across
tasks or checkpoints. Some attempted checkpoint and longer-path transfers
failed competence gates; those outcomes are indexed in the frozen summaries
rather than presented as treatment nulls.

