# Released schemas

## Natural score projection

Each natural file is a JSON object with release metadata and a `rows` array.
Every row contains:

- `row_id`: unique identifier for one state, arm, order, and exit;
- `pair_id`: opaque independent-unit identifier;
- `K`: recurrent exit requested for that system;
- `evidence_state`: original, bridge swap, or coherent repair;
- `query_arm` and `candidate_order`: counterbalancing coordinates;
- `reference_candidate_identity` and `expected_candidate_identity`: integer
  identities used to orient the contrast;
- `reference_log_odds`: full-answer reference-minus-other score;
- `reference_argmax_credit`: 1, 0.5, or 0 after tie handling;
- `state_correct_argmax_credit`: accuracy oriented to the current graph state;
- `relation_path` or `terminal_relation`: inference stratum;
- `score`: per-token log probabilities and sequence totals with token IDs
  replaced by within-row equality classes.

The reference identity is fixed to the original graph while the expected
identity follows the current state. This is why path control and state
accuracy are separate quantities.

## Fictional score rows

Fictional rows retain generated world identifiers, relation family, graph
state, query arm, answer order, K, reference and expected A/B labels, raw A/B
logits, oriented margin, pair probability, and exact-choice credit. The source
prompt is joined by `row_id` to the complete generated panel.

## Prompt-format score rows

The surface-orbit files use the fictional row schema and additionally retain
`surface_world_id`, `serialization`, `morphology_index`, `wording_index`,
`order_index`, and `design_block`. Four surface realizations share each
`world_id`; all states, arms, orders, surfaces, and exits remain coupled at the
world level. The separate competence file contains only the 24 gate worlds.
Eight treatment shards jointly contain the 48 disjoint inference worlds.

## HRM score rows

The HRM files retain the generated world and row identifiers, relation family,
graph state, query or requested-relation arm, answer order, H readout, A/B
labels, and raw A/B logits. The analysis recomputes margins, choices, and
state-oriented accuracy from those logits.

The linear release contains 384 H1/H2 gate rows and 4,800 treatment rows from
240 different worlds. The gate uses only the 192 H2 rows. The branching
release contains 384 H2 gate rows and 6,912 confirmation rows from 144
different worlds. Its source arms, requested relations, graph states, answer
orders, and H1/H2 readouts remain paired within world. Scrubbed execution
metadata retains the checkpoint, prompt-adapter, full-vocabulary H2 parity,
runner, and gate bindings without machine paths or timestamps.

## Answer-boundary rows

Each boundary row retains a pseudonymous pair identifier, an optional
pseudonymous 2Wiki relation-path stratum, K, query arm, graph cell, expected and
reference candidate identities, three score margins, the corresponding
candidate predictions, generation-adherence indicators, first-token category,
candidate-prefix ranks, and replay-parity status. Exact benchmark text,
candidate strings, generated continuations, decoded token strings, tokenizer
IDs, source identifiers, and machine paths are absent by construction.

## Factorial runs

Each file is one training seed and one of four cells:

```text
tying in {tied, untied}
supervision in {single final-exit loss, losses at every exit}
```

The file retains configuration, matched-compute contract, training history,
and per-world held-out evaluation. Analysis first forms all four paired
condition vectors within a seed, then resamples seeds and worlds.

## Frozen summaries

These files deliberately omit row-level prompts and text-bearing payloads.
They preserve the source result's summary, decision, model binding, and
analysis status where available. They support appendix auditing but are not
substitutes for the central row-level evidence.
