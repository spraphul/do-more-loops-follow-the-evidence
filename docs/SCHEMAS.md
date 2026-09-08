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

