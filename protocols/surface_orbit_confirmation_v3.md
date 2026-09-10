# Ouro-2.6B prompt-format confirmation

Status: specified before v3 panel generation and model inference

Date: 2026-09-09

## Evidence status

The v2 four-format assay stopped before treatment because intact-graph
accuracy was 0.6484 against a registered 0.6500 threshold. Its interval was
above chance and every marginal slice cleared 0.55, but the stopping rule is
honored. No v2 confirmation-world outcome was scored.

Gate-only diagnostics identified three avoidable sources of task difficulty:
the terse relation-first ledger, identifier forms containing numeric and
punctuation-heavy fragments, and two indirect question phrasings. This v3
follow-up changes those surface components before generating any new world.
It does not pool v2 gate data into v3 inference.

## Question

Does Ouro's depth-dependent control by a supplied two-hop path replicate on
fresh fictional worlds when all graph symbols are role-blind and globally
unique, and when serialization, relation names, evidence order, identifier
form, and direct query wording are independently varied?

## Independent samples

Generate 72 new worlds under seed 20260912:

- 24 gate-only worlds, used only for intact-graph K=4 competence;
- 48 confirmation-only worlds, used only for original-versus-swap inference
  at K=1 and K=4.

No visible symbol may occur twice. No failed or difficult world may be
replaced after scoring. Gate worlds and confirmation worlds are disjoint.

## Prompt formats

Each world has four isomorphic surfaces:

1. directed arrow records;
2. labeled tuples;
3. compact JSON records;
4. short natural-language mapping records.

Every surface draws all sources, bridges, terminals, and relation names from
one shuffled role-blind pool. Four neutral identifier forms vary only case and
separator: `TOK_NUVAFI`, `tok_nuvafi`, `TokNuvafi`, and `TOKNUVAFI`.
None encodes graph role. Four direct question phrasings are independently
assigned. All 16 morphology-by-wording offset cells occur exactly three times
over the 48 inference worlds, so every serialization is paired 12 times with
each morphology and wording. Eight record-order schedules occur six times per
serialization.

Original and bridge-swap prompts within a surface have the same inventory,
serialization, record order, question, and candidate display. Only the two
focal first-hop bridge assignments change. Both query arms and both candidate
orders are crossed. The response boundary remains exact one-token `A` or `B`.

## Sequential gate

Score only the 24 gate worlds in the original state at K=4. Open v3 treatment
only if:

- overall graph-correct accuracy is at least 0.65;
- the whole-world bootstrap 95% lower bound exceeds 0.50;
- every serialization, query arm, and candidate-order accuracy is at least
  0.55;
- one-token label and native-readout parity audits pass.

Failure stops v3 without swap scoring.

## Estimands and decision

Conditional on promotion, score all 48 untouched inference worlds under
original and bridge-swap states at K=1 and K=4. Average arms and candidate
orders within surface, then the four surfaces within world. The world is the
independent unit.

The three primary quantities are deep path control D(4), raw depth gain
G(4,1) = D(4) - D(1), and choice depth gain H(4,1) = C(4) - C(1). The result
confirms the surface-orbit effect only if all three 10,000-draw stratified
whole-world bootstrap 95% lower bounds exceed zero. Surface-specific results
are secondary.

## Interpretation

A positive result shows that Ouro-2.6B's path-control effect persists across
preplanned changes to graph names, record format, question wording, and
evidence order on new fictional worlds. It does not establish universal format
invariance, an internal graph algorithm, cross-model generality, or a causal
role for weight sharing.
