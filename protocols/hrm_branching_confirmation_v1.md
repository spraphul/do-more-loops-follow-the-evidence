# HRM-Text-1B branching-XOR confirmation v1

Status: frozen before any HRM-Text-1B inference on either split of the
confirmation panel. No real gate or confirmation command has been run.

This protocol is authorized by the completed 24-world development result, but
it does not reuse any development world, opaque node identifier, or target
prompt. The relation-name families and four terminal codes are intentionally
held fixed. The confirmatory claim is therefore about a new sample of graph
worlds under the prespecified relation/code inventory, not about arbitrary
relation wording or open-vocabulary generation.

## Scientific question and claim boundary

The assay asks whether HRM-Text-1B follows the interaction between a
source-selected bridge and the requested outgoing relation when both outgoing
relations from every bridge lead to plausible candidate terminals. A bridge
swap reverses the licensed answer. A topology repair retains that first-edge
swap while exchanging the bridges' terminal profiles, restoring the original
answer.

Success can establish topology-sensitive, relation-conditioned **output
control** and, under the stronger fixed-sequence criterion, an increase in that
control between retained H1 and stock H2. It cannot establish a discrete graph
algorithm, serial internal traversal, a recurrence-only causal effect, or
unrestricted generation competence.

H2 is the checkpoint's native final state. H1 is a retained intermediate state
from the same unmodified stock-H2 trajectory and is decoded with the released
language-model head only after full-vocabulary H2 parity passes. H1 is not a
separately trained native answer exit. Any H2-minus-H1 result concerns when the
answer becomes controllable at this fixed readout, not when an internal
composition algorithm first exists.

All reported accuracies are exact A/B pairwise-choice accuracies. A tied A/B
margin receives one-half credit. They are not open-vocabulary generation
accuracies.

## Prior development authorization

The only result authorized to motivate this confirmation is:

- schema: `iclr2027.hrm_text_branching_xor.development_result.v1`
- status: `development_only_not_a_paper_claim`
- file SHA-256:
  `4f0f69e9563504c107dbd13495ae05ec5fe316b1fc9e689950856a59fbd9bec0`
- decision:
  `ELIGIBLE_FOR_FRESH_CONFIRMATION_DEPTH_DEPENDENT_COMPOSITION`

The builder and every real runner invocation must validate this exact file and
decision. No other exploratory result can substitute for it.

## Frozen population and split

The panel contains 192 fresh worlds:

- **Gate:** 48 worlds. The runner scores and stores only the eight
  original-state prompts per world at H2: two source arms by two requested
  second relations by two candidate orders. This gives 384 prompts and 384
  scored rows. No H1 gate row and no gate-world swap or repair outcome is
  retained.
- **Confirmation:** 144 different worlds. The design is the complete
  `6 relation families x 12 ordered focal-terminal pairs x 2 replicates`
  cross. Every world contains all three evidence states, two source arms, two
  requested second relations, two candidate orders, and retained H1/H2 scores:
  3,456 prompt cells and 6,912 scored rows.

The two replicates in every relation-by-terminal-pair stratum have opposite
near-relation arms. Source phase, terminal phase, candidate identity, A/B label,
and physical locality are balanced by construction. All opaque identifiers are
unique and disjoint from the development panel, and every fresh target prompt
hash is absent from development.

The 144-world size is a design-stage choice: it is the smallest exact
`6 x 12` cross with two counterbalanced replicates per cell. Under the paired
world analysis, it also gives roughly 90% power for a standardized effect near
`d = 0.27`; this rationale was fixed before confirmation outcomes existed and
does not authorize sample-size re-estimation.

The confirmation split is assigned prospectively to exactly 12 shards of 12
worlds. Every shard contains:

- all 12 ordered focal-terminal pairs exactly once;
- two worlds from each of the six relation families;
- six near-X and six near-Y worlds;
- 288 prompt cells and 576 H1/H2 rows.

Shard membership is the frozen `confirmation_shard` field in each world. It
must never be recomputed from row order or `world_index` modulo a shard count.

## Gate and sealing rule

The real gate promotes only if all fixed conditions hold at H2:

- overall original-state accuracy is at least 0.65;
- its stratified whole-world bootstrap lower 95% bound is strictly above 0.50;
- accuracy is at least 0.55 separately for X and Y, source arms 0 and 1, and
  original and reversed candidate orders;
- accuracy is at least 0.50 in each of the six relation families.

H1 is not inspected or used by the gate. A failed gate means this exact
checkpoint/prompt/panel combination is not an interpretable substrate for the
confirmatory treatment. It is not a null result about recurrence or
composition.

Only an untampered real artifact with decision `PROMOTE_TO_CONFIRMATION` and
`authorizes_real_confirmation: true` may unlock real confirmation shards. The
artifact must bind the exact development result, panel, model, runner, backend,
prompt adapter, complete 384-row H2-only schedule, and recomputed gate summary.
A mock gate reports `MOCK_GATE_VALIDATION_ONLY`; it may unlock mock shards for
tests but can never authorize real inference.

Confirmation prompts remain unavailable to the model until the gate passes.
All 12 confirmation shards must bind the exact byte SHA-256 of the same passing
gate artifact.

## Estimands

Within each world and H exit, first average the two source arms and two
candidate orders. Scores are oriented to the answer licensed by the original
X-query graph. For signed exact-choice scores, use `+1`, `0`, and `-1` for
reference win, tie, and reference loss. The primary repair interaction is

```text
R_H = (repair_X - swap_X - repair_Y + swap_Y) / 4.
```

The raw A-minus-B logit-margin interaction is defined identically. The choice
scale equals one for a perfect composer. The supporting bridge interaction is

```text
B_H = (original_X - swap_X - original_Y + swap_Y) / 4.
```

Choice repair fraction at H2 is `R_H2 / B_H2` and is undefined when its paired
bootstrap denominator is zero. Relation-arm repair effects, the crossed-locality
repair effect, state accuracies, relation-family slices, ordered-terminal-pair
slices, and leave-one-relation-family-out estimates are computed from the same
paired world vectors.

## Confirmatory success rules

The analysis follows a fixed sequence. The depth-dependent claim is evaluated
only after every composition condition passes.

### Composition tier

Every condition below is claim-blocking:

- H2 primary repair-XOR choice estimate is at least 0.10 and its lower 95%
  bound is strictly above zero;
- H2 primary repair-XOR raw-margin lower 95% bound is strictly above zero;
- the X-arm and Y-arm H2 repair-choice lower 95% bounds are each strictly above
  zero;
- the H2 crossed-locality repair-choice lower 95% bound is strictly above zero;
- H2 original-state accuracy is at least 0.65, and bridge-swap and
  topology-repair accuracies are each at least 0.60; the lower 95% bound for
  every one of these three accuracies is strictly above 0.50;
- H2 choice repair fraction is defined, its point estimate is at least 0.50,
  and its lower 95% bound is at least 0.50.

### Depth-dependent tier

Conditional on the complete composition tier passing:

- H2-minus-H1 primary repair-XOR choice gain is at least 0.05 and its lower
  95% bound is strictly above zero;
- H2-minus-H1 primary repair-XOR raw-margin lower 95% bound is strictly above
  zero.

Relation-family, ordered-terminal-pair, and leave-one-family-out results are
mandatory descriptive heterogeneity checks, not additional promotion gates.
No threshold may be revised after any confirmation outcome is accessed.

The composition tier is an intersection-union success rule: every listed
one-sided scientific condition must hold before promotion. Requiring the full
intersection prevents a favorable member of a larger family from driving the
claim; no diagnostic slice is promoted into an additional hypothesis test.

## Inference

The independent unit is the world. Source arms, candidate orders, evidence
states, requested-relation arms, and H exits remain paired within world. All
intervals use exactly 10,000 shared paired whole-world bootstrap draws,
stratified so that each draw samples 24 worlds with replacement within each of
the six relation families. The fixed bootstrap salt/seed is part of the frozen
common implementation. The same draw indices are used for every effect,
difference, accuracy, and ratio so covariance is preserved.

Inference is conditional on the six fixed relation families and four terminal
codes. It does not treat those lexicons as a random sample from a broader
language population.

## No adaptation, replacement, or optional stopping

- The model revision, weights, tokenizer, prompt adapter, two solved examples,
  PrefixLM mask, H-state extraction, readout, scoring rule, panel, analysis,
  sample sizes, and thresholds are immutable after freezing.
- No confirmation world may be dropped, replaced, regenerated, or supplemented.
  The sample may not be expanded after seeing a result or interval.
- Invalid or non-finite model output is a protocol failure, not permission to
  replace a world.
- Analysis requires all 12 distinct shard indices and rejects missing,
  duplicated, mixed-mock, wrong-gate, wrong-panel, or altered rows.
- A completed shard is never rerun because of its scientific outcome. A shard
  may be retried only after an infrastructure/authentication failure or an
  incomplete/tampered artifact. The retry must use the identical frozen command,
  shard index, checkpoint, inputs, and output schedule. Record the failed
  attempt; do not inspect partial scientific outcomes; discard the invalid
  partial artifact before the one identical retry.
- There is no sequential look, early success, favorable-subset analysis, or
  sample-size re-estimation.

## Frozen bindings

- development panel file SHA-256:
  `4a5608aee11047bb6897e48ad7575c8e6f2a6dd4c14c7685bcbb3a59670f5c95`
- development panel canonical SHA-256:
  `f8c5e56965a191d391ca15965c63270dc7f890ba37b312071069afa9d50cd439`
- development result SHA-256:
  `4f0f69e9563504c107dbd13495ae05ec5fe316b1fc9e689950856a59fbd9bec0`
- confirmation builder SHA-256:
  `5f934b89abcc722ef6415b2c088312acf8a8598be59b9ecbc7afeb702ec31814`
- confirmation panel file SHA-256:
  `1e3583f34bea3f0ddf02970bd70b4711c53bff7af9b8d8db98382e21a648f41c`
- confirmation panel canonical SHA-256:
  `e76bf583223dba4f6b46e970b592009b6b7f29bd9db0e30cd8b97c3217314f31`
- HRM checkpoint revision:
  `1f82ac2b71222f0c100a224a33f24b44a3000b6d`
- HRM weights SHA-256:
  `f8fe2b2bf6948414e8e8d6538659198726d98f967c55b533b7aabe8a1fa9a584`
- inherited HRM backend SHA-256:
  `f1ee430807fa3129108c591505d40ead75b92da7137fba387d223aeaeb332970`
- frozen branching prompt-adapter SHA-256:
  `c80dfb3277af257986c5c1081ce1985c42af2ae4edaa8a760747b81e394c1a14`
- confirmation common SHA-256:
  `f47ba25ced60eff51ed6d71a0d58898892aada584ba651c7b391852d040752d0`
- confirmation runner SHA-256:
  `c960b03f3460b710ff2eb84fb1e3f60f06c00317182b2cf264c802e979c97b4e`
- confirmation analyzer SHA-256:
  `23e4ebf79474a53f6e61768337b0674fe3ba8323358dc4e450a345d5b2a7fdfa`
- confirmation mock-test SHA-256:
  `dee80f274047efc9526623b12a8bde89f2c8f0440977a2e23eab994076f0bef4`

Changing any bound file requires a new protocol version and newly generated
panel before another real command is authorized.
