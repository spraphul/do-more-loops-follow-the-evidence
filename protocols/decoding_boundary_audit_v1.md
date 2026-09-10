# Ouro-2.6B decoding-boundary audit v1

Date frozen: 2026-09-09 PDT

Status: post-outcome diagnostic, frozen before the boundary-logit replay

## Question

Why does a strong graph-dependent candidate-likelihood contrast yield much
weaker free-answer accuracy? This audit separates three possibilities on the
exact prompts already used for candidate-list-free generation:

1. the correct answer is weak even relative to the paired answer;
2. the paired-answer score is sensitive to the answer's boundary spelling,
   specifically a leading-space continuation versus the bare form emitted by
   greedy decoding;
3. the preferred candidate loses to another token under full-vocabulary
   decoding, or begins correctly and then fails during continuation.

This analysis uses already observed outcomes and does not change any registered
generation endpoint.

## Frozen population

- Ouro-2.6B at native exits `K=1` and `K=3`.
- All 992 2Wiki `NO_LIST` rows from the frozen candidate-list ablation.
- All 760 MuSiQue answer-only rows from the frozen generation experiment.
- No row, pair, output, or candidate-length filtering.

The original prompts, model revision, bfloat16 precision, readout selector,
and deterministic generation records remain fixed. The audit performs no new
generation.

## Frozen replay

For each row, retain the registered teacher-forced score of each continuation
with a leading space. Re-score each canonically cased candidate as a bare
continuation, matching the boundary form with which unrestricted generation
usually begins. From the first next-token distribution, record:

- the greedy token and its stored-generation parity;
- the full-vocabulary rank and probability of the first token of each
  candidate under both boundary spellings;
- the top ten tokens for qualitative inspection;
- the full bare-candidate sequence log probability.

Define a two-surface candidate score as the log-sum-exp over distinct
leading-space and bare token sequences. If the sequences are identical, count
the sequence once. This is a diagnostic semantic-surface score, not a revised
registered estimand.

## Frozen summaries

For registered, bare, and two-surface scores, report:

- K3 pairwise state accuracy;
- agreement with the actual greedy candidate identity among adherent K3 outputs;
- graph responses at K1 and K3 and their K1-to-K3 gain, using the original
  within-pair estimand;
- K3 decision changes between boundary spellings.

For the full-vocabulary boundary, report the proportion whose greedy first
token is unique to the correct candidate, unique to the wrong candidate,
shared by both candidates, or outside both candidate-prefix sets. Also report
candidate-rank summaries and the transition from first-token category to the
final stored generation outcome.

Uncertainty uses the same 10,000-draw independent-unit bootstrap as the source
experiments: relation-path then whole-pair resampling for 2Wiki, and whole-pair
resampling for MuSiQue. No confidence interval or original result is changed.

## Interpretation

- Better bare or two-surface accuracy indicates an answer-boundary scoring
  mismatch, not improved model reasoning.
- A non-candidate top token despite a correct paired-answer decision identifies
  full-vocabulary competition.
- Correct candidate initiation followed by a wrong or invalid final answer
  identifies continuation loss.
- None of these outcomes establishes a localized circuit, a discrete graph
  algorithm, or a causal role for any hidden unit.

This audit tests the mismatch between likelihood scoring and free generation.
Circuit-level localization is outside its scope.
