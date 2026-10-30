# B3 on schemes 2.0: controlled diagnostic experiment

Protocol frozen on 29 October 2026, before the new fits. This is an exploratory
extension selected after inspecting this benchmark, not a new confirmatory test.

## Question and scope

Does the original B3 learner improve the mapping from transferred B1 CQ signals
to the frozen fine-grained gold, relative to B1.hard on schemes 2.0?
Predictive improvement, competitiveness, and Walton validity are separate outcomes.
The contrast changes numerical thresholds and abstention handling as well as tree
structure; it cannot isolate topology alone. A negative result applies to this
transferred B1 pipeline, not to every learner or to CQ sufficiency in general.

No new LLM calls, schemes, labels, prompts, datasets, solver, or grid. Historical
evaluations are read-only. A separate command and output directory prevent a
rerun of unrelated round-two fits. Existing user changes are preserved.

## Inputs and isolation

* All 601 frozen test arguments, gold scheme supplied, canonical fine scoring.
* Five-model isolated CQ outputs: full_v2 (1.1) trains CQ-supervised B1; full_v3
  (2.0) supplies target features. B1 coefficients are global signal weights,
  transferred across versions, not human labels mapped by CQ identifier.
* The same five historical outer folds, seed 20261019; all CQ/model rows of an
  argument stay together. This retains the historical argument split and does
  not assert a new semantic-duplicate grouping. No new repeat search.
* Reconstruct B1 inside every training partition, including inner model
  selection and training-feature cross-fitting, with B1CrossFitter.fit_data.
  Globally saved OOF CQ rows are never training inputs for the downstream model.
* Save exact training/held-out IDs for upstream fits and downstream tuning.
  Held-out gold/CQ-label mutation must not change its own prediction under fixed
  folds, including cross-version transfer.
* Check run versions, hashes, five entries, complete item/CQ grids, finite target
  features, and equality of fold assignments. Record software and thread counts.

## Unchanged learner

Original CQTree/B3: one balanced Gini CART per scheme; features in YAML order are
p_first and outside-arc mass for every CQ. max_depth in {1,2,3};
min_samples_leaf in {1,3,5}. Preserve original grid order (shallower, then larger
leaves on tied inner pooled Macro-F1), four shuffled inner folds and small-sample
fallbacks. B1 retains its original grid, shrinkage, and CQ-label supervision.
Do not impose new Walton constraints during fitting: semantic rejection is an
audit outcome, not an unrecorded change to the learner.

Gold labels outside their supplied scheme's terminals are excluded only from
supervised tree training/tuning. All three known Popular Opinion/non sequitur
items remain in the primary test scoring and receive explicit audit records.

## Baseline and context

Primary pair: B3@2.0 minus B1.hard@2.0, generated from identical outer-held-out
B1 numerical signals. Reconstruct the hard baseline from the same source fits;
compare its verdicts, coverage and numerical signals against the original saved
baseline. Record any drift before interpretation; do not silently substitute a
historical snapshot with different confidence values. Historical tables remain
untouched. Qwen A0/A2@2.0 are saved contextual references only; no additional
primary hypothesis or choice of the best system after observing outcomes.

## Scoring and uncertainty

* Primary: canonical Macro-F1 on concatenated 601 OOF predictions, averaging
  classes with gold support. Abstentions count as errors, not excluded items.
* Secondary: Accuracy; also report balanced accuracy and coverage.
* One primary contrast, ordinary paired percentile IC95%, 10,000 item bootstrap
  draws from historical generator 1, seed 20261019. The historical m=4 family is
  unchanged; this extension does not inherit its multiplicity factor.
* Resample every system together. Subgroup intervals filter the same draws;
  classes absent from a resample are omitted by canonical scoring. Empty draws
  are omitted and their number reported. These intervals condition on fitted
  OOF predictions, not on retraining; they do not erase adaptive benchmark reuse.
* Practical-interest flag: observed primary difference >= +0.03 Macro-F1.
  Statistical signal: lower IC95% > 0. Report both independently; neither
  certifies normative validity or independent generalization.
* Common coverage: fixed intersection of arguments with a valid verdict from
  both primary systems. Score both on this identical selected cohort and report
  its size/support; it is a conditional sensitivity, not the primary result.
* Descriptive breakdowns: schema, class, source corpus, gold origin, no-NLAS;
  all data versus reachable-gold data, plus an item ledger for unreachable gold.
  No subgroup result promotes a new primary claim or selects another model.
* Diagnose Good dependence using per-class changes, corrections/losses, and
  Macro-F1 excluding Good from the class average while retaining all arguments
  (thus preserving false positives on Good texts). Also report the gold-non-Good
  subset separately. Origin and scheme are confounded; all Good are scheme-origin.

## Artifacts, checks, and interpretability

Save configuration/protocol snapshots and hashes before fitting; record status
running/complete/failed, timing, peak process memory, package versions, code
hashes, input/output hashes and any baseline drift. Refuse to overwrite outputs.
Save OOF predictions, full-precision distributions, features, choices, every
inner candidate's train/validation scores, final training scores, missing
training classes and excluded IDs. Serialize full trees and B1 coefficients,
not only observed paths; store raw and weighted leaf support separately.

Export every root-to-leaf rule, CQ text/version, first-arc meaning, numerical
thresholds and feature provenance. Replay exported trees on every held-out row
using sklearn's float32 input comparisons and full-precision thresholds; require
identical terminal/distribution and leaf identity. Rounded displayed paths are
not the serialization used for replay.

Audit repeated CQ tests, outside-mass conditions, tiny leaves, preconditions,
applicability, and terminal justification. Graph reachability is not semantic
admissibility. Missing prerequisite tests and probability thresholds are review
flags, not automatic proofs of a violation. Outside mass combines NA and
indeterminacy; it does not identify which occurred. Document unresolved rules
and counterexamples; no independent human adjudication is claimed. Structural
variation across the five outer models is descriptive, not repeat-fit stability.

A promising predictive result still requires argumentatively defensible rules,
adequate support, and subsequent independent validation before adoption. Do not
repair or select a new model after reading these held-out results in this run.

## Acceptance

Tests cover cross-version held-out isolation, baseline input identity, exact
tree export/replay, preserved original predictions when audit capture is enabled,
canonical abstention and common-coverage scoring, Good-excluded metrics, and
unreachable-gold retention. Lint and relevant tests precede real execution.
The final report separates predictive improvement, contextual competitiveness,
semantic concerns, and work still requiring human validation.
