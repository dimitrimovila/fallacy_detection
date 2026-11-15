# Spec 13. Pooled neural CQ aggregator

13 November 2026. Implementation experiment, not a preregistered comparison.
Confirmed scope: one neural aggregator across all schemes, using the five model
entries separately. This is an independent learner, not a replacement for A3.
The encoding and training settings below are initial implementation defaults;
no predictive advantage or final experimental protocol is established.

## Inputs and output

One row per argument, in the reference-scheme condition. A fixed feature schema
is constructed from the scheme definitions and explicitly selected model entries,
never from verdict labels. Each (scheme, CQ, entry) has four hard-answer indicators
(the two admitted answers, indeterminate, not applicable), invalid and missing
indicators, four soft probabilities, and a soft-available indicator. The preferred
soft source follows the run manifest's sampling/reasoning regime. Verbal confidence
is oriented to the first admitted answer by the existing parser and encodes an
arc-conditional split, with zero in the other two soft channels; it does not
estimate the probability of an outside answer. Hard outside answers remain
separate indicators. Missing soft
signals remain zero with availability zero; hard answers are not substituted for
them. Nonapplicable schema blocks remain zero. Scheme identity is one-hot encoded.
Question identities remain scheme-specific; CQ1 in different schemes is not the
same feature. No text, item identifier, gold verdict, human CQ annotation, direct
verdict, diagram traversal, or corpus identity enters the network.

A single PyTorch MLP maps the complete vector to canonical fine-label logits.
Output classes are those present in the training partition, without restricting
them to diagram terminals of the supplied scheme. All labelled items remain in
training, including scheme/verdict mismatches. This deliberately differs from
diagram-constrained learned aggregators. A class unseen in training receives no
predictions. Softmax distributions are scores, not certified calibrated probabilities.
The model description is not an argumentative explanation or a causal attribution.

## Training and validation

CPU execution with bounded threads, full-batch AdamW, ReLU, a small hidden layer,
class-balanced cross entropy with weights estimated on the training partition,
weight decay and gradient clipping. Training stops at the configured epoch budget
or on training-loss convergence; there is no outer-test early stopping.
Candidate widths and weight decay are selected by pooled inner out-of-fold macro
F1. Inner folds depend on scheme and item ID only. Ties choose the first candidate
(the example grid lists smaller widths and stronger shrinkage first).
Each configured initialization is evaluated as a separate system, not selected
using outer scores. Initializations share outer folds and inner partitions.

Five outer folds use the existing scheme-stratified assignment. A label held out
in an outer fold cannot affect its features, class vocabulary, weights, candidate
choice, stopping, or predictions. Nested validation is retained even though this
is exploratory work on an already inspected historical population.

## Interface and artifacts

`argfallacy evaluate-neural CONFIG --check-only` checks inputs and reports feature
dimensions, missingness, class counts and planned fit counts without fitting or
creating an evaluation. `--out DIR` is required for an actual evaluation; existing
directories are never overwritten. The configuration explicitly names the parsed
run, items file, model entries, folds, initializations and candidate settings.
The supplied example uses schemes 2.0 and the frozen historical reference labels;
it does not silently substitute the revised reference. No LLM or cluster call occurs.

An evaluation saves the resolved configuration, input/source hashes, versions,
feature names, fold membership, inner choices, per-item probabilities, canonical
point metrics per initialization and a checkpoint per outer fold. Checkpoints
retain feature identities, classes, network dimensions and fitted tensors; replay
rejects mismatched feature schemas, including CQ wording and soft-source changes.
The manifest distinguishes complete from failed
execution. This first implementation does not provide bootstrap comparisons, a
chosen final deployment fit, or an external-test result. Those need a separately
specified comparison and reference population before thesis claims are made.

## Acceptance checks

Synthetic XOR-like CQ interactions can be learned. Missing, invalid, indeterminate
and not-applicable inputs stay distinguishable; malformed probabilities, duplicate
rows, unknown questions/labels and schema mismatches fail loudly. Reordering input
rows preserves features. Perturbing only outer-held-out verdicts leaves that fold's
selected settings and probabilities unchanged. Single-class training works, empty
training fails, fixed-seed repeat fits and serialized checkpoint replay agree.
The isolated CLI preflight and a small synthetic end-to-end evaluation are checked;
these are engineering checks, not evidence of performance on real arguments.
