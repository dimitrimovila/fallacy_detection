# GOSDT@2.0 versus matched CART: controlled diagnostic protocol

Updated 30 October 2026. The scientific direction, input source, fixed thresholds,
six-candidate grid and evaluation conventions below are explicitly confirmed.
**Scientific design and technical policies accepted after synthetic checks.**
The exact-root guard, conservative incomplete-tuning treatment and per-fit limits
are accepted. Implementation and a resource-qualified local start were subsequently
authorized. A local runner has now been implemented; no fit on the 601 arguments
has started. After observing insufficient free PC memory, the execution preference
changed to evaluating the CPU cluster. That is not authorization to submit a job.
The checked supervisor currently requires Windows; Linux supervision, runtime
availability and scheduler resources must be verified before a cluster start.
B1–B5 are the current round-two names; C1–C5 are their historical aliases.

## 1. Question, scope and references

Does GOSDT improve OOF Macro-F1 over a fresh greedy CART trained on identical binary
CQ predicates, examples and partitions? Compare the final training objective and
complexity as well as prediction. This evaluates two complete learning procedures:
CART constructs a greedy Gini topology and is pruned using the target objective;
GOSDT searches more broadly for that objective. Matching their final objective
reduces confounding but does not make every internal search operation identical.

Confirmed source: the transferred CQ-supervised B1 on schemes 2.0 used by B3.
Primary contrast: GOSDT minus the newly fitted matched CART. Secondary references:
B1.hard@2.0 (manual diagram on the same B1 source) and historical B3@2.0. The latter
has continuous rather than the new binary features, and is context, not an isolated
algorithm comparison. Do not add Qwen-only inputs, other learners, new prompts,
annotations, scheme changes or LLM calls to this experiment.

Read research_findings.md F11–F18 and spec 11. Reuse verified contracts from
aggregate/crossfit.py, aggregate/tree.py and eval/b3_diagnostic.py. Preserve earlier
runs, data and evaluations. No older evaluation is rerun or overwritten.

Depth 4 extends the earlier B3 cap of 3, but a gain over historical B3 cannot be
attributed solely to depth: encoding and learning procedure also change. The
primary matched CART shares the same new depth grid.

## 2. Inputs and fixed encoding

Use full_v2 1.1 as the original B1 CQ-supervision source and full_v3 2.0 as its
prediction destination, with the same five model entries and signal ordering as
B3. Validate source versions, manifests, CQ labels and hashes in implementation.
No global saved OOF B1 rows may serve as a new downstream training matrix.

For each CQ in the loaded scheme's declared order, create exactly these predicates:

| Signal | Thresholds | Operator |
| --- | --- | --- |
| Probability of the first admitted arc, p_first | 0.25, 0.50, 0.75 | strictly greater than, p_first > t |
| Outside-arc mass | 0.10, 0.50 | strictly greater than, outside > t |

Use float64 input comparisons; threshold equality returns false. Column order is
CQ order, then the three p_first predicates in ascending threshold order, then the
two outside predicates in ascending order. Persist the exact predicate, original
arc text/polarity, source CQ version and retained column ordering. The first arc
is not universally a semantic yes. CART and GOSDT receive identical Boolean arrays.

Only remove columns constant on the applicable training feature matrix. Fit this
mask separately inside each inner training partition and again for the final outer
fit; apply that mask unchanged to validation/test, even if a removed column varies
there. Preserve removed-column IDs and their training value. Do not merge merely
correlated predicates or learn thresholds, a feature ranking, threshold guesses or
extra predictors. The current 2.0 schemes have at most seven CQs, hence at most 35
initial predicates per scheme.

Do not reconstruct NA, IDK or missing from outside mass. Original categories may
be retained in audit metadata only where actually available; no extra categorical
prediction columns are introduced in this fixed-input experiment. Retain the B1
source's documented handling of unavailable model signals. Its resulting p_first
and outside arrays must be finite and in [0,1]; otherwise fail the input check,
not silently encode missing as false, zero, NA or IDK. If all columns are removed,
use the exact no-feature root solution in section 7.

## 3. Objective, class weights and finite precision

For each actual downstream fit let T be the eligible training arguments, n=|T|,
C_T the classes present in T, K=|C_T|, and n_c their training counts. Compute these
quantities independently in each inner training fit and final outer fit, after
terminal-eligibility exclusions. Do not use validation frequencies, all-601 counts,
or absent-class pseudo-counts. For K>=2 define:

    w_i = n / (K * n_{y_i})
    R_bal(h;T) = (1/n) sum_i w_i * 1[h(x_i) != y_i]
               = (1/K) sum_{c in C_T} errors_c / n_c
    J(h;T) = R_bal(h;T) + (a/n) * L(h),  a in {2,5}

L counts leaves, including one leaf for a root-only tree. This is balanced
multiclass 0–1 training risk plus leaf penalty, not Macro-F1 optimization. Macro-F1
remains the separate inner-validation selection criterion and OOF endpoint.

GOSDT's matrix orientation is **row=predicted class, column=true class**. Order
classes canonically and pass an explicit K by K matrix with diagonal zero and
C[p,c]=1/(K*n_c) for p!=c, with balance=False to avoid a second transformation.
Its sum of per-example costs is already normalized; do not divide by n again.
Pass regularization=a/n for that fit and assert the effective parameter is unchanged.

Use the same effective cost for CART: sample_weight_i=n*C[p!=y_i,y_i], with the
training risk computed as sum of those weights on errors divided by n. GOSDT's
native FloatMatrix and regularization are float32 in the checked release. Materialize
both C and lambda to those native values once for both procedures, verify the values
read back from the API, and serialize them together with nominal lambda=a/n. Derive
the effective CART weights and pruning penalty from those same stored values,
rather than claiming bit-identical native arithmetic from independent formulas. Record both
nominal float64 R_bal and the common effective-matrix risk, their discrepancy and
J. Summation-order/native precision can still differ. Use absolute tolerance 1e-6
for cost/bound consistency checks, preserve full precision, and flag larger drift.

Only classes present in T enter the matrix. Predictions map back to canonical
labels using the saved class order. Classes absent from training remain in
validation/test scoring if gold-supported there; never drop their false negatives.

## 4. Six configurations and the matched CART pruning

Both procedures use the same six pairs (d,a):

    maximum decision depth d in {2,3,4}
    leaf coefficient a in {2,5}; regularization lambda=a/n

No additional minimum-leaf, threshold or pruning grid is introduced. GOSDT uses
its public depth_budget=d. CART uses max_depth=d. Root-only depth is zero; the
public GOSDT wrapper already translates to its native convention internally. Do
not add another level manually. Verify actual exported maximum path length.

For each d, CART grows a deterministic greedy Gini tree with the effective class
weights, splitter='best', random_state=20261019, min_samples_split=2,
min_samples_leaf=1, min_impurity_decrease=0 and ccp_alpha=0. Other parameters remain
fixed and are recorded. The Gini split choice is part of this greedy baseline,
not claimed to optimize the final 0–1 objective globally.

For each a, find the globally best rooted pruning **of that fixed greedy tree**
under the same J. Do not equate ccp_alpha with lambda: sklearn cost-complexity
pruning uses weighted impurity, not this misclassification risk. A bottom-up
dynamic program avoids any additional tuning grid. For node v:

    e_v = min_label sum_{i routed to v} C[label,y_i]
    F_stop(v) = e_v + lambda
    F(v) = min(F_stop(v), F(left(v)) + F(right(v)))

At an existing leaf only stopping is possible. Choose the minimum-cost prediction
at a stopped node; ties use canonical class order. On objective equality within
1e-12 prefer stopping, hence fewer leaves. This gives the minimum J among rooted
subtrees of the grown greedy tree, not among all possible binary trees. Verify
this claim against exhaustive pruning enumeration on small invented trees.

Record unpruned and selected CART depth, leaves, risk and J; record the same final
quantities for GOSDT. Equal depth caps need not produce equal leaf counts. Do not
infer a certified advantage from training Accuracy alone or from differently
normalized losses. Library/internal bounds and the adapter's recomputed objective
must be separate fields.

## 5. Partitions, nested B1 and hyperparameter selection

Keep the historical five outer folds on all 601 argument IDs, seed 20261019.
Verify saved fold identity before fitting. All CQ rows/model records and labels
of an argument stay together. Use one downstream tree per scheme.

For new-learner model selection, use three shuffled inner KFold partitions of
sorted eligible outer-training IDs for that scheme, random_state=20261019. Do not
stratify on rare verdict classes and thereby fail or silently change folds. With
fewer than three eligible IDs use min(3,n) where n>=2, recording this exception;
one-class/no-data cases follow section 7. Both learners share these exact folds.

The three folds concern selection of the new six-candidate learner grid. Preserve
B1's original internal fitting/feature cross-fitting settings (currently four
folds with its existing small-sample behavior); changing them would change the
agreed B3 input source. Record both nesting levels explicitly.

For an outer training context T and a scheme's inner validation IDs V, rebuild B1
with context T minus V. Build training features by upstream cross-fitting inside
that context and validation features using B1 fitted only on that context. Use
B1CrossFitter.partition/fit_data contracts. Validation IDs and outer-held IDs must
never enter a fit that produces their own held-out features, preprocessing or
hyperparameters. Other-scheme outer-training arguments may retain their original
B1 context, as in B3. B1 CQ-supervision eligibility is distinct from the downstream
terminal exclusions. Cache only by exact training context and preprocessing
configuration, and share these matrices/fits across both learners and all six
configurations; do not refit B1 per solver candidate unnecessarily.

Select by pooled inner OOF Macro-F1 across validation arguments of that scheme,
using canonical gold-supported classes, not the mean of three fold scores. On
score ties within 1e-12 prefer, in order: fewer total leaves across inner fits,
smaller maximum actual inner depth, smaller configured d, larger a, fixed grid
order. The inner candidate scores, status and structural tie-break quantities
must be retained. Do not inspect outer test performance to choose a configuration.
Refit the selected candidate on the eligible outer training set, recomputing
n, class weights, lambda and the constant-column mask there.

As in B3, three Popular Opinion/non sequitur items remain in outer scoring but
are excluded from terminal-constrained downstream training/tuning. Export IDs and
reasons per partition. Do not drop them from the 601-item primary evaluation.

## 6. GOSDT options, limits, status and certification

Technical checks used official GOSDT 1.0.4, worker_limit=1, model_limit=1,
balance=False with the explicit matrix, rule_list=False, non_binary=False,
reference labels absent, upperbound_guess=None, similar_support=False,
look_ahead=True, cancellation=True, uncertainty_tolerance=0 and verbose=False.
Retain this checked search setup; no reference ensemble or supervised guessing is
added. Record constructor and effective native configuration, including whether
an exposed Python option is actually forwarded by the installed wrapper.
allow_small_reg=False is compatible with 2/n and 5/n; assert no automatic clamp.

Accepted per-fit cap: native time_limit=60 seconds per GOSDT fit and **4 GB =
4,000,000,000 bytes** process memory, sequential fits. The user accepts the bounded design; the combined local execution preflight
remains an operational requirement, not a change to the scientific search budget. The API has no verified memory-limit constructor
argument: use an externally supervised process, e.g. a Windows Job Object for
committed/private memory plus RSS monitoring, or equivalent CPU-job controls.
Report the memory definition and both peak private memory/RSS where available.
The B1/parent process and serialized inputs are additional, not inside that
worker's cap; report full-pipeline resources separately.

GOSDT's timeout is cooperative: the one-second test returned at about 1.3 seconds.
Use an external watchdog, with a separately documented fixed grace allowance for
startup/export/cleanup; do not silently extend optimizer time. The accepted 5-second grace is for process/result handling, not a larger
scientific search budget. time_limit=0 means unlimited
in the checked release and is forbidden for bounded runs. Preserve partial result
and failure files on timeout or memory termination. Do not automatically retry
with more time, memory or another candidate grid after benchmark results.

Classify actual results, not just exceptions or a success exit code:

| Situation | Required action |
| --- | --- |
| Valid tree, CONVERGED, checked bounds and recomputed objective consistent | Use it; record certification restricted to this representation/objective and depth. |
| Valid replayable tree with TIMEOUT/NON_CONVERGENCE or nonzero unresolved gap | Use the incumbent; explicitly mark uncertified, preserve gap and elapsed time. |
| No finite, replayable, permitted-terminal tree | Solver failure; preserve native status, error, raw model and any bounds. |
| Single class, no usable features or mathematically certified root case | Exact degenerate/analytic result under section 7; no solver success/failure is invented. |

Store raw result status, model JSON, model_loss, lowerbound and upperbound. In the
checked API model_loss is unpenalized risk, while bounds concern the regularized
objective. Under timeout the returned feasible tree may improve on the separately
reported upperbound. Recompute J from that tree, preserve the raw upper bound,
and report an incumbent bound min(raw_upper,J) only when both are finite/valid.
Report a conservative incumbent-minus-lower gap with numerical tolerance; never
substitute raw model_loss for J. CONVERGED alone without a valid tree and
consistent bounds is insufficient for the adapter's certified status.

## 7. Missing classes, degenerate fits and genuine failures

Guard these cases before the library call; the stock single-class shortcut has
UNINITIALIZED status and zero loss/bounds, which are not the regularized objective.

- n=0: record insufficient training data and abstain. Never compute a/n, invent a
  training class, or infer it from held-out gold.
- K=1: output its single training label, depth 0, one leaf, R=0 and J=lambda;
  mark single_class, not solver failure or a native convergence certificate.
- All training predicates constant and removed: the exact optimum is a root
  choosing the minimum-cost training class (canonical order on ties). Export
  R and J; do not insert a dummy predictive feature just to satisfy the API.
- Large-penalty root: the current cost matrix has the same wrong-prediction cost
  within each true-class column. Let b_c=n_c*C[p!=c,c], sort these masses in
  descending order, and let b_(2) be the second largest. A root obtains risk
  sum(b)-max(b). Any L-leaf tree predicts at most L distinct classes, so compared
  with the best root it can reduce risk by at most (L-1)*b_(2). If the common
  effective lambda>=b_(2), no nontrivial tree can improve J. Return the exact
  optimal root and prefer one leaf on equality. In ideal balanced arithmetic
  b_c=1/K, so the condition is lambda>=1/K. Record the effective costs, proof
  values and analytic_root status, separately from a fallback.

The last guard is the minimal proposed compatibility remedy for tiny n: the
checked library can return FALSE_CONVERGENCE without a model at lambda=1 even
though the root is a valid optimum. It changes neither the grid nor the objective.
This explicit analytic shortcut is accepted; record it separately from solver success.
Use the common effective-matrix cost for the actual proof, with the stated
numerical tolerance; do not infer it just from nominal class balance.
The guard itself compares the stored effective values without widening the
inequality by a tolerance; tolerance is for audit consistency, not for declaring
a nearly optimal root an exact optimum.

A genuine no-incumbent failure may produce a separate operational training-only
constant predictor minimizing the same training cost, canonical order on ties;
with no training label it abstains. Mark its predictions fallback, never GOSDT
incumbents. Keep both raw solver failure and operational status visible. A tree
found only by a fallback does not satisfy optimizer validity.

Do not omit failed inner folds or select a candidate by averaging only successful
ones. A candidate lacking a valid tree or exact analytic result on any required
inner partition has incomplete selection evidence. Do not silently reduce the
six-candidate family or select using fallback scores: retain the operational
records but mark tuning incomplete and stop/defer the scientific run for review.
This conservative handling is accepted and does not authorize an automatic larger
budget. A missing valid final outer fit makes the primary scientific comparison
incomplete even if operational predictions cover all 601. It is not an ordinary
negative result for GOSDT. Uncertified but valid incumbents remain valid fits.

## 8. OOF endpoints, uncertainty and diagnostic analyses

Primary contrast: canonical OOF Macro-F1(GOSDT)-Macro-F1(matched CART) on all 601
arguments, with abstentions retained as errors. Average over gold-supported
classes following the canonical scorer. Use 10,000 shared paired argument
bootstrap draws, seed 20261019, historical generator 1. Each argument and all its
system predictions move together. Ordinary percentile IC95%, one primary contrast;
do not inherit the old m=4 correction or silently add primary tests.

Accuracy is secondary. Report coverage and a conditional common-coverage comparison
on an explicitly fixed identical classified cohort. The indicative practical
threshold is +0.03 Macro-F1, separate from an interval excluding zero. Intervals
condition on fixed OOF predictions, not repeated training. Rare classes may
disappear within bootstrap samples; report valid/empty draws under the scorer.

Descriptive diagnostics: scheme, fine class, corpus, gold origin, no-NLAS,
reachable versus all gold, Good removed only from the class average while retaining
all arguments, and gold-non-Good arguments separately. Keep B1.hard@2.0 and historical
B3 point comparisons secondary. No subgroup can select an outer-test hybrid.

If an outer fit or required tuning is incomplete, label any all-601 fallback-based
scores as operational only; do not present their delta/CI as a completed primary
optimizer comparison. Provide solver-valid counts, analytic/degenerated counts,
uncertified counts and genuine failures per scheme/fold/candidate.

## 9. Reproducibility, rules and acceptance tests

Save a new configuration/protocol snapshot before fitting; input/software/code
hashes; seed and exact partitions; B1 contexts and label provenance; all six inner
scores/statuses; selected parameters; constant masks; class orders and effective
costs; lambda/n; full trees and exact predicates/CQ text/polarity; raw and weighted
leaf supports; training R/J; actual depth/leaves; bounds/gaps; times/memory; raw
errors/fallback reasons; and all 601 OOF predictions with provenance tags.

Independent replay must match every exported verdict, including exact threshold
boundaries and constant/analytic/fallback outputs. Compare float conversion and
branch direction to the actual library; do not apply sklearn's float32 threshold
rules blindly to GOSDT's Boolean branches. GOSDT's JSON true/false branches are
explicit. Preserve exact arrays passed to each learner and hash their equality.

Audit applicability, prerequisites, CQ dependencies, outside mass, repeated CQ
conditions, leaf support and semantic justification of the terminal. Do not count
every missing manual-graph gate as a semantic violation. A sparse numerical tree
is not automatically Walton-valid. Independent adjudication and repeat-fit
stability are not claimed by this experiment.

Implementation acceptance must cover held-out CQ/gold mutation invariance with
fixed folds, nested training contexts and cache identity, train-only constant
removal, strict boundary comparisons, absent/one/zero-class behavior, no-feature
and analytic-root proof, candidate failure reporting, actual timeout/incumbent
behavior, memory termination, exact replay and canonical all-601 scoring.
Application lint/tests are required before an implementation commit. Do not
rerun real experiments merely to validate reporting or the adapter.

## 10. Technical verification completed on invented data

GOSDT 1.0.4 was loaded with an existing local CPython 3.12.14 interpreter. The
wheel and missing dependencies were placed only in scratch/gosdt_spec12_checks/vendor;
no new permanent Python environment or main-project dependency change was made.
The technical runtime uses sklearn 1.6.1, NumPy 2.3.5 and SciPy 1.16.3; record an
accepted production runtime separately. The earlier 3.13 source-build failure is
not proof of Python incompatibility and remains a separate observation.

| Check | Observed result / limit |
| --- | --- |
| Multiclass custom matrix, labels 10/20/30 and unequal supports | Correct orientation matches balanced weighted risk; transpose gives a different cost. Recomputed J agrees with converged bounds to native precision. |
| CART pruning | 72 initial checks agree with exhaustive enumeration of rooted prunings. Another 140 checks use materialized native costs/penalties; 70 also exercise the exact-root condition. |
| Depth convention | Binary parity cases require depths 1–4; public depth_budget matches CART decision depth, and exported lengths respect the cap. |
| Timeout | time_limit=0 did not cap the fit; time_limit=1 returned a valid TIMEOUT incumbent. Status, not an exception alone, determines the outcome. |
| Objective fields | model_loss is risk; bounds include leaf penalty. A timed-out incumbent can have a tighter feasible objective than raw upperbound. |
| Single class / no feature / tiny fit | Stock single-class status is UNINITIALIZED; zero columns and empty arrays are rejected. A tiny high-penalty case gives FALSE_CONVERGENCE, motivating the exact root guard. |
| Fixed predicates / constant columns | Strict equality behavior and training-only masks verified on invented arrays. |
| Three-way downstream selection and existing B1 | A 30-item invented fixture exercised three partitions, identical cached inputs and held-out-label mutation invariance; original upstream four-fold behavior retained. |
| Six-candidate resource profile | 96 invented items, 9 invented CQs (45 predicates, exceeding the actual per-scheme maximum), three inner folds: 18 fits, all CONVERGED, about 136.9 s including test-loop work; slowest fit about 34.8 s; peak working set about 395.7 MiB. |
| External memory supervision | Windows Job Object and RSS supervision verified; the resource-profile run used a 4 GiB cap, while the final unit check verifies the exact proposed 4,000,000,000-byte cap. These are process limits, not whole-pipeline memory measurements. |

The scalar scores produced on invented data test mechanics only; they are not
fallacy-detection results or a forecast of benchmark performance. An easy/small
synthetic profile does not guarantee a real fit finishes within 60 seconds or
4 GB. Raw checks, statuses, package code and resource records are local under
scratch/gosdt_spec12_checks; no real response tables, workbooks or evaluation
predictions were read by these technical scripts.

## 11. Campaign cost, freeze gate and next authorized action

With eight schemes, five outer folds and six candidates on three inner folds,
the maximum is 8*5*(6*3+1)=760 GOSDT fits, before degenerate skips. The selected
outer fit counts once. The solver-time ceiling at 60 seconds is 45,600 seconds
(12 h 40 min). B1 nested fitting/cross-fitting, imports/worker startup, encoding,
CART, exports, bootstrap and watchdog grace are additional. With a 5-second grace
for each fit, the theoretical 760*(60+5) envelope alone is about 13 h 43 min.
Caching upstream features across algorithms/candidates is necessary.

The synthetic fixture confirms that the six configurations and three-fold loop
are executable within the per-fit trial limits. It does not establish the full
real campaign's elapsed time, B1 memory peak or feasibility on a laptop with
limited free RAM. Do not extrapolate synthetic elapsed time into a guaranteed
one-hour or overnight run. No benchmark score may choose or enlarge this budget.

**Accepted execution direction, later 30 October:** local PC first, because an
existing compatible interpreter and scratch-loaded wheel have already passed the
technical checks. The 12 h 40 min figure is an aggregate worst-case solver ceiling,
not an observed duration estimate. Do not change depth, regularization, folds or
budgets in response to benchmark scores. No cluster job is submitted by this decision.

At the site check the PC had 7.88 GiB total RAM, 1.58 GiB free and four logical
processors. Historical B3 peak process memory was about 1,002 MiB and its elapsed
time about 991 s; the GOSDT synthetic profile peaked around 396 MiB. These peaks
were measured separately and do not certify the combined production run or its
worst-case memory. Do not start the long run under the observed low-free-memory
condition just because individual fixtures passed.

The implementation must include a non-benchmark resource preflight covering B1,
the GOSDT worker, process/input overhead and current available memory, plus
sequential execution and monitoring. Do not force-close user applications. If
resources are inadequate, stop before real fitting and discuss closing applications
or moving to a CPU cluster. Any low-memory stop during execution is a resource
failure with preserved artifacts, not a negative scientific result. Do not silently
reduce the agreed grid or automatically transfer work to a cluster.

**Implementation and later site preference, 30 October:** the production runner
implements the fixed design, matched binary arrays, training-only masks, exact
costs/pruning, nested upstream contexts, visible incomplete tuning, and resumable
hash-checked records. Both learners compare the exported tree replay with their
actual library predictions on inner validation and outer held-out examples.
Analytically exact roots use independent constant prediction. The code provides
rule-review flags; these do not constitute independent semantic adjudication.

The local engineering gate requires at least 3 GiB free physical memory before
benchmark loading and each outer fold, and 2 GiB before a new scheme group. Workers
are stopped if free system memory drops below 1 GiB, in addition to their hard
4,000,000,000-byte private-memory cap, RSS cap and wall-clock supervision. These
are conservative safeguards, not a measured combined peak guarantee. They do not
change the scientific grid. A local-only runtime probe uses invented inputs.

`argfallacy evaluate-gosdt configs/gosdt_schemes2_diagnostic.yaml` requires explicit
`--worker-python` and `--worker-vendor` paths. `--check-only` writes resource/runtime
readiness under scratch and performs no experimental fitting. The normal command
creates a new evaluations directory; `--resume` requires unchanged code, inputs,
runtime and recorded outputs. Completed solver checkpoints are reused only with
identical training/validation arrays and configuration; unfinished workers require
review rather than an automatic retry. Full records include continuous upstream
features, Boolean arrays/masks, all candidate fits, selected configurations, trees,
rules, predictions, status/bounds, resource supervision and fold-specific B1 choices.

The PC subsequently had only about 1.36 GiB free. The preferred next site is now
the CPU cluster. The first read-only SSH check timed out; a later connection reached the host
but noninteractive authentication was refused. No remote environment, partition
or allocation was verified and no job was submitted. Verify connectivity and Linux
resource enforcement before transferring or starting the experiment. No fit on
any of the 601 arguments has been performed.

## Sources and local evidence

- Official implementation and installation: https://github.com/ubc-systopia/gosdt-guesses
- Release files: https://pypi.org/project/gosdt/1.0.4/#files
- CART API: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- Local checked wheel source: scratch/gosdt_spec12_checks/vendor/gosdt/_classifier.py and _tree.py
- Local checks and results: scratch/gosdt_spec12_checks/checks.py, unit_results.json,
  profile_results.json, unit_supervision.json, profile_supervision.json,
  upstream_check.py and upstream_results.json
  (with precision_results.json and native_cost_pruning_results.json for numerical checks)
