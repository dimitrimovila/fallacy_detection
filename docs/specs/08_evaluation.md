# Spec 08. Evaluation

Version 2.2, 25 October 2026. Version 2.1 dates from 19 October, version 2 from 11 October 2026, version 1 from 10 and 11 September 2026. The spec holds the requirements already decided, the evaluation protocol and the implementation stages. Version 2.2 adds the scoped completion of stage two, specified after the existing round-two results and before reading the new human annotations. The changes, with their reasons, are listed at the end. Withdrawn requirements keep their number, so that references to P10, P11 and P12 elsewhere stay valid.

## Requirements already decided

1. **Test set of the experiments without the near duplicates.** All the metrics of the experiments are computed on the test set of `items.csv`, that is, the rows with `in_test_set` true. The test set does not contain the five near duplicates (`docs/data.md`, section 5.2): a near duplicate counted twice weighs double and, if a model gets one text wrong, it usually gets the other one wrong too. The near duplicates enter no experiment. The only exception is the reproduction of the numbers of Eleni's diagnosis, which serves only to validate the scorer and uses all the rows of Eleni's `results.csv` files as they are.
2. **Eleni's baseline recomputed on the test set of the experiments.** Eleni's results against which the experiments are compared (pipeline and zero-shot of her four models) are recomputed from those `results.csv` files on the items of `items.csv`, that is, on the same test set as the experiments. The exact reproduction of the numbers of the diagnosis on all of Eleni's rows remains an internal test of the scorer: it validates the computation and does not appear in the results. Eleni's two rows with the same text (ids 469 and 495, a single item in `items.csv`, `33f6dfaf9c4e41f1`) become a single row in the baseline: the row with id 469 is kept, in every model and condition. The two rows give a different verdict in one run out of eight (deepseek zero-shot: Appeal to Authority for 469, Ad Populum for 495); in the other seven the verdict is the same.

## Evaluation protocol

It rewrites the evaluation section of the thesis plan as requirements. Numbers checked on 11 October 2026 against `data/items.csv`, `data/annotations.csv` and full run v2 (`runs/20261005T193017Z_full_v2`). Unless stated otherwise, every number holds on the test set of the experiments (601 items, requirement 1), and every number that goes into the thesis comes from `argfallacy.eval`. A *system* is anything that produces a verdict per item: an aggregator (A0 to A5), the zero-shot, a comparison line. The *main table* is defined in P10.

P1. **Conditions on the scheme.** Every system is evaluated in two conditions: gold scheme, which isolates the aggregation, and scheme predicted by stage one, which measures the full system. The two conditions are always reported separately.
* A system that uses one model uses the scheme predicted by the stage one of that model (spec 04: stage two is called only for the items whose predicted scheme differs from the gold one; the other items reuse the answers of the gold condition). On full run v2 that is about 8,700 calls over the five models.
* A system that combines several models uses, for each item, the scheme predicted by most models (ties: the scheme of the reference model of P10), and combines only the models whose own prediction is that scheme. No call is needed beyond the previous point.
* An item whose predicted scheme is `none` gets no verdict and counts as an error.

P2. **Label spaces.** Two spaces, defined in `labels/fallacies.yaml` under the key `spaces`:
* `fine`: 25 classes, that is, all the terminals of the diagrams, Good Argumentation included. 23 of them appear in the gold: `definist_fallacy` and `false_attribution` have zero support.
* `collapsed_symmetric`: 20 classes. The six variants of ad hominem (abusive, circumstantial, guilt by association, tu quoque, ad fidentia, poisoning the well) merge into `ad_hominem`. 18 of them appear in the gold.
* `collapsed_compat` (21 classes, tu quoque separate) serves only the `compat` mode of the scorer, to reproduce Eleni's numbers, and does not appear in the results.

Every result is in the fine space. The collapsed space is reported for the main table only: it shows how much of the error is a confusion between variants of ad hominem.

Accuracy is computed on all the items. Macro F1 is the unweighted mean of the per-class F1 over the classes with support in the gold of the items scored. It is the mean of the F1 values, not the F1 of the mean precision and mean recall: the two definitions can differ by up to 0.5 (Opitz and Burst 2019). Balanced accuracy is the mean per-class recall over the same classes. The classes without support are listed as excluded, and a prediction of one of them remains an error.

P3. **Breakdowns by scheme and by corpus.** The main table is reported by scheme and by corpus, in the fine space, with the number of items and of classes beside each row: an accuracy on a corpus with few classes is not compared with one on a corpus with many classes without saying so. Corpora with fewer than 20 items have no breakdown of their own: they enter only the total, and the report says so with the name of the corpus and its number of items. Today, fine space with the collapsed one in parentheses: LOGIC 192 items and 18 classes (13), NLAS 197 and 16 (12), EthiX 144 and 10 (10), Reddit 67 and 4 (4); ElecDebate, with 1 item, is only in the total. By scheme: ad hominem 94, analogy 83, cause to effect 45, correlation to cause 11, example 86, expert opinion 78, popular opinion 109, slippery slope 95. The 20-item rule applies only to the corpora: every scheme is always reported, with its number of items beside it, and the report flags every scheme with fewer than 30 items (today correlation to cause).

P4. **Ablation without NLAS.** NLAS is a synthetic corpus generated by LLMs. The main table and the deciding comparisons of P10 are recomputed without the NLAS items (404 items remain) and reported next to the main result. If the conclusion changes without NLAS, the thesis says so.

P5. **Uncertainty and comparisons.** Every metric of the main table and of the breakdowns has a 95 per cent bootstrap interval (P12). Comparisons between systems are paired on the same items, with the paired bootstrap of P12 on the difference; the exact McNemar test is a check on differences in accuracy. Only the deciding comparisons of P10 can support a claim of improvement, and they carry the Bonferroni correction of P10. Every other comparison is reported as exploratory, with a 95 per cent interval, and claims nothing on its own: with many systems, some intervals exclude zero by chance (Dror et al. 2017).

P6. **Withdrawn on 11 October 2026.** It was a second evaluation on the 113 workbook items outside the test set with a terminal verdict of Enrico. Stage two never ran on those items. Their labels come from the same procedure as the gold of the test set, Enrico applying the diagrams, so they are not an independent check; and 113 items give intervals too wide to add information.

P7. **Per CQ.** For every model and every CQ, on the test items of the scheme of the CQ (11 to 109 items):
* Distribution of the hard answers (yes, no, cannot be determined, not applicable; positive and negative for ad hominem CQ1).
* Agreement among the models: Krippendorff's alpha, nominal, over the hard answers, and the share of items on which all the models give the same answer. Alpha is always read beside the distribution of the answers, because one dominant answer pushes alpha (and kappa) down even when the models agree (Feinstein and Cicchetti 1990).
* Against Dumitru's answers in `annotations.csv`, taken from the sheet of the gold scheme of the item, with `idk` read as cannot be determined:
  * the share of hard answers equal to the annotation, shown when at least 20 items are annotated;
  * the Brier score of the probability of yes (of positive for ad hominem CQ1), on the items where the annotation is yes or no (positive or negative), separately for each probability the entry has: `p_logprob`, `p_verbal`, and `p_sample` for the entries with several samples. The Brier score is a strictly proper scoring rule (Gneiting and Raftery 2007);
  * calibration shown as one reliability diagram per model and probability, pooled over the CQs, with ten bins of equal count (Nixon et al. 2019). No ECE number: the binned estimator underestimates the calibration error and changes with the number of bins (Kumar et al. 2019), and a CQ has 11 to 109 items.
* Two descriptive columns, from the answers of the models only, without gold:
  * reached: the share of items whose A0 path asks the CQ;
  * weight on the verdict: the share of items whose A0 verdict changes when the answer to that CQ alone is flipped (yes and no swapped, positive and negative swapped), averaged over the models.

P8. **Human agreement.** Krippendorff's alpha, nominal, and the raw agreement, with the distribution of the labels beside them:
* between Enrico, Eleni and Dumitru on the verdicts;
* Dumitru against Eleni per CQ, on the sample annotated by Eleni;
* Dumitru against himself, on the rows re-annotated blind.

Alpha takes any number of annotators and missing values (Hayes and Krippendorff 2007). Kappa is not reported: it repeats alpha and suffers more from the prevalence problem of P7. An empty cell is not a verdict (`docs/data.md`). Each part is computed when its data exist; the report names the parts still missing.

P9. **Withdrawn on 11 October 2026.** It was the estimate of uncertainty from the justifications of the models and its relation to human disagreement. As written it was an open research question, not a measurable requirement. The measurable signals of uncertainty (stated confidence, agreement between samples, agreement between models) are in P7 and P11. If it is taken forward, it gets its own specification.

P10. **Success criterion.** Version 1 fixed it before any result, with accuracy as the deciding metric. On 11 October 2026 the deciding metric became macro F1, after the results of A0 and of the zero-shot on full run v2 and before any result of an aggregator; the reason is in the list of changes. The criterion holds for the decision of 26 October (thesis plan, decision point).
* Baseline: A0, the fixed diagram with the hard answers of a single model, which reproduces the setting of the diagnosis.
* Reference model: the model whose A0 has the highest macro F1 in the gold scheme condition; with equal macro F1, the first model name in alphabetical order. It is written in the manifest of the evaluation. On full run v2 it is Qwen3.8 27B (macro F1 0.350), which also has the highest accuracy (0.504).
* Primary configurations: before any result of an aggregator, spec 07 names one primary configuration for each of A1 to A5. Their number m is fixed with them (5 if all five exist). An aggregator dropped later does not lower m. Spec 07 also says which deciding comparisons answer the question of 26 October.
* Round 2 has its own primary configurations and m = 4, fixed in spec 09. Its comparisons form a separate family and do not replace those of round 1.
* Deciding metric: macro F1 in the fine space (P2), gold scheme condition (P1). The same comparison in the predicted scheme condition serves as confirmation.
* Deciding comparisons: each primary configuration against the reference A0, paired on the same items, with an interval on the difference in macro F1 at level 1 − 0.05/m (99 per cent with m = 5, Bonferroni). An improvement is claimed only if that interval excludes zero. An interval that contains zero is reported as the absence of a measurable difference, not as a small improvement.
* Reported beside the deciding comparisons, without deciding: the same differences in accuracy (for comparability with Enrico's and Eleni's numbers) and in balanced accuracy; the paired difference between each primary configuration and the zero-shot of the reference model; the majority class per scheme and the in-sample ceiling, both declared as oracles.
* Diagnostic comparison: an aggregator that uses a single model is also compared with the A0 of the same model, with the same paired comparison, and the result is reported next to the first one. It does not decide success: it separates two causes. If the aggregator does not beat the A0 of its own model, the negative result comes from the aggregation; if it beats it but stays below the A0 of the reference model, it comes from the choice of the model.
* Main table: one row per system (A0 of every model, the primary configurations, the zero-shot of every model, the majority per scheme, the ceiling); columns accuracy, macro F1, balanced accuracy, recall and precision of Good Argumentation, coverage, each with its 95 per cent interval. One table for the gold condition and one, with the same rows, for the predicted condition.

P11. **Abstention curves.**
* Confidence: for a system that returns a distribution over the terminals (A2, A3, A4, and A5 when it returns a probability), the confidence of an item is the probability of the highest verdict; for the zero-shot, `p_label` of the verdict chosen. With a threshold, the system gives a verdict on the items with confidence at least equal to the threshold and abstains on the others; coverage is the share of items with a verdict. A0 and A1 are a single point, at full coverage, except when the diagram stops on `idk` with the `stop` policy, which counts as an abstention. The margin between the two most probable verdicts is computed as a check.
* Curve: the items sorted by decreasing confidence, ties by `item_id`, so that the curve is deterministic. For k from 1 to N, with e the number of errors among the first k items: coverage k/N, selective risk e/k, generalized risk e/N. Errors are in the fine space, gold scheme condition, as in P10.
* Summary number: AUGRC, the mean of the generalized risk over k, lower is better (Traub et al. 2024). It reads as the average share of items that receive a wrong verdict instead of an abstention. AURC, the mean of the selective risk (Geifman and El-Yaniv 2017), is reported beside it for comparability with earlier work, but it is not the summary number: it over-weights the errors made with high confidence, and it can improve when the classifier gets worse (Traub et al. 2024).
* Coverage levels, fixed before the results: 100, 95, 90, 80 and 70 per cent. At level c, every system gives a verdict on its first round(c·N) items; the selective accuracy at each level is reported with its 95 per cent interval, as a description. The levels stop at 70 per cent: a system that abstains on more than 30 per cent of the cases is of little practical use.
* Comparisons: two systems are compared at equal coverage, never at equal threshold, because their confidences do not have the same scale. The paired differences in AUGRC are exploratory (P5).
* Next to every curve, the composition of the abstained items: share of Good Argumentation, scheme, corpus.

P12. **Bootstrap.**
* 10,000 resamples, 95 per cent percentile interval (other levels where P10 says so).
* A single seed, written in the evaluation configuration (`configs/evaluation_v2.yaml` for full run v2) and in the manifest. The indices of the resamples are generated once and hold for all the systems and all the models: this is what makes the comparisons paired.
* The unit is the item, without strata: the items of the test set are resampled with replacement, and all the answers and verdicts of an item move together. The breakdowns too, by scheme and by corpus, are computed on the same resamples of the whole test set, so the number of items of a breakdown varies from one resample to another; a resample with no items of a breakdown does not contribute to the interval of that breakdown. Reason: stratifying would keep the number of items per scheme fixed, and the interval would not include the uncertainty on the sample size, which with the 11 items of correlation to cause is exactly what must be shown.
* The verdicts of the learned aggregators are computed once (P13) and resampled as they are, so the intervals do not include the variability of training them.
* The exact McNemar test remains as a check on the paired comparison of two accuracies at full coverage.

P13. **Out-of-fold estimation.** Every parameter estimated from labels of the test items (gold verdicts or Dumitru's per-CQ answers) is estimated out of fold, so that no verdict depends on its own label (Cawley and Talbot 2010).
* One assignment of the 601 items to 5 folds, stratified by gold scheme, drawn with the seed of P12, shared by every aggregator and every model, and written in the manifest.
* The verdict of an item in fold f comes from parameters estimated on the other four folds. Choices inside an aggregator, such as the hyperparameters of A3, are made by cross validation within those four folds.
* It applies to the node order of A1, to A3, to the weights and the stacking of A4, and to any threshold chosen on labelled items.
* The majority per scheme and the ceiling stay in sample, declared as oracles. On full run v2 the leave-one-out majority gives the same accuracy as the in-sample one, 0.579.

P14. **Output.** `argfallacy evaluate CONFIG` writes into `evaluations/<name>/`, which git ignores, as it ignores `runs/`:
* `manifest.json`: the runs used, the seed, the number of resamples, the folds, the reference model, the primary configurations and m, the commit of the code, the date.
* `verdicts.csv`: one row per system and item, with the verdict, the confidence, the abstention and the explanation (spec 07, section 9). `a1_orders.csv` and `a3_choices.csv`: the order of A1 and the C of A3 chosen in each fold and scheme.
* One CSV per table, numbers with four decimals.
* `summary.html`, one self-contained page (no external resources, English labels) with every table: metrics between 0 and 1 on a sequential color scale, differences on a diverging scale centred on zero, the interval written under each value, and under each table a note that defines its columns. Tables, in order:
  1. run card;
  2. main table, gold and predicted condition (P10);
  3. deciding, diagnostic and zero-shot comparisons (P5, P10);
  4. main table in the collapsed space (P2);
  5. breakdowns by scheme and by corpus (P3);
  6. without NLAS (P4);
  7. F1 per class and number of predictions per class;
  8. per CQ (P7), with the reliability diagrams;
  9. human agreement (P8);
  10. abstention: AUGRC, AURC, selective accuracy at the coverage levels, composition of the abstained items, and one figure with the risk coverage curves (P11);
  11. integrity: invalid samples, cannot be determined, not applicable, items without a verdict;
  12. scheme recognition of stage one.

The run report (`argfallacy report run`, spec 04) remains the diagnostic report of a single run; the evaluation report does not repeat its sections on cost and parser.

## Implementation

The protocol is implemented in two stages. The first gives what the decision of 26 October needs; the second completes the protocol in phase 4.

### Stage one, for the decision of 26 October

* What: the gold scheme condition of P1; the fine space of P2; P4; P10, that is, the main table in the gold condition, the deciding comparisons, what is reported beside them and the diagnostic comparisons; the exploratory comparisons of spec 07, section 8, with 95 per cent intervals (P5); P12; P13; from P14, the manifest, the CSV files of these tables, `verdicts.csv`, `a1_orders.csv`, `a3_choices.csv`, and `summary.md`, a markdown page with the tables of stage one, in place of `summary.html`.
* Rows of the main table: A0 of each of the five entries; the primary configurations whose results exist; the zero-shot of each entry that has a zero-shot run (`summary.md` names the entries without one); the majority per scheme; the ceiling of each entry. Columns as in P10.
* The evaluation configuration of full run v2, `configs/evaluation_v2.yaml`:

  ```yaml
  name: decision_v2
  runs:
    answers: 20261005T193017Z_full_v2
    zeroshot:
      - 20261009T221950Z_zeroshot_v1
      - 20261021T111606Z_zeroshot_k2
    judge:
      - 20261021T111110Z_judge_v1
      - 20261021T111110Z_judge_v1_answers_only
  reference_model: qwen3_8_27b
  seed: 20261019
  resamples: 10000
  folds: 5
  m: 5
  ```

  `reference_model` is a guard: the command computes the reference model as P10 defines it, and stops if the two differ. `judge` lists the runs of `configs/judge_full_v2.yaml` and `configs/judge_full_v2_answers_only.yaml` once they exist. The command reads `summary.csv` and `manifest.json` of each run, never `raw.jsonl`; of the zero-shot and judge runs, stage one reads only the hard answer.
* `runs.zeroshot` accepts one run ID (backward compatible) or a list of run IDs.
  Runs in the list must use the same scheme folder, tag, content hash and prompt versions;
  each model entry may occur in only one source. Duplicate verdict rows are rejected.
  The evaluation manifest records `zeroshot_sources`, model entry to source run ID,
  and `software_versions` for numpy, pandas and scikit-learn.
  A judge must refer to the configured answers run and its scheme folder; duplicate
  with-justifications or answers-only judge inputs are rejected. Only gold-condition
  rows of the requested verdict stage are read. Missing predictions count as errors.
  The completed configuration replaces the previous output in `decision_v2`.
* Folds (P13): three generators come from the seed, `numpy.random.SeedSequence(seed).spawn(3)`: the first for the folds, the second for the resamples of the 601 items, the third for the resamples of P4. The items of the test set, in order of `item_id`, are taken one gold scheme at a time, the schemes in alphabetical order; the items of each scheme are shuffled with the first generator and dealt in turn to folds 0, 1, 2, 3 and 4, continuing from one scheme to the next, so that the sizes of the folds differ by at most one item, and so do the sizes of each scheme in the folds.
* Resamples (P12): one array of B by N indices into the items in order of `item_id`, drawn once, used for every system. A metric on a resample counts each item as many times as the resample draws it, with the definitions of spec 06 and P2; the macro averages run over the classes with support in the resample. A vectorized implementation of the metrics is expected; on the observed sample, each item once, it equals the canonical scorer.
* Intervals: the percentiles of the B values with the default method of `numpy.percentile`: 2.5 and 97.5; for the deciding comparisons 100·α/2 and 100·(1 − α/2), with α = 0.05/m.
* P4: the same tables on the 404 items without NLAS, with the verdicts of the aggregators as they are (no training again) and the resamples drawn for those items.

Acceptance tests of stage one, on data written by hand, with no run:
1. Folds: the same seed gives the same assignment; every item is in exactly one fold; the sizes of the folds, and of each scheme in the folds, differ by at most one.
2. Resamples: drawn once, and every system is scored on the same indices.
3. Metrics: on the observed sample, the vectorized accuracy, macro F1, balanced accuracy, and precision and recall of `good_argumentation` equal those of the canonical scorer within 1e-12, on predictions with `invalid` answers, missing verdicts and classes without support; on a resample with repeated items, they equal a computation made by hand.
4. Deciding comparison: two identical systems give the interval [0, 0] and no improvement; a system right on every item, against a reference wrong on half of them, gives an improvement; the level of the interval follows m in the configuration (percentiles 0.5 and 99.5 with m = 5).
5. The reference model is computed from the A0 of the entries, and the command stops when it differs from `reference_model`.
6. Without NLAS, the frozen data give 404 items.
7. `verdicts.csv` has one row per system and item, and its rows `A0:<entry>` equal the `diagram` rule of the run report.
8. The manifest has the runs, the seed, the number of resamples, the fold of every item, the reference model, the primary configurations and m, the commit of the code and the date.

### CQ and human agreement implementation

P7 and the available parts of P8 are written by `argfallacy evaluate` alongside
its existing outputs. `argfallacy report cqs RUN_ID --out OUTPUT` refreshes only
these diagnostics and also works on revised schemes without a burden table.
Outputs: `cq_metrics.csv`, `cq_agreement.csv`, `cq_calibration.csv`,
`cq_calibration.svg` when binary human targets exist, `human_agreement.csv` for
the original-scheme evaluation, `cq_manifest.json` and `cq_report.md`.

Human CQ sheets are identified by their complete original CQ set and restricted
to the item's gold scheme, even when the annotator judges that scheme absent.
Identical duplicate ratings count once; conflicting CQ ratings stop computation.
Original annotations are not applied to revised questions. Human verdict ratings
from repeated sheets count as one per item and annotator; conflicts and nonterminal
coarse labels are excluded and counted. Missing ratings are omitted from alpha;
singletons do not enter its coincidence matrix. Alpha is undefined with no expected
disagreement; raw agreement counts observed rater pairs. Unanimity uses only items
with a valid answer from every configured model and reports that denominator.

The Brier targets are binary human answers. Parsed probabilities already point to
yes/positive; no complement of `p_verbal` is applied. Reliability bins have equal
counts (up to one), splitting tied probabilities deterministically. Flip sensitivity
is reported per model and averaged across models. Human CQ agreement below 20
annotations is withheld. Model distributions retain missing and invalid responses;
missing predictions disagree with annotated answers. Missing second-annotator CQ
ratings and unidentified blind repeats are explicitly reported. These descriptive
analyses do not change the deciding comparisons of P10.

### Visual presentation of available results

`summary.html` is a self-contained English presentation of the available saved
metric tables, comparisons, CQ diagnostics and human agreement. It is written by
`evaluate` and can be refreshed without recomputing metrics with
`argfallacy report summary EVALUATION_DIR`. Optional `--judge-audit DIR` and
`--revised-cqs DIR` include the four-judge comparison and revised questions.
The renderer keeps the saved values and intervals, embeds calibration figures,
records table fingerprints and labels oracle baselines and unavailable conditions.
It does not yet implement every table listed in P14.

### Round 2

`argfallacy evaluate configs/evaluation_round2.yaml` computes the family of spec 09 into
`evaluations/round2/`, with the folds and resamples of this spec and its own m (P10).
It implements P11 for the systems of spec 09, section 7: items sorted by decreasing
confidence, ties by `item_id`; an item with a verdict but no confidence follows every
item with one, and an item without a verdict comes last and counts as an error; the
level c keeps the first round(c·N) items, halves rounded up. `abstention.csv` has
AUGRC, AURC and the selective accuracy at each level with 95 per cent intervals, and
the AUGRC of the margin as a check for the systems with a distribution;
`abstention_curves.csv` the observed curves, drawn in `summary.html`;
`abstention_composition.csv` the abstained items by Good Argumentation, scheme, corpus
and gold origin. The breakdowns of the deciding comparisons follow P3 and P12.

The D+Q evaluation of spec 10 runs after these tables are written, in a separate
module, with the same item order, folds and bootstrap draws. The fitted round-two
objects are released before this part starts. The independent without-NLAS draws
are generated once with round two and reused here too. No D+Q system enters an
existing round-two CSV; all sixteen CSVs of spec 10 section 8.2 remain byte-identical
with the four new run keys absent, empty or populated.

`runs.zeroshot_schemes_2`, `runs.zeroshot_rerun`, `runs.dq` and
`runs.dq_schemes_2` are optional lists, initially empty. Their manifests must name
the expected schemes folder, its current content hash and the expected prompt
version. A mismatch stops evaluation. A run whose reference entry lacks verdict
rows for eligible items excludes its systems and records the missing item ids;
other run groups remain available. Present invalid or empty verdicts count as
wrong. Non-reference entries are also checked for complete item coverage.

The new systems are the declared D+Q verdict, `DQ.path` from A0 on its hard CQ
answers, and `DQ.soft` from `A2.pruned` at threshold 0.02 with the burden table of
that version. The source follows each entry's regime; entries relying on verbal
confidence have no DQ.soft because CQ rows carry no stated confidence. Consistency
requires a non-stopped path equal to the declared verdict. Departures identify the
first chosen arc losing that verdict, or the stop; absent declared verdicts have
no departure CQ and are reported explicitly. A tied positive maximum counts in
the share whose declared verdict is most probable under DQ.soft.

H_DQ compares the reference model's D+Q 2.0 with its matched zero-shot 2.0, on macro
F1 with the 2.5 and 97.5 bootstrap percentiles. It is its own family of one, separate
from m = 4. Only an interval above zero supports improvement. The reading is
`robust` only when the D+Q 1.1 minus repeated-zero-shot point estimate is positive;
otherwise an improvement is `possibly tuning`, including when that control is
missing. Other outcomes have reading `not applicable`. Accuracy, balanced accuracy,
origin and scheme breakdowns, and without-NLAS results are reported beside it.
The named secondary and exploratory comparisons of spec 10 section 6.2, including
S-soft against the zero-shot 2.0 menus, do not decide success.

For the CQ comparison, the shared binary-label selection and item-sum bootstrap
of spec 09 section 2.5 are reused. Both sources use the parsed `p_logprob` of the
first answer, including outside-arc mass, as named in spec 10; it is not conditioned
on choosing an arc. Hard agreement includes all labelled pairs; Brier excludes
missing probabilities without imputation and reports `brier_n`. Paired Brier
differences use the pairs with a probability in both sources. The new CQ file
distinguishes level and paired-difference rows. Scheme and origin shares filter
whole-test draws; empty resamples do not contribute to their intervals.

The outputs are `dq_verdicts.csv`, `dq_main_table.csv`, `dq_deciding.csv` (always
written, with outcome `missing` until its inputs exist), `dq_deciding_breakdowns.csv`,
`dq_exploratory.csv`, `dq_consistency.csv`, `dq_departures.csv`, `dq_cq_level.csv`,
`dq_cq_changes.csv`, `dq_replication.csv`, `dq_abstention.csv`,
`dq_abstention_curves.csv`, `dq_routing_curve.csv` and `dq_choices.csv`.
The zero-shot replication split is found by rendering
both prompts. `manifest.json` adds `dq`, and `summary.html` adds a final D+Q section
from these saved tables, with abstention curves as a figure. The exploratory C2
families of spec 10 section 7 use the same folds and resamples, with C1 cross-fitted
inside each training partition. S-soft enters only the family of schemes 2.0.

### Remaining stage two work

P1 with the scheme predicted by stage one: it first needs a change to the client, which today plans the predicted condition with one prediction per item for every entry and reads no prediction from a run, while P1 asks for the prediction of each entry; then the calls. Then P2 in the collapsed space, P3, P11 for the systems of the first stage, the F1 per class, the integrity table, the table of stage one and the corresponding additions to `summary.html`.

### Scoped stage-two completion (planned, 25 October 2026)

The following scope governs the analyses for thesis-plan sections 6.4.1 and 6.4.2,
with results to be frozen on 6 November. It does not change saved results, primary
configurations, the deciding families, or their conditional reading rules. New
analyses are exploratory, with 95 per cent percentile intervals. P9 stays withdrawn:
the disagreement analysis below uses measured confidence, not justifications.

**Execution and provenance.** `argfallacy analyze configs/analyses_v1.yaml` writes
to `evaluations/analyses_v1/`. The configuration names the saved evaluations and
parsed runs, seed, resamples, processing batch size, and optional external inputs
`direct_verdict.answers`, `direct_verdict.samples`, and `dq`. Optional inputs default
to null. Paths are supplied in configuration, never embedded in modules; both human
files remain outside the repository. Existing evaluation directories, frozen data,
raw runs, schemes and fitted-system definitions are not modified. Read saved
out-of-fold verdicts and confidences; recover missing zero-shot and judge confidence
from the matching parsed runs, verifying the verdicts and recording `p_label` and
its lower-bound flag. Keep evaluation, system and scheme-version identities explicit.
No fitting is performed merely to compute a subgroup or a bootstrap replicate.

The manifest records input paths and SHA-256 hashes, source manifests and code
commits, analysis commit, configuration, software versions, folds, seed, resample
generators and item order, exclusions, actual denominators, and computed, pending or
deferred parts with reasons. A missing optional input produces a pending status,
never invented observations or zero-valued results; malformed supplied inputs fail.
Analysis code and synthetic acceptance tests are completed and committed before
new human annotations are opened. New annotations enter no fitted primary system.

**Gold origin and breakdowns.** Add total, fallacy-corpus (LOGIC, Reddit,
ElecDebate) and scheme-corpus (NLAS, EthiX) results to the thesis tables only:
main tables of both rounds; deciding and exploratory comparisons; CQ-level scores;
abstention summaries (AUGRC, AURC and fixed-coverage accuracy); human path and human
agreement; and the C2 routing curve. Existing rows by scheme or CQ retain that
detail within each origin. Do not add origin to S-soft answer changes or other
minor descriptive tables. Conformal summaries retain the origin and scheme views
already required by spec 09. Report item counts, CQ-pair counts where applicable,
supported and excluded classes, and undefined metrics. Classes absent from a
subgroup do not enter its macro averages; Good Argumentation support zero is
explicit. Preserve the scorer's numeric convention, with a separate undefined flag
for a metric whose denominator is zero.

Complete main-table breakdowns under P3 and the main table without NLAS for round
two under P4. New verdict comparisons are also reported by scheme and without
NLAS, on the common eligible items; preserve P3's small-group rules. Recompute
subgroup metrics from predictions, not averages of saved metrics. For C2 retain
the pooled out-of-fold threshold grid and thresholds selected for requested shares;
report each origin's actual explained share and scores at those same thresholds,
with intervals at the selected shares. Do not retune thresholds within an origin.

**CQ and human tables.** Extend spec 09 section 2.5 to individual CQs, in total
and by origin, retaining its binary-label selection and item-level pairing. Do
not conflate that population with the broader P7 population: identify both and
their denominators. Retain P7's minimum of 20 annotated items for displayed hard
human agreement; report smaller counts and withhold the estimate. Preserve
version compatibility: no 1.1 human target is applied to a rewritten 2.0 CQ.
S-soft CQ comparisons use only observed, compatible nodes and name that population;
an unvisited node is not an incorrect answer. Extend historical P8 agreement and
the human-path reference with origins and item-bootstrap intervals; keep historical
cross-sheet verdicts separate from paths computed under the gold scheme. Missing
second-rater CQ data and blind repeats remain identified as missing.

**Abstention.** Retain P11's ordered curve and its discrete-mean AUGRC/AURC,
including its rule for items without a verdict or confidence. Also export a curve
at every distinct finite confidence threshold: keep all tied items together, require
a valid verdict and confidence, and include the empty selection. Report threshold,
kept and error counts, actual coverage, selective risk and generalized risk. At
zero coverage selective risk is undefined and generalized risk is zero. This curve
does not replace the ordered-curve indices; record missing-confidence and missing-
verdict counts and the maximum operational coverage. Do not imply that each ordered
point is attainable by a threshold. Fixed-coverage counts use half-up rounding.

Cover the available confidence-bearing systems of both rounds, including A3,
all five zero-shot entries, both judges, A2, A4, round-two soft variants, C2, C3,
S-soft and its control. A partial system uses its declared eligible population;
comparisons use the intersection. Systems without confidence are reference points,
not artificial rankings. Compute ordered curves, effective-threshold curves and
summary intervals for total and both origins. Retain the margin check where a true
terminal distribution is available; a single chosen label is not such a distribution.
Paired AUGRC differences, system minus reference, are restricted to:
`C2` against `zeroshot:qwen3_8_27b`; `C4` against `A2.pruned`; `C1` against `A2`;
`A5` against `zeroshot:qwen3_8_27b`, in total and by origin. These comparisons do
not decide success. Summary intervals and intervals at fixed coverage are required;
pointwise confidence bands over the whole curve are not required.

At fixed coverages retain composition by scheme, corpus, gold origin and gold Good
Argumentation; add human disagreement and Enrico's difficulty when available.
Report counts, shares among abstained items, and rejection rates within each group.
For partially observed attributes report known and missing counts and denominators
among kept, rejected and all eligible items; unknown is not agreement or low difficulty.
Keep difficulty levels 1 to 5, validate one unambiguous rating per item, and report
Spearman correlation of difficulty with `1 - confidence`, with an item-bootstrap
interval, on rated items only. Constant inputs give an undefined correlation.

**Future human input and D+Q.** Accept the app CSV unchanged:
`item_id,position,verdict,answered_at,revision`, including `_final.csv`. Select the
greatest revision per item, and the last physical row on equal revisions; repeated
equal revisions must have the same verdict. Validate nonnegative integer revisions,
timezone-aware timestamps, canonical terminal labels and membership in the item's
1.1 menu. Validate positions and item membership against the direct-verdict rows of
the existing `samples.csv` format (`item_id,sample,scheme,round,position`). Accept a
partial sample; do not require every sampled item or a 200-item minimum for the
reader. The planned thesis sample is at least 200 items in blocks. Report answered
and planned counts, rounds completed or partial, and counts by scheme and origin;
identify a smaller sample without presenting it as the planned completed sample.
No extrapolation to unannotated items is made.

Primary disagreement is human direct verdict versus gold. Direct verdict versus
human path and versus other annotators are separate comparisons on common items;
the same annotator's direct/path difference is a procedural comparison. Report
confidence by agreement group and AUROC of `1 - confidence` for disagreement (ties
count half), with item-bootstrap intervals, and disagreement prevalence among kept,
rejected and all annotated items. Include the paired human direct/path comparisons
and every available system scored against the direct verdict as a second reference,
on explicit common populations; do not replace gold in the primary tables. These
are associations with observed disagreement, not estimates of a well-measured human
label distribution. Single-class disagreement samples have undefined AUROC.

Save conformal sets item by item, with evaluation/system/version, item id, fold,
terminal ids, size, gold inclusion and calibration threshold, using the unchanged
rule of spec 09 section 7. Link mean set size to agreement/disagreement once the
annotation exists. Aggregate conformal summaries are insufficient to reconstruct
item sets. Recover genuine terminal distributions or export sets from the existing
evaluation calculation to the new analysis directory; do not fabricate distributions
from a chosen verdict or summary. If deterministic replay of a fitted system is
unavoidable, separate it from bootstrap work, preserve its folds and settings,
verify predictions against the saved evaluation, and record replay provenance.
D+Q adapters are prepared with synthetic normalized inputs; actual run reading
requires the format and version contract of spec 10. Until then D+Q stays pending.
Keep the declared verdict and the CQ-derived diagram verdict as distinct systems.

**Other thesis outputs.** Export per-class precision, recall, F1, gold support and
prediction counts, including Good Argumentation and unsupported classes. For the
C1 diagnosis identify rare classes by fewer than 20 gold items in the full frozen
test set, before subsetting; report the class list and mean recall over supported
rare classes alongside C1 and its references. Export integrity counts for invalid,
missing, cannot-be-determined and not-applicable answers, and absent verdicts,
with explicit sample/CQ/item denominators. Reuse the existing stage-one scorer on
test items only. A collapsed main table is optional only through existing scoring
code without new scoring machinery; otherwise defer it to level 3. The full
predicted-scheme verdict condition remains unavailable, distinct from stage-one
scheme recognition. No new inference is part of these analyses.

**Outputs and resources.** One CSV per statistical table, with explicit evaluation,
system/version and subset keys: `main_table.csv`, `comparisons.csv`, `per_class.csv`,
`integrity.csv`, `stage1.csv`, `cq_level.csv`, `cq_level_differences.csv`,
`human_path.csv`, `human_agreement.csv`, `routing_curve.csv`, `abstention.csv`,
`abstention_curves.csv`, `abstention_thresholds.csv`, `abstention_differences.csv`,
`abstention_composition.csv`, `difficulty.csv`, `human_direct.csv`,
`uncertainty_disagreement.csv`, `conformal_items.csv`, and `conformal.csv`, plus
`manifest.json` and a self-contained English `summary.html`. Pending tables have
headers and manifest status, no placeholder measurements. Aggregated metrics use
four decimals; item confidences, thresholds and probabilities retain enough
precision to reproduce ordering and ties. The report reads saved tables and shows
definitions, denominators, intervals, and pending/deferred parts.

Reuse the recorded item ordering, folds, seed and 10,000 paired draws of P12;
origins and other subgroups filter those whole-test draws, and empty replicates
are excluded metric by metric with their counts recorded. Retain the existing
generator-2 protocol for the without-NLAS analysis. Shared indices may be retained
once or replayed deterministically in batches. Sum CQ contributions per item
before resampling, as in `eval/c1.py`; process systems and batches sequentially.
Never allocate resamples-by-CQ-pairs arrays or a resamples-by-items-by-systems
array, or retain per-system copies of full bootstrap workspaces. Chunking must not
change draws, ties, estimates or intervals. Record batch size, elapsed time and
peak process memory. The local target is an 8 GB PC: provide a bounded-memory
default and measure it before a full run. If replay or analysis remains too heavy,
provide a CPU-only batch entry point for the Mathematics cluster with explicit
configuration and resource requirements; do not submit jobs without authorization.

Acceptance checks use synthetic data before new annotations are read: origin and
class denominators; original metric reproduction; paired subgroup/bootstrap
alignment; individual-CQ versus pooled totals; tied and empty threshold selections;
the four AUGRC pairs only; missing attributes and constant-input statistics; app
revision/final-export equivalence and partial rounds; pending versus malformed
inputs; per-item conformal summaries matching aggregates; D+Q version gating; and
batch-size invariance. Verify protected input hashes and report regeneration from
saved tables. Specifications and code are committed before computing new results.

## Changes

From version 2.1 to version 2.2, on 25 October 2026, after the saved round-two
results and before reading new human annotations: scoped stage-two completion,
thesis-table origins, effective-threshold curves, four exploratory AUGRC pairs,
difficulty and future-disagreement contracts, per-item conformal export, and
bounded-memory execution. Existing deciding results and rules are unchanged.

From version 2 to version 2.1, on 19 October 2026, before any result of an aggregator:
* The section on implementation: two stages, and the acceptance tests of the first. The spec had none, and the decision of 26 October needs only part of the protocol.
* P12: the evaluation configuration is named after the run it evaluates, `configs/evaluation_v2.yaml`, and the section on implementation says how folds and resamples come from the single seed.
* P14: `verdicts.csv`, `a1_orders.csv` and `a3_choices.csv` among the outputs; `evaluations/` ignored by git.
* P10: spec 07 says which deciding comparisons answer the question of 26 October.

From version 1 to version 2:
* P10, deciding metric: accuracy became macro F1. On full run v2 the majority class per scheme has accuracy 0.579 and beats the A0 of every model (best 0.504), so under accuracy an aggregator that only learned the most frequent class of each scheme would pass the criterion without using the answers to the CQs. In macro F1 the majority scores 0.216 and the A0 of Qwen3.8 27B 0.350. Accuracy depends on the prevalence of the classes (Opitz 2024); macro F1 is the usual metric in fallacy classification (Pan et al. 2024). The change was made after seeing A0 and the zero-shot, before any aggregator, and it leaves the reference model unchanged. It is reported as a deviation from the criterion fixed in advance, with its reason (Lakens 2024).
* P5 and P10: Bonferroni correction on the deciding comparisons, a primary configuration per aggregator, the zero-shot beside the deciding comparisons.
* P13, new: out-of-fold estimation with folds shared by every aggregator.
* P11: AUGRC as the summary number, AURC beside it; formal comparisons only on the summary number.
* P7: Brier score and reliability diagrams instead of ECE; alpha instead of kappa; the probabilities named; the reached and weight columns.
* P8: alpha and raw agreement; kappa removed.
* P6 and P9 withdrawn.
* P1: how the predicted scheme is chosen for the systems that combine several models.
* P2, P3 and P4: limited to the main table.
* P14, new: the output.

## References

* Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107.
* Dror, R., Baumer, G., Bogomolov, M. and Reichart, R. (2017). Replicability analysis for natural language processing: testing significance with multiple datasets. *Transactions of the ACL*, 5, 471–486.
* Feinstein, A. R. and Cicchetti, D. V. (1990). High agreement but low kappa: I. The problems of two paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543–549.
* Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
* Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
* Hayes, A. F. and Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. *Communication Methods and Measures*, 1(1), 77–89.
* Kumar, A., Liang, P. and Ma, T. (2019). Verified uncertainty calibration. *NeurIPS*.
* Lakens, D. (2024). When and how to deviate from a preregistration. *Collabra: Psychology*, 10(1), 117094.
* Nixon, J., Dusenberry, M., Zhang, L., Jerfel, G. and Tran, D. (2019). Measuring calibration in deep learning. *CVPR Workshops*.
* Opitz, J. (2024). A closer look at classification evaluation metrics and a critical reflection of common evaluation practice. *Transactions of the ACL*, 12, 820–836.
* Opitz, J. and Burst, S. (2019). Macro F1 and macro F1. arXiv:1911.03347.
* Pan, F. et al. (2024). Are LLMs good zero-shot fallacy classifiers? *EMNLP*.
* Traub, J., Bungert, T. J., Lüth, C. T., Baumgartner, M., Maier-Hein, K. H., Maier-Hein, L. and Jaeger, P. F. (2024). Overcoming common flaws in the evaluation of selective classification systems. *NeurIPS*.
