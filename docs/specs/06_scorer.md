# Spec 06. Scorer

Version 1, 24 September 2026. Component: package `src/argfallacy/eval/`, the compat functions of `src/argfallacy/labels/compat.py`, the command `argfallacy score prior`.
Depends on: 01 and the data (`docs/data.md`). Used by: 07, 08.

## 0. Purpose

The scorer turns predictions and gold labels into numbers: accuracy, precision, recall and F1 per class, macro averages. Every number of the thesis passes through it.

It has two modes, kept separate:

* **canonical**: the rules of the project (spec 01, spec 08 P2). It produces the numbers of the thesis.
* **compat**: the rules of the scoring code of the prior runs. It exists only to show that the scorer counts like that code on the same input. It never produces a number of the thesis.

The scorer is validated on the prior runs before our runs exist. Where compat reproduces the files of the prior runs exactly, the counting code is proven on real data; the canonical mode then changes only the rules, which are declared.

Out of scope: the aggregators (spec 07), intervals, paired tests, abstention curves and breakdowns (spec 08), the full parser (spec 05).

## 1. What exists

What existed before this component:

* `argfallacy.labels`: `normalize_label`, `resolve_for_scheme`, `label_matches`, `collapse`, `label_space`, `normalize_scheme_label`, `audit`, `INVALID`, and `compat_normalize`, then a fit of three affix rules that reproduced the `final_label` and `final_label (collapsed)` accuracies of the eight `metrics.csv` files within 0.0005.
* `tests/test_prior_reproduction.py`: our interpreter reproduces 2416 of the 2428 verdicts of the prior pipeline.
* No scorer module. `scikit-learn` is not a dependency.

## 2. The reference: the scoring code of the prior runs

Facts established on 23 September 2026, recorded here because the acceptance tests rest on them.

**The code.** Eleni's scoring code is `evaluation.py`, in `Lavoro di Enrico/src/` in the Thesis folder, outside the repository and outside `PRIOR_RUNS_DIR`. The file is dated 7 September 2026, the `metrics.csv` files 9 September. Run on the eight `results.csv` files, it reproduces, with differences below 1e-15:

* all 28 rows of the eight `metrics.csv` files (`n_samples`, `n_classes_gold`, `accuracy`, `macro_f1`, `precision`, `recall`); the two `scheme-correct` rows of each pipeline run are obtained by applying the same functions to the rows where the normalized predicted scheme equals the normalized gold scheme;
* all 397 rows of the eight `classwise_metrics.csv` files;
* all 48 rows of `per_dataset_micro_f1.csv` (in `PRIOR_RUNS_DIR`), grouping the rows by the part of `source` before `_` or ` (`: `logic_dev`, `logic_test` and `logic (overall)` are `logic`, `reddit_dev` and `reddit_test` are `reddit`, `elecdebate2028` stays as it is.

**Enrico's tables are the same computation.** Tables 5.1 and 5.4 of Enrico's thesis coincide with the `metrics.csv` files, table 5.3 with `per_dataset_micro_f1.csv`, tables 5.5 and 5.9 and appendix B with the `classwise_metrics.csv` files, and the first columns of table 5.6 with the `scheme-correct` rows. Appendix B was compared row by row, four models, 25 classes, two approaches: every value coincides to the third decimal. The only difference is that appendix B prints a row of zeros for False Attribution in zero-shot, a class that no zero-shot run predicts. Enrico's numbers are therefore not a second independent check: they come from the same files.

**How that code counts.** This is what compat must do.

1. Each `results.csv` is read with every column as a string and no value read as missing.
2. One row per item: duplicates on `text` are dropped, the first kept. Every file has 607 rows after this step: the texts of ids 469 and 495 differ by a double space, so they stay two rows.
3. Rows whose gold fallacy is empty or contains `?` are skipped in the fallacy tasks, rows whose gold scheme is empty in the scheme task (none today).
4. Scheme label: stripped and lower-cased; a leading `argument from ` is removed, then a leading `argument `; every run of `_`, `-`, `.` becomes a space; whitespace is squeezed and stripped; a label that starts with `good ` becomes `good argument`.
5. Fallacy label: the scheme normalization of point 4 first. Then the text is matched against a prefix pattern (`ad hominem` followed by an optional `(`, `:` or `-`, then the subtype, then an optional `)`) and, if that fails, against a suffix pattern (the subtype, an optional `(`, `:` or `-`, then `ad hominem` and an optional `)`). The captured subtype replaces the text only if it is one of the six members of the ad hominem family. In the collapsed task, the five members other than tu quoque become `ad hominem`. Finally `casual reductionism` becomes `causal reductionism`.
6. Metrics: accuracy; precision, recall and F1 per class, and their unweighted mean, over the union of the normalized labels that appear in the gold or in the predictions; a zero denominator gives zero. `n_classes_gold` is the number of distinct normalized gold labels.

Consequence of point 6: every distinct prediction string that is not a class (`appeal to populum`, `uncertain: unexpected answer for 'cq2' ...`) becomes a class with F1 zero and lowers the macro averages of that run. The macro averages of the prior runs therefore depend on how many different wrong strings a model writes: the fallacy macro of `llama3.3_70B/zeroshot` is a mean over 29 classes, six of them strings written by the model, that of `gpt-5/zero-shot` over 23. They are reproduced in compat and never compared with canonical macro averages.

Consequence of point 5: apart from `casual reductionism`, a variant spelling counts as an error even when the verdict is right. In `llama3.3_70B/zeroshot` three items with gold ad populum were answered `Appeal to Populum`: the reference counts 338 correct items out of 607 (0.557, the value of table 5.1), the canonical mode 341 (0.562). Compat keeps the reference's count, because its only job is to reproduce it.

**Where the fit used before the port differed.** On the 70 distinct strings of the eight files, `compat_normalize` and `evaluation.py` differ on two: `Casual reductionism` (evaluation.py rewrites it, the fit does not) and `Ad Hominem Red Herring` (evaluation.py leaves it whole, because red herring is not a member of the family; the fit strips `ad hominem`). The accuracy does not change in any of the 16 rows it is tested on. The macro F1 changes by up to 0.020 (`qwen3.8_27b/zero-shot`, collapsed). The fit is right for accuracy and wrong for macro averages and per-class numbers.

## 3. What to do

### 3.1 Compat mode, ported from the reference

`compat_normalize` is rewritten as a port of the two normalizations of section 2, points 4 and 5, with the same function signature. A second function, `compat_normalize_scheme`, does the same for scheme labels. On the 70 distinct fallacy strings and the 16 distinct scheme strings of the eight files the port and the reference give the same output. The label strings follow fixed rule 3: the six members and the family name come from `labels/fallacies.yaml` (the family and its members), and the strings that the reference writes in its code (`argument from`, `argument`, `good argument`, `casual reductionism` to `causal reductionism`) go into a `compat` block of `labels/fallacies.yaml`, not into the Python code. The general alias table is not used in compat.

The compat metrics follow section 2, points 1 to 3 and 6. They are computed by our own code, without `scikit-learn`.

### 3.2 Canonical mode

Input: a table with `item_id` and the raw predicted label, and the gold of `data/items.csv` (`gold_fallacy`, `gold_scheme`). Label spaces: `fine` and `collapsed_symmetric`; `collapsed_compat` belongs to compat only.

1. Gold and prediction are read with `normalize_label`, so a known variant spelling (the aliases of `labels/fallacies.yaml`) is the label it stands for.
2. The gold must be a terminal. A gold of any other kind (coarse, family, out of scope, idk) stops the scorer with an error that names the item and the label. No evaluation of the models has such a gold today: the gold of the test set and Enrico's terminal verdicts (spec 08, P6) are all terminals; `any_fallacy` comes only from Eleni's column `fallacy Eleni`, and Enrico's `false_cause` verdicts are the nine items without a scheme that P6 already excludes. There is therefore no partial credit in the scorer; `label_matches` stays in the loader, untouched.
3. A gold terminal that the diagram of the gold scheme does not produce (`resolve_for_scheme` gives `scheme_mismatch`) is kept, scored like any other, and listed in the report. Today there are three, all NLAS, in Enrico's copy of the ad populum sheet (rows 24, 29, 62): gold scheme `popular_opinion`, gold fallacy `non_sequitur`, which only expert opinion produces: `a122bba5789304c0`, `c8cdc8019c39370e`, `cf3b4d6783d834d0` (ids 114, 117, 148 in the prior runs). In the gold scheme condition no aggregator built on the diagrams can get them right.
4. Correct prediction: `collapse(prediction) == collapse(gold)` in the chosen space. A prediction that is not a class of the space is wrong: `invalid`, no verdict, and in the prior runs a family label in `fine` (in `collapsed_symmetric` the family is a class). The report counts these predictions per kind.
5. Metrics: number of items, accuracy; for every class of the space, support, precision, recall and F1; macro precision, recall and F1 as the unweighted mean over the classes with support in the gold, with the list of the classes left out. The precision of a class counts every prediction of that class. A zero denominator gives zero. A prediction that is not a class lowers only the recall of the gold class of its item: the report gives the count of these predictions next to every metric.
6. Stage one: the same metrics on scheme labels read with `normalize_scheme_label`, classes the eight schemes and `none`.
7. Every metric can be computed on a subset of the items. The `scheme-correct` rows use it, and spec 08 will use it for the breakdowns.

No rounding inside the scorer: numbers are rounded only when printed.

### 3.3 The prior runs as input

A reader for the eight runs under `PRIOR_RUNS_DIR`: model and condition from the path, predicted fallacy for every run, predicted scheme and the answers to the CQs for the pipeline runs. For the canonical mode the rows are mapped to the items through `eleni_id` of `items.csv`: id 469 stands for the item with `eleni_id` `469;495`, and the rows that have no item (495 and the five removed near duplicates, `docs/data.md` section 5.2) are dropped. Every run then has 601 items.

### 3.4 Reproduction (compat)

For every run, the tasks of its `metrics.csv` (`final_label`, `final_label (collapsed)`, and for the pipeline runs `scheme`, `final_label (scheme-correct)`, `final_label (scheme-correct, collapsed)`), the per-class rows of its `classwise_metrics.csv`, and the accuracy per source of `per_dataset_micro_f1.csv`.

### 3.5 Rescoring on the test set of the experiments (canonical)

The same five tasks on the 601 items, in `fine` and `collapsed_symmetric`, with the gold of `items.csv` (spec 08, requirement 2). The `scheme-correct` subset is the set of items whose predicted scheme id equals the gold scheme id. Per-class rows for every task. There is no reference to compare with: these are new numbers.

### 3.6 Majority per scheme and information ceiling

Both are the comparison lines of the thesis plan (section 6), defined as in the diagnosis (section 3) and written here explicitly.

* **Majority per scheme.** Every item has a scheme (the partition) and a gold label. The prediction for an item is the most frequent gold label among the items with the same scheme, computed on the same items. Ties go to the first id in alphabetical order. It is an oracle on the prior and is declared as one.
* **Information ceiling.** The same construction, with key (scheme, answer vector). The answer vector is the hard answer of the model to every CQ of the scheme; for the prior runs, `cq_answer` stripped and lower-cased, one per `cq_number`, unusable answers kept as they are. It is an oracle in sample: an upper bound for any function of the answer vector.

Both return one prediction per item (`majority_by`: key per item, gold per item, one label id per item), which the canonical scorer scores like any other prediction, in `fine` and in `collapsed_symmetric`. The prediction is a terminal id chosen in the fine space; in `collapsed_symmetric` the same prediction is scored, not a majority recomputed on the collapsed gold.

On the prior runs, 607 rows per pipeline run, partition by the scheme the model predicted, `fine` space, the diagnosis (sections 3 and 13) gives:

| model | majority | ceiling |
| --- | --- | --- |
| gpt-5 | 0.557 | 0.669 |
| qwen3.8_27b | 0.562 | 0.662 |
| llama3.3_70B | 0.558 | 0.671 |
| deepseek | 0.554 | 0.662 |
| mean | 0.558 | 0.666 |

On the 601 items the scorer also reports the majority with partition by gold scheme (one value, the same for every model) and both quantities with partition by predicted scheme, per model.

### 3.7 Command

`argfallacy score prior [--out DIR]`, default `runs/scores/prior/`. It writes:

* `compat_metrics.csv`, `compat_classwise.csv`, `compat_per_source.csv`, with the columns of the files of the prior runs, so that a diff is readable;
* `metrics.csv` and `classwise.csv` for the canonical rescoring, with the columns `run`, `condition`, `task`, `space`, `n_items`, `n_classes_gold`, `accuracy`, `macro_precision`, `macro_recall`, `macro_f1` (and `class`, `support`, `precision`, `recall`, `f1` in the per-class file). `run` is the model folder, `condition` is `pipeline` or `zero-shot`. The tasks are `final_label` and `final_label (scheme-correct)`, each in `fine` and `collapsed_symmetric`, and `scheme`, whose space is written `schemes`;
* `baselines.csv` with majority and ceiling: the columns `baseline` (`majority`, `ceiling`), `partition` (`gold_scheme`, `predicted_scheme`), `run`, `condition` (both empty for the partition by gold scheme), `space`, then the metric columns of `metrics.csv`;
* `report.md`, a short report: the prior rows that stand for no item, the `scheme_mismatch` items, the predictions that are no class of the space counted per kind next to accuracy and macro F1 of every row, the classes left out of the macro averages.

The kinds of a prediction that is no class of the space are the kinds of `normalize_label` (`invalid`, `family`, `coarse`, `out_of_scope`, `idk`) and `no_verdict`, for an empty cell or an abstention. A string the dictionary does not know stops the scorer, as everywhere else.

## 4. Acceptance tests

The tests on the prior runs are skipped when `PRIOR_RUNS_DIR` is missing, as today. The tolerance of the compat tests is 1e-9, because the reference reproduces its own files exactly.

1. **metrics.csv.** For every row of the eight files: `n_samples` and `n_classes_gold` equal; `accuracy`, `macro_f1`, `precision`, `recall` within 1e-9. 28 rows. It replaces the two compat accuracy tests of `tests/test_prior_audit.py`.
2. **classwise_metrics.csv.** For every run and task, the same set of class labels, and for every row `precision`, `recall`, `f1-score`, `support` within 1e-9. 397 rows.
3. **per_dataset_micro_f1.csv.** For every row, `n` equal and accuracy within 1e-9. 48 rows.
4. **Majority and ceiling.** On the four pipeline runs, the values of the table of section 3.6 within 0.0005.
5. **Rescoring.** Every run gives 601 items; the item with `eleni_id` `469;495` is scored once, from row 469; the rows 175, 194, 490, 495, 503 and 531 of the prior runs are dropped; the report lists the three `scheme_mismatch` items.
6. **Canonical rules, on small hand-made tables.** A variant spelling in the alias table counts as its label; an `invalid` prediction and a missing verdict are wrong in every space; a family prediction is wrong in `fine` and right in `collapsed_symmetric` when the gold is a member; a gold that is not a terminal stops the scorer with an error naming the item; the macro average leaves out the classes without gold support, and a prediction of such a class is still an error.
7. The existing test `test_canonical_normalization_is_at_least_as_generous_as_compat` keeps passing with the ported compat.

## 5. What not to do

* No `scikit-learn`, no copy of `evaluation.py` or of the files of the prior runs in the repository.
* Do not touch `data/`, `resolve_for_scheme`, `label_matches` or the interpreter.
* No compat number in the canonical outputs, and no canonical rule in compat.
* No aggregator, interval, paired test or curve.

## 6. Documents to align in the same commit

* Spec 01: section 3.3 describes the ported compat; acceptance test 8 points to this spec.
* `src/argfallacy/labels/compat.py`: the docstring no longer says that the scoring code of the prior runs was never read.
* The README: the command, and `eval/` in the package.
* `docs/data.md`: the three `scheme_mismatch` items, as a fact of the gold, with no correction.
* Spec 08, P6: the sentence "Partial credit remains in use for `any_fallacy`" goes. The gold of P6 is Enrico's terminal verdicts, and `any_fallacy` never appears among them.
* `docs/cq_proposals.md`, section 5 and "What remains unverified": the ceiling and the majority with the values of the scorer and their basis. The three numbers written there mix bases: 0.669 is the ceiling on gpt-5, 0.558 equals the four-model mean of the majority (gpt-5 alone is 0.557), and the value of the tree belongs to A0, spec 07.

## 7. Definition of done

Lint and tests pass (commands in the README); the tests of section 4 are present (1 to 5 and 7 skipped without `PRIOR_RUNS_DIR`); `argfallacy score prior` writes the files of section 3.7; summary with the canonical numbers of the rescoring and anything the tests did not cover.
