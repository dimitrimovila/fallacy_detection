# Spec 01. Label dictionary

Version 2, 9 September 2026. Replaces version 1, written before looking at the code.
Component: the existing loader, now `src/argfallacy/schemes/loader.py`, plus the module `src/argfallacy/labels/`. Data: `labels/fallacies.yaml`, `labels/schemes.yaml`.

## 1. What already exists and is not touched

The loader (`argschemes/loader.py` when this spec was written, now `src/argfallacy/schemes/loader.py`) already has the central part:

* `schemes/_vocabulary.yaml`: 25 terminals with a canonical id, the coarse label `false_cause` with `refines_to`, an alias table, the IDK markers.
* `normalize_label(raw, vocab)`: lower case and whitespace squeezed, alias table, then one deterministic rule (spaces to underscores, parentheses removed), finally a lookup in the terminals, the coarse labels, the out_of_scope labels. Unknown = `SchemeError`. No silent default.
* `resolve_for_scheme(raw, scheme, all_schemes)`: checks the label against the annotated scheme and returns `exact`, `coarse`, `scheme_mismatch` (with a reassignment proposal that is never applied), `out_of_scope`, `idk`.
* `label_matches(predicted, resolution)`: partial credit for coarse gold labels.

These functions keep the same signature and the same behaviour. The existing tests in `validate_schemes.py` must keep passing.

Decision already taken in August and confirmed: `false cause` and `false causality` are coarse labels, they are not refined, they give partial credit, and the share of coarse gold is reported separately. They are no longer "pending".

## 2. What is missing

1. Incomplete aliases. With vocabulary 1.0, strings that are in the data fail: `good argument` (124 rows of Eleni's gold), `Good Argomentation` (seven diagrams), `Abusive Ad Hominem`, `hasty generalisation`, `Appeal to Emotion`.
2. No dictionary for the scheme labels (`Argument from Expert Opinion`, `Expert opinion`, `NO`, `None`), needed to evaluate stage one.
3. No collapsed label space and no notion of an ad hominem family.
4. No compat mode that reproduces Eleni's scorer.
5. No audit tool that says, on a real column, which strings occur and how they are translated.

## 3. What to do

### 3.1 Move and update the vocabulary
`schemes/_vocabulary.yaml` is replaced by `labels/fallacies.yaml` version 1.1, already provided. Same keys as before (`terminals`, `coarse_labels`, `out_of_scope`, `aliases`, `idk_markers`), so `normalize_label` works without changes; in addition: `families`, for every terminal `family` and `schemes`, `appeal_to_emotion` among the out_of_scope labels, `spaces`, `missing_markers`. `load_vocabulary` reads from `labels/`. The old file is deleted.

### 3.2 New file `labels/schemes.yaml`
Already provided: eight schemes plus `none`, with aliases. `none` is the label "no scheme present", assigned by an annotator of the workbook when none of the eight schemes applies. It is not a ninth scheme, it is the negative class of stage one.

### 3.3 New module `labels/` with these functions

```python
from argfallacy.labels import (normalize_scheme_label, label_space, collapse,
                               compat_normalize, audit)

normalize_scheme_label("Popular Opinion")   # -> "popular_opinion"
normalize_scheme_label("NO")                # -> "none"
label_space("fine")                         # -> sorted list of the 25 terminal ids
label_space("collapsed_symmetric")          # -> 20 ids
label_space("collapsed_compat")             # -> 21 ids
collapse("ad_fidentia", "collapsed_symmetric")   # -> "ad_hominem"
compat_normalize("Ad Hominem (Tu Quoque)")       # -> "tu quoque"
audit(series, kind="fallacy")               # -> table: raw string, count, id, status
```

`audit` raises no exceptions: for every distinct raw string it returns the count, the translated id or `UNKNOWN`, and the status (`terminal`, `coarse`, `out_of_scope`, `idk`, `missing`, `UNKNOWN`).

Compat mode, in a single function with a docstring: lower case, then exactly one of three affix rules, tried in this order: if the string starts with `ad hominem (` and ends with `)` the part in parentheses is kept; otherwise, if it ends with ` ad hominem` the suffix is removed; otherwise, if it starts with `ad hominem ` the prefix is removed; finally leading and trailing spaces removed. The third rule is needed only for `qwen3.8_27b/zero-shot`, where 31 predictions carry the prefix instead of the suffix: with the first two rules alone that run is off by 0.0577. The family name comes from `labels/fallacies.yaml`, not from the code. The compat comparison is equality between strings transformed this way. No partial credit, no aliases.

### 3.4 Constraints checked at load time (to be added to the loader)
* Aliases unique after normalization; an alias cannot point to two ids.
* Every `family` and every entry of `spaces` cites existing ids.
* For every terminal, `schemes` matches the schemes that actually produce it according to the graphs in `schemes/`.

## 4. Acceptance tests

Since 20 September 2026 `tests/` keeps only the tests that defend a result, check the data or help the work. The points without a note have an automatic test; the marked points remain requirements, but today they have no test, or only a partial one.

1. `validate_schemes.py` keeps passing after the vocabulary is moved.
2. Every string of section 2 point 1 translates into the expected id; upper case and extra spaces do not change the result. *Today without an automatic test, by a choice made on 21 September 2026: every change to the aliases is reviewed by hand. Points 6 and 7 check that no string of Eleni's files is left untranslated, but not that it lands on the expected id.*
3. An invented string raises `SchemeError` with the string in the message. `false cause` returns kind `coarse`.
4. The 25 terminals of the vocabulary match the terminals of the graphs; the `schemes` list of every terminal matches the real producers.
5. `collapse` is total over the 25 ids for both spaces; the sizes are 20 and 21 (25 fine; the symmetric space merges six ids, Eleni's five). On the 23 labels with support in the current gold they become 18 and 19, and 19 is the number of classes of the collapsed row of Eleni's `metrics.csv` files.
6. Audit of `gold_fallacy` and `predicted_fallacy` in Eleni's `results.csv` files (`PRIOR_RUNS_DIR`; test skipped if missing): zero `UNKNOWN`. Every new string must be added to the aliases and reported in the summary.
7. Audit of `gold_scheme` and `predicted_scheme`: zero `UNKNOWN`.
8. Compat: on the `results.csv` of every model, the share of rows with `compat_normalize(predicted) == compat_normalize(gold)` matches the `final_label` accuracy of its `metrics.csv`, tolerance 0.0005. The same for the collapsed row, merging with `collapsed_compat` in the compat space, on the strings put through `compat_normalize`, not on the canonical ids.
9. No label string as a literal outside `labels/*.yaml`, `schemes/*.yaml` and `tests/`: a test searches the source code for `"Tu quoque"`, `"Good argument"`, `"Ad Populum"` and fails if it finds them elsewhere. *Today without an automatic test.*

## 5. What not to do
* No approximate comparison beyond the deterministic rule that already exists.
* Do not modify `resolve_for_scheme` and `label_matches`.
* Do not touch `data/` or Eleni's files to make the tests pass.

## 6. Definition of done
Lint and tests pass (commands in the README); the tests of section 4 without a note are present (6, 7 and 8 are skipped without `PRIOR_RUNS_DIR`); summary with the aliases added.
