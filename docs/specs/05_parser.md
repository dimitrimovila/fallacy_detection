# Spec 05. The parser for the aggregators

Version 1, 19 October 2026. Component: `src/argfallacy/parse/`. It replaces the minimal parser of spec 04, section 3, which stays valid for everything this spec does not change.
Depends on: 03, 04. Used by: the run report (spec 04, section 4), 07, 08.

## 1. Purpose

The parser turns `raw.jsonl` into the two tables that every later step reads: `answers.csv`, one row per call, and `summary.csv`, one row per item, stage, scheme condition, scheme, CQ and model. The minimal parser was written for the report of the pilot. The aggregators of spec 07 need columns it does not write, and three of its columns have no meaning at stage one. This spec adds the first and fixes the second. Parsing a run again regenerates both tables; no call is made again.

The module `src/argfallacy/parse/minimal.py` is renamed `src/argfallacy/parse/tables.py`, since it is no longer the minimal version. `argfallacy.parse` exports the same names as before, so no caller changes, and the command stays `argfallacy parse RUN_ID`.

## 2. What does not change

* The hard answer: sample 0 for an entry with one sample, the majority of the valid samples otherwise, with the rules for ties and for `invalid` of spec 04, section 3. A valid sample is parsed and not cut off by `max_tokens` (`valid_samples`). The hard answers of every run stay the same, so A0 stays the same: on full run v2, A0 of `qwen3_8_27b` has accuracy 0.504 and macro F1 0.350.
* `parse_ok`, `logprob_partial`, the position of the answer token, the masses at that position and `p_label` (spec 04, sections 3 and 4.2).
* The names of the existing columns. At stage two, `p_logprob`, `p_verbal` and `p_sample` keep their meaning for the entries with one sample; section 4 says what changes for the entries with several.

## 3. `answers.csv`

* Stage one gets one column `p_logprob_<scheme id>` for each scheme id of `labels/schemes.yaml`, the eight schemes and `none`: the masses that `logprob_masses` already computes over the scheme ids at the answer token, renormalised over them. The minimal parser computes them and does not write them. The names of these columns come from `labels/schemes.yaml`, not from strings in the code.
* The new columns are empty at the other stages, and the `p_logprob_<answer>` columns of stage two stay empty at stage one.

## 4. `summary.csv`, stage two

New columns:
* `p_logprob_<answer>`, one for each answer of `ALL_OPTIONS` (`yes`, `no`, `positive`, `negative`, `cannot_be_determined`, `na`): the masses of the admitted answers, renormalised over them as in `answers.csv`. With one sample, the masses of sample 0, as they are. With several, the mean, answer by answer, over the valid samples that have masses. Empty for the answers the question does not admit, and everywhere when no sample has masses. Where present, the masses of the admitted answers sum to 1 within 1e-9.
* `p_sample_<answer>`, one for each answer of `ALL_OPTIONS`: the share of the valid samples that gave that answer. Empty for an entry with one sample, and when no sample is valid. Where present, they sum to 1.
* `justification`: the justification of the hard answer. With one sample, that of sample 0. With several, that of the valid sample with the lowest index whose answer equals the hard answer. Empty when the hard answer is `invalid`.

Changed columns, for the entries with several samples only:
* `p_logprob`: the mean, over the valid samples that have masses, of the mass of the first admitted answer (`yes`, or `positive` for ad hominem CQ1), that is, the column `p_logprob_yes` (or `p_logprob_positive`) of this table. Before, the mass of sample 0 alone.
* `p_verbal`: the mean, over the valid samples whose answer is on an arc, of the confidence oriented on the first admitted answer: confidence/100 when the sample gave that answer, 1 − confidence/100 when it gave the other. A sample that answered `cannot_be_determined` or `na` gives no value, as sample 0 gave none before. Empty when no valid sample answered on an arc. Before, sample 0 alone.

The samples of an entry are drawn in the same way, so sample 0 has no special status, and the hard answer is already the majority of all of them. The mean of the stated confidence over several samples is one of the aggregations evaluated by Xiong et al. (2024).

`p_sample`, the share of the valid samples that gave the first admitted answer, does not change. `idk`, `n_samples_ok` and `invalid_count` do not change.

## 5. `summary.csv`, stages whose answer is a label

At stage one the answer is a scheme id; at the verdict stages it is a terminal id. The verdict stages are `zeroshot` and the stage `judge` that spec 07 adds; the parser lists them in one constant. At these stages the probabilities refer to the hard answer, the label chosen:
* `p_logprob`, stage one only: the mass of the hard answer, from the columns of section 3. With one sample, that of sample 0; with several, the mean over the valid samples that have masses. At the verdict stages `p_logprob` stays empty, since a name can take several tokens.
* `p_label`, a new column of `summary.csv`, verdict stages only: with one sample, `p_label` of sample 0. With several, the mean over the valid samples of `p_label` where the sample's answer is the hard answer and 0 where it is another label; a sample with the hard answer and no `p_label` is left out. `p_label_lower_bound`, also new in `summary.csv`, is true if any value used is a lower bound.
* `p_verbal`: with one sample, the confidence of sample 0 over 100. With several, the mean over the valid samples of confidence/100 where the sample's answer is the hard answer and 0 where it is another label.
* `p_sample`: the share of the valid samples whose answer is the hard answer. Empty with one sample.

All four are empty when the hard answer is `invalid`. Before, the minimal parser oriented the three columns on the first admitted answer, which at stage one is the first scheme id, so they had no meaning there; at the zero-shot stage it left them empty.

## 6. The answers that are not the argmax of `p_logprob`

On full run v2, 49 rows of sample 0 of `qwen3_8_27b` and 3 of `gemma4_31b` give an answer that is not the answer with the highest `p_logprob`. In all of them the mass of the answer given and the highest mass differ by less than 0.0004 (for example `no` 0.4947 and `yes` 0.4948). A mass sums every token that begins an answer, while the choice at temperature 0 takes the single most probable token, and the server rounds the logprobs: near a tie the two can differ with no error in the parser. The parser does not change. The hard answer stays the answer the model gave, and the run report keeps counting these rows. A difference above 0.001 would be a finding to check on `raw.jsonl`.

## 7. Out of scope

* Reading the answer from a JSON cut off by `max_tokens`. Full run v2 has no answer cut off, so the rule would change no number. If a run has answers cut off, the point is reopened.
* Any change to the client, the cache or the prompts.

## 8. Acceptance tests

On lines of `raw.jsonl` written by hand, with the response shape of the fake backend in `tests/fakes/`:
1. One sample: `summary.csv` has the same hard answers and the same `p_logprob`, `p_verbal` and `p_sample` as the minimal parser on the same lines; `p_logprob_<answer>` equals the columns of sample 0 in `answers.csv`; `justification` is that of sample 0.
2. Several samples, stage two: `p_logprob_<answer>` is the mean over the valid samples with masses, a sample that is `invalid` or cut off left out; `p_sample_<answer>` are the shares of the valid samples and sum to 1; `p_verbal` is the mean of the oriented confidences of the samples that answered on an arc, a sample that answered `na` adding nothing; `justification` is that of the valid sample with the lowest index that gave the hard answer.
3. Stage one: the nine `p_logprob_<scheme id>` columns sum to 1 on a row with masses; in `summary.csv`, `p_sample` is the share of the hard answer and `p_logprob` its mass; the run report's check `stage1_p_sample_not_modal` gives 0 on these lines.
4. Verdict stage: with one sample, `p_label` is that of sample 0; with several, the rule of section 5, a lower bound among the values used making `p_label_lower_bound` true.
5. The traversals of the `diagram` rule of the run report (spec 04, section 4) on the tables of this parser equal those on the tables of the minimal parser, on the same lines.

## 9. Documents to align in the same commit

* Spec 03, section 6: for the entries with several samples, `p_logprob` and `p_verbal` are means over the valid samples (this spec, section 4).
* Spec 04, section 3: the minimal parser is replaced by this spec, which says what changed. Section 4, parser checks: `stage1_p_sample_not_modal` checks that `p_sample` is the share of the hard answer at stage one.
* `src/argfallacy/eval/run_report.py`: the text of the parser checks no longer says that the stage-one columns have no meaning.
* The docstrings of `argfallacy.parse`, which call it the minimal version for the pilot.
* README: "the full parser" leaves the list of what is not implemented yet.

## 10. Definition of done

Lint and tests green, with the commands of the README. Full run v2 parsed again on the computer that holds its `raw.jsonl`, with A0 of `qwen3_8_27b` unchanged (accuracy 0.504, macro F1 0.350) and `stage1_p_sample_not_modal` at 0 for every entry. Summary in Italian.

## References

* Xiong, M. et al. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. *ICLR*.
