# Spec 03. Prompts and answer format

Version 1, 11 September 2026. Component: `src/argfallacy/prompts/`. Data: `prompts/`.
Depends on: 01 and the data (`docs/data.md`). Used by: 04.

## 1. Principles

* The text of every critical question comes from the scheme YAML, word for word. The code inserts it into the template; nobody rewrites it. A test compares the text in the generated prompt with the `text` field of the YAML.
* Everything around the question (instructions, format, definition of the answers) is ours and versioned. The version number is in the file name and ends up in the manifest of every run.
* One question per call. The model always sees the whole text of the argument and the scheme.
* Tolerance lives in the aggregation, not in the prompt: the model is never told what to do when it is uncertain. Only what each answer means is defined.

## 2. Files in `prompts/`

* `stage1_v1.md`: recognition of the scheme.
* `stage2_v1.md`: one critical question.
* `schemas/stage1_v1.json`, `schemas/stage2_v1.json`: JSON Schema of the answer, used for vLLM's structured output and for validation in the parser.
* A test checks that every template declared in `prompts/` compiles for every scheme and every CQ with no missing field.

The templates use double-brace placeholders (`{{text}}`, `{{scheme_name}}`, `{{schema}}`, `{{variables}}`, `{{cq_text}}`, `{{answer_options}}`), filled by `render_stage1(item)` and `render_stage2(item, scheme, cq)`. No template library: plain substitution, with an error if a placeholder stays empty.

## 3. Stage one, `stage1_v1.md`

Content, in English: role (argumentation analyst), the text, the list of the eight schemes with `name`, `schema` (the form of the argument) and `identification_question` taken from the YAML files, plus the option `none` defined as "none of these eight schemes is present". Instruction: choose only one, based on the text alone. Answer in the format of section 5, with `scheme` among the nine canonical ids of `labels/schemes.yaml`.

## 4. Stage two, `stage2_v1.md`

Content, in English:
1. Role and task: answer a single critical question about an argument that follows the given scheme.
2. The scheme: `name`, `schema` (form of the argument) and `variables` from the YAML, as a legend of the letters used in the question (S, A, D, C, and so on).
3. The text of the argument.
4. The critical question, verbatim, preceded by its id.
5. The admitted answers, with a definition:
   * `yes` and `no` (or `negative` and `positive` for ad hominem CQ1, in the order in which the question names them). The definitions are in `prompts/stage2_v1_answers.yaml`, versioned with the template; the notes of the scheme YAML files do not enter the prompt (accepted exception).
   * `cannot_be_determined`: "use this answer only if the text contains no information at all on the point asked about; if the text contains partial information, answer yes or no and lower your confidence". It is the wording of the third option printed in Enrico's thesis, made operational.
   * `na`: "use this answer only if the question has no object in this text, that is, the thing it talks about does not exist in the argument; if it exists but you cannot establish how it stands, answer cannot_be_determined and lower your confidence". Fourth option, added outside the diagram exactly like `cannot_be_determined`: never in the `answer_space` of a scheme. It tells the absence of the question's object apart from uncertainty about how that object stands.
6. Instruction to answer based on the text alone, without outside knowledge, unless the question explicitly requires it.
7. The answer format.

`na` is the fourth answer, not `not_applicable`. It is added by the code (section 5), not by the diagram: no YAML in `schemes/` gains an `na` arc, and the guard cases stay encoded as nodes (for example expert opinion CQ4). A traversal that meets `na` finds no drawn arc and stops incomplete, like any other answer outside the `answer_space`: what that outcome means for an aggregator is a decision of phase 3, not of this spec.

## 5. Answer format (JSON, structured output)

Stage one:
```json
{"scheme": "expert_opinion", "confidence": 80, "justification": "..."}
```
Stage two:
```json
{"answer": "yes", "confidence": 75, "justification": "..."}
```

Rules:
* The field `answer` (or `scheme`) is the first key of the object, so the position of the first token of the answer is the same in every output and the client can read its logprobs.
* The admitted values of `answer` are strings that begin with different tokens: `yes`, `no`, `cannot_be_determined`, `na`, and for ad hominem CQ1 `positive`, `negative`, `cannot_be_determined`, `na`. The soft probability `p_logprob` is computed from the first token of the value: take the `top_logprobs` at that position, keep the entries that begin one of the admitted answers, renormalise. If an admitted answer does not appear among the top_logprobs, its mass is zero and the row is marked `logprob_partial`.
* `confidence` is an integer from 0 to 100. `justification` is text, at most two sentences; the limit is in the prompt, not enforced by the schema.
* The JSON Schema constrains `answer` to the enumeration admitted for that CQ and `confidence` to the range. The parser validates again: an output that does not pass the schema is `invalid` (spec 01), kept and counted.

## 6. Samples

For every (item, stage, CQ, model): five calls. Sample 0 at temperature 0: it provides the hard answer and `p_logprob`. Samples 1 to 4 at temperature 0.7: together with sample 0 they provide `p_sample` = frequency of the answer `yes` (or `positive`) over the five. `p_verbal` = `confidence`/100 of sample 0, oriented: if the hard answer is `no`, `p_verbal` = 1 minus confidence/100; if it is `cannot_be_determined`, `p_verbal` is empty and the row carries `idk = true`. `na` produces no `p_verbal`, in the same way as `cannot_be_determined` — reading it as a low probability of yes would decide, silently, that `na` means `no` — but it does not raise `idk`, which stays true only for `cannot_be_determined`: the two answers remain distinguishable downstream only by looking at `answer`, the only information this parser records about them. What `na` means for the aggregation remains an open point for phase 3, to be closed on the real data: no numeric value, not a provisional rule.

The models with internal reasoning (gpt oss, K2 Horizon, and Qwen3.8 and Gemma 4 in the condition with reasoning on, spec 04) produce the same JSON; the client reads the logprobs at the position of the final answer. If logprobs are not available for a model, `p_logprob` stays empty and the manifest says so.

## 7. Acceptance tests

Since 20 September 2026 `tests/` keeps only the tests that defend a result, check the data or help the work. The points without a note have an automatic test; the marked points remain requirements, but today they have no test, or only a partial one.

1. For every scheme and every CQ, the rendered prompt contains the `text` field of the YAML as an exact substring.
2. For ad hominem CQ1 the options are `positive`, `negative`, `cannot_be_determined`, `na`; for all the other CQs `yes`, `no`, `cannot_be_determined`, `na`.
3. The stage-two JSON Schema rejects an `answer` outside the enumeration and a `confidence` outside the range; the stage-one schema rejects a non-canonical `scheme`. *Today only the stage-two part has an automatic test, checked through the parser (spec 04, point 5).*
4. Rendering is deterministic: same item, same prompt, byte for byte. *Today without an automatic test.*
5. The version of the template appears in the text of the rendered prompt as a final comment, so the manifest and the prompt cannot diverge. *Today without an automatic test.*
