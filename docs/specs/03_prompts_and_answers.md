# Spec 03. Prompts and answer format

Version 2.2, 9 October 2026 (version 1, 11 September 2026; 1.1, 26 September 2026, added the definitions of expert opinion CQ3.1; 2 describes the prompts v2 and makes the version placeholder optional, sections 2 to 4; 2.1 takes the samples per entry, section 6, and states what the logprobs are, section 5; 2.2 adds the zero-shot prompt, section 8). Component: `src/argfallacy/prompts/`. Data: `prompts/`.
Depends on: 01 and the data (`docs/data.md`). Used by: 04.

## 1. Principles

* The text of every critical question comes from the scheme YAML, word for word. The code inserts it into the template; nobody rewrites it. A test compares the text in the generated prompt with the `text` field of the YAML.
* Everything around the question (instructions, format, definition of the answers) is ours and versioned. The version number is in the file name and ends up in the manifest of every run.
* One question per call. The model always sees the whole text of the argument and the scheme.
* Tolerance lives in the aggregation, not in the prompt: the prompt defines what each answer means and how to state the confidence, and never gives a question a default answer or tells the model where its answer leads.

## 2. Files in `prompts/`

* `stage1_v2.md`: recognition of the scheme.
* `stage2_v2.md`: one critical question.
* `stage2_v2_answers.yaml`: what each stage-two answer means (section 4).
* `schemas/stage1_v2.json`, `schemas/stage2_v2.json`: JSON Schema of the answer, used for vLLM's structured output and for validation in the parser.
* `zeroshot_v1.md` and `schemas/zeroshot_v1.json`: the verdict asked directly, without the CQs (section 8).

The v1 files were replaced by the v2 files, not kept next to them: the runs made with v1 keep their prompts in `raw.jsonl` and their version in the manifest.
* A test checks that every template declared in `prompts/` compiles for every scheme and every CQ with no missing field.

The templates use double-brace placeholders (`{{text}}`, `{{scheme_name}}`, `{{schema}}`, `{{variables}}`, `{{cq_text}}`, `{{answer_options}}`), filled by `render_stage1(item)` and `render_stage2(item, scheme, cq)`. No template library: plain substitution, with an error if a placeholder stays empty or a value has no placeholder. `{{prompt_version}}` is optional: the renderer passes the version only to a template that contains it. The v2 templates do not; the version is in the file name, in the manifest of every run and in the cache key of every call.

## 3. Stage one, `stage1_v2.md`

Content, in English: role (argumentation analyst), the text, the list of the eight schemes with `name`, `schema` (the form of the argument) and `identification_question` taken from the YAML files, plus the option `none` defined as "none of these eight schemes is present". Instruction: choose only one, and if the text fits more than one scheme, the one that carries its main argument; a scheme is present when the text argues in that form, well or badly. Since v2 the template no longer says to judge from the text alone without outside knowledge. Answer in the format of section 5, with `scheme` among the nine canonical ids of `labels/schemes.yaml`.

## 4. Stage two, `stage2_v2.md`

Content, in English:
1. Role and task: answer a single critical question about an argument identified as following the given scheme, the way a careful and fair reader with a good general education would.
2. The scheme: `name`, `schema` (form of the argument) and `variables` from the YAML, as a legend of the letters used in the question (S, A, D, C, and so on).
3. The text of the argument.
4. The critical question, verbatim, preceded by its id.
5. The admitted answers, with a definition:
   * `yes` and `no` (or `negative` and `positive` for ad hominem CQ1, in the order in which the question names them). The definitions are in `prompts/stage2_v2_answers.yaml`, versioned with the template; the notes of the scheme YAML files do not enter the prompt (accepted exception). Every question gets the `default` definitions, "for the argument in this text, the answer to the question is yes" (or no), except those with an entry of their own under `per_cq`. A question gets an entry only for one of three reasons: it cannot be understood when asked alone; the entry carries a reading already used in the per-CQ annotation; the entry separates it from a sibling question the model does not see. No entry sets the standard of judgement or a default answer. The ten entries of v2:
     * ad hominem CQ1: `negative` and `positive`, the opinion about the other person and its intent on their credibility;
     * ad hominem CQ3: `yes` if the attack discredits the person through a commitment or personal tie of theirs, or through their belonging to a group, movement or category viewed negatively, rather than through what they argue;
     * cause to effect CQ1: `yes` if A can plausibly lead to B, whatever one thinks of A or B, with no certainty required;
     * cause to effect CQ4: `yes` if another factor, which the model can name, is clearly the real reason for the effect, or the text presents A as the only cause of an effect that clearly has several; `no` if A is presented as one cause among others or no other factor clearly stands out;
     * correlation to cause CQ2.1 and CQ2.2: `yes` if the text's reasoning to the causal claim, whether or not that claim is right, rests only on the order of the two events (CQ2.1) or only on their going together (CQ2.2);
     * expert opinion CQ1: `yes` if S is in a position to know whether A is true or is a genuine expert recognised in D, either being enough;
     * expert opinion CQ3.1, "Is S taking part in the discussion?": `yes` if S has expressed a position of their own on the issue, within the exchange or elsewhere (a statement, an interview, a publication the text refers to); `no` if the claim is ascribed to S without S having spoken on it. A project choice, not a correction of the thesis: read literally, the question almost never gets `yes` on these texts (none from gpt-5 in the prior runs), while the annotation of the `Annotazione Dumitru` sheets uses the broader sense; the text of the question does not change;
     * expert opinion CQ4.1: `yes` if C follows from A, including when the text's conclusion is A itself;
     * slippery slope CQ4.1: `yes` if, once the intermediate steps the sequence needs are filled in, it would plausibly lead somewhere other than Cn.

     An entry under `per_cq` must list exactly the answers of the diagram's `answer_space`, in the order the prompt shows them, otherwise rendering fails; no definition names a fallacy label.
   * `cannot_be_determined`: only if the model has no basis at all for answering the real alternatives, even after using everything the section "What to do" admits (the text, what it makes clear without saying it, general knowledge, its own judgement). The text not stating the answer explicitly is not, by itself, a reason to use it; if the model leans towards one answer, however slightly, it gives it and lowers its confidence. In v1 it meant "the text contains no information at all on the point", and the models of the first pilot read it as "the text does not say it". It is a project choice, not an inheritance from Enrico's thesis: the definitive PDF asks for a binary answer, `Yes` or `No` (§4.3.2), and names an answer that cannot be determined from the text only among the future developments of the conclusions (`docs/cq_proposals.md` §7.1, `docs/schemes_and_diagrams.md` §0.5).
   * `na`: only if the question takes for granted something that this argument does not contain, not even implicitly, so that none of the other answers makes sense. A question that itself asks whether something exists is answered normally, and if what the question takes for granted is present the model answers it, even if unsure how it stands. Fourth option, added outside the diagram exactly like `cannot_be_determined`: never in the `answer_space` of a scheme. It marks a missing presupposition of the question, not a missing mention of what the question asks about.
6. What to do: answer as a careful reader, with one's own judgement, also on points the arguer does not write down; use what the text says, what it makes clear without saying it (evident purpose, tone, intended conclusion) and general knowledge widely shared by educated people, but nothing about where the text comes from or how it has been classified; judge assessments by the standards of everyday argument, not of proof; evaluate the argument that follows the scheme when the text contains other speakers or a question to the reader; do not let one's view on the truth of the conclusion decide, unless the question asks about truth; answer this question only, without deciding whether the argument is fallacious overall.
7. The answer format, with the justification naming what decided the answer: the fact used, if general knowledge decided it; what exists, if the answer says something is present; what could not be established, for `cannot_be_determined`; what the question takes for granted, for `na`.

`na` is the fourth answer, not `not_applicable`. It is added by the code (section 5), not by the diagram: no YAML in `schemes/` gains an `na` arc, and the guard cases stay encoded as nodes (for example expert opinion CQ4). A traversal that meets `na` finds no drawn arc: the model does not choose one, and which arc to take is up to the aggregator (phase 3), not to this spec.

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
* The field `answer` (or `scheme`) is the first key of the object, so the position of the first token of the answer is the same in every output and the client can read its logprobs. The server compacts the JSON (spec 04, section 1), so every output begins with `{"answer": "` (or `{"scheme": "`) and the context before the answer token is the same in every call of a prompt.
* The admitted values of `answer` are `yes`, `no`, `cannot_be_determined`, `na`, and for ad hominem CQ1 `positive`, `negative`, `cannot_be_determined`, `na`. The soft probability `p_logprob` is computed from the first token of the value: take the `top_logprobs` at that position, keep the entries that begin one of the admitted answers, renormalise. If an admitted answer does not appear among the top_logprobs, its mass is zero and the row is marked `logprob_partial`. The values are not renamed to begin with different tokens: the token `n` begins `no`, `na` and `negative`, and counts for the first of them in the list, but its mass was at most 0.00018 in the first pilot.
* `p_logprob` is the probability at temperature 1 over the tokens the grammar of the structured output admits: vLLM renormalises the logprobs after the mask of the grammar and before the temperature of the request. It is not the raw distribution of the model over its whole vocabulary.
* `confidence` is an integer from 0 to 100. `justification` is text, at most two sentences; the limit is in the prompt, not enforced by the schema.
* The JSON Schema constrains `answer` to the enumeration admitted for that CQ and `confidence` to the range. The parser validates again: an output that does not pass the schema is `invalid` (spec 01), kept and counted.

## 6. Samples

Every entry of `serving/models.yaml` states how many times it answers every question, at both stages, and how each sample is drawn (`samples`, `temperature`, `top_p`, `top_k`; spec 04): the same parameters for every sample, with the seed equal to the sample index.

* One sample (Qwen3.8 and Gemma 4 with reasoning off, at temperature 0): the hard answer is sample 0. The soft answers are `p_logprob` and `p_verbal`; `p_sample` is empty.
* Several samples (gpt oss and K2 Horizon, five at the temperature their producers recommend): the hard answer, at stage two and at stage one alike, is the majority over the valid samples, those parsed and not cut off (`finish_reason` other than `length`). A tie goes to the answer with the higher mean `confidence` over the samples that gave it, and a tie that remains to the answer given at the lowest sample index; with no valid sample the hard answer is `invalid`. `p_sample` is the frequency of the answer `yes` (or `positive`) over the valid samples, whose number is `n_samples_ok`. `p_logprob` is still computed, on sample 0, but it is not the main soft answer: after a reasoning that has already decided, it tends to the extremes.

`p_verbal` = `confidence`/100 of sample 0, oriented on the answer of sample 0: if it is `no`, `p_verbal` = 1 minus confidence/100; if it is `cannot_be_determined`, `p_verbal` is empty. The row carries `idk = true` when the hard answer is `cannot_be_determined`. `na` produces no `p_verbal`, in the same way as `cannot_be_determined` — reading it as a low probability of yes would decide, silently, that `na` means `no` — but it does not raise `idk`, which stays true only for `cannot_be_determined`: the two answers remain distinguishable downstream only by looking at `answer`, the only information this parser records about them. What `na` means for the aggregation remains an open point for phase 3, to be closed on the real data: no numeric value, not a provisional rule.

The models with internal reasoning (gpt oss, K2 Horizon, and Qwen3.8 and Gemma 4 in the condition with reasoning on, spec 04) produce the same JSON; the client reads the logprobs at the position of the final answer. If logprobs are not available for a model, `p_logprob` stays empty and the manifest says so.

## 7. Acceptance tests

Since 20 September 2026 `tests/` keeps only the tests that defend a result, check the data or help the work. The points without a note have an automatic test; the marked points remain requirements, but today they have no test, or only a partial one.

1. For every scheme and every CQ, the rendered prompt contains the `text` field of the YAML as an exact substring.
2. For ad hominem CQ1 the options are `positive`, `negative`, `cannot_be_determined`, `na`; for all the other CQs `yes`, `no`, `cannot_be_determined`, `na`.
3. The stage-two JSON Schema rejects an `answer` outside the enumeration and a `confidence` outside the range; the stage-one schema rejects a non-canonical `scheme`. *Today only the stage-two part has an automatic test, checked through the parser (spec 04, point 5).*
4. Rendering is deterministic: same item, same prompt, byte for byte. *Today without an automatic test.*
5. A template with the `{{prompt_version}}` placeholder gets the version in the rendered prompt; one without it renders without error. The v2 templates have none. *Today without an automatic test.*

## 8. Zero-shot, `zeroshot_v1.md`

A comparison condition for A0: for every item, the model receives the text, the gold scheme and the terminals of the diagram of that scheme, each with a definition, and chooses the verdict directly. It has the information A0 has in the `gold` condition, without passing through the CQs. Version `zeroshot_v1`, rendered by `render_zeroshot(item, scheme)`.

The template has the structure and the reading instructions of `stage2_v2.md`, so that the only difference from stage two is the task:
1. The same role, with the task of choosing the one verdict that best describes the argument; the same block of the scheme (`name`, `schema`, `variables`) and of the text, filled by the same functions.
2. In place of the critical question, the verdicts: the terminals of the scheme's diagram, taken from the YAML, never written by hand, in alphabetical order of the name shown (case aside), with Good Argumentation last. Ad hominem lists none, like its diagram. Each verdict is a line ``* `name`: definition``.
3. "What to do" keeps the indications on reading: own judgement, also on points the arguer does not write down; what the text says and makes clear, and general knowledge; nothing about where the text comes from or how it has been classified; the standards of everyday argument where a verdict calls for an assessment; the argument that follows the scheme; one's view on the truth of the conclusion does not decide. Without a question, "unless the question asks about truth" goes, and so does the sentence that forbids deciding whether the argument is fallacious: the task is to choose the single verdict that best describes the argument.
4. The answer, `{"answer": <one of the verdicts>, "confidence": <0-100>, "justification": <at most two sentences>}`, `answer` first, prefix `{"answer": "` as at stage two (section 5). The JSON Schema `schemas/zeroshot_v1.json` gets, per scheme, the enumeration of the names shown.

Names and definitions come from `labels/fallacies.yaml` (version 1.7, rule 3 of spec 01). The name shown is the `label` of the terminal, except the four ad hominem variants, which have a `display_label` without the prefix: `Abusive`, `Circumstantial`, `Guilt by Association`, `Tu Quoque`; `Ad Fidentia` and the two `Appeal to` keep theirs. The `definition` of every fallacy is an English translation of the definition the thesis gives in chapter 3, under "Fallacie relative allo schema" of every scheme where the fallacy occurs (one text per fallacy, the same in every scheme), with nothing added; where the original has two sentences they are joined into one. The thesis does not define Good Argumentation: its definition, "The argument is acceptable as it stands: none of the fallacies listed applies to it.", is the project's own and is marked as such in the file. Two terminals with the same name shown, or a terminal without a definition, stop the loading of the vocabulary.

The parser takes the name shown back to the terminal id through the same file, and records the probability of the label chosen (spec 04, section 4.2).

Acceptance tests: rendering is deterministic; the admitted verdicts, in the JSON Schema and in the prompt, are the terminals of the scheme, in the order above; ad hominem has no Good Argumentation; every name shown is read back as its terminal id, and a name not shown (the full `label` of an ad hominem variant) is `invalid`.
