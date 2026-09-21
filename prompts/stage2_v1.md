You are an argumentation analyst. The text below argues in the form of a known
argumentation scheme. Your task is to answer **one** critical question about it.

## The scheme: {{scheme_name}}

{{schema}}

Where:

{{variables}}

## Text

{{text}}

## The critical question

**{{cq_id}}.** {{cq_text}}

## The answers you may give

{{answer_options}}

## What to do

Answer from the text alone. Do not use outside knowledge about the topic, the
speaker, or whether the conclusion happens to be true, unless the question
itself asks you to consider something beyond the text.

Answer this question only. Do not answer the other critical questions of the
scheme, and do not decide whether the argument is fallacious overall.

## Your answer

Reply with a single JSON object and nothing else:

```json
{"answer": "<one of the options above>", "confidence": <integer 0-100>, "justification": "<at most two sentences>"}
```

`answer` must be the first key. `confidence` is how sure you are of the answer,
0 to 100. `justification` is at most two sentences pointing at what in the text
decided it.

<!-- prompt_version: {{prompt_version}} -->
