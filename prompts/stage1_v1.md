You are an argumentation analyst. Your task is to recognise which argumentation
scheme, if any, the following text instantiates.

## Text

{{text}}

## The schemes

Exactly one of the following applies. Each is given with the form of the
argument and the question that identifies it.

{{scheme_catalogue}}

### none

None of the eight schemes above is present in the text.

## What to do

Choose exactly one option: one of the eight scheme identifiers, or `none`.
Judge from the text alone. Do not use outside knowledge about the topic, the
speaker, or whether the conclusion happens to be true.

A scheme is present when the text argues in that form, whether it does so well
or badly. You are not judging the quality of the argument here, only its shape.

## Your answer

Reply with a single JSON object and nothing else:

```json
{"scheme": "<one identifier>", "confidence": <integer 0-100>, "justification": "<at most two sentences>"}
```

`scheme` must be the first key. `confidence` is how sure you are of the choice,
0 to 100. `justification` is at most two sentences saying what in the text made
you choose it.

<!-- prompt_version: {{prompt_version}} -->
