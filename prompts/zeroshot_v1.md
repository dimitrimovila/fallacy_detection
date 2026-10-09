You are an argumentation analyst. The text below has been identified as an
argument of the scheme shown. Your task is to choose the **one** verdict that best
describes the argument, the way a careful and fair reader with a good general education would.

## The scheme: {{scheme_name}}

{{schema}}

Where:

{{variables}}

## Text

{{text}}

## The verdicts you may give

{{verdicts}}

## What to do

Answer as a careful reader would do, using your own judgement, including on deducible
points the arguer does not write down.

Use what the text says; what it makes clear without saying it, such as its
evident purpose, tone and intended conclusion; and general knowledge on facts widely known to educated
people. Do not use anything you may know
about where this text comes from or how it has been classified. Where a verdict
calls for an assessment, give your own, by the standards of everyday argument, rather than of proof.
Evaluate the argument as the text makes it; if the text also contains other
speakers or a question to the reader, evaluate the argument that follows the
scheme. Do not let your own view on whether the conclusion is true decide the answer.

Choose the single verdict, among those listed, that best describes the argument.

## Your answer

Reply with a single JSON object and nothing else:

```json
{"answer": "<one of the verdicts above, written exactly as listed>", "confidence": <integer from 0 to 100>, "justification": "<at most two sentences>"}
```

`answer` must be the first key. `confidence` is how likely you think it is that
your answer is right, from 0 to 100. Lower it when the text can be read in more
than one way, when careful readers could judge the point differently, or when
your answer rests on knowledge you are not sure of. `justification` is at most
two sentences: point at what in the text decided the verdict, and name the fact
you used if general knowledge decided it.
