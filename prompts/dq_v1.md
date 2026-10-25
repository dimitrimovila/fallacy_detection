You are an argumentation analyst. The text below has been identified as an
argument of the scheme shown. Your task is to answer every critical question of the scheme about it and then to choose the **one** verdict that best describes the argument, the way a careful and fair reader with a good general education would.

## The scheme: {{scheme_name}}

{{schema}}

Where:

{{variables}}

## Text

{{text}}

## The critical questions

Answer each of these questions about the argument in the text. Besides the options
listed under a question, every question admits two more answers:

* `cannot_be_determined`: use this only if you have no basis at all for answering with
  one of the options listed under the question, even after using everything listed in
  the section "What to do". The text not stating the answer explicitly is not, by
  itself, a reason to use it. If you lean towards one option, however slightly, give it.
* `na`: {{na_definition}}

{{questions}}

## The verdicts you may give

{{verdicts}}

## What to do

Answer as a careful reader would do, using your own judgement, including on deducible
points the arguer does not write down.

Use what the text says; what it makes clear without saying it, such as its
evident purpose, tone and intended conclusion; and general knowledge on facts widely known to educated
people. Do not use anything you may know
about where this text comes from or how it has been classified. Where a question or a verdict
calls for an assessment, give your own, by the standards of everyday argument, rather than of proof.
Evaluate the argument as the text makes it; if the text also contains other
speakers or a question to the reader, evaluate the argument that follows the
scheme. Do not let your own view on whether the conclusion is true decide an answer, unless a question asks about truth.

Answer every critical question first, in the order listed. Then, with your answers in view, choose the single verdict, among those listed, that best describes the argument.

## Your answer

Reply with a single JSON object and nothing else, with the keys in the order shown:

```json
{{answer_example}}
```

Each key `CQ...` takes one of the answers of its question, written exactly as listed;
`verdict` is one of the verdicts above, written exactly as listed.
`confidence` is how likely you think it is that
your verdict is right, from 0 to 100. Lower it when the text can be read in more
than one way, when careful readers could judge the point differently, or when
your verdict rests on knowledge you are not sure of. `justification` is at most
two sentences: point at what in the text decided the verdict, and name the fact
you used if general knowledge decided it.
