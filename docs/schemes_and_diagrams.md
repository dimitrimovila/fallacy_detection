# Argumentation schemes, critical questions and diagrams

**Source.** E. Bergamasco, *Schemi argomentativi e domande critiche nei Large Language Models come
strumento di individuazione delle fallacie* [Argumentation schemes and critical questions in Large
Language Models as a tool for fallacy detection], master's thesis, University of Padova, Department
of Linguistic and Literary Studies, academic year 2025-2026. Supervisor prof. Massimiliano Carrara,
co-supervisor prof. Giovanni Da San Martino. **Definitive PDF confirmed by the author**, 14 September
2026, 181 pages, **chapter 3**, §3.1 "Diagrammi degli schemi" [Diagrams of the schemes], printed
pp. 46-92.

**Check against the definitive PDF.** On 14 and 15 September 2026 the scheme cards were compared
block by block with the definitive PDF. The texts of the critical questions, the identification
questions and the lists of fallacies coincide word for word with what is transcribed
here, with the only exception recorded in §0.4. Two formal **Schema** blocks, ad hominem and slippery
slope, did not coincide and were replaced: see §0.5. All the page references have been
realigned to the numbering of the definitive PDF; the transcription had been made on the
135-page PDF, whose numbers were +2.

**What this document contains.** Only the schemes. Argumentation scheme, identification
question, critical questions, diagram, terminals, plus the structural properties that
follow from them. Dataset, annotation method, prompts, runs and results are in
`diagnosi_pipeline_cq.md`, outside the repository.

---

## Conventions

**The critical questions report the words of chapter 3**, with the spelling and grammar
errors corrected. No reformulation, no word changed, no content
added or removed. The corrections applied concern `committed`, `claimed`,
`claim` in the infinitive and the spelling of the terminal `Good Argumentation`.

**The text of the thesis and our observations are kept separate.** Everything that is not
transcription is marked `⟶ our observation` or collected in §12. Nothing in the cards must be
attributed to Enrico unless it is transcribed.

**The arcs of the diagrams** were read from the rasterised pages of the PDF, because the
flowcharts are vector graphics and text extraction returns the labels `yes` and `no`
in an order that cannot be reconstructed. The one of *expert opinion* was reread on this version
because its structure changed; the other seven are reported from the previous reading, whose
lists of CQs have no structural changes.

**Notation.** `CQn: answer → destination`. `⟲` flags a reconvergence arc, that is, two or
more paths that enter the same node or terminal.

**The reference version is chapter 3.** Chapter 4 of the same thesis reports, in a
prompt listing, a different version of ad hominem, and the prompts actually run in the
experiments report a third one. The comparison is in §11 and does not contaminate the cards.

**Name of the non-fallacious terminal.** In the definitive PDF the green terminal is called
`Non-fallacious Argument`, while in the previous versions it was called `Good Argument` or
`Good Argumentation` (in appendix B of the definitive PDF `Good Argument` remained). Here it stays
`Good Argumentation`: it is the same terminal, and the name changes neither the path nor the meaning.

---

## 0. Changelog with respect to the previous revision of this document

The previous revision was based on the 73-page PDF. Comparison limited to the schemes.

### 0.1 Resolved

| Problem | Status |
|---|---|
| **Comments in Italian systematically out of date.** On all 8 schemes the "Commento" [Comment] section described a set of CQs different from the one listed and diagrammed | **Resolved on all 8.** Checked by reading the opening of each Comment. They now describe the CQs actually diagrammed |
| **CQ1 of expert opinion with `and/or`**, a truth connective that is not defined | **Mitigated.** It is now `or` |
| **CQ4 of expert opinion**, a conditional with the guard inside the text of the question | **Resolved.** Split into CQ4 plus CQ4.1 |
| **Inconsistent numbering of slippery slope**, `CQ4.1` in the list and `CQ5.1` in the node | **Resolved.** Now `C.Q. 4.1` in both |

### 0.2 Modified

| Scheme | What |
|---|---|
| example, CQ2.1 | rewritten on the model of the one of cause to effect, from "selected on the basis of biased quality criteria that undermine the strength" to "selected in a clearly biased and malicious manner in order to undermine the validity" |

### 0.3 Unchanged in the thesis, corrected by us

The two inverted polarities of correlation to cause CQ5 and slippery slope CQ5 are still present
in the PDF. We corrected them by inverting the arcs, see §4, §8 and §12.1.

Ad hominem without a Good Argumentation terminal remains unchanged and uncorrected, because it is not
a sign error but a coverage choice. See §12.4.

### 0.4 Comparison with the definitive PDF (14 September 2026, 181 pages)

**correlation to cause, CQ3.** The thesis wrote `a significant number of observation?`, the
definitive PDF writes `observations`. Our transcription was already in the plural, so from a
correction of ours it becomes text of the thesis: `observations` was removed from the list of
corrections in the Conventions. The form `claimed` remains a correction of ours, the definitive PDF
still writes `Is the claim relationship`.

**The two inverted polarities of §0.3 are still inverted in the definitive PDF too.** Rechecked
on the rendered pages: correlation to cause CQ5 sends `yes` to Definist Fallacy (printed p.
64), slippery slope CQ5 sends `no` to Slippery Slope (printed p. 90). Our two arc
corrections remain necessary.

**Page references, realigned.** The page numbers of this document came from the
135-page PDF and were +2 with respect to the definitive one. They have all been brought back to the
numbering of the definitive PDF, taking them from the PDF itself and not converting them by hand.
§3.1 is at pp. 46-92.
Pages of the diagrams: 49 example, 54 analogy, 59 cause to effect, 64 correlation to cause,
70 ad hominem, 77 expert opinion, 83 popular opinion, 90 slippery slope.

**Critical questions, identification questions and lists of fallacies** coincide word for
word with what is transcribed here. As for the diagrams, the three of correlation to cause, slippery
slope and expert opinion were reread on the rendered pages of the definitive PDF; the other five
were checked on the extraction of the labels of nodes and terminals, which shows no
changes. **The formal Schema blocks had not been checked**: two of them did not coincide,
see §0.5.

### 0.5 Formal schemas realigned and source B lapsed (15 September 2026)

The comparison of §0.4 had covered critical questions, identification questions, lists
of fallacies and diagrams, but not the **Schema** blocks. Once those were checked too, two out of
eight diverged from the definitive PDF, because the transcription dated back to the 135-page PDF.

| Scheme | What changed |
|---|---|
| ad hominem, §5 | replaced with the two lines printed on p. 68 ("a is a person of bad character…"). Before, there was Walton's long ethotic formulation |
| slippery slope, §8 | replaced with the version in `C0 … Cn` notation printed on p. 88. Before, there was the `A0 … An` notation |
| slippery slope, CQ 2 | goes back to `Cn` as in the thesis. The correction `An`, applied for consistency with the old schema, was also removed from the Conventions |

**Why it matters.** The formal schema is not decoration: in the stage-one prompt it is shown
to the model together with the closed list of the eight schemes, instead of the bare label. If
our YAML files carry the old schema, the model sees a text that is not that of the thesis.
To be checked in the files of `schemes/`.

**The prompt listing no longer exists.** In the definitive PDF, §4.3.2 (pp. 109-110) describes the
prompts in words and does not report their text. It states that every critical question receives a
**binary answer, Yes or No**, with a mandatory justification anchored in the text. In all the 181
pages the string "Cannot be determined from the text" never appears. Source **B** of
§11 therefore lapses as a citable source, and the third answer of our prompt remains a project
choice of ours, to be justified as such and not as an inheritance from the thesis.

**Lesson of method.** A comparison between two versions of the thesis is not enough to validate our
files: our document must be compared with the thesis, block by block. §0.4 compared the
two drafts of the thesis with each other, and for this reason it had not seen the two divergences.

---

## 1. Argument from example (§3.1.1, pp. 47-51)

**Schema**
```
In this particular case, the individual a has property F and also property G.
a is typical of things that have F and may or may not also have G.
Therefore, generally, if x has property F then x also has property G.
```

**Identification question**
> Does the text use an example that is considered appropriate to draw a conclusion on the main topic?

**Critical questions**
- **C.Q. 1** — Is the statement illustrated by the example clearly false or unrealistic?
- **C.Q. 2** — Is the example typical of the kinds of cases that the generalization ranges over?
  - **C.Q. 2.1** — Was the example cited selected in a clearly biased and malicious manner in order to undermine the validity of the generalization?
  - **C.Q. 2.2** — Is the example given so specific that it is statistically unrepresentative and therefore cannot be used to support the generalisation?
  - **C.Q. 2.3** — Is the example given so extreme that it invalidates the generalisation?
- **C.Q. 3** — Does the example cited have characteristics that make it too different from the generalisation drawn in the conclusion?

**Diagram**
```
CQ1   yes → False Attribution
      no  → CQ2
CQ2   no  → CQ2.1
      yes → CQ3
CQ2.1 yes → Cherry Picking
      no  → CQ2.2
CQ2.2 yes → Hasty Generalization
      no  → CQ2.3
CQ2.3 yes → Appeal to Extremes
      no  → CQ3            ⟲
CQ3   yes → Weak Analogy
      no  → Good Argumentation
```

**Terminals** — False Attribution · Cherry Picking · Hasty Generalization · Appeal to Extremes ·
Weak Analogy · Good Argumentation

⟶ *our observation*. CQ1 has its polarity inverted with respect to Walton's CQ1, which the thesis
itself quotes in a footnote, "Is the proposition claimed in the premise in fact true?". Here `yes`
means a false example, hence a fallacy. The Comment of the thesis is consistent with this polarity,
so it is a choice, not an error.

---

## 2. Argument from analogy (§3.1.2, pp. 52-56)

**Schema**
```
Generally, case C1 is similar to case C2.
A is true (false) in case C1.
Therefore A is true (false) in case C2.
```

**Identification question**
> Does the text use a case considered similar to another in a certain respect to draw a conclusion about the main topic?

**Critical questions**
- **C.Q. 1** — Are there differences between C1 and C2 that would tend to undermine the force of the similarity cited?
  - **C.Q. 1.1** — Does the analogy rely on an extreme scenario to make the situation seem better or worse?
- **C.Q. 2** — Is the similarity cited relevant to A?
- **C.Q. 3** — Is A true (false) in C1?
- **C.Q. 4** — Is there some other case C3 that is also similar to C1, but in which A is false (true)?

**Diagram**
```
CQ1   yes → CQ1.1
      no  → CQ2
CQ1.1 yes → Appeal to Extremes
      no  → Weak Analogy
CQ2   no  → CQ1.1           ⟲
      yes → CQ3
CQ3   no  → False Premise
      yes → CQ4
CQ4   yes → Cherry Picking
      no  → Good Argumentation
```

**Terminals** — Appeal to Extremes · Weak Analogy · False Premise · Cherry Picking ·
Good Argumentation

---

## 3. Argument from cause to effect (§3.1.3, pp. 57-61)

**Schema**
```
Generally, if A occurs, then B will (might) occur.
In this case, A occurs (might occur).
Therefore in this case, B will (might) occur.
```

**Identification question**
> Does this text use a known causal relationship as a basis for predicting or explaining a future or past event?

**Critical questions**
- **C.Q. 1** — Is the implication that A leads to B (regardless of what judgement one might make about it) logically sound?
- **C.Q. 2** — Is the evidence cited (if there is any) strong enough to warrant the causal generalization?
  - **C.Q. 2.1** — Was the evidence cited selected in a clearly biased and malicious manner in order to undermine the validity of the generalization?
  - **C.Q. 2.2** — Is the evidence cited insufficient in quantitative terms to support the generalisation?
- **C.Q. 3** — Is the relationship between cause and effect based solely on temporal sequence?
- **C.Q. 4** — Are there other causal factors that clearly represent the real reason behind the occurrence of the effect?

**Diagram**
```
CQ1   no  → False Premise
      yes → CQ2
CQ2   no  → CQ2.1
      yes → CQ3
CQ2.1 yes → Cherry Picking
      no  → CQ2.2
CQ2.2 yes → Hasty Generalization
      no  → CQ3            ⟲
CQ3   yes → Post Hoc
      no  → CQ4
CQ4   yes → Causal Reductionism
      no  → Good Argumentation
```

**Terminals** — False Premise · Cherry Picking · Hasty Generalization · Post Hoc ·
Causal Reductionism · Good Argumentation

---

## 4. Argument from correlation to cause (§3.1.4, pp. 62-66)

**Schema**
```
There is a positive or negative correlation between A and B.
Therefore, A causes B.
```

**Identification question**
> Does this text start with two correlated events and conclude that one causes the other?

**Critical questions**
- **C.Q. 1** — Is there actually a positive or negative correlation between A and B?
- **C.Q. 2** — Is one of the two events actually the cause of the other?
  - **C.Q. 2.1** — Is the rationale behind false causation based solely on the fact that one of the two correlated events occurs before the other?
  - **C.Q. 2.2** — Is the rationale behind false causation based solely on the fact that one of the two related events is regularly associated with the other?
- **C.Q. 3** — Is the claimed relationship between A and B based on a significant number of observations?
- **C.Q. 4** — Could there be a third factor C (or a set of several factors) that is the clear cause of B or of both A and B?
- **C.Q. 5** — Can it be shown that the increase or change in B is not solely due to the way B is defined?

⟶ *version note*. `observations` in the plural is the text of the definitive PDF, not a correction
of ours. `claimed` remains a correction of ours, the thesis writes `Is the claim relationship`.
See §0.4.

**Diagram**
```
CQ1   no  → False Premise
      yes → CQ2
CQ2   no  → CQ2.1
      yes → CQ3
CQ2.1 yes → Post Hoc
      no  → CQ2.2
CQ2.2 yes → Questionable Cause
      no  → CQ3            ⟲
CQ3   no  → Hasty Generalization
      yes → CQ4
CQ4   yes → Causal Reductionism
      no  → CQ5
CQ5   yes → Good Argumentation      ⚑ arc corrected by us
      no  → Definist Fallacy        ⚑ arc corrected by us
```

⚑ **Correction applied.** In the diagram of the thesis, checked on the rasterised page, printed
p. 64, the arcs of CQ5 are the opposite, `yes` towards Definist
Fallacy and `no` towards Good Argumentation. Given the text of the question that direction is
reversed, and the Comment of the thesis itself confirms it, according to which the definist fallacy
arises when the change in B is "un effetto della definizione stessa di B, piuttosto che un mutamento
sostanziale del fenomeno osservato" [an effect of the very definition of B, rather than a substantial
change in the observed phenomenon]. We inverted the two arcs, leaving the text
of the question intact. Rechecked on the definitive PDF on 14 September 2026: the arcs are still
inverted, the correction remains necessary.

**Terminals** — False Premise · Post Hoc · Questionable Cause · Hasty Generalization ·
Causal Reductionism · Definist Fallacy · Good Argumentation

---

## 5. Argument ad hominem (§3.1.5, pp. 67-72)

**Schema**
```
a is a person of bad character.
Therefore, a's argument A should not be accepted.
```

⚑ **Schema replaced on 15 September 2026.** This is the text printed on p. 68 of the definitive
PDF, with citation [6]. The previous revision of this document reported Walton's longer ethotic
formulation ("If x is a person of good (bad) moral character, then
what x says should be accepted as more plausible…"), which came from the 135-page PDF and no longer
appears in the thesis. It matters because the formal schema is shown to the model in the stage-one
prompt. See §0.5.

**Identification question**
> Does the text shift the focus from arguments to personal judgments about the other person?

**Critical questions**
- **C.Q. 1** — Is the opinion expressed about the other person negative (and intended to undermine their credibility) or positive (and intended to enhance their credibility)?
- **C.Q. 2** — Does the direct attack on the other person amount to an accusation that they are biased?
- **C.Q. 3** — Does the direct attack on the other person consist of an accusation that they are committed?
- **C.Q. 4** — Is the direct attack on the other person an accusation of hypocrisy?
- **C.Q. 5** — Is the direct attack on the other person intended to undermine their self-confidence?
- **C.Q. 6** — Is a direct attack on the other person used as a pretext to discredit them even before the discussion has begun?

**Diagram**
```
CQ1   positive → Appeal to Authority
      negative → CQ2
CQ2   yes → Ad Hominem (Circumstantial)
      no  → CQ3
CQ3   yes → Ad Hominem (Guilt by Association)
      no  → CQ4
CQ4   yes → Ad Hominem (Tu Quoque)
      no  → CQ5
CQ5   yes → Ad Fidentia
      no  → CQ6
CQ6   yes → Poisoning the Well
      no  → Ad Hominem (Abusive)
```

**Terminals** — Appeal to Authority · Ad Hominem (Circumstantial) · Ad Hominem (Guilt by
Association) · Ad Hominem (Tu Quoque) · Ad Fidentia · Poisoning the Well · Ad Hominem (Abusive)

**Structural uniqueness.** It is the only one of the eight schemes **without a Good Argumentation
terminal**. Every path ends in a fallacy.

⟶ *our observation*. CQ1 is the only question of the whole set that does not branch on `yes` and `no`
but on `positive` and `negative`. This must be kept in mind in the encoding, because the answer
space is not uniform across the nodes.

---

## 6. Argument from expert opinion (§3.1.6, pp. 73-79)

**Schema**
```
S is an authoritative source (whether individual or collective) in the domain D.
S asserts that A is known to be true.
A is in D.
(A implies the conclusion C)
Therefore A (and/or C) may (plausibly) be taken as true.
```

**Identification question**
> Does this text use the opinion of a (supposedly) authoritative source as evidence to support a claim?

**Critical questions**
- **C.Q. 1** — Is S in a position to know whether A is true or false, or is S a genuine expert recognised by the community of experts in D?
- **C.Q. 2** — The fact that S has asserted (or claimed) A is used as the sole supporting consideration for the truth of the conclusion, without providing any other type of argument?
- **C.Q. 3** — Did S really assert (or claim) A as true?
  - **C.Q. 3.1** — Is S taking part in the discussion?
- **C.Q. 4** — Is there a conclusion C that differs from A but uses A to justify itself?
  - **C.Q. 4.1** — Is conclusion C actually implied by A?

**Variables.** `S` source, `D` domain, `A` assertion of the source, `C` conclusion of the
arguer.

**Diagram**
```
CQ1   no  → Irrelevant Authority
      yes → CQ2
CQ2   yes → Appeal to Authority
      no  → CQ3
CQ3   no  → CQ3.1
      yes → CQ4
CQ3.1 no  → False Attribution
      yes → Strawman
CQ4   no  → Good Argumentation
      yes → CQ4.1
CQ4.1 no  → Non Sequitur
      yes → Good Argumentation   ⟲
```

**Terminals** — Irrelevant Authority · Appeal to Authority · False Attribution · Strawman ·
Non Sequitur · Good Argumentation

⟶ *our observation*. With the split of CQ4, the case in which the arguer merely
reports the assertion, that is `C = A`, exits on CQ4 `no` towards Good Argumentation without needing
to encode an `na` value. The case of the **implicit** conclusion, the enthymeme, remains uncovered:
the diagram gives no rule for it. See §12.3.

---

## 7. Argument from popular opinion (§3.1.7, pp. 80-85)

**Schema**
```
If a large majority of some reference group accept A as true, then there is a
presumption in favour of A.
A large majority of the reference group accept A as true.
Therefore there is a presumption in favour of A.
```

**Identification question**
> Does this text use the fact that the conclusion is supported by a relatively large number of people as its main argument in favour of that conclusion?

**Critical questions**
- **C.Q. 1** — Are there any indications that the group of people who maintain that A is true (or false) was clearly chosen because they were already biased?
- **C.Q. 2** — Are there any indications that the group of people who maintain that A is true (or false) is clearly too small to be considered statistically significant?
- **C.Q. 3** — Is the fact that A is considered to be true (or false) by many people a good reason to believe that A is in fact true (or false)?
  - **C.Q. 3.1** — Is the presumption in favour of A based solely on the fact that it is accepted by a large majority of the reference group?
  - **C.Q. 3.2** — Does the majority of the reference group rely on the concept of tradition to assert A?
  - **C.Q. 3.3** — Does the majority of the reference group rely on the concept of nature to assert A?

**Diagram**
```
CQ1   yes → Cherry Picking
      no  → CQ2
CQ2   yes → Hasty Generalization
      no  → CQ3
CQ3   no  → CQ3.1
      yes → Good Argumentation
CQ3.1 yes → Ad Populum
      no  → CQ3.2
CQ3.2 yes → Appeal to Tradition
      no  → CQ3.3
CQ3.3 yes → Appeal to Nature
      no  → Ad Populum       ⟲
```

**Terminals** — Cherry Picking · Hasty Generalization · Ad Populum · Appeal to Tradition ·
Appeal to Nature · Good Argumentation

⟶ *our observation*. Ad Populum is reachable from two distinct arcs, CQ3.1 `yes` and CQ3.3
`no`. The terminal label alone does not identify the path.

---

## 8. Slippery slope argument (§3.1.8, pp. 86-92)

**Schema**
```
Case C0 is tentatively acceptable as an initial presumption.
There exists a series of cases, C0, C1,..., Cn−1, where each case leads to the next by a
combination of causal, precedent, and/or analogy steps.
There is a climate of social opinion such that once people come to accept each step as
plausible (or as accepted practice), they will then be led to accept the next step.
The penultimate step Cn−1 leads to a horrible outcome, C0, which is not acceptable.
Therefore, C0 is not acceptable (contrary to the presumption of the initial premise).
```

⚑ **Schema replaced on 15 September 2026.** This is the text printed on p. 88 of the definitive
PDF, with citation [9], in `C0 … Cn` notation. The previous revision reported the
version in `A0 … An` notation, which came from the 135-page PDF. As a consequence CQ 2 goes back
to saying `Cn`, which is consistent with the schema: our correction from `Cn` to `An` is no longer
needed.

⟶ *our observation*. The text of the thesis contains a typo: "The penultimate step Cn−1
leads to a horrible outcome, C0" should say `Cn`, and `Cn` is the outcome that CQ 2 refers to.
We leave it as printed and flag it here.

**Identification question**
> The text contains an argument claiming that if you allow or carry out a particular action (even if it's small or seemingly harmless), it will inevitably lead (through a chain of intermediate consequences) to an extreme or catastrophic final outcome?

**Critical questions**
- **C.Q. 1** — Do any of the causal links in the chain lack sufficient evidence to support the claim that they will (might, must) occur?
- **C.Q. 2** — Is the outcome Cn as bad as suggested?
- **C.Q. 3** — Is it possible to provide a precise definition that removes the ambiguity to such an extent as to halt the decline?
- **C.Q. 4** — Are there other steps required to fill in the sequence of events and make it plausible?
  - **C.Q. 4.1** — Would these steps lead to a different conclusion?
- **C.Q. 5** — Are there weak links in the sequence, where specific critical questions should be asked on whether one event will really lead to another?

**Diagram**
```
CQ1   yes → Slippery Slope
      no  → CQ2
CQ2   no  → Slippery Slope         ⟲
      yes → CQ3
CQ3   yes → Definist Fallacy
      no  → CQ4
CQ4   yes → CQ4.1
      no  → CQ5
CQ4.1 yes → Slippery Slope         ⟲
      no  → CQ5                    ⟲
CQ5   yes → Slippery Slope         ⟲ · ⚑ arc corrected by us
      no  → Good Argumentation     ⚑ arc corrected by us
```

⚑ **Correction applied.** In the diagram of the thesis, checked on the rasterised page, printed
p. 90, the arcs of CQ5 are the opposite, `no` towards Slippery
Slope and `yes` towards Good Argumentation. The presence of weak links in the chain is the fallacious
condition, not the sound one, and the Comment of the thesis on CQ1 confirms it: "se anche un solo
anello causale risulta privo di adeguata giustificazione, la catena si spezza e l'argomentazione
ricade nella fallacia dello slippery slope" [if even a single causal link lacks adequate
justification, the chain breaks and the argument falls into the slippery slope fallacy]. We inverted
the two arcs, leaving the text of the question intact. Rechecked on the definitive PDF on
14 September 2026: the arcs are still inverted, the correction remains necessary.

**Terminals** — Slippery Slope · Definist Fallacy · Good Argumentation

⟶ *our observation*. Slippery Slope is reachable from four distinct arcs. It is the scheme
with the greatest ambiguity between label and path.

---

## 9. Inverse index fallacy → schemes

Derived from the diagrams of this version. A consistency constraint for annotation in the direction
from a known fallacy to the scheme to be found. If a label does not appear here, none of the eight
schemes can produce it.

| Fallacy | Schemes that produce it |
|---|---|
| Ad Fidentia | ad hominem |
| Ad Hominem (Abusive) | ad hominem |
| Ad Hominem (Circumstantial) | ad hominem |
| Ad Hominem (Guilt by Association) | ad hominem |
| Ad Hominem (Tu Quoque) | ad hominem |
| Ad Populum | popular opinion |
| Appeal to Authority | expert opinion, ad hominem |
| Appeal to Extremes | example, analogy |
| Appeal to Nature | popular opinion |
| Appeal to Tradition | popular opinion |
| Causal Reductionism | cause to effect, correlation to cause |
| Cherry Picking | example, analogy, cause to effect, popular opinion |
| Definist Fallacy | correlation to cause, slippery slope |
| False Attribution | example, expert opinion |
| False Premise | analogy, cause to effect, correlation to cause |
| Hasty Generalization | example, cause to effect, correlation to cause, popular opinion |
| Irrelevant Authority | expert opinion |
| Non Sequitur | expert opinion |
| Poisoning the Well | ad hominem |
| Post Hoc | cause to effect, correlation to cause |
| Questionable Cause | correlation to cause |
| Slippery Slope | slippery slope |
| Strawman | expert opinion |
| Weak Analogy | example, analogy |

**Not reachable from any scheme** — `Appeal to Emotion`.

---

## 10. Structural properties

**The diagrams are not trees.** Six schemes out of eight contain reconvergences. Only **ad hominem**
is a pure tree.

| Scheme | Reconvergence on a decision node | Reconvergence on a terminal |
|---|---|---|
| example | CQ2.3 `no` → CQ3 | — |
| analogy | CQ2 `no` → CQ1.1 | — |
| cause to effect | CQ2.2 `no` → CQ3 | — |
| correlation to cause | CQ2.2 `no` → CQ3 | — |
| ad hominem | — | — |
| expert opinion | — | Good Argumentation, 2 arcs |
| popular opinion | — | Ad Populum, 2 arcs |
| slippery slope | CQ4.1 `no` → CQ5 | Slippery Slope, 4 arcs |

**Consequence.** Reconstructing the answer vector backwards from the pair (exit CQ,
verdict) is not unique. Example on *example*: CQ3 is reached both from `CQ2=yes` and from
`CQ2=no, CQ2.1=no, CQ2.2=no, CQ2.3=no`. The full answer vector must be **recorded**, not
reconstructed.

**Depth**, number of decision nodes on the longest path.

| Scheme | Depth | Scheme | Depth |
|---|---|---|---|
| correlation to cause | 7 | ad hominem | 6 |
| example | 6 | popular opinion | 6 |
| cause to effect | 6 | slippery slope | 6 |
| expert opinion | 5 | analogy | 4 |

**Non-uniform answer space.** All the nodes branch on `yes` and `no` except CQ1 of
ad hominem, which branches on `positive` and `negative`.

**Early exit.** The final CQs of every scheme are reached only by a fraction of the
items, and not at random. If the per-CQ gold is recorded only along the path, the
supervision on the late CQs is systematically scarce.

---

## 11. Divergences between the writings of the same schemes

The same eight schemes exist in four writings that do not coincide. **The canonical version is
the first**, and it is the one transcribed in §1-8.

| | source | role |
|---|---|---|
| **A** | thesis, chapter 3 | official and citable |
| **B** | 135-page thesis, §4.3, prompt listing | **no longer exists.** In the definitive PDF §4.3.2 describes the prompts only in words, without a listing. It stays cited below because it documents a prompt that was never run, but it is no longer a citable source |
| **C** | `Lavoro di Enrico/src/prompts/template_stage2_*.txt` | the prompts actually run |
| **D** | `Thesis/src/YAML/*.yaml` | Dumitru's encoding |

Text divergences, listed only where they are substantial. Purely orthographic differences
are not reported, because in this document and in our files the spelling is corrected.

| Scheme, CQ | A, thesis ch. 3 | C, run | D, YAML |
|---|---|---|---|
| example CQ2.1 | "**Was** the example cited selected in a clearly biased and malicious manner…" | "**Is** the example cited selected…" | "…selected **on the basis of biased quality criteria that undermine the strength**…" |
| analogy CQ4 | "…A is false (true)?" | "…A is false (**or** true)?" | same as A |
| cause to effect CQ2.1 | "…clearly biased **and malicious** manner in order to **undermine** the validity…" | same as A | "…clearly biased manner in order to **support** the generalization" |
| correlation to cause CQ5 | "Can it be shown that … is **not** solely due to the way B is defined?" | same as A | "Is the increase or change in B **solely an artefact** of the way B is defined?" |
| ad hominem CQ1 | branches on `positive` / `negative` | the text asks negative or positive but the format **imposes `Yes` or `No`** | `answer_space: ["positive", "negative"]` |
| ad hominem CQ3 | "…that they **are committed**?" | "…that they **committed [something]**?" | "…that they **are committed to a group, cause or interest**?" |
| ad hominem, number of CQs | 6 | 6 | 6. In **B** there are **7**, in a different order |
| expert opinion CQ1 | "…true or false, **or** is S…" | "…true or false, **and/or** is S…" | same as A |
| expert opinion CQ4 | split into CQ4 plus CQ4.1 | same as A, with the variable written in lower case, `uses **a** to justify itself` | single node, earlier conditional form, `answer_space: ["yes", "no", "na"]` |

**In the executed prompt the placeholder `[something]` stayed in the text** sent to the model, so
CQ3 of ad hominem was asked in an unfinished form in all the experiments.

**Confirmation that C is the version of the data.** The number of questions per scheme in the executed
prompts coincides exactly with the number of answers recorded in the results on all eight
schemes.

⟶ *our operational observation*. D adopts the words of A, with the spelling corrected. Two
points remain in which D had departed not in spelling but in meaning, cause to effect CQ2.1 and
correlation to cause CQ5. Aligning them with A, the wording goes back to that of the thesis, and with
it comes back the misalignment between the text of the question and the terminal it leads to. They
are described in §12.1 and §12.2 and must be resolved by the author, by changing the text or the
arc, not by us rewriting the question on our own.

---

## 12. Our observations on the schemes

Nothing in this section is text of the thesis.

### 12.0 Ambiguity of CQ3 of ad hominem

With the spelling corrected, the question remains ambiguous. "An accusation that they are committed"
does not say committed to what, and the terminal is Guilt by Association. If the sense is commitment
to an inconsistent position, it is tu quoque; if it is membership of a discredited group, it is
guilt by association. The diagram chooses the second, the text does not say so. To be clarified with
the author, because it is not a typo.

### 12.1 Inverted polarities, corrected

Two nodes have their arcs inverted with respect to the text of their own question. Checked by reading
the diagrams on the rasterised pages of the PDF, not from the text extraction.

**correlation to cause, CQ5**, printed p. 64. "Can it be shown that
the increase or change in B is **not** solely due to the way B is defined?" In the thesis `yes`
leads to Definist Fallacy. But `yes` means that it can be shown that the change does **not**
depend on the definition, that is, that the argument holds. The Comment of the thesis confirms it:
the definist fallacy arises when the change is "un effetto della definizione stessa di B" [an effect
of the very definition of B].

**slippery slope, CQ5**, printed p. 90. "Are there weak links in the
sequence…?" In the thesis `no` leads to Slippery Slope. But the presence of weak links is the
fallacious condition. The Comment on CQ1 confirms it: "se anche un solo anello causale risulta
privo di adeguata giustificazione … l'argomentazione ricade nella fallacia dello slippery slope"
[if even a single causal link lacks adequate justification … the argument falls into the slippery
slope fallacy].

**Correction chosen.** Invert the two arcs, leaving the text of the questions exactly as it is
in the thesis. It is the minimal correction and requires no rewording. It also aligns the
diagram with the Comment that the author wrote, so it does not introduce an interpretation
of ours.

**Status on the definitive PDF.** Both inversions are still present in the definitive PDF of
14 September 2026, rechecked on the rendered pages. The two corrections stay.

**Note for the comparison with the results.** The 607 rows of the previous runs (606 distinct texts) were evaluated with the uncorrected
arcs. But the arcs are applied by the code to the answers already recorded, not by the
prompt, so recomputing with the corrected arcs does not require a single new call to a
model. Both versions can be obtained from the same data.

### 12.2 Questions with a double condition

**example CQ2.1 and cause to effect CQ2.1** contain two joined conditions, `clearly biased`
**and** `malicious`, and attribute to the selection the intent to *undermine* the generalization,
while in the flow the node serves to identify whoever selects evidence to *support* their own thesis
unduly.

**expert opinion CQ1** joins two distinct conditions, being in a position to know and being
recognised by the community of experts. With `or` the connective is defined, but the case "competent
source not recognised by peers" remains indistinguishable from "recognised source that is not
competent".

### 12.3 Enthymemes

CQ4 of expert opinion presupposes a conclusion `C` explicit in the text. The implicit
conclusion has no rule. A useful comparison: EthiX reconstructs the enthymeme from the topic of the
debate for the purpose of annotation, but keeps the original text in the dataset.

### 12.4 Ad hominem without a non-fallacious outcome

Ad hominem is the only scheme in which no path leads to Good Argumentation. On this scheme
the accuracy of the non-fallacious class is not defined, and any macro average that includes it
must say so.

### 12.5 Ad hominem prefixes to be mapped

The variants of ad hominem are prefixed in the diagrams, `Ad Hominem (Tu Quoque)`, but in the
vocabulary of the annotations they appear without a prefix, `Tu quoque`. An explicit mapping
between the two forms is needed before computing any metric.
