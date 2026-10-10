# Argumentation schemes, critical questions and diagrams

**Source.** E. Bergamasco, *Schemi argomentativi e domande critiche nei Large Language Models come
strumento di individuazione delle fallacie* [Argumentation schemes and critical questions in Large
Language Models as a tool for fallacy detection], master's thesis, University of Padova, Department
of Linguistic and Literary Studies, academic year 2025-2026. Supervisor prof. Massimiliano Carrara,
co-supervisor prof. Giovanni Da San Martino. Published version confirmed by the author, 23 September
2026, 182 pages, **chapter 3**, §3.1 "Diagrammi degli schemi" [Diagrams of the schemes], printed
pp. 50-96. The page references below are those of this version.

On 23 September 2026 the scheme cards and the files of `schemes/` (version 1.1) were compared block
by block with it: formal schema, identification question, critical questions, terminals, and all
eight diagrams on the rendered pages. They coincide with the thesis, except for the corrections of
ours listed in the Conventions.

**What this document contains.** Only the schemes: argumentation scheme, identification question,
critical questions, diagram, terminals, and the structural properties that follow from them. Dataset
and annotation method are in `data.md`. Prompts, runs and results are in `diagnosi_pipeline_cq.md`,
outside the repository.

---

## Conventions

**The critical questions report the words of chapter 3**, with no reformulation. The corrections of
ours are the only differences from the published version and from the files of `schemes/`. Each is
made for an evident error of the thesis, with the reason given here:

1. **Expert opinion CQ3**, `claim` in the infinitive: "Did S really assert (or claim) A as true?".
   The thesis writes `(or claimed)`.
2. **Cause to effect CQ2.1**, `evidence` instead of `example`: "Was the evidence cited chosen in such
   a clearly biased and malicious manner as to undermine the validity of the generalization?". This
   scheme has no example: CQ2 and CQ2.2 speak of the evidence cited, and the Comment of the thesis
   says that question 2.1 asks whether the evidence was selected.
3. **Name of the non-fallacious terminal**, `Good Argumentation`. The published version calls the
   green terminal `Non-fallacious Argument`, and earlier versions `Good Argument`. It is the same
   terminal, and the name changes neither the path nor the meaning.
4. **Correlation to cause CQ5**, arcs inverted. The thesis sends `yes` to Definist Fallacy, against
   its own question and Comment. See §4.
5. **Ad hominem CQ3**, the missing complement: "Does the direct attack on the other person consist of
   an accusation that they are committed to a group, movement or cause (for example political,
   religious or ideological) that is viewed negatively?". The thesis stops at "an accusation that
   they are committed?", which does not say committed to what. A commitment to an inconsistent
   position would be tu quoque (CQ4), membership of a discredited group is guilt by association, and
   the diagram chooses the second. The Comment (printed p. 75) states that sense: the credibility is
   undermined by the other person's membership of a group, movement or category perceived
   negatively. The Comment's clause that the commitment motivates the person's position is left out
   on purpose: that is bias, which is CQ2 (Circumstantial), and including it would make the two
   nodes overlap. The partial overlap with CQ4 is not a defect of the text and stays.
6. **Slippery slope CQ3**, the object of the definition: "Is it possible to provide a precise
   definition of the initial action or of the concepts involved that removes the ambiguity to such
   an extent as to halt the decline?". The thesis writes "a precise definition that removes the
   ambiguity", which does not say a definition of what. The Comment (printed p. 95) states it: the
   initial act or the concepts involved.

**The text of the thesis and our observations are kept separate.** Everything that is not
transcription is marked `⟶ our observation` or collected in §12. Nothing in the cards must be
attributed to Enrico unless it is transcribed.

**The arcs of the diagrams** were read from the rasterised pages of the PDF, because the
flowcharts are vector graphics and text extraction returns the labels `yes` and `no` in an order
that cannot be reconstructed.

**Notation.** `CQn: answer → destination`. `⟲` flags a reconvergence arc, that is, two or more
paths that enter the same node or terminal.

**The reference version is chapter 3.** The prompts actually run in the earlier experiments differ
from it in a few points (§11).

**The answer space of the prompt.** The thesis (§4.3.2, printed pp. 113-114) describes the prompts
only in words: every critical question receives a binary answer, `Yes` or `No`, with a mandatory
justification anchored in the text. The third answer of our prompt, `cannot_be_determined`, is a
project choice to be justified as such, not an inheritance from the thesis.

---

## 1. Argument from example (§3.1.1, pp. 51-55)

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
  - **C.Q. 2.1** — Was the example cited chosen in such a clearly biased and malicious manner as to undermine the validity of the generalization?
  - **C.Q. 2.2** — Is the example given so specific that it is statistically unrepresentative and therefore cannot be used to support the generalization?
  - **C.Q. 2.3** — Is the example given so extreme that it invalidates the generalization?
- **C.Q. 3** — Does the example cited have characteristics that make it too different from the generalization drawn in the conclusion?

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

## 2. Argument from analogy (§3.1.2, pp. 56-60)

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

## 3. Argument from cause to effect (§3.1.3, pp. 61-65)

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
  - **C.Q. 2.1** — Was the evidence cited chosen in such a clearly biased and malicious manner as to undermine the validity of the generalization? ⚑
  - **C.Q. 2.2** — Is the evidence cited insufficient in quantitative terms to support the generalization?
- **C.Q. 3** — Is the relationship between cause and effect based solely on temporal sequence?
- **C.Q. 4** — Are there other causal factors that clearly represent the real reason behind the occurrence of the effect?

⚑ **Correction 2 of the Conventions.** The thesis writes "Was the **example** cited chosen…". We
write `evidence`.

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

## 4. Argument from correlation to cause (§3.1.4, pp. 66-70)

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

⚑ **Correction 4 of the Conventions.** In the diagram of the thesis (printed p. 68, checked on the
rendered page) the arcs of CQ5 are the opposite: `yes` towards Definist Fallacy and `no` towards
Good Argumentation. But `yes` means that the change in B does not depend on how B is defined, that
is, that the argument holds. The Comment of the thesis confirms it: the definist fallacy arises when
the change in B is an effect of the very definition of B. We inverted the two arcs and left the text
of the question intact. It is the minimal correction, it requires no rewording, and it aligns the
diagram with the Comment, so it adds no interpretation of ours.

The 607 rows of the earlier runs (606 distinct texts) were evaluated with the uncorrected arcs. The
arcs are applied by the code to the answers already recorded, not by the prompt, so recomputing with
the corrected arcs does not require a single new call to a model.

**Terminals** — False Premise · Post Hoc · Questionable Cause · Hasty Generalization ·
Causal Reductionism · Definist Fallacy · Good Argumentation

---

## 5. Argument ad hominem (§3.1.5, pp. 71-76)

**Schema**
```
a is a person of bad character.
Therefore, a's argument A should not be accepted.
```

**Identification question**
> Does the text shift the focus from arguments to personal judgments about the other person?

**Critical questions**
- **C.Q. 1** — Is the opinion expressed about the other person negative (and intended to undermine their credibility) or positive (and intended to enhance their credibility)?
- **C.Q. 2** — Does the direct attack on the other person amount to an accusation that they are biased?
- **C.Q. 3** — Does the direct attack on the other person consist of an accusation that they are committed to a group, movement or cause (for example political, religious or ideological) that is viewed negatively? ⚑
- **C.Q. 4** — Is the direct attack on the other person an accusation of hypocrisy?
- **C.Q. 5** — Is the direct attack on the other person intended to undermine their self-confidence?
- **C.Q. 6** — Is a direct attack on the other person used as a pretext to discredit them even before the discussion has begun?

⚑ **Correction 5 of the Conventions.** The thesis stops at "…an accusation that they are committed?".

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
terminal**. Every path ends in a fallacy, so on this scheme the accuracy of the non-fallacious class
is not defined, and any macro average that includes it must say so.

⟶ *our observation*. CQ1 is the only question of the whole set that does not branch on `yes` and `no`
but on `positive` and `negative`. This must be kept in mind in the encoding, because the answer
space is not uniform across the nodes.

---

## 6. Argument from expert opinion (§3.1.6, pp. 77-83)

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
- **C.Q. 1** — Is S in a position to know whether A is true or false, or is S a genuine expert recognized by the community of experts in D?
- **C.Q. 2** — Is the fact that S stated (or claimed) A used as the sole basis for supporting the truth of the conclusion, without providing any other form of argument?
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
the diagram gives no rule for it. EthiX reconstructs the enthymeme from the topic of the debate for
the purpose of annotation, but keeps the original text in the dataset.

---

## 7. Argument from popular opinion (§3.1.7, pp. 84-89)

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

## 8. Slippery slope argument (§3.1.8, pp. 90-96)

**Schema**
```
Case C0 is tentatively acceptable as an initial presumption.
There exists a series of cases, C0, C1,..., Cn−1, where each case leads to the next by a
combination of causal, precedent, and/or analogy steps.
There is a climate of social opinion such that once people come to accept each step as
plausible (or as accepted practice), they will then be led to accept the next step.
The penultimate step Cn−1 leads to a horrible outcome, Cn, which is not acceptable.
Therefore, C0 is not acceptable (contrary to the presumption of the initial premise).
```

**Identification question**
> Does the text put forward the argument that, if a particular action is permitted or carried out (even if it is minor or seemingly harmless), it will inevitably lead (through a chain of intermediate consequences) to an extreme or catastrophic final outcome?

**Critical questions**
- **C.Q. 1** — Do any of the causal links in the chain lack sufficient evidence to support the claim that they will (might, must) occur?
- **C.Q. 2** — Is the outcome Cn as bad as suggested?
- **C.Q. 3** — Is it possible to provide a precise definition of the initial action or of the concepts involved that removes the ambiguity to such an extent as to halt the decline? ⚑
- **C.Q. 4** — Are there other steps required to fill in the sequence of events and make it plausible?
  - **C.Q. 4.1** — Would these steps lead to a different conclusion?
- **C.Q. 5** — Are there weak links in the sequence, where specific critical questions should be asked on whether one event will really lead to another?

⚑ **Correction 6 of the Conventions.** The thesis writes "Is it possible to provide a precise
definition that removes the ambiguity…".

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
CQ5   yes → Slippery Slope         ⟲
      no  → Good Argumentation
```

⟶ *version note*. The presence of weak links in the chain is the fallacious condition, so `yes` on
CQ5 leads to Slippery Slope. The thesis had the opposite arcs up to the version of 14 September 2026.
The author corrected them in the published version (printed p. 94, checked on the rendered page), so
the thesis and our file coincide and this is not a correction of ours.

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

**The diagrams are not trees.** Six schemes out of eight contain reconvergences, marked `⟲` in the
diagrams. Only **ad hominem** is a pure tree.

**Consequence.** Reconstructing the answer vector backwards from the pair (exit CQ, verdict) is not
unique. Example on *example*: CQ3 is reached both from `CQ2=yes` and from
`CQ2=no, CQ2.1=no, CQ2.2=no, CQ2.3=no`. The full answer vector must be **recorded**, not
reconstructed.

**Early exit.** The final CQs of every scheme are reached only by a fraction of the
items, and not at random. If the per-CQ gold is recorded only along the path, the
supervision on the late CQs is systematically scarce.

---

## 11. Divergences between the cards and the prompts actually run

The prompts run in the earlier experiments are `Lavoro di Enrico/src/prompts/template_stage2_*.txt`
(source C). The number of questions per scheme in them coincides exactly with the number of answers
recorded in the results on all eight schemes, so C is the version of the data. The cards (source A,
chapter 3) are canonical. The table lists only the substantial differences of C from A; purely
orthographic ones are not reported.

| Scheme, CQ | A, thesis ch. 3 | C, run |
|---|---|---|
| example CQ2.1 | "**Was** the example cited chosen in such a clearly biased and malicious manner…" | "**Is** the example cited selected…" |
| analogy CQ4 | "…A is false (true)?" | "…A is false (**or** true)?" |
| ad hominem CQ1 | branches on `positive` / `negative` | the text asks negative or positive, but the format **imposes `Yes` or `No`** |
| ad hominem CQ3 | "…that they **are committed**?" | "…that they **committed [something]**?" |
| expert opinion CQ1 | "…true or false, **or** is S…" | "…true or false, **and/or** is S…" |
| expert opinion CQ4 | split into CQ4 plus CQ4.1 | same as A, with the variable written in lower case: `uses **a** to justify itself` |

**In the executed prompt the placeholder `[something]` stayed in the text** sent to the model, so
CQ3 of ad hominem was asked in an unfinished form in all the experiments.

---

## 12. Our observations on the schemes

Nothing in this section is text of the thesis.

**Questions with a double condition.** Example CQ2.1 and cause to effect CQ2.1 join two conditions,
`clearly biased` **and** `malicious`. The published version says "chosen in such a clearly biased and
malicious manner **as to** undermine the validity of the generalization": undermining the validity is
the effect of the biased selection, not its purpose. Read this way, `yes` leading to Cherry Picking
is the right answer for the fallacy, and the text stays.

Expert opinion CQ1 joins two distinct conditions, being in a position to know and being recognized by
the community of experts. With `or` the connective is defined, but the case "competent source not
recognized by peers" remains indistinguishable from "recognized source that is not competent".

**Ad hominem prefixes to be mapped.** The variants of ad hominem are prefixed in the diagrams,
`Ad Hominem (Tu Quoque)`, but in the vocabulary of the annotations they appear without a prefix,
`Tu quoque`. An explicit mapping between the two forms is needed before computing any metric.
