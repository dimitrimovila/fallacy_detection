# Classification diagrams, version 2.0

Revised critical questions and diagrams for the eight argumentation schemes, encoded in
`schemes/v2/` with `version: "2.0"`. The diagrams of version 1.1 are in `schemes/v1/` and are
described in `schemes_and_diagrams.md`; the runs made so far used them.

**Conventions**

* Every scheme is numbered in its new order. Sub-questions (CQn.m) are reached only through a specific answer to their parent question.
* "was" gives the number of the question in Enrico's thesis (published version, 23 September 2026), so that the answers of the full run v2 can be reused where the text is unchanged.
* Status of each question: **unchanged** (same text as version 1.1), **rewritten** (new text, same role), **new** (no counterpart in Enrico's set).
* The non-fallacious terminal is called Good Argumentation, as in the repository.

---

## 1. Argument from Expert Opinion

**CQ1** (was CQ1, rewritten)
Does S have competence, expertise or direct knowledge relevant to the domain D of the claim?
* no → **Irrelevant Authority**
* yes → CQ2

**CQ2** (was CQ2, rewritten)
Does the argument rest only on S's word, with no evidence or reasoning for A given by S or by the arguer?
* no → CQ3
* yes → CQ2.1

**CQ2.1** (new)
Does the text present S's word as conclusive for A?
* yes → **Appeal to Authority**
* no → CQ2.2

**CQ2.2** (new)
Does A clearly contradict the prevailing view of experts in D?
* yes → **Appeal to Authority**
* no → CQ3

**CQ3** (was CQ3, unchanged)
Did S really assert (or claim) A as true?
* no → CQ3.1
* yes → CQ4

**CQ3.1** (was CQ3.1, rewritten)
Has S expressed a position of their own on the issue under discussion, in this exchange or elsewhere?
* yes → **Strawman**
* no → **False Attribution**

**CQ4** (was CQ4, unchanged)
Is there a conclusion C that differs from A but uses A to justify itself?
* no → **Good Argumentation**
* yes → CQ4.1

**CQ4.1** (was CQ4.1, rewritten)
Assuming that A is true, does the conclusion C follow from it without an unjustified leap?
* yes → **Good Argumentation**
* no → **Non Sequitur**

---

## 2. Argument from Analogy

Structure unchanged.

**CQ1** (was CQ1, rewritten)
Are C1 and C2 so dissimilar that the comparison is inappropriate, with more differences than similarities between them?
* yes → CQ1.1
* no → CQ2

**CQ1.1** (was CQ1.1, rewritten)
Was the case used in the comparison chosen as a limit case, deliberately excessive and disproportionate to the other case, in order to influence the audience?
* yes → **Appeal to Extremes**
* no → **Weak Analogy**

**CQ2** (was CQ2, unchanged)
Is the similarity cited relevant to A?
* no → CQ1.1
* yes → CQ3

**CQ3** (was CQ3, unchanged)
Is A true (false) in C1?
* no → **False Premise**
* yes → CQ4

**CQ4** (was CQ4, rewritten)
Is there a well-known case C3, also similar to C1, but in which A is false (true)?
* yes → **Cherry Picking**
* no → **Good Argumentation**

---

## 3. Argument from Popular Opinion

New order: first whether popularity counts as evidence for A, then how the group was measured.

**CQ1** (was CQ3, rewritten)
Is the reference group in a position to know whether A is true (for example through direct experience), or does A concern a convention or shared preference for which the group's agreement is a reasonable basis for accepting A?
* yes → CQ2
* no → CQ1.1

**CQ1.1** (was CQ3.2, unchanged)
Does the majority of the reference group rely on the concept of tradition to assert A?
* yes → **Appeal to Tradition**
* no → CQ1.2

**CQ1.2** (was CQ3.3, unchanged)
Does the majority of the reference group rely on the concept of nature to assert A?
* yes → **Appeal to Nature**
* no → CQ1.3

**CQ1.3** (was CQ3.1, unchanged)
Is the presumption in favour of A based solely on the fact that it is accepted by a large majority of the reference group?
* yes → **Ad Populum**
* no → **Ad Populum**

**CQ2** (was CQ1, rewritten)
Does the argument rely on a subgroup likely to support A, while drawing a conclusion that applies to a broader population?
* yes → **Cherry Picking**
* no → CQ3

**CQ3** (was CQ2, rewritten)
Does the text refer to a group that is clearly too small to support A?
* yes → **Hasty Generalization**
* no → **Good Argumentation**

---

## 4. Argument from Example

Structure unchanged. The terminal of CQ1 changes from False Attribution to False Premise.

**CQ1** (was CQ1, rewritten)
Is the case reported as the example clearly false or unrealistic?
* yes → **False Premise**
* no → CQ2

**CQ2** (was CQ2, unchanged)
Is the example typical of the kinds of cases that the generalization ranges over?
* yes → CQ3
* no → CQ2.1

**CQ2.1** (was CQ2.1, rewritten)
Was the example selected in a biased way that makes the generalization appear more strongly supported than it actually is?
* yes → **Cherry Picking**
* no → CQ2.2

**CQ2.2** (was CQ2.2, rewritten)
Are the cases cited too few to support a conclusion as general as the one drawn?
* yes → **Hasty Generalization**
* no → CQ2.3

**CQ2.3** (was CQ2.3, rewritten)
Was the example chosen as a limit case, deliberately extreme and disproportionate to the typical cases covered by the generalization, in order to influence the audience?
* yes → **Appeal to Extremes**
* no → CQ3

**CQ3** (was CQ3, unchanged)
Does the example cited have characteristics that make it too different from the generalization drawn in the conclusion?
* yes → **Weak Analogy**
* no → **Good Argumentation**

---

## 5. Ad Hominem

Structure unchanged.

**CQ1** (was CQ1, unchanged)
Is the opinion expressed about the other person negative (and intended to undermine their credibility) or positive (and intended to enhance their credibility)?
* positive → **Appeal to Authority**
* negative → CQ2

**CQ2** (was CQ2, rewritten)
Does the attack accuse the other person of being influenced by personal interests that compromise the objectivity of their judgment?
* yes → **Ad Hominem (Circumstantial)**
* no → CQ3

**CQ3** (was CQ3, rewritten)
Does the attack discredit the other person through a commitment or tie of theirs to a person, group, movement or cause (for example political, religious or ideological) that is viewed negatively?
* yes → **Ad Hominem (Guilt by Association)**
* no → CQ4

**CQ4** (was CQ4, unchanged)
Is the direct attack on the other person an accusation of hypocrisy?
* yes → **Ad Hominem (Tu Quoque)**
* no → CQ5

**CQ5** (was CQ5, unchanged)
Is the direct attack on the other person intended to undermine their self-confidence?
* yes → **Ad Fidentia**
* no → CQ6

**CQ6** (was CQ6, rewritten)
Is the attack made before the other person has presented their argument, to make the audience distrust in advance anything they say?
* yes → **Poisoning the Well**
* no → **Ad Hominem (Abusive)**

---

## 6. Slippery Slope

Enrico's CQ1, CQ4, CQ4.1 and CQ5 are merged into the new CQ1. Enrico's CQ3 (Definist Fallacy) is removed and a new question is added.

**CQ1** (was CQ1, CQ4, CQ4.1 and CQ5, rewritten)
Is there at least one link in the chain that is implausible, potentially depending on unstated intermediate steps that are doubtful?
* yes → **Slippery Slope**
* no → CQ2

**CQ2** (was CQ2, unchanged)
Is the final outcome Cn as bad as the text suggests?
* no → **Slippery Slope**
* yes → CQ3

**CQ3** (new)
Could the chain of events be stopped or reversed at some point before reaching Cn?
* yes → **Slippery Slope**
* no → **Good Argumentation**

---

## 7. Argument from Cause to Effect

New order: specific diagnoses first, the two premises of the scheme last. Enrico's CQ2 is removed.

**CQ1** (was CQ3, unchanged)
Does the argument claim that A causes B solely because B occurred after A?
* yes → **Post Hoc**
* no → CQ2

**CQ2** (was CQ2.1, rewritten)
Does the text present a biased selection of evidence or cases that makes the causal generalization appear more strongly supported than it actually is?
* yes → **Cherry Picking**
* no → CQ3

**CQ3** (was CQ2.2, rewritten)
Does the text report observed cases for the causal generalization that are too few to support it?
* yes → **Hasty Generalization**
* no → CQ4

**CQ4** (was CQ4, rewritten)
Does the text present A as the only cause of B, while B clearly also depends on other important causes?
* yes → **Causal Reductionism**
* no → CQ5

**CQ5** (was CQ1, rewritten)
Is it plausible that A leads to B, whatever one thinks of A or B?
* no → **False Premise**
* yes → CQ5.1

**CQ5.1** (new)
Is it true that A occurs, or can occur, in the case at hand?
* no → **False Premise**
* yes → **Good Argumentation**

---

## 8. Argument from Correlation to Cause

New order. Enrico's CQ2 and CQ2.2 are removed.

**CQ1** (was CQ1, unchanged)
Is there actually a positive or negative correlation between A and B?
* no → **False Premise**
* yes → CQ2

**CQ2** (was CQ3, rewritten, polarity inverted)
Does the text report observed cases for the correlation that are too few to support it?
* yes → **Hasty Generalization**
* no → CQ3

**CQ3** (was CQ2.1, rewritten)
Does the claim that one of the two events causes the other rest only on the fact that it occurs before the other?
* yes → **Post Hoc**
* no → CQ4

**CQ4** (new)
Does the causal relationship actually go from B to A, and not just from A to B?
* yes → **Questionable Cause**
* no → CQ5

**CQ5** (was CQ4, rewritten)
Is there a third, well-known factor that causes both A and B?
* yes → **Questionable Cause**
* no → CQ6

**CQ6** (new)
Does the text present A as the only cause of B, while B clearly also depends on other important causes?
* yes → **Causal Reductionism**
* no → **Good Argumentation**

---

## Stage 2 prompt

Version 2.0 is asked with the prompt `stage2_v3`: the template is the text of `stage2_v2`, and
`stage2_v3_answers.yaml` gives definitions of their own only to the CQs that cannot be answered as
intended without one: ad hominem CQ1 (options positive and negative), expert opinion CQ2.1, analogy
CQ1.1, example CQ2.3, slippery slope CQ3, cause to effect CQ3 and CQ4, correlation to cause CQ2 and
CQ6. Every other CQ uses the default definitions. The definitions give no examples.

## Open points

* **Popular opinion CQ1.3.** Both answers lead to Ad Populum: the node does not change the verdict and
  is kept for the explanation of the path.
