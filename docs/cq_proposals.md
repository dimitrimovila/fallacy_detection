# Proposals to revise the critical questions

The numbers in this document come from the diagnosis of Eleni's runs and were
rechecked on 12 September 2026 against the frozen copy and against Eleni's `results.csv` files.
**Basis of the numbers.** Unless stated otherwise, every percentage and every count per scheme is
on the subset predicted by gpt-5: the items to which the model assigns that scheme, because that is
where its pipeline asks the CQs. The exceptions are marked next to the number, with "gold basis"
when the subset is that of the gold and with the name of the model when it is not gpt-5. The counts
of labels of the test set, for example how many items have a given gold, are by definition on a
gold basis and say so in the text.

Every number that goes into the thesis must in any case be recomputed by the scorer, which does not
exist today. Section 8 lists the errors found on 12 September 2026, with the cause of each.

**Status. None of these proposals is applied.** Neither in the file `schemes_and_diagrams.md`, nor in
the YAML files, nor in the prompts. This document is meant to be consulted **after** completing the
tests on the questions as they are written in the thesis.

**When to open it.** When the v0 baseline is closed and we move on to the reformulated version v1.

---

## 0. What is in here and what is not

The corrections fall into two categories that must not be confused.

**Corrections already applied, which are not in this file.** The two inverted polarities of
correlation to cause CQ5 and slippery slope CQ5. They are bugs of the graph, not choices of wording.
They were corrected by inverting the two arcs and leaving the text of the questions intact. They are in
`schemes_and_diagrams.md` §12.1 and in the YAML files, where they were verified on
12 September 2026: in `correlation_to_cause.yaml` CQ5 exits `yes` towards good_argumentation and `no`
towards definist_fallacy, with a note that explains the correction; in `slippery_slope.yaml` CQ5 exits
`yes` towards slippery_slope and `no` towards good_argumentation, as the Comment of the thesis says. In the
published version of the thesis (23 September 2026) the author corrected the arcs of slippery slope CQ5, which
now coincide with the YAML; the correction of correlation to cause CQ5 stays. They do not require an experiment,
because showing that a wrong arc produces wrong answers is not a result.

**Reformulation proposals, which are the ones in this file.** Seven proposals, eight questions, six schemes. Here the current
text is defensible and the rewriting is a design choice, so the difference between before and
after is measurable and counts as a result.

---

## 1. Protocol

**v0** — text of the questions as in the thesis, spelling corrected, CQ5 arcs corrected in the two
schemes. It is the current state of the YAML files and of the prompts. The baseline is our phase 2
runs on the 601 items of the test set of the experiments: the 606 distinct texts of Eleni's 607 rows,
minus five near duplicates. Eleni's runs are not v0: Eleni's prompt
admitted only `Yes` and `No`, six questions out of 48 had a text different from that of the thesis, and
Eleni's code departed from the diagram in two places (expert opinion CQ4 arcs inverted,
analogy reconvergence missing). Those results remain a rough comparison, recomputed
on the same 601 items (spec 08, requirement 2).

**v1** — v0 plus the reformulations of this document.

**What v1 costs.** Only one stage 2 pass per model, and only for the six schemes
touched. Stage 1 does not change, because no proposal modifies the identification question
of the scheme. The untouched schemes reuse the answers of v0.

**What to keep fixed.** Same test set, same models, same stage 1, same arcs, same
temperature and same answer format. The only variable is the text of the questions, otherwise
the difference cannot be attributed.

**What NOT to do.** Rewrite everything. If all the questions change, the comparison does not say which
change produced the gain. The seven below each have a written diagnosis with a
number beside it; everything else must be left identical.

---

## 2. The seven proposals

### 2.1 ad hominem, CQ1 — inconsistent answer space

**Current text (thesis)**
> Is the opinion expressed about the other person negative (and intended to undermine their
> credibility) or positive (and intended to enhance their credibility)?

**Diagnosis.** It is the only node of the whole set that does not branch on `yes` and `no` but on `positive` and
`negative`. In the executed prompt the format imposed `Yes` or `No` anyway, so the model
answered in binary terms a question with two nominal poles.

**Evidence.** `Yes` 102 times, `No` once on the 103 items on which gpt-5 predicted ad hominem. The positive branch, the only one that
leads to Appeal to Authority, never fires in the whole dataset.

**Proposal**
> Is the judgement expressed about the other person negative, that is, intended to reduce the
> credibility of what they say?

**Effect on the graph.** `yes` towards CQ2, `no` towards Appeal to Authority. Identical structure,
answer space aligned with all the other nodes. In the YAML,
`answer_space: ["positive", "negative"]` disappears.

---

### 2.2 ad hominem, CQ3 — the question does not say what it asks

**Current text (thesis)**
> Does the direct attack on the other person consist of an accusation that they are committed?

**Diagnosis.** The complement is missing, so one does not know committed to what, and the node does not
discriminate: it fires on any direct attack and sends to the Guilt by Association terminal cases
that have nothing to do with guilt by association. The overlap with CQ4 on hypocrisy,
that is, being committed to a position inconsistent with one's own conduct, remains a cause, but a
partial one: it explains less than a third of the cases.

**Evidence.** The node receives `yes` on 42 of the 103 items that gpt-5 assigns to the scheme, and 29 of
these exit on Guilt by Association; **none** of the 29 has Guilt by association in the gold. They are
9 tu quoque, which is the overlap with CQ4, plus 10 poisoning the well and 10 abusive, which have
nothing to do with CQ4. The terminal is produced 29 times, while in the test set the gold Guilt by
association labels are 8 in all (gold basis). Of the 18 gold `tu quoque` (gold basis), 9 end up
labelled Guilt by Association. In the executed prompt the question also contained the placeholder
`[something]`, left in the text.

**Proposal**, if the intended terminal stays Guilt by Association
> Does the attack discredit the person by associating them with a group, cause or individual
> regarded as disreputable, rather than by addressing their own conduct or claims?

The subordinate clause with `rather than` is the part that separates it from CQ2 and from CQ4.

**Effect on the graph.** None.

**Shorter variant**, which was already written in the YAML files and was then removed when aligning with the thesis
> Does the direct attack on the other person consist of an accusation that they are committed to
> a group, cause or interest?

It intervenes less on the original text, adds only the missing complement and chooses the sense
of guilt by association. It does not, however, separate the case from CQ4 explicitly.

**To be clarified with the author before applying.** Which of the two senses was intended. If the
intended sense was commitment to a position, then it is the terminal that is wrong, not the question.

---

### 2.3 cause to effect, CQ2 — it never fires

**Current text (thesis)**
> Is the evidence cited (if there is any) strong enough to warrant the causal generalization?

**Diagnosis.** The parenthetical `(if there is any)` collapses the absence of evidence into `no`.
In a short informal argument there is hardly any cited evidence, so the answer is `no` by
default and the whole flow ends up in the subtree of CQ2.1 and CQ2.2, which however presupposes that
some evidence exists.

**Evidence.** Rates of `yes` on the node, each on the subset that that model assigns to
cause to effect: 87 items for gpt-5, 70 for qwen.

| node | gpt-5 | qwen |
|---|---|---|
| CQ2 | 0.00 | 0.10 |
| CQ2.1 | 0.00 | 0.03 |
| CQ2.2 | 1.00 | 0.90 |
| CQ3 | 0.07 | 0.09 |

The scheme produces only two terminals out of six, False Premise and Hasty Generalization, and has an accuracy of
0.023 on the 87 items that gpt-5 assigns to it.

**Proposal A, minimal, a single node**
> Is the causal generalization supported by evidence stated in the text, rather than merely
> asserted?

It solves the default but leaves the subtree undefined when there is no evidence.

**Proposal B, correct, adds a guard node**
> CQ2 — Does the text cite any evidence for the causal generalization beyond the assertion
> itself?

`no` exits directly towards CQ3, `yes` enters the question on the strength of the evidence, which becomes
CQ2.0 and keeps the current subtree.

**Effect on the graph.** None with A. With B a node is added and the depth of the
scheme changes, so the regression snapshot must be updated.

---

### 2.4 cause to effect CQ2.1 and example CQ2.1 — two conditions and a reversed direction

**Current text (YAML), identical in the two schemes except for `evidence` versus `example`**
> Was the evidence cited chosen in such a clearly biased and malicious manner as to undermine
> the validity of the generalization?

The diagnosis below was written on the earlier wording, "selected in a clearly biased and malicious
manner **in order to** undermine", which Eleni's runs used. In the published version of the thesis
"as to" makes the undermining the effect of the biased selection rather than its purpose, which
weakens the point on the reversed direction; the two conditions remain.

**Diagnosis.** Two overlapping defects. The conditions are two, `biased` and `malicious`, and the
second is a judgement on intention that cannot be read from the text. And `undermine` is reversed,
because cherry picking selects evidence to **support** one's own thesis, not to undermine it. The
definition of cherry picking that the thesis itself gives says exactly this.

**Evidence.** On cause to effect the node receives `yes` on 0.0 per cent of the items with gpt-5
(87 items) and on 2.9 with qwen (70), so Cherry Picking is unreachable from that scheme. On
example, with the very same sentence, it fires at 10.1 per cent on both models (69 items
each). The text alone does not
explain the difference, but in the case of cause to effect it adds to the problem of CQ2 above.

**Proposal**
> Does the text present only the cases that support the conclusion, omitting comparable cases
> that would tell against it?

A single condition, verifiable on the text, consistent with the terminal.

**Effect on the graph.** None.

**Variants already written in the YAML files and then removed when aligning with the thesis.** They are more conservative than the
proposal above and worth keeping on the table.

For cause to effect
> Was the evidence cited selected in a clearly biased manner in order to support the
> generalization?

It removes only `and malicious` and turns `undermine` into `support`. It is the minimal correction: it
changes two words and leaves all the rest of Enrico's sentence.

For the example scheme, the wording that the thesis had before this revision
> Is the example cited selected on the basis of biased quality criteria that undermine the
> strength of the generalisation?

**Note.** They must be changed together, because they are the same question on two schemes. Changing
only one would introduce a difference between schemes that could not then be attributed to anything.

---

### 2.5 expert opinion, CQ1 — two conditions in a disjunction

**Current text (thesis)**
> Is S in a position to know whether A is true or false, or is S a genuine expert recognized by
> the community of experts in D?

**Diagnosis.** Two distinct conditions joined by `or`, so `no` requires both of them to fail. The
case "competent source not recognised by peers" remains indistinguishable from
"recognised source that is not competent". The terminal is Irrelevant Authority, which concerns a
more precise thing, that is, authority invoked outside its own domain.

**Evidence.** On the 72 items that gpt-5 assigns to the scheme the gold `Appeal to authority` labels are 24, and
14 of these end up on Irrelevant Authority. The Irrelevant Authority terminal is produced
32 times within the scheme, while in the test set the gold `Irrelevant authority` labels are 15 in all
(gold basis): the node emits it about twice as often as it should. The scheme has an accuracy of
0.139 on those 72 items.

**What the agreement between models does not say.** On the 64 items that gpt-5 and qwen both assign
to the scheme, the kappa between the two is 0.69 on CQ1, 0.38 on CQ4.1, 0.32 on CQ4, 0.18 on CQ3, 0.13
on CQ2 and 0.00 on CQ3.1, which is at zero because gpt-5 never answers `yes`. CQ1 is therefore the
question on which the two models agree most, not an isolated question inside a scheme that
nobody can answer: the agreement leg, on which an earlier version of this document
rested the proposal, falls. What remains is the semantic argument of the diagnosis, that is, two
conditions joined by `or` against a terminal that concerns only one of them, the confusion between
Appeal to Authority and Irrelevant Authority, the overproduction of the terminal and the accuracy of
the scheme.

**Proposal**
> Is the source's expertise in the same domain as the claim being supported?

It is exactly what irrelevant authority means, it is a single condition, and it can be answered from
the text.

**Effect on the graph.** None.

**To be considered.** With this wording the condition "recognised by the community of
experts" disappears from the diagram. If it is needed, it goes in as a node of its own, not added back inside CQ1.

---

### 2.6 analogy, CQ4 — it asks about something that is not in the text

**Current text (thesis)**
> Is there some other case C3 that is also similar to C1, but in which A is false (true)?

**Diagnosis.** It asks about a case that does not appear in the text. Two models with different
knowledge of the world answer differently by construction, so the question is not stable.

**Evidence.** It is the question with the worst agreement of the whole set. On the 85 items that gpt-5 and
qwen both assign to analogy, the raw agreement is **0.19** and the kappa 0.02. The kappa here must not
be read as a measure of agreement: the two answer distributions are very unbalanced and in
opposite directions, so the agreement expected by chance drops to 0.17 and the kappa stays squashed
near zero whatever the answers.

The evidence that counts is the two percentages of `yes` on the same basis: gpt-5 answers `yes` in
5 per cent of the cases (4 items out of 85), qwen in 86 (73 out of 85). On the same question, on the
same text, two models answer in almost opposite ways.

**Proposal**
> Does the text rely on a single favourable comparison while ignoring comparable cases that would
> point the other way?

It remains difficult, but it concerns what the text does and not what the world contains, and it is
aligned with the Cherry Picking terminal.

**Effect on the graph.** None.

---

### 2.7 slippery slope, CQ1 — it always fires

**Current text (thesis)**
> Do any of the causal links in the chain lack sufficient evidence to support the claim that they
> will (might, must) occur?

**Diagnosis.** In an informal text there is always at least one link without explicit evidence, so the
answer is `yes` by construction. The node decides the whole scheme at the first step, and the answers
to the other five questions are computed and never used.

**Evidence.** `yes` on 80 items out of 80, that is, on all those that gpt-5 assigns to the scheme. CQ4
too is at 1.00, CQ4.1 at 0.95 and CQ5 at 1.00, so four nodes out of six distinguish almost
nothing. The accuracy of 1.000 of this scheme is an artefact, and it is worth 13 per cent of the 607
rows of the test set.

**Proposal**
> Is at least one step in the chain presented as inevitable while the text gives no reason why it
> must follow?

It moves the criterion from the presence of evidence to the claim of inevitability, which is what
distinguishes the slippery slope from an ordinary conditional prediction, and it does not fire by default.

**Effect on the graph.** None.

**Caution.** If this question stops firing always, the accuracy of the scheme **will drop**
with respect to the baseline. It is not a deterioration, it is the end of an artefact. It must be said
explicitly when the numbers are reported, otherwise the comparison of v0 against v1 looks like a failure.

---

## 3. Not to be touched

**popular opinion CQ3.2 and CQ3.3**, on tradition and nature. They almost never fire: on CQ3.2 `yes`
at 1 per cent with gpt-5 (87 items) and at 6 with qwen (95), on CQ3.3 at 2 and 3 per cent. But in the
gold Appeal to Tradition counts 2 items and Appeal to Nature 1 (gold basis). The rarity is in the data, not in the question. On this dataset those two
classes cannot be evaluated, and it must be written that way instead of rewriting the questions.

**example CQ1**, with the polarity inverted with respect to Walton. It is a declared choice and the
Comment of the thesis is consistent with it, so it is not an error.

**slippery slope CQ3.** At first sight it seems reversed, but the Comment agrees with the arc.

---

## 4. Summary

| # | Scheme, CQ | Defect | Graph changes | Author needed |
|---|---|---|---|---|
| 2.1 | ad hominem CQ1 | non-binary answer space | no | no |
| 2.2 | ad hominem CQ3 | incomplete predicate, the node does not discriminate | no | **yes** |
| 2.3 | cause to effect CQ2 | never fires, the subtree presupposes evidence | only with proposal B | recommended |
| 2.4 | cause to effect CQ2.1 and example CQ2.1 | double condition, reversed direction | no | no |
| 2.5 | expert opinion CQ1 | two conditions in a disjunction | no | recommended |
| 2.6 | analogy CQ4 | requires knowledge outside the text | no | no |
| 2.7 | slippery slope CQ1 | always fires | no | no |

Seven proposals, eight questions, six schemes. Six proposals out of seven do not touch the graph, so
they are applied by changing one line of text in the YAML and regenerating the prompt.

The untouched schemes are **correlation to cause** and **popular opinion**. On those two the
answers of v0 are reused as they are.

---

## 5. What to expect from the comparison

The useful result is not "v1 does better than v0". It is the breakdown by scheme of where the gain
is and where it is not, read together with three quantities measured on v0 and on v1. The diagnosis
has already measured them on Eleni's runs, and the numbers below come from there.

**The agreement between models per single CQ.** If a reformulated question goes from a kappa near zero
to a high kappa, the reformulation has made the question answerable, regardless of how the
final accuracy goes.

**The information ceiling of the answer vector.** In the diagnosis it was 0.669 weighted on gpt-5,
against 0.558 for the majority class and 0.375 for the tree. The three numbers have not been
recomputed and remain unverified: they must be redone by the scorer on the test set of the experiments
before going into the thesis, and the recomputation joins the requirements of spec 06 when it is
written. The thesis plan reports 0.665 and 0.398 for the same quantities, so the difference must be
settled there. If v1 raises the ceiling, the questions carry more
information. If it raises only the accuracy without raising the ceiling, the gain comes from the
aggregation, not from the questions.

**The distribution of the answers per node.** A node that goes from 1.00 or 0.00 to an intermediate
value has stopped being degenerate, and it is the most direct signal that the reformulation has
worked.


---

## 6. Log of the alignments with the thesis

What was overwritten in the YAML files to bring them to the version of the thesis, and why. It serves
to know what was there before without having to search the history.

### 6.1 Text-only substitutions

| File | CQ | What was there before | Where it ended up |
|---|---|---|---|
| `example.yaml` | CQ2.1 | wording prior to the revision of the thesis | §2.4, variant |
| `cause_to_effect.yaml` | CQ2.1 | "clearly biased manner in order to support" | §2.4, variant |
| `ad_hominem.yaml` | CQ3 | "committed to a group, cause or interest" | §2.2, variant |

None of them touches the graph.

### 6.2 `expert_opinion.yaml`, structural intervention

The thesis split CQ4 into two nodes, and the YAML still had the earlier form. Aligning it:

- **CQ4** becomes "Is there a conclusion C that differs from A but uses A to justify itself?",
  with `yes` towards CQ4.1 and `no` towards good_argumentation
- **CQ4.1** is a new node, "Is conclusion C actually implied by A?", with `no` towards
  non_sequitur and `yes` towards good_argumentation in reconvergence
- **the `na` arc disappears.** It is no longer needed, because the case `C = A` now exits on CQ4 `no`
  without passing through CQ4.1. It was marked `policy: true` and was a deliberate decision, so it is
  worth knowing that it was not abandoned, it was made superfluous by the new structure
- **the case of the implicit conclusion remains uncovered.** The thesis gives no rule for
  the enthymeme. The note in the YAML says so explicitly instead of resolving it silently

**Regression snapshot.** This intervention triggers it. The expected values for
`expert_opinion` go from `(5, 4, 7, True)` to `(6, 5, 7, True)`, which is what the snapshot
records today, that is, one more CQ and one more
level of depth, while the number of paths stays 7 and Good Argumentation remains reachable.
It is still advisable to run `validate_schemes.py` and copy the values it prints, because the
convention for counting the depth is its own.

### 6.3 What was not touched

The label vocabulary, then `schemes/_vocabulary.yaml` and today `labels/fallacies.yaml`,
was already in order: the terminal is written `Good Argumentation`.

`analogy.yaml`, `popular_opinion.yaml` and `slippery_slope.yaml` were already identical to the thesis
and were not modified.

`correlation_to_cause.yaml` had been aligned earlier, restoring the text of the thesis
and inverting the two arcs of CQ5.

---

## 7. Removed from the code, to be taken up later

On 8 September the interpreter was brought back to what the diagrams actually
contain. What was removed was not wrong, it was premature: they are decisions on
cases that the diagrams do not provide for, taken before having a pipeline that reflects the
current state. They must be discussed again after the baseline, not before.

### 7.1 The answers beyond `yes` and `no`

The drawn arcs admit `yes` and `no` everywhere, except `ad_hominem` CQ1, which
branches on `positive` and `negative`. Nothing else. The possibility had been implemented
that an annotator or a model would answer `idk`, with three resolution policies.

- `charitable`, take the arcs from which Good Argumentation remains reachable,
  on the principle that an unverified condition is not an accusation
- `branch`, explore all the branches and return the set of possible verdicts
- `stop`, abstain and discard the item

With them also went `tolerant_answers`, which computed from the graph which arcs
were the most lenient, and the distinction between an absent answer and an explicitly
unknown answer.

**When to take it up again.** The prompt run by Eleni admitted only `Yes` and `No`: in the 14246
values recorded by the four pipelines (3569 with gpt-5) there are only `Yes`, `No` and ten
`[MISSING ANSWER]`. Enrico's thesis does not provide for a third answer either. An earlier
version of this paragraph said that the prompt listing printed in the thesis admits
`Cannot be determined from the text`, but in the definitive PDF that listing is no longer there, nor in
the published version of 23 September 2026: §4.3.2 (pp. 113-114) describes the prompts only in words and asks, for stage two, for a binary
answer, `Yes` or `No`, with a mandatory justification. An answer that cannot be
determined appears in the thesis only among the future developments of the conclusions (ch. 6,
pp. 154-156): "distinguere tra prove insufficienti e prove non esplicitate, per esempio con
una risposta non determinabile dal testo che non conduca automaticamente a una fallacia"
[distinguishing between insufficient evidence and evidence left unstated, for example with an
answer that cannot be determined from the text and does not automatically lead to a fallacy].

The third answer is therefore a project choice of ours, and our stage-two prompt
provides for it, `cannot_be_determined`, together with `na` (spec 03). The policy to resolve it
is needed, and these three are the candidates: they come back in phase 3 as a parameter of the
aggregator A2, which the thesis plan calls charitable, explore and stop (section 6).

**The principle worth not losing.** Tolerance lives in the aggregation, not
in the prompt. Telling a model "if you are uncertain answer X" would distort precisely the
per-CQ answers whose uncertainty we want to measure. The model says that it does not know,
and the interpreter applies the rule.

### 7.2 The policy arcs

`Edge` had a `policy` field, to mark an arc added by us to cover a case
that the diagram does not contemplate. The only one in use was `expert_opinion` CQ4 `na`, which
became superfluous when the thesis split CQ4. The mechanism was removed because no
scheme uses it any more and because an arc that is not drawn must not be able to enter the graph
covertly.

**Should it be needed again**, reintroducing it is one line in `Edge` plus the exclusion from the
reconvergence counts, but it must come with a rule on who authorises an arc that is not
present in the diagrams.

### 7.3 The statuses `abstained` and `ambiguous`

`Traversal.status` had four values. Two remain, because the others were produced
only by the IDK machinery.

`abstained` meant that the traversal had stopped without a verdict because an
answer was not available and the chosen policy was to abstain instead of guessing.
Without IDK answers there is no abstention any more.

`ambiguous` meant that the `branch` policy had explored several branches and
several possible verdicts had come out. A traversal with hard answers follows a single branch
and produces a single verdict, so it cannot be ambiguous.

**Where the disagreement ends up, now.** Not inside a traversal, which is deterministic,
but between traversals. Annotator against model, model against model, or looking at
the distributions of `propagate`. It is also the right place, because disagreement is a
property of whoever answers, not of the diagram.

---

## 8. Errors found on 12 September 2026

The document was rechecked number by number against the frozen copy and against Eleni's
`results.csv` files. The differences from the previous version are here, with their cause,
so that whoever rereads it knows what was wrong and why.

| Where | It said | It says | Cause |
|---|---|---|---|
| §1 protocol | 608 items of Eleni | 607 rows, 606 distinct texts, 601 in the experiments | off by one: none of Eleni's eight files has 608 rows, all have 607 with ids from 1 to 607; the distinct texts are 606 after normalization |
| §2.2 ad hominem CQ3 | fires on 29 items | `yes` on 42 items, 29 arrivals at the terminal | a different quantity: 29 are the items that exit on Guilt by Association, not the `yes` answers |
| §2.3 table, gpt-5 CQ3 | 0.02 | 0.07 | the true value is 6 `yes` out of 87; 0.023 is the accuracy of the scheme, reported two lines below, and seems to have ended up in the table cell |
| §2.3 table, qwen CQ2 and CQ2.2 | 0.09 and 0.91 | 0.10 and 0.90 | they correspond to 6 `yes` out of 70 instead of 7: one item off in the same direction |
| §2.5 expert opinion | 24 gold Appeal to authority | 24, with the basis stated | the 24 is right on the predicted subset; an intermediate revision had brought it to 29, which is the number on a gold basis, mixing the two bases within the same paragraph |
| §2.5 expert opinion | kappa 0.38 on CQ1, the others around zero | 0.69 on CQ1, the others between 0.00 and 0.38 | 0.380 exists, but it is the kappa of CQ4.1 on the same subset: it looks like a value taken from the wrong row |
| §2.7 slippery slope | 81 items out of 81 | 80 out of 80 | gpt-5 assigns 80 items to the scheme; no other model gives 81 (qwen 93, llama 90, deepseek 100) |
| §3 popular opinion | CQ3.2 at 1 and 5 per cent, CQ3.3 at 3 | CQ3.2 at 1 and 6, CQ3.3 at 2 and 3 | rounding, and one value per model that was missing |
| §7.1 recorded answers | 3575 values | 14246 over the four pipelines, 3569 with gpt-5 | no file gives 3575 |

**What was not touched.** The aggregate numbers of the diagnosis that appear in the thesis plan
were recomputed and come out exact: gpt-5 pipeline 0.374, zero-shot 0.565, recognition of the
scheme 0.834; on the other models the pipeline is between 0.402 and 0.412 and the scheme between
0.815 and 0.843. The errors above are therefore local to the per-node details of this document.

**What remains unverified.** The information ceiling and the two comparison terms of section 5,
0.669, 0.558 and 0.375, which require the scorer and the aggregators.

**Mixed bases.** None left: §2.6 on analogy CQ4, which was on a gold basis, is now on the
predicted subset like the rest of the document, with 0.19 agreement and the two percentages of
`yes` on the 85 items that both models assign to the scheme.
