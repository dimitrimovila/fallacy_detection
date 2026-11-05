# Argumentation schemes, critical questions and diagrams: version 2.0

**Generated file. Do not edit by hand.** It is produced from `schemes/v2/*.yaml`, which are the authoritative source; after any change to those files, regenerate it. Version 1.1 (the published thesis of E. Bergamasco, chapter 3) is described in `schemes_and_diagrams.md`. The two versions differ in the wording of many questions and, for some schemes, in the structure of the diagram: always check which version an experiment uses.


## Argument from example (`example`, version 2.0)

**Schema**
```
In this particular case, the individual a has property F and also property G.
a is typical of things that have F and may or may not also have G.
Therefore, generally, if x has property F then x also has property G.
```

**Identification question**
> Does the text use an example that is considered appropriate to draw a conclusion on the main topic?

**Variables.** `a` the individual case cited as example; `F` the property shared with the domain of the generalization; `G` the property transferred to the conclusion; `x` any individual the generalization ranges over

**Critical questions**
- **CQ1**: Is the case reported as the example clearly false or unrealistic?
- **CQ2**: Is the example typical of the kinds of cases that the generalization ranges over?
- **CQ2.1**: Was the example selected in a biased way that makes the generalization appear more strongly supported than it actually is?
- **CQ2.2**: Are the cases cited too few to support a conclusion as general as the one drawn?
- **CQ2.3**: Was the example chosen as a limit case, deliberately extreme and disproportionate to the typical cases covered by the generalization, in order to influence the audience?
- **CQ3**: Does the example cited have characteristics that make it too different from the generalization drawn in the conclusion?

**Diagram** (start: CQ1)
```
CQ1    yes      → false_premise
       no       → CQ2
CQ2    yes      → CQ3
       no       → CQ2.1
CQ2.1  yes      → cherry_picking
       no       → CQ2.2
CQ2.2  yes      → hasty_generalization
       no       → CQ2.3
CQ2.3  yes      → appeal_to_extremes
       no       → CQ3
CQ3    yes      → weak_analogy
       no       → good_argumentation
```
**Terminals**: `appeal_to_extremes`, `cherry_picking`, `false_premise`, `good_argumentation`, `hasty_generalization`, `weak_analogy`

## Argument from analogy (`analogy`, version 2.0)

**Schema**
```
Generally, case C1 is similar to case C2.
A is true (false) in case C1.
Therefore A is true (false) in case C2.
```

**Identification question**
> Does the text use a case considered similar to another in a certain respect to draw a conclusion about the main topic?

**Variables.** `C1` the source case (the analogue); `C2` the target case (the one the conclusion is about); `A` the proposition transferred from C1 to C2; `C3` a further case, similar to C1, in which A has the opposite truth value

**Critical questions**
- **CQ1**: Are C1 and C2 so dissimilar that the comparison is inappropriate, with more differences than similarities between them?
- **CQ1.1**: Was the case used in the comparison chosen as a limit case, deliberately excessive and disproportionate to the other case, in order to influence the audience?
- **CQ2**: Is the similarity cited relevant to A?
- **CQ3**: Is A true (false) in C1?
- **CQ4**: Is there a well-known case C3, also similar to C1, but in which A is false (true)?

**Diagram** (start: CQ1)
```
CQ1    yes      → CQ1.1
       no       → CQ2
CQ1.1  yes      → appeal_to_extremes
       no       → weak_analogy
CQ2    yes      → CQ3
       no       → CQ1.1
CQ3    yes      → CQ4
       no       → false_premise
CQ4    yes      → cherry_picking
       no       → good_argumentation
```
**Terminals**: `appeal_to_extremes`, `cherry_picking`, `false_premise`, `good_argumentation`, `weak_analogy`

## Argument from cause to effect (`cause_to_effect`, version 2.0)

**Schema**
```
Generally, if A occurs, then B will (might) occur.
In this case, A occurs (might occur).
Therefore in this case, B will (might) occur.
```

**Identification question**
> Does this text use a known causal relationship as a basis for predicting or explaining a future or past event?

**Variables.** `A` the putative cause; `B` the putative effect

**Critical questions**
- **CQ1**: Does the argument claim that A causes B solely because B occurred after A?
- **CQ2**: Does the text present a biased selection of evidence or cases that makes the causal generalization appear more strongly supported than it actually is?
- **CQ3**: Does the text report observed cases for the causal generalization that are too few to support it?
- **CQ4**: Does the text present A as the only cause of B, while B clearly also depends on other causes?
- **CQ5**: Is it plausible that A leads to B?

**Diagram** (start: CQ1)
```
CQ1  yes      → post_hoc
     no       → CQ2
CQ2  yes      → cherry_picking
     no       → CQ3
CQ3  yes      → hasty_generalization
     no       → CQ4
CQ4  yes      → causal_reductionism
     no       → CQ5
CQ5  yes      → good_argumentation
     no       → false_premise
```
**Terminals**: `causal_reductionism`, `cherry_picking`, `false_premise`, `good_argumentation`, `hasty_generalization`, `post_hoc`

## Argument from correlation to cause (`correlation_to_cause`, version 2.0)

**Schema**
```
There is a positive or negative correlation between A and B.
Therefore, A causes B.
```

**Identification question**
> Does this text start with two correlated events and conclude that one causes the other?

**Variables.** `A` the putative cause; `B` the second correlated event (the putative effect)

**Critical questions**
- **CQ1**: Is there actually a positive or negative correlation between A and B?
- **CQ2**: Does the text report observed cases for the correlation that are too few to support it?
- **CQ3**: Does the claim that one of the two events causes the other rest only on the fact that it occurs before the other?
- **CQ4**: Does the causal relationship actually go from B to A, and not just from A to B?
- **CQ5**: Is there a third, well-known factor that causes both A and B?
- **CQ6**: Does the text present A as the only cause of B, while B clearly also depends on other causes?

**Diagram** (start: CQ1)
```
CQ1  yes      → CQ2
     no       → false_premise
CQ2  yes      → hasty_generalization
     no       → CQ3
CQ3  yes      → post_hoc
     no       → CQ4
CQ4  yes      → questionable_cause
     no       → CQ5
CQ5  yes      → questionable_cause
     no       → CQ6
CQ6  yes      → causal_reductionism
     no       → good_argumentation
```
**Terminals**: `causal_reductionism`, `false_premise`, `good_argumentation`, `hasty_generalization`, `post_hoc`, `questionable_cause`

## Argument ad hominem (`ad_hominem`, version 2.0)

**Schema**
```
a is a person of bad character.
Therefore, a's argument A should not be accepted.
```

**Identification question**
> Does the text shift the focus from arguments to personal judgments about the other person?

**Variables.** `a` the person whose character is judged; `A` the argument put forward by a

**Critical questions**
- **CQ1**: Is the opinion expressed about the other person negative (and intended to undermine their credibility) or positive (and intended to enhance their credibility)? *(answers: positive, negative)*
  - Note in the YAML: The positive branch leaves the ad hominem cluster and shares the appeal_to_authority label with expert_opinion CQ2 and CQ3. The two are not the same situation: here the argument leverages moral character, there domain competence. Record (scheme, exit CQ), not just the terminal label.
- **CQ2**: Does the attack accuse the other person of being influenced by personal interests that compromise the objectivity of their judgment?
- **CQ3**: Does the attack discredit the other person through their commitment, connection or association with a person, group, movement or cause (for example political, religious or ideological) that is viewed negatively?
- **CQ4**: Is the direct attack on the other person an accusation of hypocrisy?
- **CQ5**: Is the direct attack on the other person intended to undermine their self-confidence?
- **CQ6**: Is the attack made before the other person has presented their argument, to make the audience distrust in advance anything they say?

**Diagram** (start: CQ1)
```
CQ1  positive → appeal_to_authority
     negative → CQ2
CQ2  yes      → ad_hominem_circumstantial
     no       → CQ3
CQ3  yes      → ad_hominem_guilt_by_association
     no       → CQ4
CQ4  yes      → ad_hominem_tu_quoque
     no       → CQ5
CQ5  yes      → ad_fidentia
     no       → CQ6
CQ6  yes      → poisoning_the_well
     no       → ad_hominem_abusive
```
**Terminals**: `ad_fidentia`, `ad_hominem_abusive`, `ad_hominem_circumstantial`, `ad_hominem_guilt_by_association`, `ad_hominem_tu_quoque`, `appeal_to_authority`, `poisoning_the_well`

**Note in the YAML.** The only scheme with no Good Argumentation terminal: every path ends in a fallacy. On this scheme the accuracy of the non-fallacious class is undefined, and any macro-averaged metric that includes it must say so.

## Argument from expert opinion (`expert_opinion`, version 2.0)

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

**Variables.** `S` the source; `D` the domain; `A` the assertion made by the source; `C` the conclusion drawn by the arguer

**Critical questions**
- **CQ1**: Does S have competence, expertise or direct knowledge relevant to the domain D of the claim?
- **CQ2**: Does the argument rest only on S's word, with no evidence or reasoning for A given by S or by the arguer?
- **CQ3**: Does A clearly contradict the prevailing view of experts in D?
- **CQ4**: Did S really assert (or claim) A as true?
- **CQ4.1**: Has S expressed a position of their own on the issue under discussion, in this exchange or elsewhere?
- **CQ5**: Is there a conclusion C that differs from A but uses A to justify itself?
- **CQ5.1**: Assuming that A is true, does the conclusion C follow from it without an unjustified leap?
  - Note in the YAML: C is never reconstructed from context. When C is left implicit (enthymeme) the diagram gives no rule, so the case is out of scope rather than answered: judging an inferential step nobody wrote would manufacture Non Sequitur verdicts out of reconstruction error.

**Diagram** (start: CQ1)
```
CQ1    yes      → CQ2
       no       → irrelevant_authority
CQ2    yes      → appeal_to_authority
       no       → CQ3
CQ3    yes      → appeal_to_authority
       no       → CQ4
CQ4    yes      → CQ5
       no       → CQ4.1
CQ4.1  yes      → strawman
       no       → false_attribution
CQ5    yes      → CQ5.1
       no       → good_argumentation
CQ5.1  yes      → good_argumentation
       no       → non_sequitur
```
**Terminals**: `appeal_to_authority`, `false_attribution`, `good_argumentation`, `irrelevant_authority`, `non_sequitur`, `strawman`

## Argument from popular opinion (`popular_opinion`, version 2.0)

**Schema**
```
If a large majority of some reference group accept A as true, then there is a
presumption in favour of A.
A large majority of the reference group accept A as true.
Therefore there is a presumption in favour of A.
```

**Identification question**
> Does this text use the fact that the conclusion is supported by a relatively large number of people as its main argument in favour of that conclusion?

**Variables.** `A` the proposition held to be true by the group

**Critical questions**
- **CQ1**: Is the reference group in a position to know whether A is true (for example through direct experience), or does A concern a convention or shared preference for which the group's agreement is a reasonable basis for accepting A?
- **CQ1.1**: Does the majority of the reference group rely on the concept of tradition to assert A?
- **CQ1.2**: Does the majority of the reference group rely on the concept of nature to assert A?
- **CQ2**: Does the argument rely on a subgroup likely to support A, while drawing a conclusion that applies to a broader population?
- **CQ3**: Is the group that supports A too small to justify the conclusion?

**Diagram** (start: CQ1)
```
CQ1    yes      → CQ2
       no       → CQ1.1
CQ1.1  yes      → appeal_to_tradition
       no       → CQ1.2
CQ1.2  yes      → appeal_to_nature
       no       → ad_populum
CQ2    yes      → cherry_picking
       no       → CQ3
CQ3    yes      → hasty_generalization
       no       → good_argumentation
```
**Terminals**: `ad_populum`, `appeal_to_nature`, `appeal_to_tradition`, `cherry_picking`, `good_argumentation`, `hasty_generalization`

## Slippery slope argument (`slippery_slope`, version 2.0)

**Schema**
```
Case C0 is tentatively acceptable as an initial presumption.
There exists a series of cases, C0, C1,..., Cn-1, where each case leads to the next
by a combination of causal, precedent, and/or analogy steps.
There is a climate of social opinion such that once people come to accept each
step as plausible (or as accepted practice), they will then be led to accept the
next step.
The penultimate step Cn-1 leads to a horrible outcome, Cn, which is not acceptable.
Therefore, C0 is not acceptable (contrary to the presumption of the initial premise).
```

**Identification question**
> Does the text put forward the argument that, if a particular action is permitted or carried out (even if it is minor or seemingly harmless), it will inevitably lead (through a chain of intermediate consequences) to an extreme or catastrophic final outcome?

**Variables.** `C0` the initial case, tentatively acceptable; `C1, ..., Cn-1` the intermediate cases, each leading to the next; `Cn` the horrible outcome that is not acceptable

**Critical questions**
- **CQ1**: Is there at least one link in the chain that is implausible, potentially depending on unstated intermediate steps that are doubtful?
- **CQ2**: Is the final outcome Cn as bad as the text suggests?
- **CQ3**: Could the chain of events be stopped or reversed at some point before reaching Cn?

**Diagram** (start: CQ1)
```
CQ1  yes      → slippery_slope
     no       → CQ2
CQ2  yes      → CQ3
     no       → slippery_slope
CQ3  yes      → slippery_slope
     no       → good_argumentation
```
**Terminals**: `good_argumentation`, `slippery_slope`

## Inverse index: fallacy → schemes (version 2.0)

| Terminal | Schemes |
|---|---|
| `ad_fidentia` | ad_hominem |
| `ad_hominem_abusive` | ad_hominem |
| `ad_hominem_circumstantial` | ad_hominem |
| `ad_hominem_guilt_by_association` | ad_hominem |
| `ad_hominem_tu_quoque` | ad_hominem |
| `ad_populum` | popular_opinion |
| `appeal_to_authority` | ad_hominem, expert_opinion |
| `appeal_to_extremes` | analogy, example |
| `appeal_to_nature` | popular_opinion |
| `appeal_to_tradition` | popular_opinion |
| `causal_reductionism` | cause_to_effect, correlation_to_cause |
| `cherry_picking` | analogy, cause_to_effect, example, popular_opinion |
| `false_attribution` | expert_opinion |
| `false_premise` | analogy, cause_to_effect, correlation_to_cause, example |
| `good_argumentation` | analogy, cause_to_effect, correlation_to_cause, example, expert_opinion, popular_opinion, slippery_slope |
| `hasty_generalization` | cause_to_effect, correlation_to_cause, example, popular_opinion |
| `irrelevant_authority` | expert_opinion |
| `non_sequitur` | expert_opinion |
| `poisoning_the_well` | ad_hominem |
| `post_hoc` | cause_to_effect, correlation_to_cause |
| `questionable_cause` | correlation_to_cause |
| `slippery_slope` | slippery_slope |
| `strawman` | expert_opinion |
| `weak_analogy` | analogy, example |

## Source files

| File | SHA-256 |
|---|---|
| `schemes/v2/example.yaml` | `09e769f523c91c38ce0128766fb935a515ed8a6ed8c25d76cdbab2b7c82be415` |
| `schemes/v2/analogy.yaml` | `3b2b61575197355554cd51157af2438d4c8a19f2e8604123005371d38d835d93` |
| `schemes/v2/cause_to_effect.yaml` | `181116cea63d7121ba4eb7ae2cfdf8ef5156d778614675bb34f1841bca9d6a4d` |
| `schemes/v2/correlation_to_cause.yaml` | `3f69828fe87029edd7e65df5cd0e652d8c59fe71f4092941b0f96e472d89ed73` |
| `schemes/v2/ad_hominem.yaml` | `2f3f288c3b2762f3a094a1b3ee391449e7d27c494ec3658ffb25b8e3dc502c88` |
| `schemes/v2/expert_opinion.yaml` | `dfb133b407f897676c5b5b7551ff828efc1c1e5441a17f3de584a0b5b13001c0` |
| `schemes/v2/popular_opinion.yaml` | `4e845113a9fd0cad3f1d723958e83bd11e748879e3fdc2e27665615a3ff2d4b3` |
| `schemes/v2/slippery_slope.yaml` | `1c76447c656e7dcd72f38f3f4dc430003ac1a1e05f0a29014c009618319ae0c0` |
