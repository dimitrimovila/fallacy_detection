# Spec 07. The aggregators

Version 1, 19 October 2026, written before any result of A1 to A5. Components: `src/argfallacy/aggregate/`; the burden table `schemes/burden/v1.yaml`; the judge prompt `prompts/judge_v1.md` with `prompts/schemas/judge_v1.json`; the stage `judge` of the client; `configs/judge_v1.yaml` and `configs/judge_v1_answers_only.yaml`.
Depends on: 01, 03, 04, 05, 06, 08 (folds of P13). Used by: 08.

## 0. Purpose, and what the primary configurations answer

An aggregator turns the answers to the critical questions of an item into a verdict (a terminal of the item's scheme) or into an abstention, with an explanation. The six aggregators share one interface (section 1). For each of A1 to A5 this spec names one primary configuration, fixed before any result as spec 08, P10 requires. They are five, so m = 5. Every other configuration is exploratory (spec 08, P5): it is named here, before the results, and never supports a claim of improvement.

The primary configurations work on full run v2 (`20261005T193017Z_full_v2`: schemes 1.1, prompts v2, gold scheme condition). Where they use one model, it is the reference model of spec 08, P10, `qwen3_8_27b`. Each answers one question, always against A0 of the reference model:

| | primary configuration | question |
|---|---|---|
| A1 | the diagram with the order of its blocks chosen out of fold; hard answers | is the order of the questions the problem? |
| A2 | probabilistic traversal of `p_logprob`, with the burden of proof of each CQ | do soft answers help? |
| A3 | multinomial logistic regression per scheme on `p_logprob` | does a weighting of soft answers learned from the gold help? |
| A4 | mean with equal weights of the five entries, then the traversal of A2 | do several models help? |
| A5 | LLM judge: the reference model reads its own answers and justifications | does an LLM combine the answers better than the diagram? |

The question of the decision of 26 October (thesis plan, decision point) is whether soft answers and several models raise the ceiling. Its answer is yes if at least one among A2, A3 and A4 passes the deciding comparison of spec 08, P10. A1 passing says that the order of the diagram was a problem. A5 passing says that an LLM judge beats the diagram, and is read against the zero-shot of the reference model, which already beats A0 (macro F1 0.478 against 0.350 on the 601 items). The decision uses the primary configurations whose results exist on 26 October; A5 enters when its run exists, and m stays 5.

## 1. The common interface

* Input: the scheme, from the folder of `schemes/` that the manifest of the run names, and the answers of the item: one row of `summary.csv` (spec 05) per CQ of the scheme and per entry used. A0, A1, A2 and A3 use one entry, A4 uses five, A5 reads the verdicts of its own run.
* Output: a distribution over the terminals of the scheme, plus a mass of abstention for A2 and A4; the verdict, the terminal with the highest mass (ties: the first terminal id in alphabetical order), or none; the confidence, the mass of the verdict (none for A0 and A1); an explanation, a short text described for each aggregator.
* The learned aggregators, A1 and A3, use the folds of spec 08, P13: the verdict of an item in fold f comes from what was estimated on the other four folds. A2 and A4 estimate nothing from labels and use no folds.
* Every aggregator works on both versions of the schemes. The only part tied to a version is the burden table (section 4.2): A2 and A4 stop with an error, naming the folder, on a version whose table does not exist.
* Labels and scheme ids come only from `labels/`. The interpreter and the loader are used as they are.

## 2. A0, the reference

The diagram traversed with the hard answers of one entry, with `traverse` of `src/argfallacy/schemes/interpreter.py`. An answer outside the arcs (`cannot_be_determined`, `na`, `invalid`) stops the traversal with no verdict, which counts as wrong (spec 08, P10) and as an abstention in the curves (P11). A0 is the `diagram` rule of the run report, and its verdicts must equal those of that rule. Explanation: the path, for example `CQ1=yes > CQ2=yes > appeal_to_authority`. A0 is computed for each of the five entries.

## 3. A1, the diagram with another order

### 3.1 Blocks

Of the eight diagrams of schemes 1.1, only ad hominem is a chain of questions, so the order is defined on blocks, derived from the graph of each diagram by one rule, the same for every version:

1. The pass arc of a node is the arc with the shortest path to `good_argumentation` (an arc that ends on `good_argumentation` counts as the shortest), ties going to the arc written first in the YAML. Where no arc reaches `good_argumentation`, it is the only arc that leads to a node; where both arcs end on terminals, the arc `no`.
2. The main path starts at the first node and follows the pass arcs. It ends on a terminal T: `good_argumentation`, or `ad_hominem_abusive` for ad hominem.
3. The block of a node of the main path is the node and the nodes reachable from its other arc without passing through a node of the main path. A node whose arcs all end on terminals may belong to two blocks (analogy CQ1.1, reached from CQ1 and from CQ2).
4. Every arc of a block ends on a terminal, on a node of the same block, or on the next node of the main path (T for the last block). A diagram that breaks this stops A1 with an error that names the arc.
5. A sub-chain is a sequence of at least two nodes of a block, other than its first node, in which each node has one arc to a terminal and one arc, with the same answer along the sequence, to the next node; the arc with that answer of the last node is the end of the sub-chain.

The blocks of schemes 1.1, in the original order (in parentheses the other nodes of a block):

| scheme | blocks | sub-chains | orders |
|---|---|---|---|
| ad hominem | CQ1, CQ2, CQ3, CQ4, CQ5, CQ6 (T is Abusive) | | 720 |
| analogy | CQ1 (CQ1.1), CQ2 (CQ1.1), CQ3, CQ4 | | 24 |
| cause to effect | CQ1, CQ2 (CQ2.1, CQ2.2), CQ3, CQ4 | CQ2.1, CQ2.2 | 48 |
| correlation to cause | CQ1, CQ2 (CQ2.1, CQ2.2), CQ3, CQ4, CQ5 | CQ2.1, CQ2.2 | 240 |
| example | CQ1, CQ2 (CQ2.1, CQ2.2, CQ2.3), CQ3 | CQ2.1, CQ2.2, CQ2.3 | 36 |
| expert opinion | CQ1, CQ2, CQ3 (CQ3.1), CQ4 (CQ4.1) | | 24 |
| popular opinion | CQ1, CQ2, CQ3 (CQ3.1, CQ3.2, CQ3.3) | CQ3.1, CQ3.2, CQ3.3 | 36 |
| slippery slope | CQ1, CQ2, CQ3, CQ4 (CQ4.1), CQ5 | | 120 |

The blocks of schemes 2.0:

| scheme | blocks | sub-chains | orders |
|---|---|---|---|
| ad hominem | CQ1, CQ2, CQ3, CQ4, CQ5, CQ6 (T is Abusive) | | 720 |
| analogy | CQ1 (CQ1.1), CQ2 (CQ1.1), CQ3, CQ4 | | 24 |
| cause to effect | CQ1, CQ2, CQ3, CQ4, CQ5 | | 120 |
| correlation to cause | CQ1, CQ2, CQ3, CQ4, CQ5, CQ6 | | 720 |
| example | CQ1, CQ2 (CQ2.1, CQ2.2, CQ2.3), CQ3 | CQ2.1, CQ2.2, CQ2.3 | 36 |
| expert opinion | CQ1, CQ2, CQ3, CQ4 (CQ4.1), CQ5 (CQ5.1) | | 120 |
| popular opinion | CQ1 (CQ1.1, CQ1.2), CQ2, CQ3 | CQ1.1, CQ1.2 | 12 |
| slippery slope | CQ1, CQ2, CQ3 | | 6 |

### 3.2 Rebuilding a diagram in another order

An order is a permutation of the blocks and, inside each block, of each sub-chain. In the rebuilt diagram, every arc that ended on the next node of the main path (on T for the last block) ends on the first node of the next block in the new order (on T for the new last block). The arc that entered a sub-chain enters its new first node, the arcs between its nodes follow the new sequence, and the last node's arc goes to the end of the sub-chain. Every other arc, every question and every terminal stay as they are. The original order rebuilds the original diagram, arc for arc.

### 3.3 Choice of the order

For each outer fold and each scheme, every order is applied to the hard answers of the training items of that scheme (the other four folds), with the rule of A0: an answer outside the arcs stops the traversal with no verdict. The order chosen has the highest macro F1 on those items (canonical scorer, classes with support among them); ties within 1e-12 go to the higher accuracy, then to the order closest to the original (fewest pairs of blocks, then of nodes of a sub-chain, in a different relative order), then to the first order in lexicographic order of the node ids. Where an answer never changes, moving its block changes no verdict, and the tie rule keeps the block in place: ad hominem CQ1 is `negative` on every item for the reference model.

A1 is the choice of the order of a decision list whose rules are fixed (Rivest 1987). With at most six blocks the exhaustive search is exact; on long lists the same choice is made by branch and bound with a penalty on complexity (Angelino et al. 2018). Correlation to cause has 11 items for 240 orders: the orders chosen in the five folds, written in `a1_orders.csv` beside the verdicts (spec 08, P14), show whether the choice is stable.

### 3.4 Output and primary configuration

Verdict through the rebuilt diagram; no confidence (a single point, spec 08, P11). Explanation: the order used and the path.

Primary configuration: the hard answers of `qwen3_8_27b`, blocks and sub-chains as above.

## 4. A2, probabilistic traversal

### 4.1 The answers

For each CQ, the distribution of the entry over its admitted answers: for the primary configuration, the columns `p_logprob_<answer>` of `summary.csv` (spec 05); for the other entries, the source of section 6. The two arcs get their masses; the mass outside the arcs, `cannot_be_determined` plus `na`, is routed by the type of the question (4.2). A CQ without masses takes its hard answer as a mass of 1; an `invalid` hard answer puts all the mass outside the arcs.

### 4.2 The burden of proof of each CQ

When a critical question remains unanswered, two theories give opposite effects: the argument falls, or the question needs evidence before it counts against the argument (Walton and Godden 2005). Carneades reconciles them: a question on a premise or on an assumption puts the burden on the party who argues, a question on an exception puts it on the party who objects, the burden of production going to the party with better access to the evidence (Gordon, Prakken and Walton 2007, sections 2 and 5; Verheij 2003). Every node gets one of four types, and the mass outside its arcs follows the type:

* **P, premise.** The question asks whether a premise of the formal schema of the scheme holds. Unanswered, the argument cannot be accepted, but no defect is shown: the mass leaves the diagram as abstention.
* **E, exception.** The question asks whether a specific circumstance defeats the argument. Unanswered, the circumstance is not shown: the mass goes to the pass arc of 3.1, toward `good_argumentation`, or toward the generic fallacy where `good_argumentation` cannot be reached.
* **G, gate.** Both arcs can still reach `good_argumentation`, and the other arc opens follow-up questions. Unanswered, the mass goes to the follow-up questions, which decide with their own answers.
* **D, two fallacies.** Both arcs end on a fallacy, and neither is the generic one. Unanswered, the mass is split in two equal parts.

The types of schemes 1.1 are in `schemes/burden/v1.yaml`. The file sits outside `schemes/v1/`, so the hash of the schemes in the manifests does not change. For the P questions, the premise in parentheses:

| scheme | P | G | D | E |
|---|---|---|---|---|
| ad hominem | CQ1 (a is a person of bad character) | | | CQ2, CQ3, CQ4, CQ5, CQ6 |
| analogy | CQ2 (C1 is similar to C2, in a respect relevant to A), CQ3 (A is true in C1) | | | CQ1, CQ1.1, CQ4 |
| cause to effect | CQ1 (generally, if A occurs, B occurs) | CQ2 | | CQ2.1, CQ2.2, CQ3, CQ4 |
| correlation to cause | CQ1 (there is a correlation between A and B), CQ3 (the observations behind the correlation) | CQ2 | | CQ2.1, CQ2.2, CQ4, CQ5 |
| example | CQ1 (a has property F and property G) | CQ2 | | CQ2.1, CQ2.2, CQ2.3, CQ3 |
| expert opinion | CQ1 (S is an authoritative source in D), CQ3 (S asserts A), CQ4.1 (A implies C) | CQ4 | CQ3.1 | CQ2 |
| popular opinion | CQ3 (acceptance by a large majority gives a presumption in favour of A) | | | CQ1, CQ2, CQ3.1, CQ3.2, CQ3.3 |
| slippery slope | CQ1 and CQ5 (each case leads to the next), CQ2 (Cn is a horrible outcome) | CQ4 | | CQ3, CQ4.1 |

14 P, 5 G, 1 D, 28 E. The routes that follow: every E goes to its arc `no`, except correlation to cause CQ5, whose arc `yes` goes to `good_argumentation`; ad hominem CQ6 goes to Abusive, analogy CQ1.1 to Weak Analogy, popular opinion CQ3.3 to Ad Populum. G goes to `no` in example, cause to effect and correlation to cause CQ2, and to `yes` in expert opinion CQ4 and slippery slope CQ4. D splits expert opinion CQ3.1 between `strawman` and `false_attribution`. G holds exactly on the nodes whose two arcs both reach `good_argumentation`, and D only on a node whose two arcs both end on a fallacy.

Six types are a judgement, recorded with their reason:
* analogy CQ1 is E: it asks for specific differences between the cases to be pointed out, although it bears on the premise of similarity;
* correlation to cause CQ5 is E by its role, a change produced by the way B is defined being an exceptional situation, although its wording, "can it be shown", puts the burden on the arguer;
* popular opinion CQ2 is E by its wording, "any indications ... clearly too small", although it bears on the premise of a large majority;
* expert opinion CQ2 is E: relying on the statement of S alone is a condition the diagram adds, not a premise of the schema;
* slippery slope CQ1 and CQ5 are P: they ask whether the links of the chain are supported, although they are worded as objections.

The two readings of `na` in the justifications of full run v2 follow from the types. "The object of the question is absent" occurs on E questions (popular opinion CQ3.3: the text does not appeal to nature; cause to effect CQ2.1: the text cites no evidence), and the mass goes to the arc without the fallacy. "The text does not follow the scheme" occurs on P questions at the top of the diagram (slippery slope CQ1 and CQ5), and the mass leaves.

The table of schemes 2.0, `schemes/burden/v2.yaml`, is written with the same rules before A2 or A4 is computed on a run of schemes 2.0.

### 4.3 Traversal

A mass of 1 enters the first node. At each node it is multiplied by the routed probabilities of the two arcs and passed on, and it collects on the terminals and in the abstention. The answers to different CQs are treated as independent given the path, as in `propagate` of `src/argfallacy/schemes/interpreter.py`. The routing with abstention is written in `aggregate/` without changing `propagate`; with no mass outside the arcs, its distribution equals that of `propagate`. The masses of the terminals and of the abstention sum to 1. The verdict is the terminal with the highest mass, its confidence that mass; with all the mass in the abstention, no verdict. Explanation: the most probable path, with the probability of each arc, and the three terminals with the highest mass.

Primary configuration: `p_logprob` of `qwen3_8_27b`, burden of proof of each CQ.

## 5. A3, weighted checklist per scheme

* One model per scheme, trained on the training items of that scheme (the other four folds).
* Features: for each CQ of the scheme, in the order of the YAML, the two masses of its arcs from `p_logprob_<answer>` of the entry, in the order of the answer space; what is missing to 1 is the mass outside the arcs. A CQ without masses takes its hard answer as 1 and 0; an answer outside the arcs gives 0 and 0.
* Target: the gold fallacy. Classes: the terminals of the scheme present among the training items. An item whose gold is not a terminal of its scheme (the three `scheme_mismatch` items of spec 06) stays out of training, and in the evaluation it is an error like any other.
* Model: multinomial logistic regression of scikit-learn (`LogisticRegression`, solver `lbfgs`, L2 penalty, `class_weight="balanced"`, `max_iter` 10000, fixed `random_state`), on the features as they are, with no scaling. C is chosen among 0.01, 0.1, 1 and 10 by a 4-fold cross validation within the training items (`KFold`, shuffled with the seed of spec 08, P12; not stratified, because some classes have one or two items). The criterion is the macro F1 of the pooled predictions of the four inner folds; ties go to the smaller C. With one class among the training items, that class with probability 1.
* Why: with 11 to 109 items per scheme, logistic regression does better than trees on small training sets (Perlich, Provost and Simonoff 2003); it takes the probabilities as they are; its weights are the explanation. The pattern, a model that answers questions and a linear model that combines the answers, is that of CHiLL (McInerney et al. 2023) and of concept bottleneck models on text (Ludan et al. 2023). The class weights align the training with macro F1, the deciding metric: without them a model with weak features drifts toward the most frequent class of the scheme, which on full run v2 has accuracy 0.579 and macro F1 0.216.
* Output: the probabilities of the classes, the verdict and its confidence. Explanation: the three largest contributions, coefficient times feature, to the class chosen, each with its CQ and arc. The C chosen in each fold and scheme is written in `a3_choices.csv` beside the verdicts (spec 08, P14).

Primary configuration: `p_logprob` of `qwen3_8_27b`.

## 6. A4, several models

* Entries: the five of full run v2. The source of each entry follows its regime, read from the manifest of the run:
  * one sample, with `enable_thinking` false in the `reasoning` of the entry (`qwen3_8_27b`, `gemma4_31b`): `p_logprob_<answer>`;
  * several samples (`gpt_oss_20b`, `gpt_oss_120b`): `p_sample_<answer>`;
  * one sample, with reasoning (`k2_horizon_32b_high`): the stated confidence. With the hard answer on an arc, the first admitted answer gets `p_verbal` and the other `1 − p_verbal`; with `cannot_be_determined` or `na`, that answer gets all the mass.

  After a reasoning that has already decided, the logprob of the answer token goes to the extremes (spec 03, section 6): on full run v2, `p_logprob` is above 0.99 or below 0.01 on 96 to 100 per cent of the rows of every entry except `qwen3_8_27b`.
* For each CQ, the mean with equal weights of the distributions of the entries, each built as A2 builds it for that entry (section 4.1); an entry whose answer is `invalid` and has no masses is left out of the mean, and if every entry is, the whole mass is outside the arcs. Equal weights estimate nothing, so A4 uses no folds. With little data, weights estimated from the data are usually beaten by equal weights, because the error in estimating them cancels their gain (Smith and Wallis 2009); simple combinations of several LLMs have matched the accuracy of human crowds in forecasting (Schoenegger et al. 2024).
* Then the traversal of A2, with the burden of proof of each CQ.
* Explanation: that of A2, with the hard answers of the five entries at each node of the most probable path.

Primary configuration: as above.

## 7. A5, LLM judge

### 7.1 The prompt

`prompts/judge_v1.md` is `prompts/zeroshot_v1.md` with two changes, so that the only difference from the zero-shot is the answers. Between the text and the verdicts, a section:

~~~~text
## Answers to the critical questions

Each critical question of this scheme was answered about the text, one question at
a time, without seeing the verdicts. The answers may contain mistakes.
`cannot_be_determined` means that no basis was found for answering;
`na` means that the question takes for granted something the argument does not contain.

{{answers}}
~~~~

and, in "What to do", before "Choose the single verdict, among those listed, that best describes the argument.", the sentence: "Weigh the answers to the critical questions against the text: they are evidence, not a verdict, and you may disagree with any of them."

`{{answers}}` lists every CQ of the scheme, in the order of the YAML, word for word, followed by the hard answer of the source entry, as its canonical value, and its justification:

~~~~text
**CQ1.** <the text of the question, word for word>
Answer: `<the hard answer>`. <the justification of the hard answer>
~~~~

An `invalid` hard answer is written "Answer: none usable." With `justifications: false` every line stops after the answer. The judge does not see the arcs of the diagram: it combines the answers freely, which is what A5 sets against the fixed diagram. The JSON Schema `prompts/schemas/judge_v1.json` is that of `zeroshot_v1` with its own title, and `answer` is narrowed per scheme in the same way.

### 7.2 The stage `judge` in the client

* A configuration with `stages: [judge]` and `prompt_version: v1` plans one call per item of the test set, under its gold scheme, with the samples of each entry, as the zero-shot does.
* Keys of the configuration, under `judge`: `answers_from`, the run whose `summary.csv` gives the answers; `answers_model`, the entry whose answers are shown; `justifications`, true or false. The plan reads `runs/<answers_from>/summary.csv`, which the parser of spec 05 must have written, and stops with an error naming the item and the CQ if a CQ of the item's scheme has no row.
* The answers enter the rendered prompt, so the cache key covers them; the manifest records them with the rest of the configuration.
* The lines of `raw.jsonl` have `stage` `judge`, the scheme of the prompt and an empty `cq_id`. The parser reads them as zero-shot rows (spec 05, section 5); the run report does not read this stage.


For full run v3, `configs/judge_full_v3.yaml` and
`configs/judge_full_v3_answers_only.yaml` use the same `judge_v1` prompt, with
`schemes: v2` and the answers of `20261019T151336Z_full_v3`. Each plans 601 calls.
The source manifest must name the same schemes folder; when it records a content hash,
that hash must match the current scheme files. A mismatch stops the plan before calls.
A legacy source manifest without a folder means `schemes/v1/`.
The v3 pairing rule applies to the CQ prompts, not to a judge-only run.

### 7.3 Configurations

* `configs/judge_v1.yaml`, the primary configuration: the 601 items of the test set, `qwen3_8_27b`, the answers of `qwen3_8_27b` in `20261005T193017Z_full_v2`, with justifications. 601 calls.
* `configs/judge_v1_answers_only.yaml`, exploratory: the same without justifications. 601 calls.

Both run with `serving/slurm/pilot.sbatch`, like every other configuration. The verdict of A5 is the hard answer of the parsed run, its confidence `p_label` (spec 05). The judge has what the zero-shot has plus the answers, so A5 against the zero-shot of the same model measures what the answers add, reported beside the deciding comparison (spec 08, P10). An LLM used as a judge: Zheng et al. (2023).

## 8. Exploratory configurations

Named before the results; reported as exploratory comparisons (spec 08, P5), never as an improvement:
* A2 with one policy on every CQ: `stop`, every CQ as P (with this policy A2 differs from A0 only in using soft answers); `branch`, the mass outside the arcs split in two equal parts at every CQ; `charitable`, the mass outside the arcs to the pass arc at every CQ, split in two at the D node.
* A2 for each of the other four entries, with the source of section 6 and the burden of proof of each CQ: the diagnostic comparison of spec 08, P10, against the A0 of the same entry.
* A5 without justifications, `configs/judge_v1_answers_only.yaml`.

Named here and computed in phase 4 only if time allows: A3 as a tree of depth at most 3 on the hard answers; A4 with weights per CQ from the agreement of each entry with the per-CQ annotation of the `Annotazione Dumitru` sheets, estimated out of fold; A4 by stacking (Wolpert 1992).

## 9. Output

The aggregators are functions of `argfallacy.aggregate`. `argfallacy evaluate` (spec 08) calls them and writes `verdicts.csv` (spec 08, P14): one row per system and item, with `system`, `item_id`, `fold`, `verdict`, `confidence`, `abstention` and `explanation`. The names of the systems: `A0:<entry>`, `A1`, `A2`, `A3`, `A4`, `A5`, `zeroshot:<entry>`, `majority`, `ceiling:<entry>`; for the exploratory ones `A2.stop`, `A2.branch`, `A2.charitable`, `A2:<entry>` and `A5.answers_only`.

## 10. Acceptance tests

On the diagrams of `schemes/v1/` and `schemes/v2/`, and on answers written by hand; no run and no network:
1. Blocks: the derivation gives the blocks, sub-chains and number of orders of the two tables of section 3.1. The original order rebuilds every diagram arc for arc. A diagram written by hand with an arc from a block to a later node of the main path stops A1 with an error that names the arc.
2. A1: with the original order, the verdicts of A0 on every answer vector of a test; on a scheme written by hand where the last block alone decides correctly, the order chosen moves it first; on a tie, the original order.
3. Burden table: every CQ of `schemes/v1/` has one type and the file names no other CQ; G holds exactly where both arcs reach `good_argumentation`; D only where both arcs end on a fallacy; the routes of section 4.2.
4. A2: with no mass outside the arcs, the distribution equals that of `propagate` within 1e-12; with all the mass outside the arcs at a P node, abstention; at an E node, the pass arc; at a G node, the arc to the follow-up questions; at the D node, half and half; terminals and abstention sum to 1. The three exploratory policies route as section 8 says.
5. A3: on data written by hand where one CQ separates the classes, the model learns it and predicts out of fold; with one class in training, that class; ties in the inner criterion go to the smaller C; the `scheme_mismatch` items stay out of training.
6. A4: with five identical entries, A4 equals A2 of one entry; the source of each entry follows its regime, on a manifest written by hand; an entry whose answer at a CQ is `invalid` with no masses is left out of the mean.
7. A5: rendering is deterministic; the prompt contains every CQ of the scheme word for word and the answers, and no justification with `justifications: false`; a source run without a CQ of an item stops the plan; `configs/judge_v1.yaml` plans 601 calls; a judge line parses to a terminal id with its `p_label`.
8. A1 and A3 give the same verdicts in two runs with the same seed.

## 11. What not to do

* No change to `schemes/v1/`, `schemes/v2/`, the interpreter or the loader.
* No change to a primary configuration after a result: a new idea becomes an exploratory configuration, named as such.
* No gold label in A2, A4 or A5.
* No label or scheme name written in the code.
* No job launched on the cluster.

## 12. Documents to align in the same commit

* README: the aggregators in the package; the stage `judge` and its two configurations; the aggregators leave the list of what is not implemented yet.
* Spec 03: a section on `judge_v1`, after the one on `zeroshot_v1`.
* Spec 04: the stage `judge` in sections 2.3 and 2.5, and in 4.2 beside the zero-shot.
* `pyproject.toml`: scikit-learn among the dependencies.

## 13. Definition of done

Lint and tests green, with the commands of the README. A0 to A4 computed on full run v2 by `argfallacy evaluate` (spec 08), A0 equal to the `diagram` rule of the run report. A5 ready: `argfallacy run plan configs/judge_v1.yaml` counts 601 calls. Summary in Italian.

## References

* Angelino, E., Larus-Stone, N., Alabi, D., Seltzer, M. and Rudin, C. (2018). Learning certifiably optimal rule lists for categorical data. *Journal of Machine Learning Research*, 18(234), 1–78.
* Gordon, T. F., Prakken, H. and Walton, D. (2007). The Carneades model of argument and burden of proof. *Artificial Intelligence*, 171(10–15), 875–896.
* Ludan, J. M. et al. (2023). Interpretable-by-design text understanding with iteratively generated concept bottleneck. arXiv:2310.19660.
* McInerney, D. J. et al. (2023). CHiLL: zero-shot custom interpretable feature extraction from clinical notes with large language models. *Findings of EMNLP*.
* Perlich, C., Provost, F. and Simonoff, J. S. (2003). Tree induction vs. logistic regression: a learning-curve analysis. *Journal of Machine Learning Research*, 4, 211–255.
* Rivest, R. L. (1987). Learning decision lists. *Machine Learning*, 2(3), 229–246.
* Schoenegger, P. et al. (2024). Wisdom of the silicon crowd: LLM ensemble prediction capabilities rival human crowd accuracy. *Science Advances*, 10(45).
* Smith, J. and Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3), 331–355.
* Verheij, B. (2003). Dialectical argumentation with argumentation schemes: an approach to legal logic. *Artificial Intelligence and Law*, 11(2–3), 167–195.
* Walton, D. and Godden, D. (2005). The nature of critical questions in argumentation schemes. In D. Hitchcock (ed.), *The Uses of Argument*, McMaster University, 476–484.
* Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
* Zheng, L. et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS Datasets and Benchmarks*.
