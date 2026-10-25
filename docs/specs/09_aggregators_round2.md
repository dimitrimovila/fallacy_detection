# Spec 09. Aggregators, round 2

Version 1, 22 October 2026, written before any result of round 2. Components: the round 2 modules of `src/argfallacy/aggregate/`; the burden table `schemes/burden/v2.yaml`; the stage `ssoft` of the client, with `prompts/ssoft_v1.md`, `prompts/schemas/ssoft_v1.json`, `configs/ssoft_full_v3.yaml` and `configs/ssoft_nohistory_full_v3.yaml`; the evaluation configuration `configs/evaluation_round2.yaml`.
Depends on: 01, 03, 04, 05, 06, 07, 08. Used by: 08.

## 0. Purpose, and what the primary configurations answer

Round 1 (spec 07) fixed one primary configuration for each of A1 to A5 before any result. None of A1 to A4 beat A0 of the reference model at the level of spec 08, P10; A5 beat it, but matched the zero-shot of the same model. Round 1 stays as it is, and is reported as it was pre-registered. Round 2 is a second family of aggregators, fixed here before any of its results.

Four facts about full run v2, computed for the diagnosis of round 1 before this spec, motivate its design. They use no output of round 2.
* Equal weights lose information. Per CQ, the hard answers of the five entries agree with Dumitru's binary answers on 68 to 78 per cent of the 2,656 labelled pairs, the majority of the five on 76.9 per cent, and the best entry chosen per CQ in sample on 84.7 per cent.
* A3 fits eight models on 11 to 109 items each, on the answers of one entry.
* The probabilities are not calibrated. The AUROC of the confidence of the answer given, for agreeing with Dumitru, is 0.78 and 0.79 for the logprobs of Qwen and Gemma, but Gemma puts 96 per cent of its answers at or above 0.99; after reasoning, the logprob of the answer token carries almost no signal of its own: K2 gives 1.00 to every answer (0.46), and the logprob of the two gpt oss, averaged over their five samples, is almost exactly the frequency of their answers (correlation 1.00); the verbal confidence of the three reasoning entries reaches 0.64 to 0.67.
* No aggregator uses the zero-shot. The diagram and the zero-shot of the reference model agree on 58 per cent of the items, with accuracy 0.72 on them; where they disagree, the zero-shot is right about twice as often as the diagram.

Round 2 has five components. Four have a primary configuration, so m = 4; the fifth is descriptive.

| | component | primary configuration | question |
|---|---|---|---|
| C1 | combination of the five entries at the level of each CQ | stacking supervised by Dumitru's CQ answers, soft traversal of the diagram 1.1 | does a combination learned at the level of the CQs give better verdicts through the diagram? |
| C2 | routing between the diagram and the zero-shot | C1 against the zero-shot of the reference model, threshold 0 | how much does staying explainable cost, and does the combination beat the zero-shot? |
| C3 | learned model that stays explainable | a tree of depth at most 3 per scheme on the combined answers of C1 | does a tree learned from the gold beat the diagram drawn by hand on the same answers? |
| C4 | S-soft | `qwen3_8_27b` on schemes 2.0 | do answers conditioned on the earlier answers of the path beat answers given in isolation? |
| C5 | uncertainty | none | is the uncertainty of the systems higher on their errors and where humans disagree? |

The reference model is that of spec 08, P10, `qwen3_8_27b`.

## 1. Common rules

* Condition and metrics: the 601 items of the test set, gold scheme condition, fine space (spec 08, requirement 1, P1, P2). The deciding metric is macro F1.
* Runs: full run v2 (`20261005T193017Z_full_v2`, schemes 1.1) for C1, C2 and C3; full run v3 (`20261019T151336Z_full_v3`, schemes 2.0) for C4 and for the exploratory configurations on schemes 2.0; zero-shot v1 (`20261009T221950Z_zeroshot_v1`) for the zero-shot of the reference model; the S-soft run of section 6 and the run of its control without history (section 6.5).
* The distribution of an entry at a CQ: as A4 builds it (spec 07, sections 4.1 and 6): `p_logprob_<answer>` for `qwen3_8_27b` and `gemma4_31b`, `p_sample_<answer>` for the two gpt oss, the stated confidence for `k2_horizon_32b_high`. Its mass outside the arcs is the mass of `cannot_be_determined` and `na`. An entry whose answer is `invalid` and has no masses is left out at that CQ, as in A4 (spec 07, section 6): it takes no part in the mass outside the arcs or in any combiner.
* The signals of an entry at a CQ: each probability column of `summary.csv` (spec 05, section 4) that the entry has, read as the split between the two arcs, the mass of the first admitted answer over the mass of the two arcs: `p_logprob` for every entry, `p_sample` for the two gpt oss, `p_verbal` for every entry. Twelve signals in all; a signal without a value is missing, and so are all the signals of an entry left out.
* The combined answer at a CQ: the mass outside the arcs is the mean, with equal weights, of the masses outside the arcs of the entries not left out, as in A4, and the whole mass when every entry is left out; a combiner decides only how the rest of the mass is split between the two arcs, as the probability `p_first` of the first admitted answer.
* CQ labels: Dumitru's answers in `annotations.csv`, from the sheet of the gold scheme of the item and a row whose scheme judgement also matches the gold, binary only (`yes` and `no`; `positive` and `negative` for ad hominem CQ1). `idk`, `na` and empty cells are not labels. On full run v2 there are 2,656 labelled pairs of item and CQ.
* Folds: those of spec 08, P13. Every parameter estimated from a label of a test item, a gold verdict or a CQ label, is estimated on the other four folds. Hyperparameters are chosen by a 4-fold cross validation within the four training folds (`KFold`, shuffled with the seed of spec 08, P12, as A3), on the grids written in each section; ties go to the stronger shrinkage, then to the first value of the grid.
* Cross-fitting. When a component learns from the output of another learned component (C2 and C3 from C1), the inputs of its training items come from the first component fitted by an inner 4-fold cross-fitting within the four training folds, and the inputs of the held-out fold from the first component fitted on all four training folds. No verdict depends on the labels of its own item.
* Estimates without labels (Dawid and Skene, section 2.1) use all 601 items and no folds.
* Interface: that of spec 07, section 1: a distribution over the terminals of the scheme with an abstention mass, a verdict or none, a confidence, an explanation.
* Optimisation: the models that scikit-learn does not provide (stacking with a centre of shrinkage, section 2.1; the pooled recalibration, section 4) are fitted with L-BFGS of `scipy.optimize`. scipy is already installed with scikit-learn and becomes a declared dependency.

## 2. C1, combination at the level of each CQ

### 2.1 Combiners

* **Majority.** The hard answers on the arcs of the entries not left out vote: `p_first` is the share that goes to the first answer, 0.5 when no entry answers on an arc. An arc tie goes to the hard answer of the reference model when it is on an arc, otherwise to the first admitted answer. The hard combined answer follows section 2.3, as for every combiner; outside hard answers do not vote as a separate option.
* **Dawid and Skene** (1979). The true answer z is one of the two arcs. Each entry e has one confusion matrix π_e from z to its hard answer, among the first arc, the second arc and outside the arcs, shared by every CQ of every scheme; each CQ q has its prior ρ_q over z; an entry left out at a CQ is a missing observation there. EM starts from the majority, stops after 500 iterations or when the log-likelihood changes by less than 1e-8, and gives `p_first` as the posterior of the first arc. The method assumes that the errors of the entries are independent given the true answer; LLMs that share data and architectures can break that assumption (Balasubramanian et al. 2026), and the comparison with the supervised combiners measures how much it costs.
* **A4 with weights per CQ.** `p_first` = Σ_e w_qe · r_e · p_e / Σ_e w_qe · r_e, over the entries not left out (0.5 when the denominator is 0), where p_e is the split of one signal of entry e, chosen per entry by the inner cross validation among its signals, and r_e is the mass that the distribution of e puts on the two arcs: an entry that puts most of its mass outside the arcs weighs less on the split, as in the mean of A4. The weights shrink toward equal weights: w_qe = (1 − λ)/5 + λ · ŵ_qe, with ŵ_qe proportional to max(0, logit(a_qe)), the weights of a weighted majority of independent voters, and a_qe = (k + 1)/(n + 2) the agreement of the hard answer of e with the CQ labels of q on the training items (k agreements out of n labels); equal weights when every ŵ is 0. Grid: λ in 0, 0.25, 0.5, 0.75 and 1; λ = 0, with the signal of the distribution of each entry, gives exactly the split of A4. Estimated weights often lose to equal weights on small samples, because the error of the estimate cancels the gain (Smith and Wallis 2009); λ chosen out of fold decides how much of the estimate to keep.
* **Stacking** (Wolpert 1992). `p_first` = σ(β_0 + Σ_s β_s x_s + Σ_s γ_s u_s), over the twelve signals s, with x_s the logit of the split clipped to [0.005, 0.995] (0 when missing) and u_s 1 when the signal is missing. The penalty shrinks the coefficients toward a centre, not toward zero: (1/(2C)) times the squared distance from β⁰, where β⁰ gives 1/5 to the signal of each entry named in section 1 for its distribution and 0 to every other coefficient, the logarithmic pool with equal weights (Genest, Weerahandi and Zidek 1984). Grid: C in 0.01, 0.1, 1 and 10. The model learns the weights of the entries and their calibration together.

### 2.2 Supervision

The source of supervision is a factor of the experiment, as in concept bottleneck models, where a model first predicts interpretable concepts and the decision depends only on them (Koh et al. 2020):
* **(a) None.** Majority and Dawid and Skene.
* **(b) CQ labels**, the supervision of the concepts. A4 with weights per CQ and stacking are fitted on the labelled pairs of the training items: stacking minimises the penalised log loss of `p_first` against the label; A4 chooses λ and the signals by the Brier score. Inner criterion: the log loss of the inner predictions for stacking, the Brier score for A4.
* **(c) Gold verdicts through the diagram**, the supervision of the task. Stacking, with the same features and the same centre as in (b), minimises −(1/N) Σ_i log(m_i(g_i) + 1e-6) + (1/(2CN)) ‖β − β⁰‖², where m_i(g_i) is the mass that the soft traversal of section 2.3 gives to the gold verdict of training item i; the items whose gold is not a terminal of their scheme stay out, as in A3. Inner criterion: the macro F1 of the pooled inner verdicts. The CQ labels are not used.

(b) and (c) share the family and the centre of the stacking, so their difference isolates the supervision. Reading: similar verdicts mean that the CQ labels and the gold ask the same of the answers; (c) above (b) with answers that agree less with the CQ labels means that the gain comes from bending the answers toward the verdict, so that an answer no longer means what its question asks (Margeloiu et al. 2021): the explanation becomes less faithful; (b) above (c) means that the CQ labels teach what 601 verdicts cannot.

### 2.3 Traversal

* **Soft.** The traversal of A2 (spec 07, section 4.3) on the combined answers, with the burden table of the version of the schemes: `schemes/burden/v1.yaml` for schemes 1.1, `schemes/burden/v2.yaml` for schemes 2.0 (section 5). Verdict, confidence and explanation as A2; the explanation also gives, at each node of the most probable path, the hard answers of the five entries and the combined `p_first`.
* **Hard.** The combined hard answer is the arc with the larger split, or `cannot_be_determined` when the mass outside the arcs is above 0.5, traversed with the rule of A0: an answer outside the arcs stops the traversal with no verdict.

### 2.4 Primary configuration and exploratory configurations

Primary configuration C1: stacking, supervision (b), soft traversal, full run v2.

Exploratory, named here:
* every other combination of section 2.1 to 2.3 on full run v2: majority (a), Dawid and Skene (a) and A4 with weights per CQ (b), each soft and hard; stacking (b) hard, named `C1.hard`; stacking (c), soft and hard;
* on full run v3 with the burden table of schemes 2.0: majority, Dawid and Skene, stacking (b) with the coefficients fitted on full run v2 with the CQ labels of schemes 1.1 (the signals keep their meaning across runs; the 2.0 annotation, when it exists, checks the transfer), and stacking (c);
* stacking (b) without the labels of example, cause to effect and correlation to cause, whose sheets went through a different checking procedure, scored on the other five schemes;
* stacking (b) without the CQs whose labels prove unstable in the re-annotation of schemes 1.1 (agreement below 0.6 with at least 10 paired answers), if the re-annotation exists before the results are frozen;
* C1 against A4 of spec 07, which isolates the gain of a learned combination over equal weights;
* C1 against A2 of spec 07, `qwen3_8_27b` alone with the same soft traversal and the burden table of schemes 1.1, which isolates the gain of combining the five entries; H1.1 measures that gain and the soft traversal together.

### 2.5 Evaluation at the level of each CQ

On the 2,656 labelled pairs, for each combiner of full run v2 and for each entry with the signal named in section 1: the agreement of the hard answer with the label, an answer outside the arcs counting as a disagreement, and the Brier score of `p_first`. Paired differences against the reference model and among the three supervisions, with the resamples of spec 08, P12 (all the pairs of an item move together), and 95 per cent intervals. Reported in total and by scheme; exploratory, as every comparison not named in section 8. The path of Dumitru's answers through the diagram 1.1 (438 items) is reported as a reference line, declared as such.

## 3. C2, routing between the diagram and the zero-shot

* Two sides: the diagram, C1, and the zero-shot of the reference model (`zeroshot:qwen3_8_27b`).
* When the two verdicts agree, the verdict is theirs, and it counts as explained by the diagram. When one side has no verdict, the verdict of the other; with none on both sides, no verdict.
* Where they disagree, two logistic regressions (scikit-learn, L2, C in 0.01, 0.1, 1 and 10, inner criterion the log loss) estimate the probability that each side is right, P_d for the diagram and P_z for the zero-shot. Features: the confidence of C1, the margin between its two most probable terminals, its abstention mass, the mean binary entropy of the combined answers on its most probable path, the `p_label` of the zero-shot (missing: the mean of the training items, with an indicator), and the scheme. They are fitted on the training items where the two sides disagree, with the inputs of C1 cross-fitted (section 1).
* Rule: the diagram when P_d ≥ P_z − τ, the zero-shot otherwise. The primary configuration C2 uses τ = 0. Confidence: the estimated probability of the side chosen.
* Curve: τ from −1 to 1 moves the share of the verdicts explained by the diagram from the share of agreement to 1. Reported: macro F1 and accuracy against that share, with 95 per cent intervals at the shares of 60, 70, 80, 90 and 100 per cent, as a description; the value of τ for each share is read on the pooled out-of-fold estimates. The precedent is the hybrid model, in which an interpretable model covers part of the inputs and a black box the rest (Wang 2019; Wang and Lin 2021).
* Explanation: the side chosen, its explanation, and P_d and P_z.
* Exploratory: A0 of the reference model as the diagram side; on schemes 2.0, stacking (b) of full run v3 against the zero-shot with the 2.0 menus (spec 10), when it exists; more sides, D+Q (spec 10) and S-soft, when their runs exist, with one regression per side and the side with the highest estimate chosen.

## 4. C3, a learned model that stays explainable

* One decision tree per scheme (scikit-learn `DecisionTreeClassifier`, criterion gini, `class_weight="balanced"`, fixed `random_state`), on the training items of the scheme. Features: for each CQ of the scheme, in the order of the YAML, the combined `p_first` and the mass outside the arcs of C1, cross-fitted (section 1). Target: the gold verdict; the items whose gold is not a terminal of their scheme stay out of training, as in A3. Grid: `max_depth` in 1, 2 and 3; `min_samples_leaf` in 1, 3 and 5; inner criterion the macro F1 of the pooled inner verdicts. With one class among the training items, that class.
* Confidence: the share of the verdict among the training items of the leaf. Explanation: the path of the tree, each split with its CQ and its threshold.
* Primary configuration C3: the tree above.
* Exploratory: a pooled recalibration of the terminal masses of C1, one model for every scheme in place of the eight small models of A3: for an item with masses m_t over the terminals of its scheme, P(t) ∝ exp(a · log(m_t + 1e-6) + b_ℓ(t)), with one slope a and one bias per label ℓ shared by every scheme; L2 toward a = 1 and b = 0, C in 0.01, 0.1, 1 and 10, inner criterion the log loss of the gold; and the tree on the answers of the reference model alone, named in spec 07, section 8.

## 5. The burden table of schemes 2.0

`schemes/burden/v2.yaml` gives one type to every CQ of `schemes/v2/`, with the rules of spec 07, section 4.2. The table was approved on 23 October 2026, before any result on schemes 2.0 that depends on it. For the P questions, the premise in parentheses:

| scheme | P | G | D | E |
|---|---|---|---|---|
| ad hominem | CQ1 (a is a person of bad character) | | | CQ2, CQ3, CQ4, CQ5, CQ6 |
| analogy | CQ2 (C1 is similar to C2, in a respect relevant to A), CQ3 (A is true in C1) | | | CQ1, CQ1.1, CQ4 |
| cause to effect | CQ5 (generally, if A occurs, B occurs) | | | CQ1, CQ2, CQ3, CQ4 |
| correlation to cause | CQ1 (there is a correlation between A and B) | | | CQ2, CQ3, CQ4, CQ5, CQ6 |
| example | | CQ2 | | CQ1, CQ2.1, CQ2.2, CQ2.3, CQ3 |
| expert opinion | CQ1 (S is an authoritative source in D), CQ2 (A rests on evidence or reasoning, not on the word of S alone), CQ4 (S asserts A), CQ5.1 (A implies C) | CQ5 | CQ4.1 | CQ3 |
| popular opinion | CQ1 (acceptance by a large majority gives a presumption in favour of A) | | | CQ1.1, CQ1.2, CQ2, CQ3 |
| slippery slope | CQ1 (each case leads to the next), CQ2 (Cn is a horrible outcome), CQ3 (the climate of opinion carries each step to the next) | | | |

13 P, 2 G, 1 D, 27 E. The routes that follow: every E goes to its arc `no`, toward the next node, toward `good_argumentation` (analogy CQ4, correlation to cause CQ6, example CQ3, popular opinion CQ3), or toward the generic fallacy (ad hominem CQ6 to Abusive, analogy CQ1.1 to Weak Analogy, popular opinion CQ1.2 to Ad Populum). G goes to `no` in example CQ2 and to `yes` in expert opinion CQ5. D splits expert opinion CQ4.1 between `strawman` and `false_attribution`. G holds exactly on the two nodes whose two arcs both reach `good_argumentation`, and D only on a node whose two arcs both end on a fallacy other than the generic one.

The new diagrams move some types without a new judgement: the premise question of cause to effect moved from the first node to the last (CQ5), and its gate is gone; the question on the observations of correlation to cause became CQ2, a defect to be shown, and its gate is gone; in expert opinion the assertion is CQ4, the node with two fallacies CQ4.1, the gate CQ5 and the implication CQ5.1; in popular opinion the presumption is the first question, and tradition and nature are two new E questions.

Eight types are a judgement, recorded with their reason:
* analogy CQ1 is E, as in schemes 1.1: it asks whether the cases are so dissimilar that the comparison is inappropriate, an objection to be shown; relevant similarity, the premise, is asked by CQ2;
* expert opinion CQ2 is P, unlike schemes 1.1: it asks whether A is backed by evidence, and Carneades models the backup evidence question of this scheme as an assumption, with the burden on the proponent (Gordon, Prakken and Walton 2007, section 5); the same source sets the types of CQ1 and CQ3;
* expert opinion CQ3 is E: consistency with the other experts is an exception of the scheme in Carneades (Gordon, Prakken and Walton 2007, section 5);
* example CQ1 is E, unlike schemes 1.1: its wording asks whether the example is "clearly false or unrealistic", a defect the critic must show; it bears on the premise that a has F and G, and its terminal is False Premise, which would support P, but the other P questions with that terminal ask in positive form whether the premise holds;
* popular opinion CQ3 is E, as CQ2 of schemes 1.1: it asks whether the group is too small, a defect to be shown, although it bears on the premise of a large majority;
* cause to effect CQ3 and correlation to cause CQ2 are E by their answer definitions (`prompts/stage2_v3_answers.yaml`), which send a text with no observed cases to `no`;
* slippery slope CQ1 is P, as in schemes 1.1: it asks whether the links of the chain are supported, although it is worded as an objection;
* slippery slope CQ3 is P: its answer definitions require the text to make it plausible that the chain would not be stopped before `no` lets the argument pass; worded as an objection, it would otherwise be E.

The file sits outside `schemes/v2/`, so the hash of the schemes in the manifests does not change.

## 6. C4, S-soft

S-soft asks the CQs along the diagram, with the earlier answers of the path in view, and keeps every branch whose mass is large enough. It is the probabilistic form of a single path that follows the answers of the model: that path comes out of the same calls. A decision tree whose nodes are calls to a language model has a precedent in Tree Prompting (Singh et al. 2023); sequential sub-questions answered with the earlier answers in view help in other tasks (Zhou et al. 2023).

### 6.1 The prompt

`prompts/ssoft_v1.md` is `prompts/stage2_v3.md` with the heading "The critical question" renamed "The critical question you must answer", and with two sections added before it:

~~~~text
## All the critical questions of this scheme

These are all the questions asked about this argument, one at a time. Answer only
the question indicated below.

{{all_cqs}}

## Earlier questions on this argument, and their answers

{{history}}
~~~~

and one sentence in "What to do", before its last sentence: "Take the earlier answers as given."

`{{all_cqs}}` lists every CQ of the scheme in the order of the YAML, word for word, as `**CQ1.** <text>`. `{{history}}` lists the CQs of the path that leads to the node, in the order of the path, each as `**CQ2.** <text>` followed by `Answer: <answer>.` When there is no earlier answer (at the first node, and at every node of the control without history, section 6.5), the section on the earlier questions and the sentence in "What to do" are left out, so that the prompt differs from `prompts/stage2_v3.md` only by the list of the CQs and the heading. The history holds answers only: on a branch the model did not choose, no justification exists. The answers are those of the arcs of the path; the model sees neither the arcs nor the terminals. The answer definitions are those of `prompts/stage2_v3_answers.yaml`. `prompts/schemas/ssoft_v1.json` is `prompts/schemas/stage2_v3.json` with its own title, narrowed per CQ in the same way.

### 6.2 Expansion

For each item, under its gold scheme, on schemes 2.0:
1. The first node of the diagram gets a path mass of 1 and an empty history.
2. A node gets one call, with the history of its path. The answer gives a distribution over the admitted answers: the entry's distribution of section 1 for an entry read through its logprobs; for an entry read through its stated confidence, the confidence on the arc answered and the rest on the other arc, or all the mass outside the arcs for an answer outside them.
3. The mass outside the arcs is routed by the type of the CQ (section 5 and spec 07, section 4.2): P leaves the diagram as abstention, E goes to the arc of its route, G to the arc toward the follow-up questions, D is split in two.
4. Each arc carries the path mass times its routed mass. An arc to a terminal adds its mass to the terminal. An arc to a node is expanded, with the history extended by the CQ and the answer of the arc, when its mass is at least 0.02; below, its mass goes to a pruned bucket.
5. Terminals, abstention and pruned bucket sum to 1. The verdict is the terminal with the highest mass (ties: the first terminal id in alphabetical order), none when no terminal has mass; the confidence is its mass. Explanation: the most probable path, with the probability of each arc, and the three terminals with the highest mass.
6. The single path, `C4.path`, follows at each node the arc with the larger routed mass.

Two paths that reach the same node with different histories are two calls. On full run v2, the logprobs of the reference model under the assumption of independence give about 5 expanded nodes per item, about 3,000 calls for one entry.

### 6.3 The stage `ssoft` in the client

* A configuration with `stages: [ssoft]` plans the first node of every item; the later calls are planned during the run, level by level, from the parsed answers of the level before, with the function of section 6.2, shared with the aggregator.
* Every call goes through the cache. The history is part of the rendered prompt, so it enters the key. A resubmitted job replays the expansion from the cache and makes only the missing calls; `raw.jsonl` gets no duplicate line.
* The lines of `raw.jsonl` have `stage` `ssoft`, the `cq_id` of the node and a field `path` with the history, for example `CQ1=no>CQ2=yes`, empty at the first node.
* The parser writes `ssoft.csv`: one row per item, entry and node of a path, with the columns of a stage-two row of `summary.csv` (spec 05, section 4) and `path` and `depth`.
* `configs/ssoft_full_v3.yaml`: `stages: [ssoft]`, `prompt_version: v1`, `schemes: v2`, the burden table of schemes 2.0, the 601 items of the test set under their gold scheme, threshold 0.02, entry `qwen3_8_27b` with its sampling (one sample at temperature 0). `argfallacy run plan` counts the 601 first calls and says that the rest is planned during the run.
* The stage `ssoft` has its own prompt version, written for schemes 2.0, and reads the answer definitions of `stage2_v3`. The rule that pairs `schemes: v2` with `prompt_version: v3` concerns the CQ runs of stages one and two; as for the judge, it does not apply to this stage.
* An entry whose answer-token logprob carries almost no signal of its own after reasoning (the two gpt oss, K2 Horizon, section 0) is read through its stated confidence, with one sample per node at the temperature of its producer, so that each node costs one call as for the reference model.

### 6.4 Primary configuration and exploratory configurations

Primary configuration C4: `qwen3_8_27b`, schemes 2.0, `p_logprob` as they are, the burden table of schemes 2.0, threshold 0.02. Its comparison system, `A2.pruned`, is A2 on the answers of `qwen3_8_27b` in full run v3, with the same burden table and the same pruning rule. Calibration enters neither side. The prompt of S-soft differs from that of full run v3 in two ways, the list of all the CQs of the scheme and the history of the path, so H4 measures the two together: the context of the diagram against answers given in isolation. The control `C4.nohistory` (section 6.5) separates them.

Exploratory, named here:
* S-soft with the logprobs of `qwen3_8_27b` scaled by a temperature (Guo et al. 2017), fitted out of fold on full run v2 against the CQ labels of schemes 1.1;
* `C4.path` against the diagram 2.0, A0 of `qwen3_8_27b` on full run v3; and, on the nodes of the most probable path, the answers that change between full run v3 and S-soft;
* S-soft against the zero-shot with the 2.0 menus (spec 10) and against A0 on full run v3. The questions of schemes 2.0 were written while looking at the errors of full run v2 on the test set: a comparison with a system that does not use them favours S-soft, and is reported with that note;
* S-soft for the other four entries, with configurations `configs/ssoft_full_v3_<entry>.yaml`: `gemma4_31b` through its logprobs, the two gpt oss and K2 Horizon through their stated confidence, each scaled by a temperature fitted as above.

### 6.5 The control without history

`C4.nohistory` asks every CQ of the scheme of the item with the prompt of S-soft, the list of all the CQs included, and without the section on the earlier questions and its sentence in "What to do" (section 6.1). Each CQ is asked once per item, with no expansion during the run, so the calls do not depend on the earlier answers: one call per CQ, 3,162 calls for `qwen3_8_27b` on the 601 items. The answers go through the traversal of `A2.pruned`, with the burden table of schemes 2.0 and threshold 0.02.

Configuration `configs/ssoft_nohistory_full_v3.yaml`: as `configs/ssoft_full_v3.yaml`, with `history: false`. With `history: false` the stage `ssoft` plans every call at the start, and `argfallacy run plan` counts all of them. The lines of `raw.jsonl` have `stage` `ssoft` and an empty `path`, and the parser writes them to `ssoft.csv` with `depth` empty.

Exploratory comparisons, named here: C4 against `C4.nohistory`, the effect of the history alone; `C4.nohistory` against `A2.pruned`, the effect of the list of the CQs alone. Both paired, macro F1, 95 per cent intervals.

## 7. C5, uncertainty

* Abstention curves of spec 08, P11, for every system of round 2 that has a confidence (C1 and its soft variants, C2, C3, C4), beside A2, A4, the zero-shot and A5.
* Conformal prediction sets, for the systems with a distribution over the terminals (the soft forms of C1, the recalibration of C3, C4, A2, A4). The score of item i and terminal t is 1 − P_i(t), with P_i renormalised over the terminals of the scheme of the item, abstention and pruned mass removed; with no mass on any terminal, the set is every terminal. For the items of fold f, the threshold q is the ⌈(n + 1)(1 − α)⌉-th smallest score of the gold verdicts of the n items of the other four folds, and the set holds the terminals with a score at most q. α = 0.10 (Angelopoulos and Bates 2023). For a system estimated without labels the coverage is guaranteed on average over the items; for a learned system, whose scores are out of fold, approximately.
* Reported: the coverage, the mean size of the sets and the share of sets with one verdict, in total, by gold origin and by scheme; on the items of the direct-verdict annotation, the mean size where the human direct verdict differs from the gold against where it equals it. All exploratory.

## 8. Deciding comparisons

* Deciding metric: macro F1, fine space, gold scheme condition; paired bootstrap on the resamples of spec 08, P12.
* m = 4, one family per component, Bonferroni across components: each interval at level 1 − 0.05/4, the 0.625 and 99.375 percentiles.
* H1.1: C1 against A0 of `qwen3_8_27b` on full run v2. H1.2, read only if H1.1 shows an improvement: C1 against `zeroshot:qwen3_8_27b`.
* H2: C2 against `zeroshot:qwen3_8_27b`.
* H3.1: C3 against `C1.hard`, the diagram on the same combined answers. H3.2, read only if H3.1 shows an improvement: C3 against `zeroshot:qwen3_8_27b`.
* H4: C4 against `A2.pruned`.
* Within a component the second comparison is read only after the first passes, so the chance of any false claim over the six comparisons stays at most 0.05.
* An improvement is claimed only if the interval excludes zero. Against the zero-shot, an interval below zero is reported as the cost of the diagram, and an interval that contains zero as no measurable difference, not as equivalence.
* Reported beside, without deciding: the same differences in accuracy and in balanced accuracy; each comparison by gold origin (LOGIC, Reddit and ElecDebate against NLAS and EthiX), by scheme, and without NLAS (spec 08, P4), with 95 per cent intervals. When the predicted scheme condition exists (spec 08, P1), the same comparisons serve as confirmation.
* A component not computed does not lower m.

## 9. Output

`argfallacy evaluate configs/evaluation_round2.yaml` writes into `evaluations/round2/`:
* `manifest.json`: the runs, the seed, the resamples, the folds, m, the primary configurations, the grids, the commit of the code, the date;
* `verdicts.csv` (spec 07, section 9), with the systems `C1`, `C2`, `C3`, `C4`, `C1.hard`, `C4.path`, `C4.nohistory`, `A2.pruned`, and the exploratory systems named `C1.<combiner>.<supervision>.<traversal>`, with `@2.0` for those on schemes 2.0;
* `cq_combined.csv`: one row per item, CQ, combiner and supervision, with the fold, `p_first` and the mass outside the arcs;
* `round2_choices.csv`: the hyperparameters chosen in each fold, component and scheme;
* `cq_level.csv`, `cq_level_differences.csv` and `human_path.csv` (section 2.5), `routing_curve.csv` (section 3), `ssoft_changes.csv` (the answers that change, section 6.4), `abstention.csv`, `abstention_curves.csv`, `abstention_composition.csv` and `conformal.csv` (section 7), `main_table.csv` with every system, and one CSV per table of section 8: `deciding.csv`, `deciding_breakdowns.csv`, `exploratory.csv`;
* the sections of round 2 in `summary.html`: deciding comparisons, the level of each CQ, the supervision factor, the routing curve as a figure, S-soft, abstention curves as a figure, conformal sets, and what was not computed.

A configuration is read as one of round 2 when its `runs` name `answers_schemes_2`, `ssoft` or `ssoft_nohistory`. The schemes left out by the exploratory C1 of section 2.4 are written in the configuration, since no scheme name is written in the code.

An incomplete S-soft history or control run excludes its systems and comparisons without lowering m, records "incomplete S-soft run" and the affected item ids as missing, stores their count and ids separately in `manifest.json` as `ssoft_pending_items` and `ssoft_nohistory_pending_items`, and identifies the incomplete run and count in `summary.html`, while `A2.pruned` and the rest of the evaluation remain computed.

The configuration:

```yaml
name: round2
runs:
  answers: 20261005T193017Z_full_v2
  answers_schemes_2: 20261019T151336Z_full_v3
  zeroshot:
    - 20261009T221950Z_zeroshot_v1
    - 20261021T111606Z_zeroshot_k2
  judge:
    - 20261021T111110Z_judge_v1
    - 20261021T111110Z_judge_v1_answers_only
  ssoft: []          # the run of configs/ssoft_full_v3.yaml, once it exists
  ssoft_nohistory: []  # the run of configs/ssoft_nohistory_full_v3.yaml, once it exists
reference_model: qwen3_8_27b
seed: 20261019
resamples: 10000
folds: 5
m: 4
cq_label_exclusion: [example, cause_to_effect, correlation_to_cause]
```

## 10. Acceptance tests

On the diagrams of `schemes/v1/` and `schemes/v2/`, on answers written by hand and on the fake backend; no run and no network:
1. Burden table 2.0: every CQ of `schemes/v2/` has one type and the file names no other CQ; G holds exactly where both arcs reach `good_argumentation`; D only where both arcs end on a fallacy other than the generic one; the routes of section 5.
2. Combined answers: the mass outside the arcs is the mean of the entries not left out, and an `invalid` entry without masses changes no combined answer. With five identical, coherent entries, the hard combined answer is that answer for majority and A4 with weights per CQ for any weights and shrinkage, for stacking at its centre with intercept 0, and for Dawid and Skene with every confusion matrix better than chance and a non-extreme prior. Fitted stacking and Dawid and Skene need not preserve unanimity; no override is applied. A4 with λ = 0, with the signal of the distribution of each entry, gives the split of A4 within 1e-12, also when the entries put different masses outside the arcs and when one entry is left out. Stacking with the coefficients at β⁰ gives the logarithmic pool with equal weights. On data where one entry equals the label and the others are noise, stacking (b) puts the largest weight on that entry. On a scheme where the verdict depends on one CQ, stacking (c) gives the gold verdicts a larger mean log mass than β⁰.
3. Out of fold: changing the label of an item, gold or CQ, changes no verdict of that item, in C1, C2 and C3.
4. Dawid and Skene: the same input gives the same output; on data drawn from known confusion matrices, the estimates come close to them.
5. Routing: equal verdicts give the diagram; τ = −2 gives the zero-shot on every item where the two disagree, τ = 2 the diagram; the share explained by the diagram does not decrease with τ.
6. Tree: depth at most 3; one tree per scheme; on data where one CQ separates the classes, the tree splits on it.
7. S-soft: with a fake model whose answers do not depend on the history, S-soft equals `A2.pruned` within 1e-12; terminals, abstention and pruned bucket sum to 1; the routing of P, E, G and D as in A2; a node reached by two paths gets two calls; the pruning threshold holds.
8. S-soft prompt: rendering is deterministic; every CQ of the scheme is listed word for word; the history lists the CQs of the path in order, with the answers of the arcs and no justification; two histories give two cache keys.
9. Stage `ssoft` on the fake backend: the expansion ends; the number of calls equals the count made by hand for a fake model with fixed masses; a run interrupted and resubmitted completes without duplicate lines and without repeating a call. With `history: false`, the plan has one call per CQ of the scheme of each item, no prompt has the section on the earlier questions or its sentence, and the traversal of the answers equals that of `A2.pruned` on the same answers within 1e-12.
10. Conformal sets: on scores written by hand, the threshold is the order statistic of section 7.
11. Deciding comparisons: the percentiles follow m; the second comparison of a component is marked as not read when the first does not pass.
12. The same seed gives the same outputs.

## 11. What not to do

* No change to `schemes/v1/`, `schemes/v2/`, the interpreter, the loader, the aggregators of round 1 or `evaluations/decision_v2/`.
* No change to a primary configuration after a result: a new idea becomes an exploratory configuration, named as such.
* No CQ label in (a) and (c), no gold verdict in (a) and (b); no verdict that depends on the labels of its own item.
* The new human annotations (direct verdict, re-annotation of schemes 1.1, annotation of schemes 2.0) enter no primary configuration.
* No label or scheme name written in the code.
* No job launched on the cluster.

## 12. Documents to align in the same commit

* README: the components of round 2, the stage `ssoft`, `configs/ssoft_full_v3.yaml`, `configs/ssoft_nohistory_full_v3.yaml` and `configs/evaluation_round2.yaml`.
* Spec 03: a section on `ssoft_v1`, after the one on `judge_v1`.
* Spec 04: the stage `ssoft`, its planning during the run and the field `path`.
* Spec 05: the table `ssoft.csv`.
* Spec 08, P10: round 2 has its own primary configurations and m, in this spec.
* `pyproject.toml`: scipy among the dependencies.

## 13. Definition of done

Lint and tests green, with the commands of the README. `schemes/burden/v2.yaml` in place. C1, C2, C3 and C5 computed on full run v2 by `argfallacy evaluate configs/evaluation_round2.yaml`, with the exploratory configurations on full run v3. S-soft ready: `argfallacy run plan configs/ssoft_full_v3.yaml` counts 601 first calls, and `argfallacy run plan configs/ssoft_nohistory_full_v3.yaml` counts 3,162 calls; C4 and its control are computed once their runs exist. Summary in Italian.

## References

* Angelopoulos, A. N. and Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
* Balasubramanian, K., Podkopaev, A. and Kasiviswanathan, S. (2026). Dependence-aware label aggregation for LLM-as-a-judge via Ising models. *ICML*, PMLR 306, 5748–5785.
* Dawid, A. P. and Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Journal of the Royal Statistical Society, Series C*, 28(1), 20–28.
* Genest, C., Weerahandi, S. and Zidek, J. V. (1984). Aggregating opinions through logarithmic pooling. *Theory and Decision*, 17(1), 61–70.
* Gordon, T. F., Prakken, H. and Walton, D. (2007). The Carneades model of argument and burden of proof. *Artificial Intelligence*, 171(10–15), 875–896.
* Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, PMLR 70, 1321–1330.
* Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B. and Liang, P. (2020). Concept bottleneck models. *ICML*, PMLR 119, 5338–5348.
* Margeloiu, A., Ashman, M., Bhatt, U., Chen, Y., Jamnik, M. and Weller, A. (2021). Do concept bottleneck models learn as intended? arXiv:2105.04289.
* Singh, C., Morris, J., Rush, A., Gao, J. and Deng, Y. (2023). Tree prompting: efficient task adaptation without fine-tuning. *EMNLP*, 6253–6267.
* Smith, J. and Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3), 331–355.
* Wang, T. (2019). Gaining free or low-cost interpretability with interpretable partial substitute. *ICML*, PMLR 97, 6505–6514.
* Wang, T. and Lin, Q. (2021). Hybrid predictive models: when an interpretable model collaborates with a black-box model. *Journal of Machine Learning Research*, 22(137), 1–38.
* Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
* Zhou, D. et al. (2023). Least-to-most prompting enables complex reasoning in large language models. *ICLR*.
