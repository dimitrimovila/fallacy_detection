# Research findings and confirmed decisions

Updated 30 October 2026. This is the public, readable record of the evidence accumulated in this project. It consolidates saved evaluations, diagnostic reports and previously unrecorded case discussions; it introduces no new experiment or metric calculation. It is intended to make the scientific reasoning recoverable without inspecting all result tables or conversation histories.

## How to read and maintain this record

**Measured** identifies a reported calculation or saved execution check. **Interpretation** identifies what the evidence plausibly supports. **Hypothesis** identifies an explanation not established by those results. **Proposed** identifies a possible action without a confirmed decision. **Confirmed decision** is reserved for an explicitly agreed choice. A proposal, an implementation's existence, or an unchallenged recommendation is not a decision. A decision and its execution status are separate facts.

For every material new finding, record the question, date, population, scheme/prompt version, systems and comparator, result and uncertainty where available, interpretation, limitations, canonical evidence location, and consequence for the next question. Record verified findings without waiting for a methodological decision. Before recording a new methodological decision as confirmed, obtain explicit agreement on that specific choice; existing explicit agreement suffices. Keep an unresolved recommendation marked Proposed. If a later finding changes an interpretation, retain the earlier result and identify the superseding interpretation. Update affected overview documents in the same task.

This record is versioned. The machine-readable `evaluations/`, raw `runs/`, scratch files, external workbooks and detailed local diagnostic ledgers are not published with it. Evidence paths below are **local artifact identifiers**, written as code rather than broken public hyperlinks. `workspace:` means relative to the parent thesis workspace, outside this repository. Public readers can assess the reported reasoning, but cannot reproduce the calculations from this document alone. See [data provenance](data.md) and the [evaluation protocol](specs/08_evaluation.md) for the versioned contracts. A local manifest remains authoritative for the exact inputs of its own run; it does not automatically supersede later scientific interpretation.

### Reading order

The organization is thematic, following the research progression rather than the order in which files or conversations were inspected. It is a findings register, not a complete chronological experiment log:

| Sections | Scientific question |
| --- | --- |
| F01–F03 | How were model regimes chosen, what happened in round 1, and how reliable are CQ confidence signals? |
| F04–F06 | Did learned combinations, sequential questions or D+Q improve the verdict? |
| F07–F11 | What do human answers, substitutions, collisions and initial learned mappings reveal about the bottleneck? |
| F12–F15 | What changed with 2.0, and how do graph structure, gold provenance and individual cases affect interpretation? |
| F16–F17 | Which analyses actually completed, what remains unreproduced, and what did C3@2.0 establish? |
| F18 and decision table | Which directions remain open, and which choices have actually been confirmed? |

These identifiers are stable so later corrections can cite them. Evidence lists deliberately point to detailed local records rather than copy every secondary table. Recovered observations with an unresolved denominator or source convention remain explicitly qualified; they must not be silently promoted to frozen-benchmark results.

## Current state and scientific objective

The objective is a fallacy verdict computed from Walton critical-question answers by understandable, inspectable and argumentatively justified rules. Predictive improvement, competitiveness with other systems, numerical faithfulness of an explanation, and validity under Walton are distinct outcomes. A path reproducing the program is not automatically a sound argumentative justification.

The reference task generally supplies the gold scheme. It is therefore conditional fallacy classification, not a demonstrated end-to-end system with predicted schemes. There are 601 frozen test arguments, eight schemes and 23 gold-supported fine verdict classes, including Good Argumentation. Class support is very uneven. The earlier 607 rows contain 606 distinct texts; five near-duplicate pairs reduce the reference to 601. The full frozen item table has 773 items. Canonical scoring and historical compatibility scoring must not be mixed.

Completed work includes pilots; five-model isolated-CQ runs on 1.1 and 2.0; direct verdicts; judge variants; round-1/round-2 aggregators; Qwen S-soft and its no-history control; Qwen D+Q with matched direct controls; saved-result analyses; human-CQ and graph diagnostics; and C3@2.0. Existing output directories do not imply that every optional analysis has been executed.

**Confirmed direction, 30 October:** prioritize trying structure-learning algorithms on existing CQ outputs. Another manual redesign or new annotation campaign is not a prerequisite. The existing 2.0 diagrams remain an important candidate representation, not a demonstrated universal improvement over 1.1. The next algorithm, encoding, runtime, search budget and protocol have not yet been agreed. GOSDT, DL8.5, constrained structures, rule lists, concept bottlenecks and CQ prediction remain alternatives, not an obligatory architecture.

## F01 — Pilot choices and model operating regimes

**Measured, pilot phase.** Within the 50-item pilot, the reasoning ablations use the same 163 annotated CQ–argument pairs. Qwen reasoning-on/off agreement is 0.810/0.767 (exact paired McNemar p = 0.230) and Gemma 0.779/0.755 (p = 0.481). The predefined rule retained reasoning off. These results do not establish equivalence or that reasoning never helps. The later K2 comparison favored high over low effort: 127/163 versus 111/163 matches with the human answers, with 22 versus 6 discordant pairs (p = 0.004). The recorded low-effort reference in the deciding comparison uses logprob argmax; the historical supplementary comparison against low-effort sample majority gave p = 0.043. These are pilot CQ-agreement results, not 601-item verdict scores or independent CQ truth labels.

Early pilots also exposed almost constant affirmative answers on some small scheme samples and frequent missing-information justifications: a recorded check found 223 IDK answers, 200 justified by missing information. This motivated explicit conventions about reasonable reconstruction, available knowledge and applicability. Those changes are methodological provenance, not evidence of a later accuracy gain in isolation. K2 low-effort empty reasoning in 1,740 calls was investigated as model/template behavior rather than assumed to be a client failure.

**Confirmed operating choices:** Qwen and Gemma without reasoning, one deterministic sample; GPT-OSS with five samples; K2 high effort with one sample. Exact temperatures, revisions and serving arguments belong to each manifest and `serving/models.yaml`.

Evidence: local pilot reports under `runs/`; local `docs/proposals.md`, `docs/changelog.md`; versioned `serving/README.md` and `docs/specs/04_client_serving_pilot.md`. The pilot score convention is CQ agreement with the reference annotations, not an independent proof that those annotations are correct.

## F02 — Round 1: soft aggregation did not solve the verdict gap

**Measured, 21 October; schemes 1.1, 601 arguments.** Qwen A0 has Accuracy 0.5042 and Macro-F1 0.3503. Its original direct-verdict condition has 0.6256/0.4778. The judge reading answers and justifications has 0.6290/0.4811. A1–A4 do not establish their intended improvements over A0 under the original deciding comparisons. The round-1 family uses five deciding comparisons and 99% intervals.

The judge improves Macro-F1 over A0 by about +0.131, 99% CI [0.030, 0.203]. Its difference from the direct verdict is about +0.003, ordinary 95% CI [-0.032, 0.064]. **Interpretation:** the judge recovers predictive performance without an established advantage over direct classification. The latter interval does not demonstrate equivalence. A judge's readable rationale does not establish that the verdict was derived faithfully from the CQ answers.

Answers-only judge variants score 0.445 Macro-F1 on 1.1 and 0.465 on 2.0; the 2.0 judge with justifications scores about 0.466. These are contextual variants, not additional independent successful primary tests.

Evidence: `evaluations/decision_v2/{main_table.csv,deciding.csv,summary.md}`; `evaluations/_archive/judge_audit_20261021/`; local thesis plan section 4.

## F03 — Confidence ranking, calibration and CQ agreement are different

**Measured, 1.1 CQ diagnostics.** On 2,656 selected binary human–model pairs, agreement is 0.7775 for Qwen, 0.7748 for Gemma, 0.7669 for K2, 0.6796 for GPT-OSS 20B and 0.6849 for GPT-OSS 120B. The five-model majority is about 0.769. Selecting the best model separately for each CQ on the same sample reaches about 0.847, an optimistic in-sample diagnostic, not a deployable validated selection rule.

For predicting agreement with the human answer, Qwen answer-logprob confidence has AUROC about 0.775, with 18% of values at least 0.99; Gemma has AUROC about 0.789 despite 96% at least 0.99. Thus a badly scaled signal may still rank answers usefully. K2 logprob is effectively 1 throughout, AUROC about 0.463. The proposed mechanism is that its reasoning settles the answer before the scored answer token. GPT-OSS mean logprob is almost identical to sample frequency (reported correlation 1.00). Verbal confidence AUROCs are roughly 0.58–0.67 and were the more useful signals for K2/GPT-OSS; sample-frequency AUROCs are about 0.62 and 0.58 for the two GPT-OSS sizes.

**Interpretation:** confidence sources must be chosen and calibrated deliberately. Agreement with one annotator is the diagnostic target here; these AUROCs do not measure final-verdict accuracy or absolute CQ correctness.

Do not infer the primary evaluation's chosen source from this later signal ranking. The saved round-1 manifest uses logprob for Qwen/Gemma, sample frequency for both GPT-OSS entries, and verbal confidence for K2. A retrospective comparison of AUROCs does not silently change those systems.

Evidence: `evaluations/decision_v2/cq_report.md`; local thesis plan section 4; `docs/specs/07_aggregators.md`.

**Recovered earlier annotation discussion, not recalculated on the frozen evaluation cohort.** The earlier review reported Popular Opinion CQ3 with 56 SUBJ uncertainty codes; Expert Opinion CQ1 with 46 AMB, 18 SUBJ and six DIFF; Analogy CQ1 with 26 SUBJ and CQ4 with 21 DIFF. Popular Opinion tradition/nature subquestions had five and two affirmative answers among 115 non-NA responses. These historical counts describe the reviewed annotation sample, whose selection is not established here as identical to the 2,656-pair cohort. They motivate investigating wording and applicability, not deletion or a claim about intrinsic CQ quality. A small affirmative count can reflect a rare phenomenon. Proposed wording distinguished representativeness from sample size and excessive breadth; a single case is not automatically a fallacious generalization. These were proposals, not newly approved changes to 2.0.

## F04 — Round 2: better CQ answers need not yield better verdicts

**Measured, 25 October; 601 arguments.** The four original deciding comparisons use 98.75% intervals: C1 versus Qwen A0, Macro-F1 delta about -0.053 [-0.114, 0.023]; C2 versus direct verdict, -0.003 [-0.022, 0.015]; C3 versus C1.hard, about -0.100 [-0.175, 0.017]; S-soft versus A2.pruned, -0.065 [-0.122, -0.003]. Follow-on tests governed by the original hierarchy were not interpreted when their first comparison failed.

C1 stacking improves selected binary CQ agreement to about 0.800; CQ-specific weighted combination reaches 0.809 versus Qwen 0.778. Yet C1's final Macro-F1 is 0.2975 versus A0 0.3503, at similar Accuracy 0.5008 versus 0.5042. Good predictions become more frequent and mean recall over classes with fewer than 20 gold items falls from about 0.41 to 0.30. **Interpretation:** local agreement gains and the downstream mapping can interact adversely. **Hypothesis:** imbalance, source differences, applicability and model calibration may contribute; no unique cause has been established.

C2's saved curve illustrates a predictive tradeoff: increasing the share of verdicts selected from the diagram to roughly 70%, 80%, 90% and 100% yields Macro-F1 about 0.444, 0.399, 0.344 and 0.298. Actual achieved shares are discrete. Routing selects which system decides; it does not make a direct model's internal reasoning a faithful CQ derivation.

Evidence: `evaluations/round2/{main_table.csv,deciding.csv,summary.html}`; `docs/specs/09_aggregators_round2.md`. These historical comparisons must not be substituted with C3@2.0 results.

## F05 — S-soft control and a corrected interpretation

**Measured.** Qwen S-soft on 2.0 reaches Accuracy 0.5258 / Macro-F1 0.3415 versus 0.5591/0.4068 for its matched A2.pruned reference. The no-history control permits two further ordinary 95% comparisons: S-soft minus control is -0.009 [-0.036, 0.018]; control minus A2.pruned is -0.057 [-0.094, -0.011].

**Interpretation:** the loss is already present in the condition exposing the full CQ list without earlier answers. The initial hypothesis that early answers propagate errors is not established by these comparisons. The history interval contains zero, so it does not establish that history has no effect. Earlier wording that definitively attributed the whole loss to the list is superseded by this qualified interpretation.

**Confirmed decision:** discontinue further S-soft experimentation; retain the implementation, completed runs and controls for reproducibility.

Evidence: original `evaluations/round2` and the S-soft/no-history manifests listed in `configs/evaluation_round2.yaml`.

## F06 — D+Q improves CQ agreement without an established verdict gain

**Measured, completed before 30 October; Qwen, matched direct controls, 601 arguments.**

| Schemes | D+Q Macro-F1 | Matched direct Macro-F1 | Difference, ordinary 95% CI |
| --- | ---: | ---: | --- |
| 2.0, original primary D+Q comparison | 0.4463 | 0.4713 | -0.025 [-0.062, +0.033] |
| 1.1, control | 0.4679 | 0.4748 | -0.007 [-0.043, +0.054] |

On the 2,656 binary 1.1 human-reference pairs, agreement rises from 77.75% with isolated questions to 82.68% with D+Q, about +4.9 percentage points, ordinary paired 95% CI about [3.6, 6.2] points. The declared D+Q verdict agrees with its CQ-derived 2.0 diagram verdict in 74.2% of items. That is path consistency, not verdict accuracy or causal faithfulness. The full prompt contains the CQs; their answers do not mechanically determine the declared verdict.

**Interpretation:** D+Q gives no demonstrated verdict advantage over its matched direct condition. This is not evidence of equivalence. Better agreement on CQ answers and incomplete consistency with the declared verdict can coexist. A later routing combination reaches about 0.496 Macro-F1 but has an uncertain advantage and was not the primary D+Q test.

Evidence: `evaluations/round2/dq_{main_table,deciding,cq_level,consistency,exploratory}.csv`; `workspace:reports/call_prof_2026-10-27.html`. Use original round2 as the selected source; a later recomputation changed some confidences despite identical verdicts (F16).

## F07 — Human annotations improve the fixed diagram but do not remove its errors

**Measured, 29 October diagnosis; 1.1.** Human annotations on the positive gold-scheme sheet give 476 complete vectors. Of these, 438 have resolvable diagram paths; 430 are arc-admissible throughout the entire vector. These are different cohorts. The primary matched comparison uses the same 438 arguments for every system.

| CQ answers through diagram 1.1 | Correct / 438 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Human | 238 | 0.5434 | 0.5198 |
| Qwen | 199 | 0.4543 | 0.3673 |
| Gemma | 150 | 0.3425 | 0.2394 |
| GPT-OSS 20B | 97 | 0.2215 | 0.1860 |
| GPT-OSS 120B | 138 | 0.3151 | 0.2647 |
| K2 | 140 | 0.3196 | 0.2932 |

Human minus Qwen Macro-F1 is +0.1525, ordinary paired 95% CI [0.0834, 0.1949]; the Accuracy difference is +0.0890 [0.0355, 0.1406]. The aggregate human–other-model intervals are also positive. These bootstrap intervals condition on the saved predictions and omit retraining variability and multiplicity correction.

There remain 200 resolved human errors and 38 unresolved paths. In 35 unresolved cases, Cause to Effect reaches CQ2=NA because the question presupposes evidence that is absent; their golds include 27 Good, six cherry picking and two false premise. The other three stops involve indeterminacy. Unresolved paths are not 38 adjudicated wrong argumentative conclusions. Coding NA as no would change the meaning of the annotation.

Human verdict agreement also differs markedly by annotator pair: reported alpha/raw agreement are 0.83/0.89 for Enrico–Eleni (89 items), 0.49/0.55 for Enrico–Dumitru (483), and 0.32/0.49 for Eleni–Dumitru (89). These do not have identical cohorts or independent gold provenance. Some saved human-path rows measure agreement with the human-derived verdict rather than accuracy against gold; read the reference column before quoting them.

Evidence: `evaluations/analyses_v1/human_path.csv`; `workspace:reports/Diagnosi CQ e diagrammi.md` sections 1–2; `workspace:research_notes/Diagnosi CQ e diagrammi/` matched comparisons and provenance.

## F08 — CQ substitution has non-additive downstream effects

**Measured, diagnostic interventions on saved outputs.** Replacing available admissible human-reference answers within Qwen's 601-item grid changes 182 verdicts: 90 wrong-to-correct and 51 correct-to-wrong. Accuracy rises from 0.5042 to 0.5691; Macro-F1 from 0.3503 to 0.4797. The saved delta intervals are [0.0250, 0.1032] and [0.0674, 0.1680]. This remains a hybrid system: 125 arguments lack the selected human vector and 128 human responses are outside the arcs. It is not the all-human 438-item system.

The five-model extension gives positive point differences for all models, but individual corrections can hurt: replacing Example CQ2 loses seven correct verdicts without gains. Analogy CQ2 disagrees on 58/81 pairs yet changing it alone changes no verdict, because earlier gates block it. Replacing Analogy CQ1 and CQ2 jointly changes 23 verdicts, 13 untouched by either intervention alone. The sum of individual Qwen gains is 27 correct versus 39 for full substitution.

**Interpretation:** answer agreement, reachability, class support and interactions jointly matter. Neither a low marginal influence nor statistical redundancy justifies deleting a CQ. These interventions are diagnostic, not a decomposition into independent causal contributions or Shapley values. The substitution curve's bands summarize 50 sampled correction subsets, not argument-bootstrap confidence intervals.

Evidence: `evaluations/round2/substitution*.csv`; `workspace:reports/Diagnosi CQ e diagrammi.md` section 3; five-model substitution and Qwen pair-interaction ledgers in the corresponding research_notes folder.

## F09 — Identical human CQ vectors can have different gold labels

**Measured.** Among 476 complete human vectors, 84 distinct scheme/vector combinations occur; 39 are conflicting and contain 397 arguments. Assigning each observed vector its sample-majority label can reproduce at most 336/476 = 70.59% of those gold labels. On the 430 fully admissible vectors, 36 conflicting groups contain 358 items and the analogous ceiling is 300/430 = 69.77%.

**Interpretation:** a deterministic function of those fixed inputs cannot distinguish identical vectors. This is a retrospective empirical Accuracy ceiling on that cohort, not a Macro-F1 bound, a population limit, or a prospective validation score. Conflicts can reflect representation, missing context, annotation conventions, or gold taxonomy; they do not by themselves prove that Walton's questions are wrong. The result is not solely an artifact of NA/IDK encoding.

Evidence: human-vector collision tables and members under `workspace:research_notes/Diagnosi CQ e diagrammi/`; report section 4.

## F10 — Human-only diagnostic learners and CQ complementarity

**Measured, nested OOF diagnostic probes.** On the same 438 resolved items, balanced logistic regression reaches Accuracy/Macro-F1 0.4475/0.4383 and CART 0.4703/0.4605 versus manual-human 0.5434/0.5198. Their paired Macro-F1 differences are -0.0814 [-0.1278, -0.0004] and -0.0593 [-0.1142, +0.0307]. Both beat a training-majority-per-scheme baseline in Macro-F1 (about +0.230 and +0.252), showing predictive information in the CQs without beating the manual mapping.

On all 476 items, unresolved paths remain unclassified for manual scoring: manual Macro-F1 0.5098 at coverage 92.02%, logistic 0.4632 and CART 0.4633 at full coverage. These diagnostic learners can predict training-observed labels outside manual terminals, so their task is not identical to a terminal-constrained Walton system. Small per-scheme cohorts, imbalance and tuning instability limit conclusions about other learners.

The 48-CQ information/ablation study finds complementarity as well as apparent redundancy. Analogy CQ1.1 removal hurts both learners; Ad Hominem CQ2/CQ4 jointly contribute information; Example CQ2.3 removal improves fitted performance despite adding conditional information with CQ2.2. Cause CQ2.1/CQ2.2 equality is dominated by NA on 42 vectors; Correlation has only nine complete vectors. Constant answers or perfect dependence on these samples do not establish semantic equivalence or dispensability. These findings concern 1.1: Expert CQ3 here is not Expert CQ3 in 2.0 (F15).

Evidence: human learning manifest, metrics, ablations and information tables in the same research_notes folder; report sections 5–6. Mutual-information estimates are descriptive; sparse conditioning and singleton groups limit them.

## F11 — Historical C3 is one learner, not a test of all learned diagrams

**Measured, 1.1.** Historical C3 uses one balanced Gini CART per scheme, probability of the first answer plus outside-arc mass per CQ, depth {1,2,3}, minimum leaf {1,3,5}, inner Macro-F1 selection and nested C1 construction. On 601 items it reaches 0.4393 Accuracy / 0.2550 Macro-F1 versus C1.hard 0.5025/0.3555. The Macro-F1 difference is -0.1005, adjusted 98.75% CI [-0.1753, +0.0170].

Depth 3 is selected in 25/40 scheme/fold fits; 301/601 saved paths use outside mass and 167/601 repeat a CQ. The original export lacks complete fitted trees and inner/train scores. **Hypotheses:** search limits, representation, small samples or overfitting may contribute; none is proven by these counts. A repeated probability threshold can define an interval rather than literally ask the same question twice.

A counterexample to blanket CQ insufficiency is the post-results A3.combined variant: majority-combined LLM inputs with balanced logistic regression give 0.5374/0.3915 versus 0.4309/0.2876 for hard traversal of those same inputs; paired Macro-F1 delta +0.1040, ordinary 95% CI [0.0373, 0.1766]. This is a post-results finding, not a newly confirmed advantage or proof that an optimal tree will recover it.

Evidence: original round2, spec 09 section 14, and human diagnosis section 7. Do not substitute Qwen A0 for the same-input C1.hard comparator.

## F12 — Version 2.0 changes help unevenly

**Measured, matched 601 arguments per model and aggregator.**

| A0 model | Macro-F1 1.1 → 2.0 | Paired difference, ordinary 95% CI |
| --- | --- | --- |
| Qwen | 0.3503 → 0.4028 | +0.0525 [-0.0099, +0.1045] |
| Gemma | 0.2419 → 0.2614 | +0.0195 [-0.0159, +0.0561] |
| GPT-OSS 20B | 0.1828 → 0.2943 | +0.1115 [+0.0361, +0.1775] |
| GPT-OSS 120B | 0.2583 → 0.2752 | +0.0169 [-0.0338, +0.0607] |
| K2 | 0.2897 → 0.2502 | -0.0395 [-0.0847, +0.0417] |

Qwen A0 correct counts increase 303→328: 94 errors corrected and 69 introduced. Cause to Effect improves 7→30/45 correct, while Example falls 38→26/86, Macro-F1 delta -0.1134 [-0.2199, -0.0121]. The total net gain of 25 equals both the gain in correct Good predictions (25→50) and the net NLAS gain (79→104/197). These are intersecting breakdowns, not two independently additive explanations. Outside NLAS, total correct remains 224/404, despite individual corrections and regressions. Scheme and class Macro-F1 can change even when net Accuracy is unchanged.

All 124 Good golds are from the scheme-origin datasets: 80 NLAS and 44 EthiX. On fallacy-origin data, the direct model has a clearer advantage over diagram 2.0 than on scheme-origin data. **Interpretation:** source, scheme and gold construction interact; 2.0 is not universally superior. The comparison changes wording, answer definitions, ordering and sometimes terminals together, so it cannot isolate graph topology as the cause.

The saved comparison with the original Qwen direct baseline also shows complementarity: on 601 items, 65 are correct only for A0@2.0 and 113 only for the direct model. Their labels disagree on 245 items, including 67 where both are wrong. An oracle choosing the correct member of these two saved predictions would get 441/601 right. That is an in-sample diagnostic Accuracy bound for selection between this fixed pair, not an achieved routing score, a Macro-F1 ceiling, or a bound on a system allowed to predict a third label. The ordinary intervals in the earlier direction audit differ slightly from the later paired audit because their resampling specifications differ; retain each interval with its own source rather than treating them as contradictory estimates.

Evidence: `evaluations/decision_v2`, `evaluations/exploratory_aggregators_v3`; `workspace:reports/Diagnosi diagrammi 2 e prospettive.md` sections 1–2 and its paired tables; historical direction audit under `evaluations/_archive/`.

## F13 — Structural, error and CQ-transfer diagnosis on 2.0

**Measured graph audit.** Ad Hominem and Analogy graphs are unchanged between versions. Example changes only its CQ1=yes terminal from false attribution to false premise; its representativeness bypass remains. Other graphs change ordering or branching. Every declared terminal is formally reachable, but the same three Popular Opinion/non sequitur golds are unavailable as terminals. Formal reachability does not establish a semantically possible answer combination.

Cause removes an evidence-presupposition gate but replaces it with different criteria; Popular's no branch can bypass sampling questions; Expert's new consensus exit precedes attribution/inference checks. **Interpretation:** applicability and priority among simultaneous defects remain important. These observations do not constitute approved graph edits.

The saved replay reproduces all 3,005 A0 and 3,005 A2 verdicts across five models. Qwen has 273 A0 errors, 265 A2 errors and a union of 279 affected arguments; A2.pruned has the same 601 verdicts as A2. The detailed Qwen version-comparison report covers 367 arguments erroneous in at least one version. Replay establishes computational consistency, not independent adjudication of gold labels or explanations. A node frequently visited by errors is exposed to disagreement; it is not thereby the cause of those errors.

**Measured explanation-fidelity issue:** for Qwen A2@2.0, seven of 601 winning verdicts differ from the terminal of the single most probable path. A2 sums probability mass from multiple paths. Showing only that path can therefore misrepresent the computation, even if its CQ tests are individually understandable. An A2 explanation must expose the relevant competing/aggregated masses; the most probable path alone is not always its decision rule. This is distinct from whether the winning verdict is correct against gold.

Only 12 textually identical 1.1/2.0 CQs support direct annotation transfer in the audit. Six further mappings are candidates requiring validation, not equivalent merely because their IDs match. On 936 safely mapped binary pairs, Qwen agreement changes 0.7981→0.7906: no general CQ-quality improvement is established. No complete human 2.0 vectors were documented. A 94-item/396-pair screening was proposed, not completed or required before trying algorithms.

Evidence: 2.0 diagnosis sections 2–6 and its graph, mapping and error-ledger manifests; `workspace:reports/qwen_errori_diagrammi_1.1_2.0/resoconto.html`.

## F14 — Gold provenance and the Enrico comparison

**Measured provenance checks.** Gold labels were developed or selected partly alongside the schemes/diagrams. A historical file named after DeepSeek holds the input gold in the relevant column; matching that column is not agreement with independent model predictions. The 601 imported gold labels match that source. The audit traces 307 explicit Enrico records and 34 historical Slippery Slope records, with differing provenance strength. A high match with labels already incorporated into gold cannot independently validate the annotator or the normative correctness of the diagrams.

The 307+34 breakdown concerns the 341 **scheme-origin** test items, not all 601. The 34 Slippery records lack an independent Enrico verdict cell; their historical agreement does not establish either model-generated labels or independent human validation. Nor is a graph hash documented per historical annotation, so these golds cannot all be attributed specifically to the current frozen 1.1 graph.

Another dependency matters for human-verdict comparisons: the historical Dumitru verdict agrees with the recomputed human-CQ path on all 437 common usable cases in the diagnostic audit. It must not be treated as a second independent direct-verdict annotation. Among the 438 resolved paths, the fallacy-origin subset is 180 items (Accuracy 0.5667 / Macro-F1 0.5580) and scheme-origin 258 (0.5271/0.4133); these are selected subsets, not scores on all 260/341 original items.

The later original-versus-Enrico comparison normalizes spelling and deduplicates within sections. It reports:

| Dataset | Label compared | Compared records | Different |
| --- | --- | ---: | ---: |
| NLAS | Scheme | 152 | 2 (1.3%) |
| EthiX | Scheme | 162 | 0 |
| LOGIC | Fallacy | 191 | 17 (8.9%) |
| Reddit | Fallacy | 75 | 0 |
| ElecDebate | Fallacy | 8 | 1 (12.5%) |

The 17 LOGIC differences comprise nine category changes, five refinements of generic false causality and three post hoc→questionable cause changes. The latter are not automatically incompatible. ElecDebate's difference refines generic false causality to post hoc. NLAS's two changes are popular→expert opinion. For NLAS/EthiX the original scheme is inferred from worksheet membership, not a separately verified source annotation. EthiX has 161 distinct texts and Reddit 74; comparisons count section records. Two texts have inconsistent initial labels across sections.

**Recovered provenance warning.** Ad Hominem is excluded because the original versus revised columns cannot be confidently attributed. A recorded comparison of column D (`gold_fallacy`) with an unnamed G column finds 28 matches, 66 differences and four cases with `?` in D, among 98 records. Imported entries called Enrico match D in 98/98 cases. This describes the import and column disagreement, not authorship: it must not be reported as 70.2% Enrico–dataset disagreement without resolving G's origin. Earlier near-perfect agreement claims against incorporated gold are not independent validation.

Evidence: `docs/data.md`; 2.0 diagnosis gold-lineage tables; `workspace:reports/Controllo divergenze Enrico.md`; local recheck scripts and historical 98-column comparison. Two report filenames once cited during review were not found; the existing report above preserves the aggregate results. No workbook cells or frozen labels were changed by this consolidation.

## F15 — Recovered case discussions: scope, applicability and a low-impact CQ

### Schema-NO cases

**Recovered historical observation, not a new adjudication.** A workbook review found 33 items marked schema=NO on the human sheets despite a dataset scheme (explicit labels where available; worksheet membership for NLAS/EthiX). Counts by sheet were Analogy 3, Expert Opinion 6, Ad Hominem 2, Popular Opinion 2, Example 19, Cause to Effect 1. LOGIC/Reddit fallacy-only labels were excluded as insufficient evidence of a gold scheme. “Disagreement” is therefore appropriate; “human annotation error” is not established.

The Example cases illustrate that mentioning a category, empirical study or phenomenon is not sufficient for Argument from Example: individual cases must support a generalization. Some texts may instead invoke expert testimony or causation. Suggested alternative readings were WHO/CDC support (Expert Opinion), benefits of remote work inferred from observations (Correlation to Cause), and electric-car features supporting predicted benefits (Cause to Effect). These are context-limited interpretations, not replacements approved for the dataset.

Two Analogy cases, IDs `fc667929ada47d92` and `5ab9c79508d4cc70`, compare causing death with failing to prevent death. The review first entertained a broad analogy reading, then corrected it: a disanalogy or comparison does not automatically instantiate the operational 1.1 transfer of a property through relevant similarity. Forcing the diagram can produce Weak Analogy for the very difference that the argument intentionally defends. When the property A and required transfer cannot be identified, dependent CQs are NA rather than IDK. Do not preserve the initial forced-path example as an endorsed annotation.

A separate review of literal `none` in the Correlation worksheet found 28 rows / 26 distinct items, not an additional 28 schema-NO items. The main lesson is that “none of these eight schemes” is not “no reasoning.” Practical reasoning, inference from signs and applying a generalization to an individual can fall outside this scheme inventory. Deduction of identity from a shared property is not automatically analogy; a stereotype or observed association alone is not a causal inference. The teenager-credibility case remained ambiguous about Ad Hominem because no rejected argument was available; the missing-context liberals comparison could not support a reconstructed analogy. No confident relabeling or correction campaign was agreed.

The case-level rationale is retained below by historical worksheet row, without reproducing corpus texts or asserting a new annotation:

| Correlation worksheet row(s) | Recovered interpretation and limit |
| --- | --- |
| 2 | Teenager credibility: a premise is present; possible preventive discrediting, but Ad Hominem uncertain without a rejected position. |
| 3 | Incomplete quiz fragment; no recoverable inference. |
| 4, 7, 12, 13, 15, 16, 18–19, 21–23, 30 | Unsupported general assertions, judgments or stereotypes; no identified inference instantiating the eight schemes. Rows 18/19 duplicate one item. Some assertions could serve a scheme in missing context. |
| 5 | Shopkeeper inference from a sign to suspected theft; inferring a cause from an effect is not Cause to Effect. |
| 6 | Favorable appearance supports a purchase proposal: practical reasoning outside this inventory. |
| 8, 20 | Generalization applied to an individual, then a request; not Example's direction from cases to a generalization. Duplicate item. |
| 9 | Stereotypical association supports an expectation, without a causal conclusion. |
| 10 | Methodological flaws undermine evidence for a test; neither case-to-population generalization nor reliance on expert status is established. |
| 14 | Invalid identity inference from a shared property; not automatically analogical transfer. |
| 17 | Inference from clothing as a sign of preference, without a causal claim. |
| 24 | Hashtag/rhetorical question without available supporting argument. |
| 25 | Generalizing presupposition without an argument supporting it. |
| 26 | Claim about a poem without the needed source context. |
| 27 | Multiple implicit claims and religious reference, but no clear expert-based normative inference; higher uncertainty. |
| 28 | Extends an unavailable description; analogy cannot be reconstructed safely. |
| 29 | Stereotypical joke with a sign/recognition structure; humor does not itself imply absence of inference. |


Recovered schema-NO inventory (historical worksheet rows; source dataset and stable ID retained without reproducing texts):

| No. | Human worksheet scheme | Row | Source | Item ID |
| --- | --- | ---: | --- | --- |
| 1 | Analogy | 15 | Ethix | `c6d6a06f6f6e39d0` |
| 2 | Analogy | 26 | Ethix | `fc667929ada47d92` |
| 3 | Analogy | 28 | Ethix | `5ab9c79508d4cc70` |
| 4 | Expert Opinion | 54 | Ethix | `56ed61184509ffa2` |
| 5 | Expert Opinion | 63 | Ethix | `f52d317e726ed40d` |
| 6 | Expert Opinion | 64 | Ethix | `e5d452f6be52cc11` |
| 7 | Expert Opinion | 71 | Ethix | `4cadbd64b2566207` |
| 8 | Expert Opinion | 72 | Ethix | `f24965ce5ead9a36` |
| 9 | Expert Opinion | 74 | Ethix | `097a831e0583649d` |
| 10 | Ad Hominem | 71 | Logic test | `bf13dbd0e3e008d1` |
| 11 | Ad Hominem | 98 | Logic test | `48ebd3f161c137d8` |
| 12 | Popular Opinion | 7 | Ethix | `8dab1ff77b609c7e` |
| 13 | Popular Opinion | 10 | Ethix | `fbbfd63fa69b5575` |
| 14 | Example | 10 | Ethix | `006fc3d81e7b1a67` |
| 15 | Example | 13 | Ethix | `651bcd3ecc151320` |
| 16 | Example | 15 | Ethix | `53e55e300e3ded46` |
| 17 | Example | 20 | Ethix | `8ebbd3b485d0d35c` |
| 18 | Example | 28 | Ethix | `3bb81bd1e7325ef1` |
| 19 | Example | 31 | Ethix | `2e1b56ea711f1d62` |
| 20 | Example | 32 | Ethix | `e176549d0dad79f3` |
| 21 | Example | 40 | Ethix | `3fccbe0c7d4b6f40` |
| 22 | Example | 41 | Ethix | `b5135bc054922198` |
| 23 | Example | 48 | NLAS | `3982639560a85fe5` |
| 24 | Example | 49 | NLAS | `3e518532e2e4f30f` |
| 25 | Example | 50 | NLAS | `f8364c6ed565537e` |
| 26 | Example | 52 | NLAS | `312bfd2ce258caa6` |
| 27 | Example | 54 | NLAS | `8bfef3a2395b3f3d` |
| 28 | Example | 60 | NLAS | `1e7f21304b6b8321` |
| 29 | Example | 62 | NLAS | `612742242a2593ed` |
| 30 | Example | 67 | NLAS | `a5f84f252764b872` |
| 31 | Example | 68 | NLAS | `a795033fb742a703` |
| 32 | Example | 70 | NLAS | `e7d9c6de6611e713` |
| 33 | Cause to Effect | 49 | Ethix | `f4b9b15b3ea2eb00` |

### Expert Opinion CQ3 on 2.0

**Recovered static output inspection.** Among the 78 Expert Opinion arguments, the numbers reaching 2.0 CQ3 with yes were one each for Qwen/Gemma, two for GPT-OSS 20B, zero for K2 and one for GPT-OSS 120B; the latter also had one unresolved response at that node. These are at most potentially affected cases per model if CQ3 were bypassed in the fixed diagram. Continuing to later questions could still yield the same verdict. This was not an evaluated deletion, a measured metric gain, or a claim about soft/learned aggregators. For D+Q/judges/S-soft, removing a question changes context and would require a separate condition; isolated-CQ outputs have different reuse properties.

**Decision status:** removing CQ3 or making a 2.1 diagram was a proposal only. No such change was approved or executed in the audited work. The negative 1.1 Expert CQ3 ablation in F10 concerns a different question/version and is not contradictory evidence.

Evidence: recovered worksheet review and saved-output inspection; local audit `scratch/findings_audit_20261030/` preserves the source history and turn identifiers. These recovered observations were not rechecked by rerunning analyses during consolidation. Raw conversations and corpus excerpts are not part of the public documentation.

## F16 — Completed saved-result analysis and unresolved replay

**Measured execution status.** The real `analyses_v1` saved-input analysis completed and wrote 39 tables. It used original round2 D+Q and historical human annotations; new direct-verdict answer logs and newly drawn sample-sheet responses were not read. Optional empty tables are pending inputs, not zero effects. The pipeline's existence and synthetic acceptance checks are not additional completed real-data findings.

Post-results class-weighting, prior-adjusted aggregators and frequent-class summaries were also retained, rather than becoming new primary successes. The frequent-class endpoint uses 12 classes with at least ten gold examples; the earlier rare-class recall diagnostic uses fewer than 20. These cutoffs answer different questions and are not interchangeable. The remaining eleven supported classes have one to eight examples. Bootstrap intervals over fixed OOF predictions do not include refitting uncertainty; per-scheme stratification does not guarantee rare-class representation in every training fold.

Original round2 and the archived spec-10 recomputation have identical verdicts and main/deciding tables, but base confidence values differ beyond serialization precision (reported maximum absolute differences: C2 0.0014600673485322124; C1 0.000009284341750293024). Exact C2 routing/conformal item exports remain unaccepted because isolated replay did not match the selected saved outputs. Rounded curves cannot recover genuine item-level routing or conformal sets. Preserve original confidences and source hashes; do not substitute a convenient recomputation to obtain agreement.

Human sample sheets were prepared: 111 items for 2.0 CQ annotation and 375 for direct verdict, with recorded seed and disjoint allocation. Their existence does not demonstrate completed answers. Independent human adjudication, predicted-scheme verdict evaluation, and retraining stability remain open. Synthetic resource measurements do not establish the cost of real fitted replay.

Evidence: `evaluations/analyses_v1/execution_audit.json` and manifest; `evaluations/_archive/relocation_manifest.json`; analysis and replay source records; sample manifest outside the repository. Current execution records supersede older implementation-only status paragraphs.

Technical checks also delimit what can be reproduced. A Unicode line-splitting problem in raw JSONL reading required streaming by actual file lines; this is a parser lesson, not a new model result. Near ties can make the generated hard answer differ from the answer with the largest summed token mass, so those two signals must not be silently equated. The local full-v2 raw file is kept on the cluster; its local stage-two summary was reconstructed from saved answer tables. A fresh full parse still requires the raw source. Legacy agreement on 2,416 of 2,428 prior verdicts is a compatibility check on those cases, not proof of general correctness. These issues belong to the client/parser/serving provenance, separate from the reported classifier comparisons.

## F17 — C3@2.0: completed controlled diagnostic

**Confirmed protocol, 29 October.** Compare original C3 with C1.hard on identical transferred C1 features for 2.0; preserve CART/grid and the five historical argument folds with fully nested upstream fitting. Primary endpoint: canonical OOF Macro-F1 on all 601 arguments; Accuracy secondary; ordinary paired 95% intervals; +0.03 as practical-interest threshold. Retain abstentions and unreachable gold in primary scoring, with coverage/common-coverage and schema/class/origin/Good sensitivities. Qwen A0/A2 are context only. Export full fitted trees and inspect numerical and argumentative rules separately. The protocol was fixed before these fits but after substantial benchmark exploration, so this is not independent confirmation.

**Measured, completed 29 October.**

| System | Accuracy | Macro-F1 | Coverage |
| --- | ---: | ---: | ---: |
| C1.hard@2.0 | 0.4942 | 0.3893 | 0.9800 |
| C3@2.0 | 0.4825 | 0.2853 | 1.0000 |
| Qwen A0@2.0, context | 0.5458 | 0.4028 | 0.9900 |
| Qwen A2@2.0, context | 0.5591 | 0.4068 | 1.0000 |

Primary Macro-F1 delta **-0.1040 [-0.1539, -0.0019]**; practical-interest threshold not reached. Common coverage (589): -0.1046 [-0.1600, -0.0023]. Removing Good only from the class average while retaining all texts: -0.1084 [-0.1598, -0.0024]. On 598 reachable-gold items: -0.1012 [-0.1511, +0.0013]. The three unavailable gold labels remain errors for both systems in the primary endpoint. Excluding NLAS yields -0.0534 [-0.1177, +0.0316]. These sensitivities are not independent confirmations or equivalent-effect proofs.

C3 corrects 115 baseline errors and loses 122 successes: 290 versus 297 correct. Expert Opinion and Example show positive local Macro-F1 differences; Ad Hominem and Cause deteriorate strongly. Their inspection does not authorize a hybrid chosen from outer-test predictions. The decline is not confined to Good or to singleton classes.

All 40 outer trees are retained, giving 184 leaf rules; exact exported replay matches all 601 items. Of the rules, 95 use outside mass, 55 repeat a CQ, 57 have fewer than five training arguments, and 67 have no zero-outside Boolean-corner witness. The 172 rules omitting a manual-graph gate are routing-difference flags, not 172 demonstrated semantic violations. Lack of a Boolean-corner witness does not make a probabilistic region impossible.

One actual Expert rule predicts Strawman from high CQ5.1 outside mass with support three, without a source-attribution test. This numerical rule does not by itself establish misrepresentation. Other rules use dependent CQ5.1 without establishing CQ5's prerequisite answer. Export fidelity is established; independent adjudication of every rule and repeat-fit stability are not.

**Interpretation:** this transferred C1/CART pipeline fails to improve its matched baseline and has lower point scores than the saved Qwen contextual references. Those context rows were not additional primary paired tests. The result does not rule out all optimized trees, other encodings, direct model CQ inputs or other rule architectures. Thresholding, outside mass and coverage differ as well as topology. An earlier recommendation to require manual semantic screening before any further learner is superseded by the confirmed 30 October direction.

Evidence: `evaluations/c3_schemes2_diagnostic_20261029/{protocol.md,summary.md,review.md,manifest.json,metrics.csv,differences.csv,rules.json}`. Its separate failed-preflight sibling is not an experimental result. The C3 implementation and some analysis improvements may be local changes not yet included in the published code; publishing this account does not assert they are all committed.

## F18 — Alternatives, feasibility and unresolved choices

**Proposed.** A controlled globally optimized tree experiment could compare GOSDT with a fresh CART on identical Boolean CQ predicates and comparable complexity, alongside the version-matched manual baseline. Old continuous C3 alone would not isolate the solver effect. Encoding thresholds, selection, class weighting, stopping/timeout policy, budgets and nested upstream isolation must be specified before fitting. Weighted classification loss plus leaf penalty is not a claimed native Macro-F1 training objective.

The methodological review also distinguishes library capability from a demonstrated experiment: standard CORELS uses a binary target, so a reduction would change the fine multiclass question; GOSDT/DL8.5 capability does not itself impose Walton prerequisites. A timeout incumbent must not be called a certified optimum without the corresponding termination/gap evidence. Transforming probabilities into deterministic threshold tests does not propagate uncertainty down branches, and using the same CQ twice does not create independent evidence. These are design constraints from the prior methods review, not results of an installed solver comparison.

A 30 October metadata-only feasibility check found neither GOSDT nor PyDL8.5 installed in the active Windows Python 3.13.12 environment. The checked GOSDT 1.0.4 release listed Windows wheels through Python 3.12, and PyDL8.5 0.1.8 older Windows wheels; a separate compatible environment was proposed. This is a dated observation, not current package advice, successful installation or fit. No solver choice or runtime protocol is yet confirmed.

Rule lists, constrained edits, concept bottlenecks and a smaller encoder such as RoBERTa for CQ answering remain open. Each needs a distinct scientific question. External corpora discussed in the dataset survey can provide additional evaluation material, but ordinary fallacy labels do not automatically provide Walton schemes, complete CQ labels or independent normative validation. The few remaining weeks favor controlled reuse of existing outputs, not simultaneous implementation of every candidate.

Evidence: `workspace:reports/Apprendere diagrammi dalle domande critiche.md`; `workspace:reports/Dataset annotati per nuove valutazioni.md`; local roadmap and `scratch/optimal_tree_feasibility_20261030.md`. These are feasibility/research directions, not completed comparisons.

## Confirmed decisions and their boundaries

| Date / phase | Confirmed choice | Reason and execution status |
| --- | --- | --- |
| Pilot/full-run preparation | Preserve chosen model regimes and immutable versioned prompts/schemes per run | Pilot evidence and reproducibility; completed runs retain their original manifests. |
| Original evaluation protocols | Keep canonical scoring, argument-level folds, prescribed references and multiplicity rules | Avoid selecting comparisons after outcomes; round-1/round-2 complete. |
| After S-soft controls | Stop further S-soft experiments | Inferior matched result without established benefit; preserve existing work. |
| Saved-analysis phase | Reuse saved outputs; preserve original round2 confidences | Replay discrepancy; 39-table analysis complete, exact exports pending. |
| 29 October | Freeze and execute C3@2.0 versus C1.hard@2.0 with original algorithm/grid | Controlled first structure-learning diagnostic; completed, negative result (F17). |
| 30 October | Prioritize algorithms on existing CQ outputs; do not require another manual redesign/annotation campaign first | Current research direction; next solver/protocol remains open. |
| 30 October | Consolidate and publish readable evidence; document future findings and only explicitly agreed decisions | This register and aligned local guidance implement that request; no new scientific method selected by documentation alone. |

Not confirmed: deleting Expert 2.0 CQ3; a 2.1 graph; a per-scheme post-hoc hybrid; GOSDT as the chosen next solver; a new grid/encoding; new annotation collection or external evaluation. Historical recommendations to do these things are not authorization.

## Practical reading map

For conclusions and choices, start with this file. For exact local outputs, select the relevant collection:

| Local collection | Purpose |
| --- | --- |
| `evaluations/decision_v2` | Original round 1, CQ diagnostics and five-model/direct references |
| `evaluations/round2` | Original round 2, S-soft controls, D+Q, post-results additions |
| `evaluations/exploratory_aggregators_v3` | Later saved aggregator comparisons on 2.0 |
| `evaluations/analyses_v1` | Cross-round saved-result analysis and execution audit |
| `evaluations/c3_schemes2_diagnostic_20261029` | Completed C3@2.0 protocol, trees, sensitivities and review |
| `evaluations/_archive` | Historical audits and preserved alternative snapshots; relocation manifest resolves old paths |

The two local diagnostic reports and their `workspace:research_notes/` contain the detailed human/graph analyses consolidated in F07–F15. The roadmap organizes alternatives; it is not a substitute for this result/status record. Future updates must keep this map, the local evaluation catalog and strategic overview consistent without rewriting historical measured results.
