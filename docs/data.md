# The data

Log of the project's data. It says where the data come from, what the two files of `data/` contain, what was corrected or removed and why, and how Dumitru's annotations are updated. It replaces spec 02, which described the extraction code now deleted.

## 1. The two files

* `data/items.csv`: one row per item. The text of the argument and the labels it came with.
* `data/annotations.csv`: one row for every annotated cell. Who wrote it, where, what it says.

Both were frozen on 21 September 2026. Since that day no program rebuilds them: they are corrected by hand, recording the correction in this document, or, for Dumitru's rows, with `argfallacy annotations update` (section 6).

## 2. Where they come from

Two sources, both outside the repository, at the paths of the `.env`.

* **Eleni's `results.csv` files** (`PRIOR_RUNS_DIR`): one per model and condition, eight in all. They have the same 607 rows, with the columns `id`, `source`, `text`, `gold_scheme`, `gold_fallacy`. The gold is that of the run `deepseek/pipeline`. The other runs sometimes write it differently (15 cells with `Casual reductionism` instead of `Causal reductionism`), but after label normalization they all coincide.
* **The workbook `Dati_da_annotare.xlsx`** (`WORKBOOK_PATH`), version of 21 September 2026, with row 5 of `Annotazione Dumitru Expert Opin` already corrected. 14 sheets were read: the base sheets `new_analogy`, `new_expert_opinion`, `ad hominem`, `new_ad_populum`, `example_final`, `cause_to_effect`, `hasty_generalization`, `Slippery slope`; Enrico's copies `Copia di new_ad_populum_Enrico`, `Copia di example_final_Enrico`, `Copia di cause_to_effect_Enrico`, `Copia di hasty_generalization_E`; Dumitru's sheets `Annotazione Dumitru Analogy` and `Annotazione Dumitru Expert Opin`. The red herring and worse problems sheets have no scheme and stay out of the project.

The workbook on the Drive remains the shared reference. The repository keeps a readable copy of it, and does not replace it.

## 3. `items.csv`

773 items: 601 from Eleni's test set and 172 that are only in the workbook.

| column | what it contains |
| --- | --- |
| `item_id` | identifier, 16 hexadecimal characters |
| `text` | the text of the argument |
| `source` | the corpus: `logic`, `nlas`, `ethix`, `reddit`, `elecdebate` |
| `in_test_set` | `True` if the item is among Eleni's rows |
| `eleni_id` | the `id` in Eleni's files; two ids separated by `;` when Eleni has the same text twice |
| `gold_scheme` | Eleni's gold scheme, empty outside the test set |
| `gold_fallacy` | Eleni's gold fallacy, empty outside the test set |

By corpus: LOGIC 271 items (192 in the test set), EthiX 178 (144), NLAS 209 (197), Reddit 107 (67), ElecDebate 8 (1).

**The identifier.** At the freeze it was computed from the text: the first 16 characters of the sha256 of the normalized text (Unicode NFC, leading and trailing whitespace removed, internal whitespace squeezed, typographic quotes made plain). It has been fixed ever since. If a text is corrected, the id stays the same. No program recomputes it any more.

**The text.** For the items of the test set it is Eleni's text, word for word: Eleni's models read that one, and the comparison with those results holds only if ours read the same. For the others it is the text of the first sheet that contains them.

**The test set of the experiments** is the set of rows with `in_test_set` true: 601 items. Eleni has 606 distinct ones. The five missing ones are near duplicates (section 5.2). Eleni's text `33f6dfaf9c4e41f1` appears twice, with ids 469 and 495, and is a single item with `eleni_id` equal to `469;495`.

**The 172 items outside the test set** come from the base sheets of the workbook. 170 have a verdict by Enrico and form the second evaluation of spec 08 (P6). The other two are the new texts of Slippery slope (section 5.5).

## 4. `annotations.csv`

4470 rows, one per annotated cell.

| column | what it contains |
| --- | --- |
| `item_id` | the item |
| `annotator` | who wrote the cell |
| `sheet`, `row` | the sheet of the workbook and the Excel row |
| `field` | what the cell annotates |
| `value` | the value read, in canonical form |
| `raw` | the cell as it is written in the workbook |

With `sheet` and `row`, two rows of the same item stay distinct. An item present both in the base sheet and in Enrico's copy has the rows of both.

The file must be read with `argfallacy.annotations.load_annotations`, or with pandas passing `dtype=str, keep_default_na=False`: `na` and `none` are values, not empty cells.

### 4.1 Who wrote what

* `enrico`: the copies `Copia di ..._Enrico` and Enrico's columns inside `new_analogy`, `new_expert_opinion`, `ad hominem`, `new_ad_populum`, `cause_to_effect`. 1932 rows.
* `eleni`: Eleni's columns in `new_analogy` and `new_expert_opinion`. 382 rows.
* `dumitru`: the `Annotazione Dumitru ...` sheets. 1761 rows.
* `base`: the starting label, that is, the fallacy column of the base sheets and of Dumitru's sheets, before Enrico's revision. It is not an annotator's judgement. 372 rows.
* `unknown`: the `doubt` column of `new_analogy` and `new_expert_opinion`, which has no author. 23 rows.

### 4.2 What it annotates, and how to read `value`

* `scheme`. `YES` means the scheme of the sheet, `NO` means `none`, `SNI` and `?` mean `idk`, a scheme name means that scheme (`labels/schemes.yaml`). The dash means a cell not filled in, and produces no row.
* `verdict`. The fallacy, with the canonical id of `labels/fallacies.yaml`. In the column `fallacy Eleni`, `NO` means `good_argumentation` and `YES` means `any_fallacy`, a coarse label satisfied by every fallacious terminal of the scheme (spec 01). In Dumitru's sheets `n.a. (schema assente)` [n.a. (scheme absent)] is not a verdict: the row has the scheme `none` and no `verdict`.
* `starting_label`, only for `base`. The starting label, with the canonical id. One item has two different ones: `4000a372e40713ec` starts as `appeal_to_authority` in Dumitru's sheet on expert opinion and as `ad_populum` in `new_ad_populum`.
* `CQ1`, `CQ2.1` and so on, only for `dumitru`. `yes` and `no` are the answers. `na` means that the diagram does not reach the CQ: in the workbook it is written `-` or `na`, and `n.a.(C=A)` when the conclusion coincides with the assertion. `?` and `idk` mean `idk`. For ad hominem CQ1 the answers are `positive` and `negative`.
* `uncertainty`, only for `dumitru`. The column `incertezza` [uncertainty] as it is written: `CQn:CODE` separated by `;`, with the code `AMB`, `DIFF` or `SUBJ`.
* `exit_cq`, `difficulty` (from 1 to 5), `comment`, for Enrico; `comment` also for Eleni; `note` for Dumitru; `doubt` (`s`, `f`, `x`) for `unknown`. All as they are written.

## 5. What was corrected or removed at the freeze

### 5.1 Two copying errors

Two rows of the workbook have a text slightly different from that of their item. Their annotations are on the right item.

* `Annotazione Dumitru Analogy`, row 86: two long dashes turned into short ones. The item is `97695052f03f5191`, the same as row 13 of `new_analogy`.
* `cause_to_effect`, row 74: the base sheet repeats at the end the sentence "During the time that the vice president and the president have been in office, 4 million more Americans have fallen into poverty.". The item is `74fb573039aeaf58`; Enrico's copy and the test set do not repeat it.

### 5.2 Five near duplicates

Five pairs of almost identical texts are all in Eleni's test set. For each pair `items.csv` keeps a single item, and the annotations of the other one do not enter `annotations.csv`. The two halves always have the same gold, and Enrico, where both were annotated, gave the same verdict. The first in alphabetical order is kept.

| removed | kept | Eleni's ids (removed, kept) | where | what changes |
| --- | --- | --- | --- | --- |
| `689d10927ad646e4` | `3847c80729510e82` | 194, 160 | `new_ad_populum` rows 131 and 77 | `Iphone` versus `iPhone` |
| `b4f5e0db2e57aa4e` | `57df90e878a69523` | 503, 496 | `hasty_generalization` rows 71 and 60 | `Obviously the` versus `Obviously, the` |
| `a1bc855f65cb00ba` | `3fbae950e4e60365` | 175, 186 | `new_ad_populum` rows 100 and 118 | quotation marks, dash, apostrophe, double spaces, `That is` versus `This is` |
| `c536608a8e446a3d` | `579a2e83b3554b69` | 490, 488 | `hasty_generalization` rows 53 and 50 | `all little girls` versus `all girls` |
| `44baba98b841dc30` | `17ea0841b588dc4f` | 531, 512 | `Slippery slope` rows 23 and 2 | `won't get into` versus `can't get into`, double spaces, `class...` versus `class.` |

The reproduction of Eleni's numbers uses Eleni's files as they are, near duplicates included (spec 08, requirements 1 and 2).

### 5.3 Two items removed by Enrico

Enrico marked two rows to be removed from the dataset. Neither of them is in the test set. The items are not in `items.csv`, and none of their annotations is in `annotations.csv`, including Dumitru's at the same row of Dumitru's sheets.

* `10a2cd8c906211ce`, `new_analogy` row 44, `eliminare` [delete] in the column `fallacy Enrico`: "Is it ethically wrong to watch pornography? By the same reasoning we should ban entire industries...".
* `e4a405952d48fdbe`, `new_expert_opinion` row 16, `togliere` [remove] in the column `comment Enrico`: "The word fascism has now no meaning except in so far as it signifies something not desirable", George Orwell.

### 5.4 Rows repeated within a sheet

Some sheets contain the same item twice. Both rows stay in `annotations.csv`, and the annotations coincide. In Enrico's copy of `hasty_generalization` Enrico marked them with the comment `doppia` [duplicate].

* `new_analogy` rows 10 and 90, which in Dumitru's sheet are rows 83 and 90.
* `cause_to_effect` rows 70 and 72.
* `hasty_generalization` rows 7 and 8, 48 and 49, 22 and 58, 16 and 59.

### 5.5 The Slippery slope sheet

The `Slippery slope` sheet arrived on 21 September 2026. It contains the 96 slippery slope items of the test set, with the same gold as Eleni's, plus two LOGIC texts that are not in the test set: `9ede7cb26b5bc4f7` (psychologists and drugs) and `2338a8fbf845d621` (discounts for patients). The `gold_fallacy` column of the sheet is read as a starting label (`base`). The `gold_scheme` column says `Slippery Slope` on every row and adds nothing.

The two new texts have empty `gold_scheme` and `gold_fallacy`, because the gold of `items.csv` is only Eleni's. So they do not enter the pilot.

## 6. How Dumitru's annotations are updated

The workflow:

1. The annotations are made locally, then the columns are pasted on the Drive together with the `item_id` column.
2. When a sheet is finished, it is copied into the local `Dati_da_annotare.xlsx`, the one given by `WORKBOOK_PATH`.
3. `argfallacy annotations update` is run.

The command reads every sheet whose name starts with `Annotazione Dumitru` and replaces in `annotations.csv` the `dumitru` rows of that sheet. The rows of the other annotators and of the other sheets stay as they are.

What it expects from a sheet:

* an `item_id` column, in text format;
* the `text` column;
* the CQ columns, with the same names as in the YAML of the scheme (`CQ1`, `CQ1.1`, ...). The scheme of the sheet is recognised from these: it is the only scheme with exactly those CQs;
* the columns `scheme` (if there are two, the last one counts), `verdetto` [verdict], and if needed `incertezza` [uncertainty] and `nota` [note].

A row with `item_id` equal to `ESCLUSO` [EXCLUDED] is skipped. The command stops without writing anything if it finds an id that is not in `items.csv`, a text without an id, an answer it does not know, or a text different from that of the item. The last check tolerates a copying slip, not a column pasted one row too high or too low.

The `item_id` values to paste are in `item_id_per_foglio.xlsx`, in the `Thesis` folder, outside the repository: one sheet for each of Dumitru's two sheets and of the eight base sheets, with the rows in the same order as in the workbook. Enrico's copies have the same order as their base sheet. The `nota` column flags the `ESCLUSO` rows, the repeated rows and the two copying errors. Whoever prepares a new Dumitru sheet starting from a base sheet pastes the ids right away, before reordering the rows.

## 7. Checks made at the freeze

* Every verdict, scheme and CQ answer of `annotations.csv` coincides with the tables produced by the previous extraction code. The only difference is row 5 of `Annotazione Dumitru Expert Opin`, corrected on 21 September 2026 from `Irrelevant Authority` to `Appeal to Authority`, in agreement with the answer `no` to CQ4.1 of the same row.
* The reader of section 6, run on a copy of the workbook with the `item_id` values inserted, reproduces Dumitru's rows byte for byte.
* The 182 rows of Dumitru with all the CQs filled in and no `idk` give, along the diagram, the verdict written in the sheet. All 182 of them.
* On the 507 items of the test set with a verdict by Enrico, the verdict coincides with Eleni's gold 505 times. The two exceptions: `8fdfc761e4aab007`, gold `slippery_slope`, Enrico `appeal_to_authority` in `new_expert_opinion` row 23; `f1023bd0186f33c5`, gold `ad_populum`, Enrico `hasty_generalization` in `Copia di new_ad_populum_Enrico` row 14.
* Five items have different schemes by Enrico in different rows: `2991ee41f2652b99` and `581504d112994e8f` (popular opinion in the base sheet, expert opinion in the copy), `4000a372e40713ec` (popular opinion and `SNI`), `f1023bd0186f33c5` and `fbbfd63fa69b5575` (popular opinion and expert opinion).

The old `human_cq.csv` misread the `incertezza` column when it contained more than one code: it attributed the whole rest of the cell to the first CQ. `annotations.csv` keeps the whole cell.

## 8. What is no longer there

With the freeze, the extraction code (`src/argfallacy/data/`), the column map `data/workbook_columns.yaml`, `data/item_aliases.yaml`, `data/duplicates.yaml`, `labels/sources.yaml`, the tables of `data/processed/` (`items.csv`, `verdicts.csv`, `human_cq.csv`, `build_report.md`) and the command `argfallacy data build` left the repository. Their useful content is in this file. The code remains in the git history and in the backup archive, and it is the only way to redo the freeze from the workbook.
