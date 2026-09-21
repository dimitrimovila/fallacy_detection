# I dati

Registro dei dati del progetto. Dice da dove vengono, cosa contengono i due file di `data/`, cosa è stato corretto o tolto e perché, e come si aggiornano le annotazioni di Dumitru. Sostituisce la specifica 02, che descriveva il codice di estrazione ora eliminato.

## 1. I due file

* `data/items.csv`: una riga per item. Il testo dell'argomento e le etichette con cui è arrivato.
* `data/annotations.csv`: una riga per ogni cella annotata. Chi l'ha scritta, dove, cosa dice.

Tutti e due sono stati congelati il 21 settembre 2026. Da quel giorno nessun programma li ricostruisce: si correggono a mano, registrando la correzione in questo documento, oppure, per le righe di Dumitru, con `argfallacy annotations update` (sezione 6).

## 2. Da dove vengono

Due sorgenti, tutte e due fuori dal repository, ai percorsi del `.env`.

* **I `results.csv` di Eleni** (`PRIOR_RUNS_DIR`): uno per modello e condizione, otto in tutto. Hanno le stesse 607 righe, con colonne `id`, `source`, `text`, `gold_scheme`, `gold_fallacy`. Il gold è quello del run `deepseek/pipeline`. Gli altri run lo scrivono a volte in modo diverso (15 celle con `Casual reductionism` invece di `Causal reductionism`), ma dopo la normalizzazione delle etichette coincidono tutti.
* **Il workbook `Dati_da_annotare.xlsx`** (`WORKBOOK_PATH`), versione del 21 settembre 2026, con la riga 5 di `Annotazione Dumitru Expert Opin` già corretta. Sono stati letti 14 fogli: i fogli base `new_analogy`, `new_expert_opinion`, `ad hominem`, `new_ad_populum`, `example_final`, `cause_to_effect`, `hasty_generalization`, `Slippery slope`; le copie di Enrico `Copia di new_ad_populum_Enrico`, `Copia di example_final_Enrico`, `Copia di cause_to_effect_Enrico`, `Copia di hasty_generalization_E`; i fogli di Dumitru `Annotazione Dumitru Analogy` e `Annotazione Dumitru Expert Opin`. I fogli di red herring e worse problems non hanno uno schema e restano fuori dal progetto.

Il workbook sul Drive resta il riferimento condiviso. Il repository ne tiene una copia leggibile, non lo sostituisce.

## 3. `items.csv`

773 item: 601 del test set di Eleni e 172 che stanno solo nel workbook.

| colonna | cosa contiene |
| --- | --- |
| `item_id` | identificativo, 16 caratteri esadecimali |
| `text` | il testo dell'argomento |
| `source` | il corpus: `logic`, `nlas`, `ethix`, `reddit`, `elecdebate` |
| `in_test_set` | `True` se l'item è fra le righe di Eleni |
| `eleni_id` | l'`id` nei file di Eleni; due id separati da `;` quando Eleni ha lo stesso testo due volte |
| `gold_scheme` | lo schema gold di Eleni, vuoto fuori dal test set |
| `gold_fallacy` | la fallacia gold di Eleni, vuota fuori dal test set |

Per corpus: LOGIC 271 item (192 nel test set), EthiX 178 (144), NLAS 209 (197), Reddit 107 (67), ElecDebate 8 (1).

**L'identificativo.** Al congelamento è stato calcolato dal testo: i primi 16 caratteri di sha256 del testo normalizzato (Unicode NFC, spazi in testa e in coda tolti, spazi interni compressi, apici tipografici resi semplici). Da allora è fisso. Se un testo si corregge, l'id resta quello. Nessun programma lo ricalcola più.

**Il testo.** Per gli item del test set è il testo di Eleni, parola per parola: i suoi modelli hanno letto quello, e il confronto con i suoi risultati regge solo se i nostri leggono lo stesso. Per gli altri è il testo del primo foglio che li contiene.

**Il test set degli esperimenti** è l'insieme delle righe con `in_test_set` vero: 601 item. Eleni ne ha 606 distinti. I cinque che mancano sono doppioni (sezione 5.2). Il suo testo `33f6dfaf9c4e41f1` compare due volte, con id 469 e 495, ed è un item solo con `eleni_id` uguale a `469;495`.

**I 172 item fuori dal test set** vengono dai fogli base del workbook. 170 hanno un verdetto di Enrico e formano la seconda valutazione della specifica 08 (P6). Gli altri due sono i testi nuovi di Slippery slope (sezione 5.5).

## 4. `annotations.csv`

4470 righe, una per cella annotata.

| colonna | cosa contiene |
| --- | --- |
| `item_id` | l'item |
| `annotator` | chi ha scritto la cella |
| `sheet`, `row` | il foglio del workbook e la riga Excel |
| `field` | cosa annota la cella |
| `value` | il valore letto, in forma canonica |
| `raw` | la cella come è scritta nel workbook |

Con `sheet` e `row`, due righe dello stesso item restano distinte. Un item presente nel foglio base e nella copia di Enrico ha le righe di tutti e due.

Il file va letto con `argfallacy.annotations.load_annotations`, oppure con pandas passando `dtype=str, keep_default_na=False`: `na` e `none` sono valori, non celle vuote.

### 4.1 Chi ha scritto

* `enrico`: le copie `Copia di ..._Enrico` e le colonne di Enrico dentro `new_analogy`, `new_expert_opinion`, `ad hominem`, `new_ad_populum`, `cause_to_effect`. 1932 righe.
* `eleni`: le colonne di Eleni in `new_analogy` e `new_expert_opinion`. 382 righe.
* `dumitru`: i fogli `Annotazione Dumitru ...`. 1761 righe.
* `base`: l'etichetta di partenza, cioè la colonna di fallacia dei fogli base e dei fogli di Dumitru, prima della revisione di Enrico. Non è il giudizio di un annotatore. 372 righe.
* `unknown`: la colonna `doubt` di `new_analogy` e `new_expert_opinion`, che non ha un autore. 23 righe.

### 4.2 Cosa annota, e come si legge `value`

* `scheme`. `YES` vale lo schema del foglio, `NO` vale `none`, `SNI` e `?` valgono `idk`, un nome di schema vale quello schema (`labels/schemes.yaml`). Il trattino vuol dire cella non compilata, e non produce nessuna riga.
* `verdict`. La fallacia, con l'id canonico di `labels/fallacies.yaml`. Nella colonna `fallacy Eleni` `NO` vale `good_argumentation` e `YES` vale `any_fallacy`, un'etichetta a grana grossa soddisfatta da ogni terminale fallace dello schema (specifica 01). Nei fogli di Dumitru `n.a. (schema assente)` non è un verdetto: la riga ha lo schema `none` e nessun `verdict`.
* `starting_label`, solo per `base`. L'etichetta di partenza, con l'id canonico. Un item ne ha due diverse: `4000a372e40713ec` parte come `appeal_to_authority` nel foglio di Dumitru su expert opinion e come `ad_populum` in `new_ad_populum`.
* `CQ1`, `CQ2.1` e così via, solo per `dumitru`. `yes` e `no` sono le risposte. `na` vuol dire che il diagramma non raggiunge la CQ: nel workbook si scrive `-` oppure `na`, e `n.a.(C=A)` quando la conclusione coincide con l'asserzione. `?` e `idk` valgono `idk`. Per CQ1 di ad hominem le risposte sono `positive` e `negative`.
* `uncertainty`, solo per `dumitru`. La colonna `incertezza` come è scritta: `CQn:CODICE` separati da `;`, con codice `AMB`, `DIFF` o `SUBJ`.
* `exit_cq`, `difficulty` (da 1 a 5), `comment`, per Enrico; `comment` anche per Eleni; `note` per Dumitru; `doubt` (`s`, `f`, `x`) per `unknown`. Tutti come sono scritti.

## 5. Cosa è stato corretto o tolto al congelamento

### 5.1 Due errori di copiatura

Due righe del workbook hanno un testo leggermente diverso da quello del loro item. Le loro annotazioni stanno sull'item giusto.

* `Annotazione Dumitru Analogy`, riga 86: due trattini lunghi diventati corti. L'item è `97695052f03f5191`, lo stesso della riga 13 di `new_analogy`.
* `cause_to_effect`, riga 74: il foglio base ripete in coda la frase "During the time that the vice president and the president have been in office, 4 million more Americans have fallen into poverty.". L'item è `74fb573039aeaf58`; la copia di Enrico e il test set non la ripetono.

### 5.2 Cinque doppioni

Cinque coppie di testi quasi uguali stanno tutte nel test set di Eleni. Per ogni coppia `items.csv` tiene un item solo, e le annotazioni dell'altro non entrano in `annotations.csv`. Le due metà hanno sempre lo stesso gold, ed Enrico, dove le ha annotate entrambe, ha dato lo stesso verdetto. Si tiene il primo in ordine alfabetico.

| tolto | tenuto | id di Eleni (tolto, tenuto) | dove | cosa cambia |
| --- | --- | --- | --- | --- |
| `689d10927ad646e4` | `3847c80729510e82` | 194, 160 | `new_ad_populum` righe 131 e 77 | `Iphone` contro `iPhone` |
| `b4f5e0db2e57aa4e` | `57df90e878a69523` | 503, 496 | `hasty_generalization` righe 71 e 60 | `Obviously the` contro `Obviously, the` |
| `a1bc855f65cb00ba` | `3fbae950e4e60365` | 175, 186 | `new_ad_populum` righe 100 e 118 | virgolette, trattino, apostrofo, doppi spazi, `That is` contro `This is` |
| `c536608a8e446a3d` | `579a2e83b3554b69` | 490, 488 | `hasty_generalization` righe 53 e 50 | `all little girls` contro `all girls` |
| `44baba98b841dc30` | `17ea0841b588dc4f` | 531, 512 | `Slippery slope` righe 23 e 2 | `won't get into` contro `can't get into`, doppi spazi, `class...` contro `class.` |

La riproduzione dei numeri di Eleni usa i suoi file così come sono, doppioni compresi (specifica 08, requisiti 1 e 2).

### 5.3 Due item tolti da Enrico

Enrico ha segnato due righe da togliere dal dataset. Nessuna delle due sta nel test set. Gli item non sono in `items.csv`, e nessuna delle loro annotazioni è in `annotations.csv`, comprese quelle di Dumitru alla stessa riga dei suoi fogli.

* `10a2cd8c906211ce`, `new_analogy` riga 44, `eliminare` nella colonna `fallacy Enrico`: "Is it ethically wrong to watch pornography? By the same reasoning we should ban entire industries...".
* `e4a405952d48fdbe`, `new_expert_opinion` riga 16, `togliere` nella colonna `comment Enrico`: "The word fascism has now no meaning except in so far as it signifies something not desirable", George Orwell.

### 5.4 Righe ripetute dentro un foglio

Alcuni fogli contengono lo stesso item due volte. Le righe restano tutte e due in `annotations.csv`, e le annotazioni coincidono. Nella sua copia di `hasty_generalization` Enrico le ha segnate con il commento `doppia`.

* `new_analogy` righe 10 e 90, che nel foglio di Dumitru sono le righe 83 e 90.
* `cause_to_effect` righe 70 e 72.
* `hasty_generalization` righe 7 e 8, 48 e 49, 22 e 58, 16 e 59.

### 5.5 Il foglio Slippery slope

Il foglio `Slippery slope` è arrivato il 21 settembre 2026. Contiene i 96 item di slippery slope del test set, con lo stesso gold di Eleni, più due testi di LOGIC che nel test set non ci sono: `9ede7cb26b5bc4f7` (psicologi e farmaci) e `2338a8fbf845d621` (sconti ai pazienti). La colonna `gold_fallacy` del foglio è letta come etichetta di partenza (`base`). La colonna `gold_scheme` vale `Slippery Slope` su ogni riga e non aggiunge niente.

I due testi nuovi hanno `gold_scheme` e `gold_fallacy` vuoti, perché il gold di `items.csv` è solo quello di Eleni. Quindi non entrano nel pilota.

## 6. Come si aggiornano le annotazioni di Dumitru

Il flusso:

1. Le annotazioni si fanno in locale, poi le colonne si incollano sul Drive insieme alla colonna `item_id`.
2. Quando un foglio è finito, si copia nel `Dati_da_annotare.xlsx` locale, quello indicato da `WORKBOOK_PATH`.
3. Si lancia `argfallacy annotations update`.

Il comando legge ogni foglio il cui nome comincia con `Annotazione Dumitru` e sostituisce in `annotations.csv` le righe di `dumitru` di quel foglio. Le righe degli altri annotatori e degli altri fogli restano come sono.

Cosa si aspetta da un foglio:

* una colonna `item_id`, in formato testo;
* la colonna `text`;
* le colonne delle CQ, con gli stessi nomi dello YAML dello schema (`CQ1`, `CQ1.1`, ...). Lo schema del foglio si riconosce da qui: è l'unico schema con esattamente quelle CQ;
* le colonne `scheme` (se ce ne sono due, vale l'ultima), `verdetto`, e se servono `incertezza` e `nota`.

Una riga con `item_id` uguale a `ESCLUSO` si salta. Il comando si ferma senza scrivere niente se trova un id che non è in `items.csv`, un testo senza id, una risposta che non conosce, oppure un testo diverso da quello dell'item. L'ultimo controllo tollera un errore di copiatura, non una colonna incollata una riga più in alto o più in basso.

Gli `item_id` da incollare stanno in `item_id_per_foglio.xlsx`, nella cartella `Thesis`, fuori dal repository: un foglio per ciascuno dei due fogli di Dumitru e degli otto fogli base, con le righe nello stesso ordine del workbook. Le copie di Enrico hanno lo stesso ordine del loro foglio base. La colonna `nota` segnala gli `ESCLUSO`, le righe ripetute e i due errori di copiatura. Chi prepara un foglio nuovo di Dumitru partendo da un foglio base incolla gli id subito, prima di riordinare le righe.

## 7. Controlli fatti al congelamento

* Ogni verdetto, schema e risposta alle CQ di `annotations.csv` coincide con le tabelle prodotte dal codice di estrazione precedente. L'unica differenza è la riga 5 di `Annotazione Dumitru Expert Opin`, corretta il 21 settembre 2026 da `Irrelevant Authority` ad `Appeal to Authority`, in accordo con la risposta `no` a CQ4.1 della stessa riga.
* Il lettore della sezione 6, lanciato su una copia del workbook con gli `item_id` inseriti, riproduce le righe di Dumitru byte per byte.
* Le 182 righe di Dumitru con tutte le CQ compilate e nessun `idk` danno, lungo il diagramma, il verdetto scritto nel foglio. Tutte e 182.
* Sui 507 item del test set con un verdetto di Enrico, il verdetto coincide con il gold di Eleni 505 volte. Le due eccezioni: `8fdfc761e4aab007`, gold `slippery_slope`, Enrico `appeal_to_authority` in `new_expert_opinion` riga 23; `f1023bd0186f33c5`, gold `ad_populum`, Enrico `hasty_generalization` in `Copia di new_ad_populum_Enrico` riga 14.
* Cinque item hanno schemi diversi di Enrico in righe diverse: `2991ee41f2652b99` e `581504d112994e8f` (popular opinion nel foglio base, expert opinion nella copia), `4000a372e40713ec` (popular opinion e `SNI`), `f1023bd0186f33c5` e `fbbfd63fa69b5575` (popular opinion ed expert opinion).

Il vecchio `human_cq.csv` leggeva male la colonna `incertezza` quando conteneva più di un codice: attribuiva alla prima CQ tutto il resto della cella. `annotations.csv` tiene la cella intera.

## 8. Cosa non c'è più

Con il congelamento sono usciti dal repository il codice di estrazione (`src/argfallacy/data/`), la mappa delle colonne `data/workbook_columns.yaml`, `data/item_aliases.yaml`, `data/duplicates.yaml`, `labels/sources.yaml`, le tabelle `data/processed/` (`items.csv`, `verdicts.csv`, `human_cq.csv`, `build_report.md`) e il comando `argfallacy data build`. Il loro contenuto utile è in questo file. Il codice resta nella storia di git e nell'archivio di backup, ed è l'unico modo di rifare il congelamento dal workbook.
