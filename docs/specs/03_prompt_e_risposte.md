# Specifica 03. Prompt e formato di risposta

Versione 1, 11 settembre 2026. Componente: `src/argfallacy/prompts/`. Dati: `prompts/`.
Dipende da: 01 e dai dati (`docs/dati.md`). Chi la usa: 04.

## 1. Principi

* Il testo di ogni domanda critica viene dallo YAML dello schema, parola per parola. Il codice lo inserisce nel template; nessuno lo riscrive. Un test confronta il testo nel prompt generato con il campo `text` dello YAML.
* Tutto ciò che sta intorno alla domanda (istruzioni, formato, definizione delle risposte) è nostro e versionato. Il numero di versione sta nel nome del file e finisce nel manifest di ogni esecuzione.
* Una domanda per chiamata. Il modello vede sempre il testo intero dell'argomento e lo schema.
* La tolleranza sta nell'aggregazione, non nel prompt: non si dice mai al modello cosa fare quando è incerto. Si definisce solo cosa significa ogni risposta.

## 2. File in `prompts/`

* `stage1_v1.md`: riconoscimento dello schema.
* `stage2_v1.md`: una domanda critica.
* `schemas/stage1_v1.json`, `schemas/stage2_v1.json`: JSON Schema della risposta, usato per l'output strutturato di vLLM e per la validazione nel parser.
* Un test verifica che ogni template dichiarato in `prompts/` si compili per ogni schema e ogni CQ senza campi mancanti.

I template usano segnaposto con doppie graffe (`{{text}}`, `{{scheme_name}}`, `{{schema}}`, `{{variables}}`, `{{cq_text}}`, `{{answer_options}}`), riempiti da `render_stage1(item)` e `render_stage2(item, scheme, cq)`. Nessuna libreria di template: sostituzione semplice, con errore se un segnaposto resta vuoto.

## 3. Stadio uno, `stage1_v1.md`

Contenuto, in inglese: ruolo (analista di argomentazione), il testo, l'elenco degli otto schemi con `name`, `schema` (la forma dell'argomento) e `identification_question` presi dagli YAML, più l'opzione `none` definita come "nessuno di questi otto schemi è presente". Istruzione: scegliere uno solo, in base al solo testo. Risposta nel formato della sezione 5, con `scheme` fra i nove id canonici di `labels/schemes.yaml`.

## 4. Stadio due, `stage2_v1.md`

Contenuto, in inglese:
1. Ruolo e compito: rispondere a una sola domanda critica su un argomento che segue lo schema indicato.
2. Lo schema: `name`, `schema` (forma dell'argomento) e `variables` dallo YAML, come legenda delle lettere usate nella domanda (S, A, D, C, eccetera).
3. Il testo dell'argomento.
4. La domanda critica, verbatim, preceduta dal suo id.
5. Le risposte ammesse, con definizione:
   * `yes` e `no` (oppure `negative` e `positive` per ad hominem CQ1, nell'ordine in cui la domanda le nomina). Le definizioni stanno in `prompts/stage2_v1_answers.yaml`, versionato con il template; le note degli YAML degli schemi non entrano nel prompt (deroga accettata).
   * `cannot_be_determined`: "usa questa risposta solo se il testo non contiene nessuna informazione sul punto chiesto; se il testo contiene informazione parziale, rispondi yes o no e abbassa la sicurezza". È la formulazione della terza opzione stampata nella tesi di Enrico, resa operativa.
   * `na`: "usa questa risposta solo se la domanda non ha oggetto in questo testo, cioè la cosa di cui parla non esiste nell'argomento; se esiste ma non si riesce a stabilire come stia, rispondi cannot_be_determined e abbassa la sicurezza". Quarta opzione, aggiunta fuori dal diagramma esattamente come `cannot_be_determined`: mai nell'`answer_space` di uno schema. Distingue l'assenza dell'oggetto della domanda dall'incertezza su come stia.
6. Istruzione a rispondere in base al solo testo, senza conoscenza esterna, salvo che la domanda la richieda esplicitamente.
7. Il formato di risposta.

`na` è la quarta risposta, non `not_applicable`. È aggiunta dal codice (sezione 5), non dal diagramma: nessuno YAML di `schemes/` guadagna un arco `na`, e i casi di guardia restano codificati come nodi (per esempio expert opinion CQ4). Una traversata che incontra `na` non trova un arco disegnato e si ferma incompleta, come qualunque altra risposta fuori dall'`answer_space`: cosa significhi quell'esito per un aggregatore è una decisione della fase 3, non di questa specifica.

## 5. Formato di risposta (JSON, output strutturato)

Stadio uno:
```json
{"scheme": "expert_opinion", "confidence": 80, "justification": "..."}
```
Stadio due:
```json
{"answer": "yes", "confidence": 75, "justification": "..."}
```

Regole:
* Il campo `answer` (o `scheme`) è la prima chiave dell'oggetto, così la posizione del primo token della risposta è la stessa in ogni output e il client può leggerne i logprob.
* I valori ammessi di `answer` sono stringhe che iniziano con token diversi: `yes`, `no`, `cannot_be_determined`, `na`, e per ad hominem CQ1 `positive`, `negative`, `cannot_be_determined`, `na`. La probabilità morbida `p_logprob` si calcola dal primo token del valore: si prendono i `top_logprobs` a quella posizione, si tengono le voci che iniziano una delle risposte ammesse, si rinormalizza. Se una risposta ammessa non compare fra i top_logprobs, la sua massa è zero e la riga viene marcata `logprob_partial`.
* `confidence` è un intero da 0 a 100. `justification` è testo, massimo due frasi; il limite è nel prompt, non imposto dallo schema.
* Lo JSON Schema vincola `answer` all'enumerazione ammessa per quella CQ e `confidence` all'intervallo. Il parser valida di nuovo: un output che non passa lo schema è `invalid` (specifica 01), conservato e contato.

## 6. Campioni

Per ogni (item, stadio, CQ, modello): cinque chiamate. Campione 0 a temperatura 0: fornisce la risposta secca e `p_logprob`. Campioni da 1 a 4 a temperatura 0.7: forniscono, insieme allo 0, `p_sample` = frequenza della risposta `yes` (o `positive`) sui cinque. `p_verbal` = `confidence`/100 del campione 0, orientata: se la risposta secca è `no`, `p_verbal` = 1 meno confidence/100; se è `cannot_be_determined`, `p_verbal` è vuota e la riga porta `idk = true`. `na` non produce `p_verbal` allo stesso modo di `cannot_be_determined` — leggerlo come una bassa probabilità del sì deciderebbe, in silenzio, che `na` vale `no` — ma non alza `idk`, che resta vero solo per `cannot_be_determined`: le due risposte restano distinguibili a valle solo guardando `answer`, l'unica informazione che questo parser registra su di esse. Cosa significhi `na` per l'aggregazione resta un punto aperto per la fase 3, da chiudere sui dati veri: nessun valore numerico, non una regola provvisoria.

I modelli con ragionamento interno (gpt oss, K2 Horizon) producono lo stesso JSON; il client legge i logprob alla posizione della risposta finale. Se per un modello i logprob non sono disponibili, `p_logprob` resta vuota e il manifest lo dichiara.

## 7. Test di accettazione

Dal 20 settembre 2026 `tests/` tiene solo le prove che difendono un risultato, verificano i dati o servono a lavorare. I punti senza nota hanno un test automatico; i punti segnati restano requisiti, ma oggi non ce l'hanno, o ce l'hanno solo in parte.

1. Per ogni schema e ogni CQ, il prompt renderizzato contiene il campo `text` dello YAML come sottostringa esatta.
2. Per ad hominem CQ1 le opzioni sono `positive`, `negative`, `cannot_be_determined`, `na`; per tutte le altre CQ `yes`, `no`, `cannot_be_determined`, `na`.
3. Lo JSON Schema dello stadio due rifiuta un `answer` fuori enumerazione e un `confidence` fuori intervallo; quello dello stadio uno rifiuta uno `scheme` non canonico. *Oggi ha un test automatico solo la parte sullo stadio due, verificata attraverso il parser (specifica 04, punto 5).*
4. Il rendering è deterministico: stesso item, stesso prompt, byte per byte. *Oggi senza test automatico.*
5. La versione del template compare nel testo del prompt renderizzato come commento finale, così il manifest e il prompt non possono divergere. *Oggi senza test automatico.*
