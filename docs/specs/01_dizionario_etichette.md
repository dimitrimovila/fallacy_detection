# Specifica 01. Dizionario delle etichette

Versione 2, 9 settembre 2026. Sostituisce la versione 1, scritta prima di vedere il codice.
Componente: il loader esistente, oggi `src/argfallacy/schemes/loader.py`, più il modulo `src/argfallacy/labels/`. Dati: `labels/fallacies.yaml`, `labels/schemes.yaml`.

## 1. Cosa esiste già e non si tocca

Il loader (`argschemes/loader.py` quando questa specifica è stata scritta, oggi `src/argfallacy/schemes/loader.py`) ha già la parte centrale:

* `schemes/_vocabulary.yaml`: 25 terminali con id canonico, l'etichetta a grana grossa `false_cause` con `refines_to`, una tabella di alias, i marcatori IDK.
* `normalize_label(raw, vocab)`: minuscole e spazi compressi, tabella alias, poi una regola deterministica (spazi in trattino basso, parentesi tolte), infine lookup nei terminali, nei coarse, negli out_of_scope. Sconosciuto = `SchemeError`. Nessun default silenzioso.
* `resolve_for_scheme(raw, scheme, all_schemes)`: controlla l'etichetta contro lo schema annotato e restituisce `exact`, `coarse`, `scheme_mismatch` (con proposta di riassegnazione mai applicata), `out_of_scope`, `idk`.
* `label_matches(predicted, resolution)`: credito parziale per i gold a grana grossa.

Queste funzioni restano con la stessa firma e lo stesso comportamento. I test esistenti in `validate_schemes.py` devono continuare a passare.

Decisione già presa ad agosto e confermata: `false cause` e `false causality` sono etichette a grana grossa, non si raffinano, danno credito parziale, e la quota di gold grosso si riporta a parte. Non sono più "in sospeso".

## 2. Cosa manca

1. Alias incompleti. Con il vocabolario 1.0 falliscono stringhe che nei dati ci sono: `good argument` (124 righe del gold di Eleni), `Good Argomentation` (sette diagrammi), `Abusive Ad Hominem`, `hasty generalisation`, `Appeal to Emotion`.
2. Nessun dizionario per le etichette di schema (`Argument from Expert Opinion`, `Expert opinion`, `NO`, `None`), necessario per valutare lo stadio uno.
3. Nessuno spazio di etichette collassato e nessuna nozione di famiglia ad hominem.
4. Nessun modo compat che riproduca lo scorer di Eleni.
5. Nessuno strumento di audit che dica, su una colonna reale, quali stringhe compaiono e come vengono tradotte.

## 3. Cosa fare

### 3.1 Spostare e aggiornare il vocabolario
`schemes/_vocabulary.yaml` viene sostituito da `labels/fallacies.yaml` versione 1.1, già fornito. Stesse chiavi di prima (`terminals`, `coarse_labels`, `out_of_scope`, `aliases`, `idk_markers`), quindi `normalize_label` funziona senza modifiche; in più: `families`, per ogni terminale `family` e `schemes`, `appeal_to_emotion` fra le out_of_scope, `spaces`, `missing_markers`. `load_vocabulary` legge da `labels/`. Il vecchio file si elimina.

### 3.2 Nuovo file `labels/schemes.yaml`
Già fornito: otto schemi più `none`, con alias. `none` è l'etichetta "nessuno schema presente", assegnata da un annotatore del workbook quando nessuno degli otto schemi si applica. Non è un nono schema, è la classe negativa dello stadio uno.

### 3.3 Nuovo modulo `labels/` con queste funzioni

```python
from argfallacy.labels import (normalize_scheme_label, label_space, collapse,
                               compat_normalize, audit)

normalize_scheme_label("Popular Opinion")   # -> "popular_opinion"
normalize_scheme_label("NO")                # -> "none"
label_space("fine")                         # -> lista ordinata dei 25 id dei terminali
label_space("collapsed_symmetric")          # -> 20 id
label_space("collapsed_compat")             # -> 21 id
collapse("ad_fidentia", "collapsed_symmetric")   # -> "ad_hominem"
compat_normalize("Ad Hominem (Tu Quoque)")       # -> "tu quoque"
audit(series, kind="fallacy")               # -> tabella: stringa grezza, conteggio, id, stato
```

`audit` non solleva eccezioni: per ogni stringa grezza distinta restituisce conteggio, id tradotto oppure `UNKNOWN`, e lo stato (`terminal`, `coarse`, `out_of_scope`, `idk`, `missing`, `UNKNOWN`).

Modo compat, in una sola funzione con docstring: minuscole; se la stringa inizia con `ad hominem (` e finisce con `)` si tiene la parte fra parentesi; se finisce con ` ad hominem` si toglie il suffisso; spazi iniziali e finali tolti. Il confronto compat è uguaglianza fra stringhe così trasformate. Niente credito parziale, niente alias.

### 3.4 Vincoli controllati al caricamento (da aggiungere al loader)
* Alias unici dopo normalizzazione; un alias non può puntare a due id.
* Ogni `family` e ogni voce di `spaces` cita id esistenti.
* Per ogni terminale, `schemes` coincide con gli schemi che lo producono davvero secondo i grafi in `schemes/`.

## 4. Test di accettazione

Dal 20 settembre 2026 `tests/` tiene solo le prove che difendono un risultato, verificano i dati o servono a lavorare. I punti senza nota hanno un test automatico; i punti segnati restano requisiti, ma oggi non ce l'hanno, o ce l'hanno solo in parte.

1. `validate_schemes.py` continua a passare dopo lo spostamento del vocabolario.
2. Ogni stringa nella sezione 2 punto 1 si traduce nell'id atteso; maiuscole e spazi in più non cambiano il risultato. *Oggi senza test automatico, per scelta del 21 settembre 2026: ogni modifica agli alias si rivede a mano. I punti 6 e 7 controllano che nessuna stringa dei file di Eleni resti senza traduzione, ma non che finisca sull'id atteso.*
3. Una stringa inventata solleva `SchemeError` con la stringa nel messaggio. `false cause` restituisce kind `coarse`.
4. I 25 terminali del vocabolario coincidono con i terminali dei grafi; la lista `schemes` di ogni terminale coincide con i produttori reali.
5. `collapse` è totale sui 25 id per entrambi gli spazi; le taglie sono 20 e 21 (25 fine; il simmetrico fonde sei id, quello di Eleni cinque). Sulle 23 etichette con supporto nel gold attuale diventano 18 e 19, e 19 è il numero di classi della riga collassata dei `metrics.csv` di Eleni.
6. Audit su `gold_fallacy` e `predicted_fallacy` dei `results.csv` di Eleni (`PRIOR_RUNS_DIR`; test saltato se manca): zero `UNKNOWN`. Ogni stringa nuova va aggiunta agli alias e riportata nel riepilogo.
7. Audit su `gold_scheme` e `predicted_scheme`: zero `UNKNOWN`.
8. Compat: sui `results.csv` di ogni modello, la quota di righe con `compat_normalize(predetto) == compat_normalize(gold)` coincide con l'accuratezza `final_label` del suo `metrics.csv`, tolleranza 0.0005. Stessa cosa per la riga collassata usando `collapsed_compat` sugli id canonici.
9. Nessuna stringa di etichetta come letterale fuori da `labels/*.yaml`, `schemes/*.yaml` e `tests/`: un test cerca `"Tu quoque"`, `"Good argument"`, `"Ad Populum"` nel sorgente e fallisce se le trova altrove. *Oggi senza test automatico.*

## 5. Cosa non fare
* Nessun confronto approssimato oltre la regola deterministica già esistente.
* Non modificare `resolve_for_scheme` e `label_matches`.
* Non toccare `data/` né i file di Eleni per far passare i test.

## 6. Definizione di fatto
Lint e test passano (comandi nel README); presenti i test della sezione 4 senza nota (6, 7 e 8 saltano senza `PRIOR_RUNS_DIR`); riepilogo con gli alias aggiunti.
