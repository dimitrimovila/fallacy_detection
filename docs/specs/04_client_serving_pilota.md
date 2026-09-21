# Specifica 04. Intervistatore, serving, parser minimo e pilota

Versione 1, 11 settembre 2026. Componenti: `src/argfallacy/client/`, `src/argfallacy/parse/` (versione minima), `serving/`.
Dipende da: 01, 03 e dai dati (`docs/dati.md`). Produce: `runs/<run_id>/`.

## 1. Serving sul cluster (`serving/`)

* `serving/models.yaml`: una voce per modello con `model_id` (il nome con cui vLLM lo serve), `display_name`, `supports_logprobs`, `is_reasoning`, `max_concurrency`, `notes`. Prime voci: Qwen3.6 27B, Gemma 4 31B; poi gpt oss 120b, K2 Horizon 32B. I nomi esatti dei repository Hugging Face si fissano in `serving/models.yaml`.
* `serving/README.md`: runbook in dieci righe. Avvio di vLLM con l'API compatibile OpenAI e l'output strutturato abilitato, con un esempio:
  ```
  vllm serve <model_id> --dtype auto --max-model-len 8192 --port 8000
  ```
  più le istruzioni per raggiungere l'endpoint dal proprio computer se il cluster richiede un tunnel SSH, e il modo per lanciarlo come job se il cluster usa uno scheduler. La parte specifica del cluster resta da completare.
* `serving/smoke_test.py`: manda una chat completion con `logprobs=true`, `top_logprobs=20` e `response_format` con lo schema JSON dello stadio due su un item di prova, e stampa la risposta, i top_logprobs alla posizione del primo token di `answer`, e la latenza. Se i logprob non arrivano lo dice in chiaro. Questo è il primo comando da eseguire su ogni modello nuovo.

Variabili d'ambiente: `LLM_BASE_URL` (per esempio `http://localhost:8000/v1`), `LLM_API_KEY` (qualunque stringa per vLLM; la chiave vera per un provider commerciale). Nessuna nel codice.

## 2. Il client (`argfallacy.client`)

### 2.1 Chiamata
Una funzione `ask(request) -> RawResponse` costruita sul pacchetto `openai` (chat completions), con: `model_id`, messaggi, `temperature`, `seed`, `max_tokens`, `response_format` (JSON Schema), `logprobs` e `top_logprobs` quando il modello li supporta. `RawResponse` conserva la risposta completa così come arriva (contenuto, blocco logprobs, usage, `finish_reason`), più latenza e timestamp. Mai riassunta, mai ripulita.

### 2.2 Cache
SQLite in `runs/cache.sqlite`, tabella con chiave = sha256 di (`model_id`, `prompt_version`, testo del prompt renderizzato, parametri di generazione, `sample_index`) e valore = `RawResponse` serializzato. Prima di ogni chiamata si consulta la cache; una chiamata già fatta non si ripete. La cache non si cancella mai automaticamente. Cambiare il prompt cambia la chiave, quindi una nuova versione del prompt genera chiamate nuove solo per quel prompt.

### 2.3 Piano ed esecuzione
* `argfallacy run plan CONFIG` legge un file di configurazione YAML (item da includere, stadio, condizione di schema `gold` o `predicted`, modelli, numero di campioni, versione dei prompt) e stampa quante chiamate servono per modello, quante sono già in cache, e una stima di durata data la concorrenza. Non chiama nulla.
* `argfallacy run execute CONFIG` crea `runs/<run_id>/` con `run_id` = data e ora più nome della configurazione, scrive `manifest.json`, esegue le chiamate mancanti e appende ogni risposta a `raw.jsonl`. Ordine deterministico: per modello, per item, per CQ, per campione.
* Concorrenza con un limite per modello (da `models.yaml`), tentativi ripetuti con attesa crescente su errori 429 e 5xx, timeout per chiamata. Un errore definitivo produce una riga in `raw.jsonl` con `error` valorizzato e la chiave non entra in cache, così alla ripresa si ritenta.
* Ripresa: rilanciare lo stesso `execute` dopo un'interruzione riprende dalle chiamate mancanti, senza duplicare righe in `raw.jsonl` (si scrive solo ciò che non è già in cache).
* Condizione `predicted`: il piano usa le predizioni di stadio uno del modello stesso e genera chiamate di stadio due solo per gli item in cui lo schema predetto differisce dal gold; per gli altri i risultati della condizione `gold` sono validi anche qui, e il parser lo sa.

### 2.4 Manifest
`manifest.json` con: `run_id`, configurazione completa, `model_id` e `display_name`, versione dei prompt, versione degli schemi (tag e hash del contenuto di `schemes/`), parametri di generazione, numero di campioni, timestamp di inizio e fine, numero di chiamate previste, eseguite, servite da cache, fallite, e la nota `logprobs_available` per modello.

### 2.5 `raw.jsonl`
Una riga per chiamata: `run_id`, `item_id`, `stage`, `scheme_condition`, `scheme`, `cq_id` (vuoto per lo stadio uno), `sample_index`, `model`, `model_id`, `revision`, `tier`, `reasoning`, `prompt_version`, `json_schema` (lo schema mandato al modello con quella chiamata), `params`, `cache_key`, `from_cache`, `raw_response`, `latency_s`, `error`. Il file è append-only.

## 3. Parser minimo (`argfallacy.parse`, versione per il pilota)

`argfallacy parse RUN_ID` legge `raw.jsonl` e produce `answers.csv`, una riga per chiamata: le chiavi di cui sopra, più `answer` (canonico o `invalid`), `confidence`, `justification`, `p_logprob` per ogni risposta ammessa, `logprob_partial`, `parse_ok`, `finish_reason`. `parse_ok` è vero solo se l'oggetto della risposta rispetta lo `json_schema` della sua riga (specifica 03 sezione 5): una risposta fuori enumerazione, con `confidence` fuori da 0 e 100, senza un campo richiesto o con un campo in più è `invalid`, e la riga resta. Poi `summary.csv`, una riga per (item, stadio, scheme, cq_id, modello): risposta secca dal campione 0, `p_logprob`, `p_verbal`, `p_sample`, `n_samples_ok`, `idk`, `invalid_count`, secondo la specifica 03 sezione 6. La versione completa del parser è la specifica 05; questa deve solo bastare per il rapporto del pilota e non va oltre.

## 4. Il pilota

`configs/pilot.yaml`: 50 item scelti da `items.csv` con schema gold diverso da `none`, stratificati per schema in proporzione con almeno 4 per schema, seme fisso; stadio uno e stadio due in condizione `gold`; due modelli; cinque campioni.

`items.csv` non contiene i doppioni né gli item tolti (`docs/dati.md`, sezione 5), quindi il pilota pesca fra tutti gli item con uno schema gold, cioè il test set degli esperimenti. L'estrazione con seme corre dentro ogni strato (stesso schema).

`argfallacy pilot report RUN_ID...` produce `pilot_report.md` con, per modello:
* stadio uno: accuratezza sullo schema gold e matrice di confusione ridotta;
* per ogni CQ: distribuzione delle risposte, quota di `cannot_be_determined`, quota di `na`, quota di `invalid`;
* accordo fra la risposta del campione 0 e la maggioranza dei cinque;
* correlazione fra `p_logprob`, `p_verbal` e `p_sample` per CQ;
* quante traversate per schema finiscono incomplete perché una risposta è `na` (specifica 03), con il nodo su cui si fermano;
* latenza media e token per chiamata, e la proiezione di tempo per l'esecuzione completa (tutti gli item di `items.csv`, oggi 773, tutte le CQ, cinque campioni).

Regola d'arresto scritta nel rapporto: ogni CQ con più del 50 per cento di `cannot_be_determined` o più del 10 per cento di `invalid` su un modello va elencata in testa al rapporto; la stessa soglia del 50 per cento vale per `na`. La decisione su cosa fare non è automatica: si prende leggendo il rapporto.

## 5. Test di accettazione

Dal 20 settembre 2026 `tests/` tiene solo le prove che difendono un risultato, verificano i dati o servono a lavorare. I punti senza nota hanno un test automatico; i punti segnati restano requisiti, ma oggi non ce l'hanno, o ce l'hanno solo in parte.

Tutti con il backend finto in `tests/fakes/llm.py`, che restituisce risposte deterministiche in funzione del prompt e del `sample_index`, con un blocco logprobs verosimile; nessuna chiamata di rete nei test.

1. Cache: due `ask` identiche producono una sola chiamata al backend; cambiare `prompt_version`, un parametro o `sample_index` ne produce un'altra.
2. Ripresa: un'esecuzione interrotta a metà (il backend finto solleva un errore dopo N chiamate) e rilanciata completa il piano senza righe duplicate in `raw.jsonl` e senza rifare le chiamate riuscite.
3. Piano: su 10 item di prova con schema gold noto, `plan` conta chiamate = somma sulle CQ dello schema di ogni item, per 5 campioni, per modello, più le chiamate di stadio uno.
4. Manifest: tutti i campi della sezione 2.4 presenti; l'hash degli schemi cambia se si modifica un byte di uno YAML. *Oggi ha un test automatico solo la parte sui campi.*
5. Parser: da un `raw.jsonl` finto con risposte valide, non valide e con logprobs parziali, `answers.csv` e `summary.csv` hanno i valori attesi; `p_logprob` si rinormalizza sulle risposte ammesse; un output che non rispetta lo schema è `invalid` e non fa saltare la riga. *Oggi hanno un test automatico la risposta valida, quella fuori schema che resta come `invalid` e le tre probabilità del sommario; il caso con logprob parziali no.*
6. Condizione `predicted`: il piano genera chiamate di stadio due solo per gli item con schema predetto diverso dal gold. *Oggi senza test automatico.*
7. `smoke_test.py` gira contro il backend finto (`--model fake`, il valore predefinito) e stampa il blocco logprobs. *Oggi senza test automatico: si lancia a mano.*

## 6. Cosa non fare
* Niente parsing dentro il client: il client salva e basta.
* Niente pulizia automatica di `runs/`.
* Niente prompt scritto nel codice: solo i template di `prompts/`.
* Niente chiamate ai modelli nei test.

## 7. Definizione di fatto
Lint e test verdi (comandi nel README); `plan` e `execute` funzionano contro il backend finto e contro vLLM sul cluster con un modello (verifica con `smoke_test.py`); `pilot_report.md` prodotto su 50 item e 2 modelli; riepilogo con i numeri del rapporto e le CQ sopra soglia.
