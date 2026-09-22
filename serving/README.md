# Runbook: servire un modello e parlarci

Dieci righe, nell'ordine in cui si eseguono.

1. **Avvia vLLM** sul nodo con la GPU. `<model_id>` e `<revision>` vengono da
   `serving/models.yaml`; finche' `revision` e' `PLACEHOLDER` il client non parte:
   ```
   vllm serve <model_id> --revision <revision> --dtype auto --max-model-len 8192 --port 8000
   ```
   L'API compatibile OpenAI e l'output strutturato sono attivi di default; se la tua
   versione di vLLM chiede di abilitarlo, aggiungi `--guided-decoding-backend outlines`.
   Se le note del modello in `models.yaml` elencano flag in piu', vanno aggiunti qui.
   Qwen e Gemma hanno una voce con ragionamento spento e una con ragionamento acceso
   (`_think`), servite dallo stesso server: si avviano sempre con il parser di ragionamento
   indicato nelle note del modello e con `--max-model-len 16384`, perche' le voci `_think`
   hanno `max_tokens` 8192. Senza parser di ragionamento l'output strutturato vincola il JSON
   dal primo token e il modello non ragiona.
2. **Dichiara l'endpoint** nel `.env` alla radice del repository, mai nel codice:
   ```
   LLM_BASE_URL=http://localhost:8000/v1
   LLM_API_KEY=qualunque-stringa
   ```
   vLLM non controlla la chiave, ma il client `openai` pretende che ci sia.
3. **Verifica il modello** prima di qualunque esecuzione:
   ```
   python serving/smoke_test.py
   ```
   Stampa la risposta, il token della risposta dentro il JSON finale con la sua
   posizione, i `top_logprobs` a quel token e la latenza. Se prima del JSON c'e'
   ragionamento (gpt oss, voci `_think`) e un lettore ingenuo sarebbe finito li', lo dice. Su una
   voce `_think` controlla anche che il ragionamento ci sia stato e che la risposta non sia troncata. Se i logprob non arrivano lo dice in chiaro: allora `supports_logprobs` in
   `models.yaml` va messo a `false` e `p_logprob` restera' vuota per quel modello.
4. **Conta le chiamate** prima di farle: `argfallacy run plan configs/pilot.yaml`.
5. **Esegui**: `argfallacy run execute configs/pilot.yaml`. Interrompibile: rilanciando
   lo stesso comando riprende dalle chiamate mancanti, senza duplicare righe.

## >>> DA COMPLETARE <<<

Questa sezione dipende dal cluster. Serve una riga per ognuno dei due
punti, con i comandi esatti:

* **Come raggiungere l'endpoint dal proprio computer.** Se il nodo GPU non e' raggiungibile
  direttamente, il tunnel SSH ha questa forma, da completare con nodo, utente e porta reali:
  ```
  ssh -N -L 8000:<nodo-gpu>:8000 <utente>@<host-di-accesso>
  ```
  Con il tunnel aperto, `LLM_BASE_URL` resta `http://localhost:8000/v1`.

* **Come lanciarlo come job.** Se il cluster usa uno scheduler (Slurm o altro), qui va lo
  script di sottomissione: partizione, GPU richieste, tempo massimo, modulo o container da
  caricare, e come si scopre su quale nodo e' finito il job per aprirci il tunnel.

Finche' queste due righe non ci sono, tutto si esegue contro il backend finto
(`tests/fakes/llm.py`), che e' quello che usano i test: nessuna chiamata di rete.
