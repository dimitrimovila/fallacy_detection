# Proposte di revisione delle domande critiche

I numeri di questo documento vengono dalla diagnosi delle esecuzioni di Eleni e sono stati
ricontrollati il 12 settembre 2026 contro la copia congelata e contro i suoi `results.csv`.
**Base dei numeri.** Salvo dove è detto altro, ogni percentuale e ogni conteggio per schema sta
sul sottoinsieme predetto da gpt-5: gli item a cui il modello assegna quello schema, perché è lì
che la sua pipeline pone le CQ. Le eccezioni sono marcate accanto al numero, con «base gold»
quando il sottoinsieme è quello del gold e con il nome del modello quando non è gpt-5. I conteggi
di etichette del test set, per esempio quanti item hanno un certo gold, sono per definizione su
base gold e lo dicono nel testo.

Ogni numero che entra in tesi va comunque ricalcolato dallo scorer, che oggi non esiste. La
sezione 8 elenca gli errori trovati il 12 settembre 2026, con la causa di ognuno.

**Stato. Nessuna di queste proposte è applicata.** Né nel file `schemi_e_diagrammi.md`, né nei
YAML, né nei prompt. Questo documento serve da consultare **dopo** aver completato i test sulle
domande così come sono scritte nella tesi.

**Quando aprirlo.** Quando il baseline v0 è chiuso e si passa alla versione riformulata v1.

---

## 0. Cosa sta qui dentro e cosa no

Le correzioni si dividono in due categorie che non vanno confuse.

**Correzioni già applicate, e non sono in questo file.** Le due polarità invertite di
correlation to cause CQ5 e slippery slope CQ5. Sono bug del grafo, non scelte di formulazione.
Sono state corrette invertendo i due archi e lasciando intatto il testo delle domande. Stanno in
`schemi_e_diagrammi.md` §12.1 e nei YAML, dove sono state verificate il
12 settembre 2026: in `correlation_to_cause.yaml` CQ5 esce `yes` verso good_argumentation e `no`
verso definist_fallacy, con una nota che spiega la correzione; in `slippery_slope.yaml` CQ5 esce
`yes` verso slippery_slope e `no` verso good_argumentation, come dice il Commento della tesi. Non richiedono un esperimento, perché mostrare che un
arco sbagliato produce risposte sbagliate non è un risultato.

**Proposte di riformulazione, e sono queste.** Sette proposte, otto domande, sei schemi. Qui il testo
attuale è difendibile e la riscrittura è una scelta di design, quindi la differenza fra prima e
dopo è misurabile e vale come risultato.

---

## 1. Protocollo

**v0** — testo delle domande come nella tesi, ortografia corretta, archi di CQ5 corretti nei due
schemi. È lo stato attuale dei YAML e dei prompt. Il baseline sono le nostre esecuzioni della
fase 2 sui 601 item del test set degli esperimenti: i 606 testi distinti delle 607 righe di
Eleni, meno cinque quasi duplicati. Le esecuzioni di Eleni non sono la v0: il suo prompt
ammetteva solo `Yes` e `No`, sei domande su 48 avevano un testo diverso da quello della tesi, e
il suo codice si scostava dal diagramma in due punti (archi di CQ4 di expert opinion invertiti,
riconvergenza di analogy assente). I suoi risultati restano un confronto di massima, ricalcolato
sugli stessi 601 item (specifica 08, requisito 2).

**v1** — v0 più le riformulazioni di questo documento.

**Cosa costa la v1.** Solo una passata di stage 2 per modello, e solo per i sei schemi
toccati. Lo stage 1 non cambia, perché nessuna proposta modifica la domanda di identificazione
dello schema. Gli schemi non toccati riusano le risposte di v0.

**Cosa tenere fisso.** Stesso test set, stessi modelli, stesso stage 1, stessi archi, stessa
temperatura e stesso formato di risposta. L'unica variabile è il testo delle domande, altrimenti
la differenza non è attribuibile.

**Cosa NON fare.** Riscrivere tutto. Se cambiano tutte le domande, il confronto non dice quale
modifica ha prodotto il guadagno. Le sette qui sotto hanno ciascuna una diagnosi scritta con un
numero accanto, il resto va lasciato identico.

---

## 2. Le sette proposte

### 2.1 ad hominem, CQ1 — spazio delle risposte incoerente

**Testo attuale (tesi)**
> Is the opinion expressed about the other person negative (and intended to undermine their
> credibility) or positive (and intended to enhance their credibility)?

**Diagnosi.** È l'unico nodo di tutto il set che non si dirama su `yes` e `no` ma su `positive` e
`negative`. Nel prompt eseguito il formato imponeva comunque `Yes` o `No`, quindi il modello ha
risposto in binario a una domanda a due poli nominali.

**Evidenza.** `Yes` 102 volte, `No` 1 volta sui 103 item in cui gpt-5 ha predetto ad hominem. Il ramo positivo, unico che
porta ad Appeal to Authority, non si attiva mai in tutto il dataset.

**Proposta**
> Is the judgement expressed about the other person negative, that is, intended to reduce the
> credibility of what they say?

**Effetto sul grafo.** `yes` verso CQ2, `no` verso Appeal to Authority. Struttura identica,
spazio delle risposte allineato a tutti gli altri nodi. Nel YAML sparisce
`answer_space: ["positive", "negative"]`.

---

### 2.2 ad hominem, CQ3 — la domanda non dice cosa chiede

**Testo attuale (tesi, ortografia corretta)**
> Does the direct attack on the other person consist of an accusation that they are committed?

**Diagnosi.** Manca il complemento, quindi non si sa committed a che cosa, e il nodo non
discrimina: si attiva su qualunque attacco diretto e manda al terminale Guilt by Association casi
che con la colpa per associazione non c'entrano. La sovrapposizione con la CQ4 sull'ipocrisia,
cioè l'essere impegnato in una posizione incoerente con la propria condotta, resta una causa, ma
parziale: spiega meno di un terzo dei casi.

**Evidenza.** Il nodo riceve `yes` su 42 dei 103 item che gpt-5 assegna allo schema, e 29 di
questi escono su Guilt by Association; **nessuno** dei 29 ha Guilt by association nel gold. Sono
9 tu quoque, che è la sovrapposizione con CQ4, più 10 poisoning the well e 10 abusive, che con
CQ4 non c'entrano. Il terminale viene prodotto 29 volte mentre nel test set i gold Guilt by
association sono 8 in tutto (base gold). Dei 18 gold `tu quoque` (base gold), 9 finiscono
etichettati Guilt by Association. Nel prompt eseguito la domanda conteneva anche il segnaposto
`[something]` rimasto nel testo.

**Proposta**, se il terminale voluto resta Guilt by Association
> Does the attack discredit the person by associating them with a group, cause or individual
> regarded as disreputable, rather than by addressing their own conduct or claims?

La subordinata con `rather than` è la parte che la separa da CQ2 e da CQ4.

**Effetto sul grafo.** Nessuno.

**Variante più corta**, che era già scritta nei YAML e poi è stata rimossa allineando alla tesi
> Does the direct attack on the other person consist of an accusation that they are committed to
> a group, cause or interest?

Interviene meno sul testo originale, aggiunge solo il complemento mancante e sceglie il senso
della colpa per associazione. Non separa però il caso da CQ4 in modo esplicito.

**Da chiarire con l'autore prima di applicare.** Quale dei due sensi intendeva. Se intendeva
l'impegno in una posizione, allora è il terminale a essere sbagliato, non la domanda.

---

### 2.3 cause to effect, CQ2 — si attiva mai

**Testo attuale (tesi)**
> Is the evidence cited (if there is any) strong enough to warrant the causal generalization?

**Diagnosi.** La parentetica `(if there is any)` fa collassare l'assenza di prove dentro il `no`.
In un argomento informale breve prove citate quasi non ce ne sono, quindi la risposta è `no` per
default e tutto il flusso finisce nel sottoalbero di CQ2.1 e CQ2.2, che però presuppone che
delle prove esistano.

**Evidenza.** Tassi di `yes` sul nodo, ciascuno sul sottoinsieme che quel modello assegna a
cause to effect: 87 item per gpt-5, 70 per qwen.

| nodo | gpt-5 | qwen |
|---|---|---|
| CQ2 | 0.00 | 0.10 |
| CQ2.1 | 0.00 | 0.03 |
| CQ2.2 | 1.00 | 0.90 |
| CQ3 | 0.07 | 0.09 |

Lo schema produce due soli terminali su sei, False Premise e Hasty Generalization, e ha accuracy
0.023 sugli 87 item che gpt-5 gli assegna.

**Proposta A, minima, un nodo solo**
> Is the causal generalization supported by evidence stated in the text, rather than merely
> asserted?

Risolve il default ma lascia indefinito il sottoalbero quando prove non ce ne sono.

**Proposta B, corretta, aggiunge un nodo di guardia**
> CQ2 — Does the text cite any evidence for the causal generalization beyond the assertion
> itself?

`no` esce direttamente verso CQ3, `yes` entra nella domanda sulla forza delle prove, che diventa
CQ2.0 e conserva il sottoalbero attuale.

**Effetto sul grafo.** Nessuno con A. Con B si aggiunge un nodo e cambia la profondità dello
schema, quindi va aggiornato lo snapshot di regressione.

---

### 2.4 cause to effect CQ2.1 ed example CQ2.1 — due condizioni e direzione rovesciata

**Testo attuale (tesi), identico nei due schemi salvo `evidence` contro `example`**
> Was the evidence cited selected in a clearly biased and malicious manner in order to undermine
> the validity of the generalization?

**Diagnosi.** Due difetti sovrapposti. Le condizioni sono due, `biased` e `malicious`, e la
seconda è un giudizio sull'intenzione che dal testo non si legge. E `undermine` è rovesciato,
perché il cherry picking seleziona prove per **sostenere** la propria tesi, non per minarla. La
definizione che la tesi stessa dà del cherry picking dice esattamente questo.

**Evidenza.** Su cause to effect il nodo riceve `yes` sullo 0.0 per cento degli item con gpt-5
(87 item) e sul 2.9 con qwen (70), quindi Cherry Picking è irraggiungibile da quello schema. Su
example, con la stessa identica frase, si attiva al 10.1 per cento su entrambi i modelli (69 item
per ciascuno). Il testo da solo non
spiega la differenza, ma nel caso di cause to effect si somma al problema di CQ2 sopra.

**Proposta**
> Does the text present only the cases that support the conclusion, omitting comparable cases
> that would tell against it?

Una condizione sola, verificabile sul testo, coerente con il terminale.

**Effetto sul grafo.** Nessuno.

**Varianti già scritte nei YAML e poi rimosse allineando alla tesi.** Sono più conservative della
proposta sopra e vale la pena tenerle sul tavolo.

Per cause to effect
> Was the evidence cited selected in a clearly biased manner in order to support the
> generalization?

Toglie solo `and malicious` e gira `undermine` in `support`. È la correzione minima, cambia due
parole e lascia tutto il resto della frase di Enrico.

Per example, la formulazione che la tesi aveva prima di questa revisione
> Is the example cited selected on the basis of biased quality criteria that undermine the
> strength of the generalisation?

**Nota.** Vanno cambiate insieme, perché sono la stessa domanda su due schemi. Cambiarne una sola
introdurrebbe una differenza fra schemi che poi non si sa a cosa attribuire.

---

### 2.5 expert opinion, CQ1 — due condizioni in disgiunzione

**Testo attuale (tesi)**
> Is S in a position to know whether A is true or false, or is S a genuine expert recognised by
> the community of experts in D?

**Diagnosi.** Due condizioni distinte unite da `or`, quindi il `no` richiede che falliscano
entrambe. Il caso «fonte competente ma non riconosciuta dai pari» resta indistinguibile da
«fonte riconosciuta ma non competente». Il terminale è Irrelevant Authority, che riguarda una
cosa più precisa, cioè l'autorità invocata fuori dal proprio dominio.

**Evidenza.** Sui 72 item che gpt-5 assegna allo schema i gold `Appeal to authority` sono 24, e
14 di questi finiscono su Irrelevant Authority. Il terminale Irrelevant Authority viene prodotto
32 volte dentro lo schema, mentre nel test set i gold `Irrelevant authority` sono 15 in tutto
(base gold): il nodo lo emette circa il doppio delle volte che dovrebbe. Lo schema ha accuracy
0.139 su quei 72 item.

**Quello che l'accordo fra modelli non dice.** Sui 64 item che gpt-5 e qwen assegnano entrambi
allo schema, il kappa fra i due vale 0.69 su CQ1, 0.38 su CQ4.1, 0.32 su CQ4, 0.18 su CQ3, 0.13
su CQ2 e 0.00 su CQ3.1, che sta a zero perché gpt-5 non risponde mai `yes`. CQ1 è quindi la
domanda su cui i due modelli vanno più d'accordo, non una domanda isolata dentro uno schema che
nessuno sa rispondere: la gamba dell'accordo, su cui una versione precedente di questo documento
poggiava la proposta, cade. Restano l'argomento semantico della diagnosi, cioè due condizioni
unite da `or` contro un terminale che ne riguarda una sola, la confusione fra Appeal to Authority
e Irrelevant Authority, la sovrapproduzione del terminale e l'accuratezza dello schema.

**Proposta**
> Is the source's expertise in the same domain as the claim being supported?

È esattamente ciò che significa autorità irrilevante, è una condizione sola, si risponde dal
testo.

**Effetto sul grafo.** Nessuno.

**Da considerare.** Con questa formulazione la condizione «riconosciuto dalla comunità di
esperti» sparisce dal diagramma. Se serve, va come nodo proprio, non riaggiunta dentro CQ1.

---

### 2.6 analogy, CQ4 — chiede una cosa che nel testo non c'è

**Testo attuale (tesi)**
> Is there some other case C3 that is also similar to C1, but in which A is false (true)?

**Diagnosi.** Chiede di un caso che nel testo non compare. Due modelli con conoscenza del mondo
diversa rispondono diverso per costruzione, quindi la domanda non è stabile.

**Evidenza.** È la domanda con il peggior accordo di tutto il set. Sugli 85 item che gpt-5 e
qwen assegnano entrambi ad analogy, l'accordo grezzo è **0.19** e il kappa 0.02. Il kappa qui non
va letto come misura di accordo: le due distribuzioni di risposta sono molto sbilanciate e in
versi opposti, quindi l'accordo atteso per caso scende a 0.17 e il kappa resta schiacciato vicino
a zero comunque vadano le risposte.

L'evidenza che conta sono le due percentuali di `yes` sulla stessa base: gpt-5 risponde `yes` nel
5 per cento dei casi (4 item su 85), qwen nell'86 (73 su 85). Sulla stessa domanda, sullo stesso
testo, due modelli rispondono in modo quasi opposto.

**Proposta**
> Does the text rely on a single favourable comparison while ignoring comparable cases that would
> point the other way?

Resta difficile, ma riguarda quello che il testo fa e non quello che il mondo contiene, ed è
allineata al terminale Cherry Picking.

**Effetto sul grafo.** Nessuno.

---

### 2.7 slippery slope, CQ1 — si attiva sempre

**Testo attuale (tesi)**
> Do any of the causal links in the chain lack sufficient evidence to support the claim that they
> will (might, must) occur?

**Diagnosi.** In un testo informale c'è sempre almeno un anello senza prove esplicite, quindi la
risposta è `yes` per costruzione. Il nodo decide l'intero schema al primo passo e le risposte
alle altre cinque domande vengono calcolate e mai usate.

**Evidenza.** `yes` su 80 item su 80, cioè su tutti quelli che gpt-5 assegna allo schema. Anche
CQ4 sta a 1.00, CQ4.1 a 0.95 e CQ5 a 1.00, quindi quattro nodi su sei non distinguono quasi
nulla. L'accuracy 1.000 di questo schema è un artefatto e vale il 13 per cento delle 607 righe
del test set.

**Proposta**
> Is at least one step in the chain presented as inevitable while the text gives no reason why it
> must follow?

Sposta il criterio dalla presenza di prove alla pretesa di inevitabilità, che è ciò che
distingue lo slippery slope da una normale previsione condizionale, e non si attiva per default.

**Effetto sul grafo.** Nessuno.

**Attenzione.** Se questa domanda smette di attivarsi sempre, l'accuracy dello schema **scenderà**
rispetto al baseline. Non è un peggioramento, è la fine di un artefatto. Va detto esplicitamente
quando si riportano i numeri, altrimenti il confronto v0 contro v1 sembra un fallimento.

---

## 3. Da non toccare

**popular opinion CQ3.2 e CQ3.3**, su tradizione e natura. Si attivano quasi mai: su CQ3.2 `yes`
all'1 per cento con gpt-5 (87 item) e al 6 con qwen (95), su CQ3.3 al 2 e al 3 per cento. Ma nel
gold Appeal to Tradition vale 2 item e Appeal to Nature 1 (base gold). La rarità sta nei dati, non nella domanda. Su questo dataset quelle due
classi non sono valutabili, e va scritto così invece di riscrivere le domande.

**example CQ1**, con la polarità invertita rispetto a Walton. È una scelta dichiarata e il
Commento della tesi vi è coerente, quindi non è un errore.

**slippery slope CQ3.** A prima vista sembra girata, ma il Commento concorda con l'arco.

---

## 4. Riassunto

| # | Schema, CQ | Difetto | Grafo cambia | Serve l'autore |
|---|---|---|---|---|
| 2.1 | ad hominem CQ1 | spazio risposte non binario | no | no |
| 2.2 | ad hominem CQ3 | predicato incompleto, il nodo non discrimina | no | **sì** |
| 2.3 | cause to effect CQ2 | si attiva mai, il sottoalbero presuppone prove | solo con la proposta B | consigliato |
| 2.4 | cause to effect CQ2.1 ed example CQ2.1 | doppia condizione, direzione rovesciata | no | no |
| 2.5 | expert opinion CQ1 | due condizioni in disgiunzione | no | consigliato |
| 2.6 | analogy CQ4 | richiede conoscenza fuori dal testo | no | no |
| 2.7 | slippery slope CQ1 | si attiva sempre | no | no |

Sette proposte, otto domande, sei schemi. Sei proposte su sette non toccano il grafo, quindi si applicano
cambiando una riga di testo nel YAML e rigenerando il prompt.

Gli schemi non toccati sono **correlation to cause** e **popular opinion**. Su quei due le
risposte di v0 si riusano tali e quali.

---

## 5. Cosa aspettarsi dal confronto

Il risultato utile non è «v1 fa meglio di v0». È la scomposizione per schema di dove il guadagno
c'è e dove no, letta insieme a tre grandezze misurate su v0 e su v1. La diagnosi le ha già
misurate sulle esecuzioni di Eleni, e i numeri qui sotto vengono da lì.

**L'accordo fra modelli per singola CQ.** Se una domanda riformulata passa da kappa vicino a zero
a kappa alto, la riformulazione ha reso la domanda rispondibile, indipendentemente da come va
l'accuracy finale.

**Il tetto informativo del vettore di risposte.** Nella diagnosi valeva 0.669 pesato su gpt-5,
contro 0.558 della classe maggioritaria e 0.375 dell'albero. I tre numeri non sono stati
ricalcolati e restano non verificati: vanno rifatti dallo scorer sul test set degli esperimenti
prima di entrare in tesi, e il ricalcolo entra fra i requisiti della specifica 06 quando verrà
scritta. Il piano di tesi riporta per le stesse grandezze 0.665 e 0.398, quindi la differenza va
chiusa lì. Se v1 alza il tetto, le domande portano più
informazione. Se alza solo l'accuracy senza alzare il tetto, il guadagno viene dall'aggregazione,
non dalle domande.

**La distribuzione delle risposte per nodo.** Un nodo che passa da 1.00 o 0.00 a un valore
intermedio ha smesso di essere degenere, ed è il segnale più diretto che la riformulazione ha
funzionato.


---

## 6. Registro degli allineamenti alla tesi

Cosa è stato sovrascritto nei YAML per portarli alla versione della tesi, e perché. Serve a
sapere cosa c'era prima senza dover cercare nella cronologia.

### 6.1 Sostituzioni di solo testo

| File | CQ | Cosa c'era prima | Dove è finito |
|---|---|---|---|
| `example.yaml` | CQ2.1 | formulazione precedente alla revisione della tesi | §2.4, variante |
| `cause_to_effect.yaml` | CQ2.1 | «clearly biased manner in order to support» | §2.4, variante |
| `ad_hominem.yaml` | CQ3 | «committed to a group, cause or interest» | §2.2, variante |

Nessuna tocca il grafo.

### 6.2 `expert_opinion.yaml`, intervento strutturale

La tesi ha sdoppiato CQ4 in due nodi, e il YAML aveva ancora la forma precedente. Allineandolo

- **CQ4** diventa «Is there a conclusion C that differs from A but uses A to justify itself?»,
  con `yes` verso CQ4.1 e `no` verso good_argumentation
- **CQ4.1** è un nodo nuovo, «Is conclusion C actually implied by A?», con `no` verso
  non_sequitur e `yes` verso good_argumentation in riconvergenza
- **l'arco `na` sparisce.** Non serve più, perché il caso `C = A` esce ora su CQ4 `no` senza
  passare da CQ4.1. Era marcato `policy: true` ed era una decisione presa apposta, quindi vale la
  pena sapere che non è stata abbandonata, è stata resa superflua dalla struttura nuova
- **il caso della conclusione implicita resta scoperto.** La tesi non dà una regola per
  l'entimema. La nota nel YAML lo dice esplicitamente invece di risolverlo in silenzio

**Snapshot di regressione.** Questo intervento lo fa scattare. I valori attesi per
`expert_opinion` passano da `(5, 4, 7, True)` a `(6, 5, 7, True)`, che è quanto lo snapshot
registra oggi, cioè una CQ in più e una
profondità in più, mentre il numero di percorsi resta 7 e Good Argumentation resta raggiungibile.
Conviene comunque eseguire `validate_schemes.py` e copiare i valori che stampa lui, perché la
convenzione di conteggio della profondità è la sua.

### 6.3 Cosa non è stato toccato

Il vocabolario delle etichette, allora `schemes/_vocabulary.yaml` e oggi `labels/fallacies.yaml`,
era già a posto: il terminale è scritto `Good Argumentation`.

`analogy.yaml`, `popular_opinion.yaml` e `slippery_slope.yaml` erano già identici alla tesi e non
sono stati modificati.

`correlation_to_cause.yaml` era stato allineato in precedenza, ripristinando il testo della tesi
e invertendo i due archi di CQ5.

---

## 7. Rimosso dal codice, da riprendere in seguito

Il 8 settembre l'interprete è stato riportato a quello che i diagrammi contengono
davvero. Quello che è stato tolto non era sbagliato, era prematuro: sono decisioni su
casi che i diagrammi non prevedono, prese prima di avere una pipeline che riflette lo
stato attuale. Vanno ridiscusse dopo il baseline, non prima.

### 7.1 Le risposte oltre `yes` e `no`

Gli archi disegnati ammettono `yes` e `no` ovunque, tranne `ad_hominem` CQ1 che si
dirama su `positive` e `negative`. Niente altro. Era stata implementata la possibilità
che un annotatore o un modello rispondesse `idk`, con tre politiche di risoluzione.

- `charitable`, prendere gli archi da cui Good Argumentation resta raggiungibile,
  sul principio che una condizione non accertata non è un'accusa
- `branch`, esplorare tutti i rami e restituire l'insieme dei verdetti possibili
- `stop`, astenersi e scartare l'item

Con esse sono spariti anche `tolerant_answers`, che calcolava dal grafo quali archi
fossero i più indulgenti, e la distinzione fra risposta assente e risposta esplicitamente
ignota.

**Quando riprenderla.** Il prompt eseguito da Eleni ammetteva solo `Yes` e `No`: nei 14246
valori registrati dalle quattro pipeline (3569 con gpt-5) ci sono solo `Yes`, `No` e dieci
`[MISSING ANSWER]`. Nemmeno la tesi di Enrico prevede una terza risposta. Una versione
precedente di questo paragrafo diceva che il listato di prompt stampato nella tesi ammette
`Cannot be determined from the text`, ma nel PDF definitivo quel listato non c'è più: il
§4.3.2 (pp. 109-110) descrive i prompt solo a parole e chiede per lo stadio due una risposta
binaria, `Yes` oppure `No`, con una giustificazione obbligatoria. Una risposta non
determinabile compare nella tesi solo fra gli sviluppi futuri delle conclusioni (cap. 6,
pp. 149-151): «distinguere tra prove insufficienti e prove non esplicitate, per esempio con
una risposta non determinabile dal testo che non conduca automaticamente a una fallacia».

La terza risposta è quindi una scelta di progetto nostra, e il nostro prompt dello stadio due
la prevede, `cannot_be_determined`, insieme a `na` (specifica 03). La politica per risolverla
serve, e queste tre sono le candidate: tornano nella fase 3 come parametro dell'aggregatore
A2, che il piano di tesi chiama indulgente, esplora e fermati (sezione 6).

**Il principio che vale la pena non perdere.** La tolleranza sta nell'aggregazione, non
nel prompt. Dire a un modello «se sei incerto rispondi X» distorcerebbe proprio le
risposte per-CQ di cui vogliamo misurare l'incertezza. Il modello dice che non sa,
e la regola la applica l'interprete.

### 7.2 Gli archi di policy

`Edge` aveva un campo `policy`, per marcare un arco aggiunto da noi a coprire un caso
che il diagramma non contempla. L'unico in uso era `expert_opinion` CQ4 `na`, diventato
superfluo quando la tesi ha sdoppiato CQ4. Il meccanismo è stato tolto perché nessuno
schema lo usa più e perché un arco non disegnato non deve poter entrare nel grafo di
nascosto.

**Se dovesse riservire**, la reintroduzione è una riga in `Edge` più l'esclusione dai
conteggi di riconvergenza, ma va accompagnata da una regola su chi autorizza un arco non
presente nei diagrammi.

### 7.3 Gli status `abstained` e `ambiguous`

`Traversal.status` aveva quattro valori. Ne restano due, perché gli altri erano prodotti
soltanto dalla macchina IDK.

`abstained` significava che la traversata si era fermata senza verdetto perché una
risposta non era disponibile e la politica scelta era di astenersi invece di indovinare.
Senza risposte IDK non esiste più nessuna astensione.

`ambiguous` significava che la politica `branch` aveva esplorato più rami e ne erano
usciti più verdetti possibili. Una traversata con risposte secche percorre un ramo solo
e produce un verdetto solo, quindi non può essere ambigua.

**Dove finisce il disaccordo, adesso.** Non dentro una traversata, che è deterministica,
ma fra traversate. Annotatore contro modello, modello contro modello, oppure guardando
le distribuzioni di `propagate`. È anche il posto giusto, perché il disaccordo è una
proprietà di chi risponde, non del diagramma.

---

## 8. Errori trovati il 12 settembre 2026

Il documento è stato ricontrollato numero per numero contro la copia congelata e contro i
`results.csv` di Eleni. Le differenze rispetto alla versione precedente stanno qui, con la causa,
così chi rilegge sa cosa era sbagliato e perché.

| Dove | Diceva | Dice | Causa |
|---|---|---|---|
| §1 protocollo | 608 item di Eleni | 607 righe, 606 testi distinti, 601 negli esperimenti | fuori di uno: nessuno degli otto file di Eleni ha 608 righe, tutti ne hanno 607 con id da 1 a 607; i testi distinti sono 606 dopo la normalizzazione |
| §2.2 ad hominem CQ3 | si attiva su 29 item | `yes` su 42 item, 29 arrivi al terminale | quantità diversa: 29 sono gli item che escono su Guilt by Association, non le risposte `yes` |
| §2.3 tabella, gpt-5 CQ3 | 0.02 | 0.07 | il valore vero è 6 `yes` su 87; 0.023 è l'accuratezza dello schema, riportata due righe sotto, e sembra finita nella cella della tabella |
| §2.3 tabella, qwen CQ2 e CQ2.2 | 0.09 e 0.91 | 0.10 e 0.90 | corrispondono a 6 `yes` su 70 invece di 7: un item di scarto nello stesso verso |
| §2.5 expert opinion | 24 gold Appeal to authority | 24, con la base dichiarata | il 24 è giusto sul sottoinsieme predetto; una revisione intermedia lo aveva portato a 29, che è il numero su base gold, mescolando le due basi dentro lo stesso paragrafo |
| §2.5 expert opinion | kappa 0.38 su CQ1, le altre attorno a zero | 0.69 su CQ1, le altre fra 0.00 e 0.38 | 0.380 esiste, ma è il kappa di CQ4.1 sullo stesso sottoinsieme: sembra un valore preso dalla riga sbagliata |
| §2.7 slippery slope | 81 item su 81 | 80 su 80 | gpt-5 assegna allo schema 80 item; nessun altro modello ne dà 81 (qwen 93, llama 90, deepseek 100) |
| §3 popular opinion | CQ3.2 all'1 e al 5 per cento, CQ3.3 al 3 | CQ3.2 all'1 e al 6, CQ3.3 al 2 e al 3 | arrotondamento, e un valore per modello che mancava |
| §7.1 risposte registrate | 3575 valori | 14246 sulle quattro pipeline, 3569 con gpt-5 | nessun file dà 3575 |

**Cosa non è stato toccato.** I numeri aggregati della diagnosi che compaiono nel piano di tesi
sono stati ricalcolati e tornano esatti: gpt-5 pipeline 0.374, zero shot 0.565, riconoscimento
dello schema 0.834; sugli altri modelli la pipeline sta fra 0.402 e 0.412 e lo schema fra 0.815 e
0.843. Gli errori sopra sono quindi locali ai dettagli per nodo di questo documento.

**Cosa resta non verificato.** Il tetto informativo e i due termini di confronto della sezione 5,
0.669, 0.558 e 0.375, che richiedono lo scorer e gli aggregatori.

**Basi miste.** Nessuna rimasta: il §2.6 su analogy CQ4, che era su base gold, sta ora sul
sottoinsieme predetto come il resto del documento, con 0.19 di accordo e le due percentuali di
`yes` sugli 85 item che entrambi i modelli assegnano allo schema.
