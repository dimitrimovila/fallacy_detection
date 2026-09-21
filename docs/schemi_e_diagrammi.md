# Schemi argomentativi, domande critiche e diagrammi

**Fonte.** E. Bergamasco, *Schemi argomentativi e domande critiche nei Large Language Models come
strumento di individuazione delle fallacie*, tesi magistrale, Università di Padova, Dip. Studi
Linguistici e Letterari, A.A. 2025-2026. Relatore prof. Massimiliano Carrara, correlatore prof.
Giovanni Da San Martino. **PDF definitivo confermato dall'autore**, 14 settembre 2026, 181
pagine, **capitolo 3**, §3.1 «Diagrammi degli schemi», pp. stampate 46-92.

**Verifica sul PDF definitivo.** Il 14 e il 15 settembre 2026 le schede sono state confrontate
blocco per blocco con il PDF definitivo. I testi delle domande critiche, le domande di
identificazione e gli elenchi delle fallacie coincidono parola per parola con quanto trascritto
qui, con la sola eccezione registrata nel §0.4. Due **Schema** formali, ad hominem e slippery
slope, non coincidevano e sono stati sostituiti: vedi §0.5. Tutti i riferimenti di pagina sono
stati riallineati alla numerazione del definitivo; la trascrizione era stata fatta sul PDF da
135 pagine, i cui numeri erano +2.

**Cosa contiene questo documento.** Solo gli schemi. Schema argomentativo, domanda di
identificazione, domande critiche, diagramma, terminali, più le proprietà strutturali che ne
derivano. Dataset, metodo di annotazione, prompt, esecuzioni e risultati stanno in
`diagnosi_pipeline_cq.md`, fuori dal repository.

---

## Convenzioni

**Le domande critiche riportano le parole del capitolo 3**, con gli errori ortografici e
grammaticali corretti. Nessuna riformulazione, nessuna parola cambiata, nessun contenuto
aggiunto o tolto. Le correzioni applicate riguardano `committed`, `claimed`,
`claim` all'infinito e l'ortografia del terminale `Good Argumentation`.

**Il testo della tesi e le nostre osservazioni sono separati.** Tutto ciò che non è trascrizione
è marcato `⟶ nostra osservazione` oppure raccolto nel §12. Nulla di ciò che sta nelle schede va
attribuito a Enrico se non è trascritto.

**Gli archi dei diagrammi** sono stati letti dalle pagine rasterizzate del PDF, perché i
flowchart sono grafica vettoriale e l'estrazione testuale restituisce le etichette `yes` e `no`
in ordine non ricostruibile. Quello di *expert opinion* è stato riletto su questa versione perché
la sua struttura è cambiata, gli altri sette sono riportati dalla lettura precedente, i cui
elenchi CQ non hanno cambiamenti strutturali.

**Notazione.** `CQn: risposta → destinazione`. `⟲` segnala un arco di riconvergenza, cioè due o
più percorsi che entrano nello stesso nodo o terminale.

**La versione di riferimento è il capitolo 3.** Il capitolo 4 della stessa tesi riporta, in un
listato di prompt, una versione diversa di ad hominem, e i prompt effettivamente eseguiti negli
esperimenti ne riportano una terza. Il confronto è nel §11 e non contamina le schede.

**Nome del terminale non fallace.** Nel PDF definitivo il terminale verde si chiama
`Non-fallacious Argument`, mentre nelle versioni precedenti si chiamava `Good Argument` o
`Good Argumentation` (nell'appendice B del definitivo è rimasto `Good Argument`). Qui resta
`Good Argumentation`: è lo stesso terminale, il nome non cambia né il percorso né il senso.

---

## 0. Changelog rispetto alla revisione precedente di questo documento

La revisione precedente era basata sul PDF da 73 pagine. Confronto limitato agli schemi.

### 0.1 Risolto

| Problema | Stato |
|---|---|
| **Commenti in italiano sistematicamente obsoleti.** Su tutti e 8 gli schemi la sezione «Commento» descriveva un insieme di CQ diverso da quello elencato e diagrammato | **Risolto su tutti e 8.** Verificato leggendo l'incipit di ciascun Commento. Ora descrivono le CQ effettivamente diagrammate |
| **CQ1 di expert opinion con `and/or`**, connettivo di verità non definita | **Attenuato.** Ora è `or` |
| **CQ4 di expert opinion**, condizionale con la guardia dentro il testo della domanda | **Risolto.** Spezzata in CQ4 più CQ4.1 |
| **Numerazione incoerente di slippery slope**, `CQ4.1` nell'elenco e `CQ5.1` nel nodo | **Risolto.** Ora `C.Q. 4.1` in entrambi |

### 0.2 Modificato

| Schema | Cosa |
|---|---|
| example, CQ2.1 | riscritta ricalcando quella di cause to effect, da «selected on the basis of biased quality criteria that undermine the strength» a «selected in a clearly biased and malicious manner in order to undermine the validity» |

### 0.3 Invariato nella tesi, corretto da noi

Le due polarità invertite di correlation to cause CQ5 e slippery slope CQ5 sono ancora presenti
nel PDF. Le abbiamo corrette invertendo gli archi, vedi §4, §8 e §12.1.

Resta invariato e non corretto ad hominem privo di terminale Good Argumentation, perché non è un
errore di segno ma una scelta di copertura. Vedi §12.4.

### 0.4 Confronto con il PDF definitivo (14 settembre 2026, 181 pagine)

**correlation to cause, CQ3.** La tesi scriveva `a significant number of observation?`, il
definitivo scrive `observations`. La nostra trascrizione era già al plurale, quindi da correzione
nostra diventa testo della tesi: `observations` è stato tolto dall'elenco delle correzioni nelle
Convenzioni. La forma `claimed` resta una nostra correzione, il definitivo scrive ancora
`Is the claim relationship`.

**Le due polarità invertite del §0.3 sono ancora invertite anche nel definitivo.** Riverificato
sulle pagine renderizzate: correlation to cause CQ5 manda `yes` a Definist Fallacy (p. stampata
64), slippery slope CQ5 manda `no` a Slippery Slope (p. stampata 90). Le nostre due correzioni
di arco restano necessarie.

**Riferimenti di pagina, riallineati.** I numeri di pagina di questo documento venivano dal PDF
da 135 pagine ed erano +2 rispetto al definitivo. Sono stati tutti riportati alla numerazione
del definitivo, ricavandoli dal PDF stesso e non convertendoli a mano. §3.1 sta alle pp. 46-92.
Pagine dei diagrammi: 49 example, 54 analogy, 59 cause to effect, 64 correlation to cause,
70 ad hominem, 77 expert opinion, 83 popular opinion, 90 slippery slope.

**Domande critiche, domande di identificazione ed elenchi delle fallacie** coincidono parola per
parola con quanto trascritto qui. Sui diagrammi, i tre di correlation to cause, slippery slope
ed expert opinion sono stati riletti sulle pagine renderizzate del definitivo; gli altri cinque
sono stati controllati sull'estrazione delle etichette di nodi e terminali, che non mostra
cambiamenti. **Gli Schema formali non erano stati controllati**: due di essi non coincidevano,
vedi §0.5.

### 0.5 Schemi formali riallineati e sorgente B decaduta (15 settembre 2026)

Il confronto del §0.4 aveva riguardato domande critiche, domande di identificazione, elenchi
delle fallacie e diagrammi, ma non i blocchi **Schema**. Controllati anche quelli, due su otto
divergevano dal PDF definitivo, perché la trascrizione risaliva al PDF da 135 pagine.

| Schema | Che cosa è cambiato |
|---|---|
| ad hominem, §5 | sostituito con le due righe stampate a p. 68 («a is a person of bad character…»). Prima c'era la formulazione etotica lunga di Walton |
| slippery slope, §8 | sostituito con la versione in notazione `C0 … Cn` stampata a p. 88. Prima c'era la notazione `A0 … An` |
| slippery slope, CQ 2 | torna a `Cn` come nella tesi. La correzione `An`, applicata per coerenza con il vecchio schema, è stata tolta anche dalle Convenzioni |

**Perché conta.** Lo schema formale non è decorazione: nel prompt dello stadio uno viene mostrato
al modello insieme all'elenco chiuso degli otto schemi, al posto della semplice etichetta. Se i
nostri YAML portano lo schema vecchio, il modello vede un testo che non è quello della tesi.
Da verificare nei file di `schemes/`.

**Il listato dei prompt non esiste più.** Nel PDF definitivo il §4.3.2 (pp. 109-110) descrive i
prompt a parole e non ne riporta il testo. Dichiara che ogni domanda critica riceve una risposta
**binaria, Yes oppure No**, con giustificazione obbligatoria ancorata al testo. In tutte le 181
pagine non compare mai la stringa «Cannot be determined from the text». La sorgente **B** del
§11 decade quindi come fonte citabile, e la terza risposta del nostro prompt resta una scelta di
progetto nostra, da giustificare come tale e non come eredità della tesi.

**Lezione di metodo.** Un confronto fra due versioni della tesi non basta a validare i nostri
file: va confrontato il nostro documento con la tesi, blocco per blocco. Il §0.4 confrontava le
due stesure della tesi fra loro, e per questo non aveva visto le due divergenze.

---

## 1. Argument from example (§3.1.1, pp. 47-51)

**Schema**
```
In this particular case, the individual a has property F and also property G.
a is typical of things that have F and may or may not also have G.
Therefore, generally, if x has property F then x also has property G.
```

**Domanda di identificazione**
> Does the text use an example that is considered appropriate to draw a conclusion on the main topic?

**Domande critiche**
- **C.Q. 1** — Is the statement illustrated by the example clearly false or unrealistic?
- **C.Q. 2** — Is the example typical of the kinds of cases that the generalization ranges over?
  - **C.Q. 2.1** — Was the example cited selected in a clearly biased and malicious manner in order to undermine the validity of the generalization?
  - **C.Q. 2.2** — Is the example given so specific that it is statistically unrepresentative and therefore cannot be used to support the generalisation?
  - **C.Q. 2.3** — Is the example given so extreme that it invalidates the generalisation?
- **C.Q. 3** — Does the example cited have characteristics that make it too different from the generalisation drawn in the conclusion?

**Diagramma**
```
CQ1   yes → False Attribution
      no  → CQ2
CQ2   no  → CQ2.1
      yes → CQ3
CQ2.1 yes → Cherry Picking
      no  → CQ2.2
CQ2.2 yes → Hasty Generalization
      no  → CQ2.3
CQ2.3 yes → Appeal to Extremes
      no  → CQ3            ⟲
CQ3   yes → Weak Analogy
      no  → Good Argumentation
```

**Terminali** — False Attribution · Cherry Picking · Hasty Generalization · Appeal to Extremes ·
Weak Analogy · Good Argumentation

⟶ *nostra osservazione*. CQ1 ha polarità invertita rispetto alla CQ1 di Walton citata in nota
dalla tesi stessa, «Is the proposition claimed in the premise in fact true?». Qui `yes` significa
esempio falso, quindi fallacia. Il Commento della tesi è coerente con questa polarità, quindi è
una scelta, non un errore.

---

## 2. Argument from analogy (§3.1.2, pp. 52-56)

**Schema**
```
Generally, case C1 is similar to case C2.
A is true (false) in case C1.
Therefore A is true (false) in case C2.
```

**Domanda di identificazione**
> Does the text use a case considered similar to another in a certain respect to draw a conclusion about the main topic?

**Domande critiche**
- **C.Q. 1** — Are there differences between C1 and C2 that would tend to undermine the force of the similarity cited?
  - **C.Q. 1.1** — Does the analogy rely on an extreme scenario to make the situation seem better or worse?
- **C.Q. 2** — Is the similarity cited relevant to A?
- **C.Q. 3** — Is A true (false) in C1?
- **C.Q. 4** — Is there some other case C3 that is also similar to C1, but in which A is false (true)?

**Diagramma**
```
CQ1   yes → CQ1.1
      no  → CQ2
CQ1.1 yes → Appeal to Extremes
      no  → Weak Analogy
CQ2   no  → CQ1.1           ⟲
      yes → CQ3
CQ3   no  → False Premise
      yes → CQ4
CQ4   yes → Cherry Picking
      no  → Good Argumentation
```

**Terminali** — Appeal to Extremes · Weak Analogy · False Premise · Cherry Picking ·
Good Argumentation

---

## 3. Argument from cause to effect (§3.1.3, pp. 57-61)

**Schema**
```
Generally, if A occurs, then B will (might) occur.
In this case, A occurs (might occur).
Therefore in this case, B will (might) occur.
```

**Domanda di identificazione**
> Does this text use a known causal relationship as a basis for predicting or explaining a future or past event?

**Domande critiche**
- **C.Q. 1** — Is the implication that A leads to B (regardless of what judgement one might make about it) logically sound?
- **C.Q. 2** — Is the evidence cited (if there is any) strong enough to warrant the causal generalization?
  - **C.Q. 2.1** — Was the evidence cited selected in a clearly biased and malicious manner in order to undermine the validity of the generalization?
  - **C.Q. 2.2** — Is the evidence cited insufficient in quantitative terms to support the generalisation?
- **C.Q. 3** — Is the relationship between cause and effect based solely on temporal sequence?
- **C.Q. 4** — Are there other causal factors that clearly represent the real reason behind the occurrence of the effect?

**Diagramma**
```
CQ1   no  → False Premise
      yes → CQ2
CQ2   no  → CQ2.1
      yes → CQ3
CQ2.1 yes → Cherry Picking
      no  → CQ2.2
CQ2.2 yes → Hasty Generalization
      no  → CQ3            ⟲
CQ3   yes → Post Hoc
      no  → CQ4
CQ4   yes → Causal Reductionism
      no  → Good Argumentation
```

**Terminali** — False Premise · Cherry Picking · Hasty Generalization · Post Hoc ·
Causal Reductionism · Good Argumentation

---

## 4. Argument from correlation to cause (§3.1.4, pp. 62-66)

**Schema**
```
There is a positive or negative correlation between A and B.
Therefore, A causes B.
```

**Domanda di identificazione**
> Does this text start with two correlated events and conclude that one causes the other?

**Domande critiche**
- **C.Q. 1** — Is there actually a positive or negative correlation between A and B?
- **C.Q. 2** — Is one of the two events actually the cause of the other?
  - **C.Q. 2.1** — Is the rationale behind false causation based solely on the fact that one of the two correlated events occurs before the other?
  - **C.Q. 2.2** — Is the rationale behind false causation based solely on the fact that one of the two related events is regularly associated with the other?
- **C.Q. 3** — Is the claimed relationship between A and B based on a significant number of observations?
- **C.Q. 4** — Could there be a third factor C (or a set of several factors) that is the clear cause of B or of both A and B?
- **C.Q. 5** — Can it be shown that the increase or change in B is not solely due to the way B is defined?

⟶ *nota di versione*. `observations` al plurale è il testo del PDF definitivo, non una nostra
correzione. `claimed` resta una nostra correzione, la tesi scrive `Is the claim relationship`.
Vedi §0.4.

**Diagramma**
```
CQ1   no  → False Premise
      yes → CQ2
CQ2   no  → CQ2.1
      yes → CQ3
CQ2.1 yes → Post Hoc
      no  → CQ2.2
CQ2.2 yes → Questionable Cause
      no  → CQ3            ⟲
CQ3   no  → Hasty Generalization
      yes → CQ4
CQ4   yes → Causal Reductionism
      no  → CQ5
CQ5   yes → Good Argumentation      ⚑ arco corretto da noi
      no  → Definist Fallacy        ⚑ arco corretto da noi
```

⚑ **Correzione applicata.** Nel diagramma della tesi, verificato sulla pagina rasterizzata
p. stampata 64, gli archi di CQ5 sono l'opposto, `yes` verso Definist
Fallacy e `no` verso Good Argumentation. Con il testo della domanda quella direzione è rovesciata,
e lo conferma il Commento della tesi stessa, secondo cui la definist fallacy si configura quando
il mutamento di B è «un effetto della definizione stessa di B, piuttosto che un mutamento
sostanziale del fenomeno osservato». Abbiamo invertito i due archi lasciando intatto il testo
della domanda. Riverificato sul PDF definitivo il 14 settembre 2026: gli archi sono ancora
invertiti, la correzione resta necessaria.

**Terminali** — False Premise · Post Hoc · Questionable Cause · Hasty Generalization ·
Causal Reductionism · Definist Fallacy · Good Argumentation

---

## 5. Argument ad hominem (§3.1.5, pp. 67-72)

**Schema**
```
a is a person of bad character.
Therefore, a's argument A should not be accepted.
```

⚑ **Schema sostituito il 15 settembre 2026.** Questo è il testo stampato a p. 68 del PDF
definitivo, con citazione [6]. La revisione precedente di questo documento riportava la
formulazione etotica più lunga di Walton («If x is a person of good (bad) moral character, then
what x says should be accepted as more plausible…»), che veniva dal PDF da 135 pagine e non
compare più nella tesi. Conta perché lo schema formale viene mostrato al modello nel prompt
dello stadio uno. Vedi §0.5.

**Domanda di identificazione**
> Does the text shift the focus from arguments to personal judgments about the other person?

**Domande critiche**
- **C.Q. 1** — Is the opinion expressed about the other person negative (and intended to undermine their credibility) or positive (and intended to enhance their credibility)?
- **C.Q. 2** — Does the direct attack on the other person amount to an accusation that they are biased?
- **C.Q. 3** — Does the direct attack on the other person consist of an accusation that they are committed?
- **C.Q. 4** — Is the direct attack on the other person an accusation of hypocrisy?
- **C.Q. 5** — Is the direct attack on the other person intended to undermine their self-confidence?
- **C.Q. 6** — Is a direct attack on the other person used as a pretext to discredit them even before the discussion has begun?

**Diagramma**
```
CQ1   positive → Appeal to Authority
      negative → CQ2
CQ2   yes → Ad Hominem (Circumstantial)
      no  → CQ3
CQ3   yes → Ad Hominem (Guilt by Association)
      no  → CQ4
CQ4   yes → Ad Hominem (Tu Quoque)
      no  → CQ5
CQ5   yes → Ad Fidentia
      no  → CQ6
CQ6   yes → Poisoning the Well
      no  → Ad Hominem (Abusive)
```

**Terminali** — Appeal to Authority · Ad Hominem (Circumstantial) · Ad Hominem (Guilt by
Association) · Ad Hominem (Tu Quoque) · Ad Fidentia · Poisoning the Well · Ad Hominem (Abusive)

**Unicità strutturale.** È l'unico degli otto schemi **privo di terminale Good Argumentation**.
Ogni percorso termina in una fallacia.

⟶ *nostra osservazione*. CQ1 è l'unica domanda dell'intero set che non si dirama su `yes` e `no`
ma su `positive` e `negative`. Va tenuto presente in codifica, perché lo spazio delle risposte
non è uniforme fra i nodi.

---

## 6. Argument from expert opinion (§3.1.6, pp. 73-79)

**Schema**
```
S is an authoritative source (whether individual or collective) in the domain D.
S asserts that A is known to be true.
A is in D.
(A implies the conclusion C)
Therefore A (and/or C) may (plausibly) be taken as true.
```

**Domanda di identificazione**
> Does this text use the opinion of a (supposedly) authoritative source as evidence to support a claim?

**Domande critiche**
- **C.Q. 1** — Is S in a position to know whether A is true or false, or is S a genuine expert recognised by the community of experts in D?
- **C.Q. 2** — The fact that S has asserted (or claimed) A is used as the sole supporting consideration for the truth of the conclusion, without providing any other type of argument?
- **C.Q. 3** — Did S really assert (or claim) A as true?
  - **C.Q. 3.1** — Is S taking part in the discussion?
- **C.Q. 4** — Is there a conclusion C that differs from A but uses A to justify itself?
  - **C.Q. 4.1** — Is conclusion C actually implied by A?

**Variabili.** `S` fonte, `D` dominio, `A` affermazione della fonte, `C` conclusione di chi
argomenta.

**Diagramma**
```
CQ1   no  → Irrelevant Authority
      yes → CQ2
CQ2   yes → Appeal to Authority
      no  → CQ3
CQ3   no  → CQ3.1
      yes → CQ4
CQ3.1 no  → False Attribution
      yes → Strawman
CQ4   no  → Good Argumentation
      yes → CQ4.1
CQ4.1 no  → Non Sequitur
      yes → Good Argumentation   ⟲
```

**Terminali** — Irrelevant Authority · Appeal to Authority · False Attribution · Strawman ·
Non Sequitur · Good Argumentation

⟶ *nostra osservazione*. Con lo sdoppiamento di CQ4 il caso in cui chi argomenta si limita a
riportare l'affermazione, cioè `C = A`, esce su CQ4 `no` verso Good Argumentation senza bisogno
di codificare un valore `na`. Resta scoperto il caso della conclusione **implicita**, l'entimema,
per cui il diagramma non dà una regola. Vedi §12.3.

---

## 7. Argument from popular opinion (§3.1.7, pp. 80-85)

**Schema**
```
If a large majority of some reference group accept A as true, then there is a
presumption in favour of A.
A large majority of the reference group accept A as true.
Therefore there is a presumption in favour of A.
```

**Domanda di identificazione**
> Does this text use the fact that the conclusion is supported by a relatively large number of people as its main argument in favour of that conclusion?

**Domande critiche**
- **C.Q. 1** — Are there any indications that the group of people who maintain that A is true (or false) was clearly chosen because they were already biased?
- **C.Q. 2** — Are there any indications that the group of people who maintain that A is true (or false) is clearly too small to be considered statistically significant?
- **C.Q. 3** — Is the fact that A is considered to be true (or false) by many people a good reason to believe that A is in fact true (or false)?
  - **C.Q. 3.1** — Is the presumption in favour of A based solely on the fact that it is accepted by a large majority of the reference group?
  - **C.Q. 3.2** — Does the majority of the reference group rely on the concept of tradition to assert A?
  - **C.Q. 3.3** — Does the majority of the reference group rely on the concept of nature to assert A?

**Diagramma**
```
CQ1   yes → Cherry Picking
      no  → CQ2
CQ2   yes → Hasty Generalization
      no  → CQ3
CQ3   no  → CQ3.1
      yes → Good Argumentation
CQ3.1 yes → Ad Populum
      no  → CQ3.2
CQ3.2 yes → Appeal to Tradition
      no  → CQ3.3
CQ3.3 yes → Appeal to Nature
      no  → Ad Populum       ⟲
```

**Terminali** — Cherry Picking · Hasty Generalization · Ad Populum · Appeal to Tradition ·
Appeal to Nature · Good Argumentation

⟶ *nostra osservazione*. Ad Populum è raggiungibile da due archi distinti, CQ3.1 `yes` e CQ3.3
`no`. L'etichetta terminale da sola non identifica il percorso.

---

## 8. Slippery slope argument (§3.1.8, pp. 86-92)

**Schema**
```
Case C0 is tentatively acceptable as an initial presumption.
There exists a series of cases, C0, C1,..., Cn−1, where each case leads to the next by a
combination of causal, precedent, and/or analogy steps.
There is a climate of social opinion such that once people come to accept each step as
plausible (or as accepted practice), they will then be led to accept the next step.
The penultimate step Cn−1 leads to a horrible outcome, C0, which is not acceptable.
Therefore, C0 is not acceptable (contrary to the presumption of the initial premise).
```

⚑ **Schema sostituito il 15 settembre 2026.** Questo è il testo stampato a p. 88 del PDF
definitivo, con citazione [9], in notazione `C0 … Cn`. La revisione precedente riportava la
versione in notazione `A0 … An`, che veniva dal PDF da 135 pagine. Di conseguenza la CQ 2 torna
a dire `Cn`, che è coerente con lo schema: la nostra correzione da `Cn` ad `An` non serve più.

⟶ *nostra osservazione*. Il testo della tesi contiene un refuso: «The penultimate step Cn−1
leads to a horrible outcome, C0» dovrebbe dire `Cn`, ed è `Cn` l'esito a cui si riferisce la
CQ 2. Lo lasciamo come è stampato e lo segnaliamo qui.

**Domanda di identificazione**
> The text contains an argument claiming that if you allow or carry out a particular action (even if it's small or seemingly harmless), it will inevitably lead (through a chain of intermediate consequences) to an extreme or catastrophic final outcome?

**Domande critiche**
- **C.Q. 1** — Do any of the causal links in the chain lack sufficient evidence to support the claim that they will (might, must) occur?
- **C.Q. 2** — Is the outcome Cn as bad as suggested?
- **C.Q. 3** — Is it possible to provide a precise definition that removes the ambiguity to such an extent as to halt the decline?
- **C.Q. 4** — Are there other steps required to fill in the sequence of events and make it plausible?
  - **C.Q. 4.1** — Would these steps lead to a different conclusion?
- **C.Q. 5** — Are there weak links in the sequence, where specific critical questions should be asked on whether one event will really lead to another?

**Diagramma**
```
CQ1   yes → Slippery Slope
      no  → CQ2
CQ2   no  → Slippery Slope         ⟲
      yes → CQ3
CQ3   yes → Definist Fallacy
      no  → CQ4
CQ4   yes → CQ4.1
      no  → CQ5
CQ4.1 yes → Slippery Slope         ⟲
      no  → CQ5                    ⟲
CQ5   yes → Slippery Slope         ⟲ · ⚑ arco corretto da noi
      no  → Good Argumentation     ⚑ arco corretto da noi
```

⚑ **Correzione applicata.** Nel diagramma della tesi, verificato sulla pagina rasterizzata
p. stampata 90, gli archi di CQ5 sono l'opposto, `no` verso Slippery
Slope e `yes` verso Good Argumentation. La presenza di anelli deboli nella catena è la condizione
fallace, non quella sana, e lo conferma il Commento della tesi sulla CQ1, «se anche un solo
anello causale risulta privo di adeguata giustificazione, la catena si spezza e l'argomentazione
ricade nella fallacia dello slippery slope». Abbiamo invertito i due archi lasciando intatto il
testo della domanda. Riverificato sul PDF definitivo il 14 settembre 2026: gli archi sono ancora
invertiti, la correzione resta necessaria.

**Terminali** — Slippery Slope · Definist Fallacy · Good Argumentation

⟶ *nostra osservazione*. Slippery Slope è raggiungibile da quattro archi distinti. È lo schema
con la maggiore ambiguità fra etichetta e percorso.

---

## 9. Indice inverso fallacia → schemi

Ricavato dai diagrammi di questa versione. Vincolo di consistenza per l'annotazione in direzione
fallacia nota verso schema da trovare. Se un'etichetta non compare qui, nessuno degli otto schemi
può produrla.

| Fallacia | Schemi che la producono |
|---|---|
| Ad Fidentia | ad hominem |
| Ad Hominem (Abusive) | ad hominem |
| Ad Hominem (Circumstantial) | ad hominem |
| Ad Hominem (Guilt by Association) | ad hominem |
| Ad Hominem (Tu Quoque) | ad hominem |
| Ad Populum | popular opinion |
| Appeal to Authority | expert opinion, ad hominem |
| Appeal to Extremes | example, analogy |
| Appeal to Nature | popular opinion |
| Appeal to Tradition | popular opinion |
| Causal Reductionism | cause to effect, correlation to cause |
| Cherry Picking | example, analogy, cause to effect, popular opinion |
| Definist Fallacy | correlation to cause, slippery slope |
| False Attribution | example, expert opinion |
| False Premise | analogy, cause to effect, correlation to cause |
| Hasty Generalization | example, cause to effect, correlation to cause, popular opinion |
| Irrelevant Authority | expert opinion |
| Non Sequitur | expert opinion |
| Poisoning the Well | ad hominem |
| Post Hoc | cause to effect, correlation to cause |
| Questionable Cause | correlation to cause |
| Slippery Slope | slippery slope |
| Strawman | expert opinion |
| Weak Analogy | example, analogy |

**Non raggiungibili da alcuno schema** — `Appeal to Emotion`.

---

## 10. Proprietà strutturali

**I diagrammi non sono alberi.** Sei schemi su otto contengono riconvergenze. Solo **ad hominem**
è un albero puro.

| Schema | Riconvergenza su nodo decisionale | Riconvergenza su terminale |
|---|---|---|
| example | CQ2.3 `no` → CQ3 | — |
| analogy | CQ2 `no` → CQ1.1 | — |
| cause to effect | CQ2.2 `no` → CQ3 | — |
| correlation to cause | CQ2.2 `no` → CQ3 | — |
| ad hominem | — | — |
| expert opinion | — | Good Argumentation, 2 archi |
| popular opinion | — | Ad Populum, 2 archi |
| slippery slope | CQ4.1 `no` → CQ5 | Slippery Slope, 4 archi |

**Conseguenza.** La ricostruzione a ritroso del vettore di risposte a partire dalla coppia (CQ di
uscita, verdetto) non è univoca. Esempio su *example*, si arriva a CQ3 sia da `CQ2=yes`, sia da
`CQ2=no, CQ2.1=no, CQ2.2=no, CQ2.3=no`. Il vettore completo delle risposte va **registrato**, non
ricostruito.

**Profondità**, numero di nodi decisionali sul percorso più lungo.

| Schema | Profondità | Schema | Profondità |
|---|---|---|---|
| correlation to cause | 7 | ad hominem | 6 |
| example | 6 | popular opinion | 6 |
| cause to effect | 6 | slippery slope | 6 |
| expert opinion | 6 | analogy | 4 |

**Spazio delle risposte non uniforme.** Tutti i nodi si diramano su `yes` e `no` tranne la CQ1 di
ad hominem, che si dirama su `positive` e `negative`.

**Uscita anticipata.** Le CQ finali di ogni schema sono raggiunte solo da una frazione degli
item, e non in modo casuale. Se il gold per-CQ viene registrato solo lungo il percorso, la
supervisione sulle CQ tardive è scarsa in modo sistematico.

---

## 11. Divergenze fra le scritture degli stessi schemi

Gli stessi otto schemi esistono in quattro scritture che non coincidono. **La versione canonica è
la prima**, ed è quella trascritta nei §1-8.

| | sorgente | ruolo |
|---|---|---|
| **A** | tesi, capitolo 3 | ufficiale e citabile |
| **B** | tesi da 135 pagine, §4.3, listato di prompt | **non esiste più.** Nel PDF definitivo il §4.3.2 descrive i prompt solo a parole, senza listato. Resta citato qui sotto perché documenta un prompt mai eseguito, ma non è più una fonte citabile |
| **C** | `Lavoro di Enrico/src/prompts/template_stage2_*.txt` | i prompt effettivamente eseguiti |
| **D** | `Thesis/src/YAML/*.yaml` | la codifica di Dumitru |

Divergenze di testo, elencate solo dove sono sostanziali. Le differenze puramente ortografiche
non sono riportate, perché in questo documento e nei nostri file l'ortografia è corretta.

| Schema, CQ | A, tesi cap. 3 | C, eseguito | D, YAML |
|---|---|---|---|
| example CQ2.1 | «**Was** the example cited selected in a clearly biased and malicious manner…» | «**Is** the example cited selected…» | «…selected **on the basis of biased quality criteria that undermine the strength**…» |
| analogy CQ4 | «…A is false (true)?» | «…A is false (**or** true)?» | come A |
| cause to effect CQ2.1 | «…clearly biased **and malicious** manner in order to **undermine** the validity…» | come A | «…clearly biased manner in order to **support** the generalization» |
| correlation to cause CQ5 | «Can it be shown that … is **not** solely due to the way B is defined?» | come A | «Is the increase or change in B **solely an artefact** of the way B is defined?» |
| ad hominem CQ1 | si dirama `positive` / `negative` | il testo chiede negativo oppure positivo ma il formato **impone `Yes` o `No`** | `answer_space: ["positive", "negative"]` |
| ad hominem CQ3 | «…that they **are committed**?» | «…that they **committed [something]**?» | «…that they **are committed to a group, cause or interest**?» |
| ad hominem, numero di CQ | 6 | 6 | 6. In **B** sono **7**, in ordine diverso |
| expert opinion CQ1 | «…true or false, **or** is S…» | «…true or false, **and/or** is S…» | come A |
| expert opinion CQ4 | spezzata in CQ4 più CQ4.1 | come A, con la variabile scritta minuscola, `uses **a** to justify itself` | nodo unico, forma condizionale precedente, `answer_space: ["yes", "no", "na"]` |

**Nel prompt eseguito il segnaposto `[something]` è rimasto nel testo** inviato al modello, quindi
la CQ3 di ad hominem è stata posta in forma incompiuta in tutti gli esperimenti.

**Conferma che C è la versione dei dati.** Il numero di domande per schema nei prompt eseguiti
coincide esattamente con il numero di risposte registrate nei risultati su tutti e otto gli
schemi.

⟶ *nostra osservazione operativa*. D adotta le parole di A, con l'ortografia corretta. Restano
due punti in cui D si era discostato non per ortografia ma per senso, cause to effect CQ2.1 e
correlation to cause CQ5. Allineandoli ad A la formulazione torna quella della tesi, e con essa
torna il disallineamento fra il testo della domanda e il terminale a cui porta. Sono descritti
in §12.1 e §12.2 e vanno risolti dall'autore, cambiando il testo oppure l'arco, non da noi
riscrivendo la domanda per conto nostro.

---

## 12. Osservazioni nostre sugli schemi

Nulla in questo paragrafo è testo della tesi.

### 12.0 Ambiguità di CQ3 di ad hominem

Corretta l'ortografia, la domanda resta ambigua. «An accusation that they are committed» non
dice committed a che cosa, e il terminale è Guilt by Association. Se il senso è l'impegno in una
posizione incoerente si tratta di tu quoque, se è l'appartenenza a un gruppo screditato si tratta
di colpa per associazione. Il diagramma sceglie la seconda, il testo non lo dice. Da chiarire con
l'autore, perché non è un refuso.

### 12.1 Polarità invertite, corrette

Due nodi hanno gli archi invertiti rispetto al testo della propria domanda. Verificato leggendo i
diagrammi sulle pagine rasterizzate del PDF, non dall'estrazione testuale.

**correlation to cause, CQ5**, p. stampata 64. «Can it be shown that
the increase or change in B is **not** solely due to the way B is defined?» Nella tesi `yes`
porta a Definist Fallacy. Ma `yes` significa che si può dimostrare che il mutamento **non**
dipende dalla definizione, cioè che l'argomento regge. Il Commento della tesi lo conferma, la
definist fallacy si configura quando il mutamento è «un effetto della definizione stessa di B».

**slippery slope, CQ5**, p. stampata 90. «Are there weak links in the
sequence…?» Nella tesi `no` porta a Slippery Slope. Ma la presenza di anelli deboli è la
condizione fallace. Il Commento della CQ1 lo conferma, «se anche un solo anello causale risulta
privo di adeguata giustificazione … l'argomentazione ricade nella fallacia dello slippery slope».

**Correzione scelta.** Invertire i due archi, lasciando il testo delle domande esattamente come è
nella tesi. È la correzione minima e non richiede di riformulare nulla. Allinea inoltre il
diagramma al Commento che l'autore stesso ha scritto, quindi non introduce un'interpretazione
nostra.

**Stato sul PDF definitivo.** Entrambe le inversioni sono ancora presenti nel PDF definitivo del
14 settembre 2026, riverificate sulle pagine renderizzate. Le due correzioni restano.

**Nota per il confronto con i risultati.** I 608 item sono stati valutati con gli archi non
corretti. Ma gli archi vengono applicati dal codice sulle risposte già registrate, non dal
prompt, quindi il ricalcolo con gli archi corretti non richiede una sola nuova chiamata a un
modello. Entrambe le versioni sono ottenibili dagli stessi dati.

### 12.2 Domande a doppia condizione

**example CQ2.1 e cause to effect CQ2.1** contengono due condizioni congiunte, `clearly biased`
**and** `malicious`, e attribuiscono alla selezione l'intento di *minare* la generalizzazione,
mentre nel flusso il nodo serve a individuare chi seleziona prove per *sostenere* indebitamente
la propria tesi.

**expert opinion CQ1** unisce due condizioni distinte, essere in posizione di sapere ed essere
riconosciuto dalla comunità di esperti. Con `or` il connettivo è definito, ma il caso «fonte
competente ma non riconosciuta dai pari» resta indistinguibile da «fonte riconosciuta ma non
competente».

### 12.3 Entimemi

CQ4 di expert opinion presuppone una conclusione `C` esplicita nel testo. La conclusione
implicita non ha una regola. Confronto utile, EthiX ricostruisce l'entimema dal tema del dibattito
ai fini dell'annotazione ma conserva nel dataset il testo originale.

### 12.4 Ad hominem senza esito non fallace

Ad hominem è l'unico schema in cui nessun percorso porta a Good Argumentation. Su questo schema
l'accuratezza della classe non fallace non è definita, e qualunque media macro che la includa
deve dirlo.

### 12.5 Prefissi di ad hominem da mappare

Le varianti di ad hominem sono prefissate nei diagrammi, `Ad Hominem (Tu Quoque)`, ma nel
vocabolario delle annotazioni compaiono senza prefisso, `Tu quoque`. Serve una mappatura
esplicita fra le due forme prima di calcolare qualunque metrica.