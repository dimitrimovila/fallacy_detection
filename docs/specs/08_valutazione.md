# Specifica 08. Valutazione

Bozza, 10 settembre 2026, aggiornata l'11 settembre 2026. Contiene i requisiti già decisi e il protocollo di valutazione. Curve di astensione e parametri del bootstrap sono fissati in P11 e P12.

## Requisiti già decisi

1. **Test set degli esperimenti senza i doppioni.** Tutte le metriche degli esperimenti si calcolano sul test set di `items.csv`, cioè le righe con `in_test_set` vero, che non contiene i cinque doppioni (`docs/dati.md`, sezione 5.2). Un doppione contato due volte pesa doppio e, se un modello sbaglia su un testo, di solito sbaglia anche sull'altro. I doppioni non entrano in nessun esperimento. L'unica eccezione è la riproduzione dei numeri della diagnosi di Eleni, che serve solo a validare lo scorer e usa tutte le righe dei suoi `results.csv` così come sono.
2. **Baseline di Eleni ricalcolata sul test set degli esperimenti.** I risultati di Eleni con cui si confrontano quelli degli esperimenti (pipeline e zero shot dei suoi quattro modelli) si ricalcolano dai suoi `results.csv` sugli item di `items.csv`, cioè sullo stesso test set degli esperimenti. La riproduzione esatta dei numeri della diagnosi su tutte le sue righe resta un test interno dello scorer, che ne valida il calcolo, e non compare nei risultati. Le due righe di Eleni con lo stesso testo (id 469 e 495, un item solo in `items.csv`, `33f6dfaf9c4e41f1`) diventano una riga sola nella baseline: si tiene la riga con id 469, in ogni modello e condizione. Le due righe danno un verdetto diverso in un run su otto (deepseek zero-shot: Appeal to Authority la 469, Ad Populum la 495); negli altri sette il verdetto coincide.

## Protocollo di valutazione

Riscrive in forma di requisiti il §7 del piano di tesi. Numeri verificati l'11 settembre 2026 contro `labels/fallacies.yaml` e la copia congelata, e di nuovo il 21 settembre 2026 contro `data/items.csv` e `data/annotations.csv`. Salvo dove è detto altro, ogni numero vale sul test set degli esperimenti (601 item, requisito 1), e ogni numero che entra in tesi viene da `argfallacy.eval`.

P1. **Condizioni sullo schema.** Ogni aggregatore si valuta in due condizioni: schema gold, che isola l'aggregazione, e schema predetto dallo stadio uno, che misura il sistema completo. Le due condizioni si riportano sempre separate.

P2. **Spazi di etichette.** Due spazi, definiti in `labels/fallacies.yaml` alla chiave `spaces`:
* `fine`: 25 classi, cioè tutti i terminali dei diagrammi, Good Argumentation compresa. Nel gold ne compaiono 23: `definist_fallacy` e `false_attribution` hanno supporto zero. Il piano diceva "23 classi" contando le classi con supporto; lo spazio è definito sui terminali, quindi le classi sono 25 e 23 quelle con supporto.
* `collapsed_symmetric`: 20 classi. Le sei varianti di ad hominem (abusive, circumstantial, guilt by association, tu quoque, ad fidentia, poisoning the well) confluiscono in `ad_hominem`, tu quoque compreso. Nel gold ne compaiono 18.
* `collapsed_compat` (21 classi, tu quoque separato) serve solo al modo `compat` dello scorer per riprodurre i numeri di Eleni e non compare nei risultati.

L'accuratezza si calcola su tutti gli item. Le medie macro (F1 macro) si calcolano sulle classi con supporto nel gold, dichiarando quali classi sono escluse e perché; una predizione di una classe senza supporto resta un errore.

P3. **Spaccati per schema e per corpus.** Ogni risultato principale si riporta per schema e per corpus, a parità di spazio di etichette, con il numero di item e di classi accanto: un'accuratezza su un corpus con poche classi non si confronta con una su un corpus con molte classi senza dirlo. I corpus con meno di 20 item non hanno uno spaccato proprio: entrano solo nel totale, e il rapporto lo dichiara con il nome del corpus e il numero di item. Oggi, spazio fine con il collassato fra parentesi: LOGIC 192 item e 18 classi (13), NLAS 197 e 16 (12), EthiX 144 e 10 (10), Reddit 67 e 4 (4); ElecDebate, con 1 item, sta solo nel totale. Per schema: ad hominem 94, analogy 83, cause to effect 45, correlation to cause 11, example 86, expert opinion 78, popular opinion 109, slippery slope 95. La regola dei 20 item vale solo per i corpus: ogni schema si riporta sempre, anche correlation to cause con 11 item, con il numero di item accanto, e il rapporto segnala ogni schema con meno di 30 item (oggi correlation to cause).

P4. **Ablazione senza NLAS.** Ogni risultato principale si ripete senza gli item NLAS, corpus sintetico generato da LLM: 404 item.

P5. **Incertezza e confronti.** Intervalli di confidenza bootstrap su accuratezza e F1 macro. I confronti fra aggregatori sono appaiati sugli stessi item, con il bootstrap appaiato di P12 e il test esatto di McNemar come controllo.

P6. **Seconda valutazione sul workbook.** Gli item di `items.csv` fuori dal test set che hanno un verdetto terminale di Enrico in `annotations.csv` sono un secondo insieme di valutazione, con etichette prodotte dentro la tassonomia dei diagrammi: oggi 113 item, nessuno con due verdetti terminali diversi. I verdetti `idk` e quelli fuori ambito non entrano. I 9 item con etichetta di partenza (`starting_label`) uguale a `false_cause` sono esclusi anche loro: quelle righe non hanno uno schema, quindi non esiste l'insieme di terminali su cui calcolare il credito parziale della specifica 01. Conteggio ricostruibile: 170 item fuori dal test set con un verdetto di Enrico, meno gli `idk` restano 122, meno i 9 `false_cause` restano 113. Il credito parziale resta in uso per `any_fallacy`.

P7. **Per CQ.** Per ogni modello: accuratezza e calibrazione (Brier, ECE) contro le risposte di Dumitru in `annotations.csv`, annotate secondo la mappa; accordo fra modelli con kappa, letto sempre insieme alla distribuzione delle risposte; quota di "non determinabile dal testo".

P8. **Accordo umano.** Alpha di Krippendorff e kappa fra Enrico, Eleni e Dumitru sui verdetti; Dumitru contro Eleni per CQ sul campione annotato da Eleni; coerenza interna di Dumitru sul dieci per cento delle righe riannotato a freddo. Una cella vuota non è un verdetto (`docs/dati.md`).

P9. **Incertezza dalle giustificazioni.** Stima dell'incertezza dalle giustificazioni dei modelli (marcatori linguistici, similarità fra modelli: proposta di Eleni) e sua relazione con il disaccordo umano.

P10. **Criterio di successo.** Si fissa prima di scrivere lo scorer e prima di guardare qualunque risultato degli esperimenti, e vale per la decisione del 26 ottobre (piano di tesi, punto di decisione).
* Baseline: A0, il diagramma fisso con le risposte secche di un solo modello, che riproduce l'impostazione della diagnosi.
* Metrica che decide: accuratezza nello spazio fine (P2) in condizione schema gold (P1). L'accuratezza in condizione schema predetto si calcola con lo stesso confronto e fa da conferma.
* Confronto: appaiato sugli stessi item del test set degli esperimenti, fra l'aggregatore e A0, con intervallo di confidenza sulla differenza di accuratezza (P5).
* Si parla di miglioramento solo se l'intervallo del confronto appaiato esclude lo zero. Un intervallo che contiene lo zero si riporta come assenza di una differenza misurabile, non come un miglioramento piccolo.
* Modello di riferimento: l'A0 di riferimento è quella del modello con l'accuratezza più alta sul test set degli esperimenti, misurata come A0 con la metrica che decide. Il modello si sceglie prima di calcolare i risultati degli aggregatori, e la scelta si scrive nel manifest della valutazione; a parità di accuratezza vale il primo nome di modello in ordine alfabetico.
* Confronto diagnostico: un aggregatore che usa un solo modello si confronta anche con l'A0 dello stesso modello, con lo stesso confronto appaiato, e il risultato si riporta accanto al primo. Non decide il successo: separa due cause. Se l'aggregatore non batte l'A0 del suo stesso modello, il risultato negativo viene dall'aggregazione; se lo batte ma resta sotto l'A0 del modello migliore, viene dalla scelta del modello.

P11. **Curve di astensione.**
* Confidenza: per un aggregatore che restituisce una distribuzione sui terminali (A2, A3, A4, e A5 quando restituisce una probabilità), la confidenza di un item è la probabilità del verdetto più alto. Con una soglia, l'aggregatore si pronuncia sugli item con confidenza almeno pari alla soglia e si astiene sugli altri; la copertura è la quota di item su cui si pronuncia. A0 e A1 sono un punto solo, a copertura piena, salvo quando il diagramma si ferma su `idk` con la politica `stop`, che conta come astensione. Il margine fra i due verdetti più probabili si calcola come controllo.
* Assi: in ascissa la copertura, da 0 a 1; in ordinata l'accuratezza sugli item su cui l'aggregatore si pronuncia (accuratezza selettiva), nello spazio fine e in condizione schema gold, come in P10. La curva ordina gli item per confidenza decrescente e prende i primi k, per ogni k da 1 a N; a parità di confidenza l'ordine è per `item_id`, così la curva è deterministica.
* Numero riassuntivo: area sotto la curva errore contro copertura (AURC), più bassa è meglio, con intervallo di confidenza (P12).
* Confronto: due aggregatori si confrontano a parità di copertura, mai di soglia, perché le loro confidenze non hanno la stessa scala. Livelli fissati prima di guardare i risultati: 100, 95, 90, 80 e 70 per cento; ci si ferma al 70 perché un sistema che rinuncia su metà dei casi non è utile. A ogni livello c ciascun aggregatore si pronuncia sui suoi primi round(c·N) item per confidenza. Gli intervalli della differenza di accuratezza selettiva e della differenza di AURC vengono dal bootstrap appaiato di P12, che ricalcola entrambe le curve sugli stessi ricampionamenti; una differenza si dichiara solo se l'intervallo esclude lo zero.
* Accanto a ogni curva, la composizione degli item sospesi: quota di Good Argumentation, schema, corpus.

P12. **Bootstrap.**
* 10000 ricampionamenti, intervallo percentile al 95 per cento.
* Un solo seme, fissato nella configurazione della valutazione e scritto nel manifest. Gli indici dei ricampionamenti si generano una volta e valgono per tutti gli aggregatori e tutti i modelli: è questo che rende appaiati i confronti.
* L'unità è l'item, senza strati: si ricampionano con reinserimento gli item del test set degli esperimenti, e tutte le risposte e i verdetti di un item si muovono insieme. Anche gli spaccati, per schema e per corpus, si calcolano sugli stessi ricampionamenti dell'intero test set, quindi il numero di item di uno spaccato varia da un ricampionamento all'altro; un ricampionamento senza item di uno spaccato non contribuisce all'intervallo di quello spaccato. Motivo: stratificando si terrebbe fisso il numero di item per schema e l'intervallo non includerebbe l'incertezza sulla numerosità, che con gli 11 item di correlation to cause è proprio quello che va mostrato.
* Il test esatto di McNemar resta come controllo sul confronto appaiato di due accuratezze a copertura piena.
