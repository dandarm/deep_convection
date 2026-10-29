# Handoff per Codex — Self-Supervised Learning delle dinamiche atmosferiche da SEVIRI

## 1. Obiettivo scientifico

Il progetto dispone già di un'implementazione funzionante di VideoMAE, di un dataset collaudato di sequenze multispettrali SEVIRI e di esperimenti di pretraining self-supervised già avviati.

Vogliamo ora estendere gli esperimenti per verificare **quali obiettivi di apprendimento favoriscano la rappresentazione delle dinamiche atmosferiche**, in particolare delle trasformazioni delle strutture nuvolose, oltre alla semplice ricostruzione delle immagini.

L'ipotesi principale è che un modello addestrato a prevedere l'evoluzione futura delle osservazioni apprenda rappresentazioni spazio-temporali più informative rispetto al solo masked video modeling.

Un'ulteriore ipotesi riguarda la distinzione tra:

- **Advezione:** spostamento orizzontale apparente delle strutture nuvolose.
- **Evoluzione intrinseca:** cambiamenti radiometrici e morfologici non spiegati dalla sola advezione, potenzialmente associati a formazione, crescita verticale, raffreddamento, organizzazione e dissipazione delle nubi.

Non assumiamo che tali trasformazioni identifichino univocamente la convezione profonda. In questa fase vogliamo apprenderne le rappresentazioni senza etichette e confrontare quantitativamente differenti strategie self-supervised.

## 2. Esperimenti di training

Utilizzare l'implementazione esistente di VideoMAE come punto di partenza, preservando la compatibilità con il dataset, i checkpoint e i training già eseguiti.

### A. Masked Video Modeling — baseline esistente

Mantenere il modello attuale come riferimento sperimentale.

Verificare quale strategia di masking sia attualmente implementata e valutare, se non già presenti, le seguenti varianti:

- Random spatiotemporal masking.
- Tube masking, con regioni spaziali nascoste attraverso più frame consecutivi.
- Block masking spazio-temporale.

L'obiettivo è verificare quanto la ricostruzione richieda effettivamente informazione temporale, anziché essere risolta prevalentemente attraverso correlazioni spaziali.

### B. Causal Future Prediction

Implementare un secondo obiettivo self-supervised che utilizzi una sequenza di osservazioni passate per prevedere uno o più frame futuri:

$$
\widehat X_{t+\Delta t}=D_\phi(E_\theta(X_{t-k:t}))
$$

Utilizzare inizialmente orizzonti di +15 e +30 minuti, compatibilmente con la cadence e la lunghezza delle clip disponibili.

Riutilizzare l'encoder VideoMAE e introdurre una prediction head adeguata alla ricostruzione spaziale multispettrale.

Il modello deve ricevere soltanto i frame storici; i frame futuri costituiscono i target del training.

Come loss iniziale utilizzare MSE o Huber sui valori radiometrici normalizzati, monitorando separatamente l'errore per canale e orizzonte temporale.

Valutare se la previsione tende a produrre strutture troppo smussate, perdendo minimi termici, gradienti o variazioni rapide.

### C. Joint Masked Modeling + Future Prediction

Sperimentare un training multi-obiettivo:

$$
\mathcal L=\lambda_M\mathcal L_{\mathrm{MVM}}+\lambda_F\mathcal L_{\mathrm{future}}
$$

Utilizzare un encoder condiviso con due head, una per la ricostruzione dei token storici mascherati e una per la previsione futura.

I pesi delle loss devono essere configurabili; monitorare separatamente i due contributi.

Confrontare questo modello con le due configurazioni precedenti, mantenendo per quanto possibile condizioni di training comparabili.

### D. Advection-Residual Prediction

Questo esperimento intende verificare se separare il trasporto apparente dalle trasformazioni radiometriche migliori l'apprendimento.

Stimare inizialmente un campo di movimento dalle immagini IR storiche, utilizzando optical flow o un metodo equivalente.

Costruire una previsione puramente advettiva:

$$
\widetilde X^{adv}_{t+\Delta t}=\mathcal W(X_t,\widehat{\mathbf u}_t)
$$

dove $\mathcal W$ è l'operatore di warping.

Definire il residuo osservato:

$$
R_{t+\Delta t}=X_{t+\Delta t}-\widetilde X^{adv}_{t+\Delta t}
$$

Addestrare quindi una prediction head a stimare tale residuo dalle rappresentazioni latenti:

$$
\widehat R_{t+\Delta t}=D_\phi(Z_t)
$$

ricostruendo il futuro mediante:

$$
\boxed{\widehat X_{t+\Delta t}=\widetilde X^{adv}_{t+\Delta t}+\widehat R_{t+\Delta t}}
$$

Utilizzare inizialmente una singola loss di previsione sul campo ricostruito: con la medesima loss e un baseline advettivo fissato, penalizzare il residuo previsto o il frame finale è matematicamente equivalente.

Il campo di movimento utilizzato dal predittore deve essere calcolato esclusivamente dalle osservazioni disponibili fino a $t$.

Confrontare questa configurazione con la future prediction diretta e con la pura advezione, verificando se il miglioramento si concentri nelle regioni soggette a trasformazioni radiometriche.

Considerare il residuo come una quantità osservativa: comprende anche errori dell'optical flow e altre variazioni non direttamente riconducibili alla convezione.

## 3. Dataset e campionamento

Il dataset esistente, costituito da clip SEVIRI campionate casualmente, è sufficiente per avviare gli esperimenti, dopo aver verificato che ciascuna clip contenga i frame storici e futuri richiesti.

Riutilizzare la pipeline attuale, estendendola dove necessario per produrre coppie:

$$
(\underbrace{X_{t-k:t}}_{\mathrm{input}},\underbrace{X_{t+1:t+h}}_{\mathrm{target}})
$$

Mantenere la struttura multispettrale originale e la normalizzazione già collaudata, verificando che questa conservi l'informazione sulle differenze termiche assolute e temporali.

Per l'eventuale espansione del dataset, affiancare al campionamento casuale una strategia che privilegi sequenze caratterizzate da significativa evoluzione radiometrica.

Un primo indicatore può essere:

$$
V(x,y,t)=|T_{10.8}(x,y,t)-T_{10.8}(x,y,t-\Delta t)|
$$

Da questo ricavare statistiche per clip, quali variazione media, quantili elevati e frazione di pixel soggetti a cambiamenti importanti.

In una seconda variante, calcolare la stessa variabilità dopo compensazione dell'advezione apparente.

**Mantenere comunque una componente di campionamento uniforme**, affinché il dataset rappresenti anche sistemi nuvolosi stabili, cirri, nubi stratiformi e altre dinamiche non necessariamente convettive.

Le sequenze molto variabili devono essere sovracampionate per rendere più informativo il training, senza introdurre implicitamente una classificazione convettiva.

Evitare sovrapposizioni temporali tra training e validation/test, mantenendo gli stessi split nei differenti esperimenti.

## 4. Valutazione degli esperimenti

Il confronto deve verificare sia la qualità della previsione sia l'effettivo utilizzo dell'informazione temporale.

### Baseline di previsione

Implementare o utilizzare:

$$
\widehat X_{t+\Delta t}=X_t
$$

come persistence baseline, e la previsione ottenuta mediante sola advezione.

Confrontare con entrambe i modelli predittivi neurali.

### Metriche

Calcolare, separatamente per canale e orizzonte:

- MAE e RMSE, preferibilmente anche in kelvin.
- Errore sui gradienti spaziali.
- Errore sulle variazioni temporali di brightness temperature.
- Capacità di preservare minimi locali e contrasti termici.
- Prestazioni condizionate al livello di variabilità radiometrica delle sequenze.

Oltre alle metriche aggregate, produrre visualizzazioni comparative di osservazioni future, persistence, advection e previsioni neurali, usando scale termiche coerenti.

### Analisi delle rappresentazioni

Per verificare se gli encoder apprendono effettivamente informazione dinamica:

1. Confrontare modelli addestrati con e senza informazione temporale.
2. Valutare l'effetto della permutazione dell'ordine dei frame storici.
3. Confrontare le rappresentazioni di sequenze reali con quelle ottenute mediante sola advezione.
4. Estrarre embedding spazio-temporali dagli encoder preaddestrati.
5. Utilizzare, se utile, sonde leggere su encoder congelati per verificare quanto le rappresentazioni permettano di recuperare variazioni termiche, gradienti e altre caratteristiche dinamiche.

Per il confronto fra gli encoder, utilizzare sonde con identica architettura e protocollo di training.

Valutare la predizione futura anche tramite una prediction head comune addestrata sulle embedding congelate. Questo consente di confrontare un encoder MVM e un encoder predittivo sullo stesso downstream task self-supervised, indipendentemente dalle differenti loss originali.

L'obiettivo è stabilire se la qualità delle rappresentazioni migliori realmente, anziché confrontare soltanto errori di ricostruzione e previsione relativi a task differenti.

## 5. Procedura operativa e risultati attesi

Procedere incrementalmente:

**Step 1 — Verifica del setup esistente.** Identificare checkpoint, configurazioni e risultati dei training VideoMAE già completati o in corso. Stabilire un baseline riproducibile senza modificare inutilmente l'infrastruttura.

**Step 2 — Future Prediction.** Implementare il nuovo obiettivo, eseguire un training pilota e confrontarlo con persistence e advection. Questa è la prima estensione prioritaria.

**Step 3 — Joint Training.** Integrare masked modeling e previsione futura, verificando se l'obiettivo congiunto migliora le prestazioni rispetto agli obiettivi separati.

**Step 4 — Advection-Residual Learning.** Implementare la decomposizione proposta e verificare sperimentalmente se migliora la rappresentazione e la previsione delle trasformazioni delle nubi.

**Step 5 — Representation Analysis.** Confrontare gli encoder e condurre le analisi temporali e radiometriche descritte nella sezione precedente.

Per ogni esperimento salvare configurazione, checkpoint, metriche e un insieme riproducibile di visualizzazioni.

Mantenere confrontabili dataset, architetture e budget di training, segnalando esplicitamente eventuali differenze.

Il risultato atteso è una valutazione sperimentale delle seguenti ipotesi:

$$
\boxed{\text{Masked reconstruction vs Future prediction vs Joint learning vs Advection-residual prediction}}
$$

La questione scientifica centrale è determinare **se un obiettivo predittivo, eventualmente strutturato per distinguere advezione e trasformazione, favorisca l'apprendimento di rappresentazioni delle dinamiche atmosferiche più informative rispetto al masked video modeling tradizionale**.

Questa fase deve concludersi con encoder riutilizzabili e risultati quantitativi comparabili, costituendo la base per una successiva analisi fisica delle rappresentazioni e per la detection della convezione profonda.

---

**Nota sulla priorità:** iniziare concretamente dallo **Step 2**, perché VideoMAE e il masked modeling sono già operativi. La future prediction è il primo esperimento capace di verificare l'ipotesi fondamentale senza richiedere modifiche sostanziali al dataset o l'introduzione immediata di optical flow.
