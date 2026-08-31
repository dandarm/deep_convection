# Handoff verso Windows

Questo file permette di riprendere il feasibility study su un altro computer
anche se la cronologia locale della task Codex non è disponibile.

## Stato verificato

- Repository: `https://github.com/dandarm/deep_convection`
- Branch principale: `main`
- Pilot: Mediterraneo ed Europa meridionale, maggio-settembre 2020
- Dominio: 10°W-45°E, 25°N-55°N
- Tolleranza temporale EMMA-GPM: massimo 30 minuti, con offset effettivo
  conservato nei cataloghi
- Stato dei test al primo rilascio: 5 test superati
- I file grezzi e intermedi non sono versionati su GitHub

Il pilot pubblicato usa il GPM Precipitation Feature Database. I risultati PF
sono un primo catalogo riproducibile e non equivalgono alla copertura completa
degli swath GMI/DPR. Consultare `REPORT.md` e `PLAN_COMPLETAMENTO.md` prima di
estendere o interpretare le metriche.

## 1. Recuperare gli artefatti versionati

In PowerShell:

```powershell
git clone https://github.com/dandarm/deep_convection.git D:\Progetti\deep_convection
cd D:\Progetti\deep_convection
```

Cambiare `D:\Progetti` con la cartella desiderata. Aprire poi questa directory
come progetto nella app Codex desktop.

## 2. Trasferire il dataset non versionato

La copia sorgente Linux si trova in:

```text
/home/daniele/Datasets/Deep_convection
```

Copiarne ricorsivamente il contenuto sul computer Windows, per esempio in:

```text
D:\Datasets\Deep_convection
```

Il trasferimento può avvenire tramite disco esterno, condivisione di rete o
SSH. Non aggiungere questi file al repository Git.

Impostare la variabile per la sessione PowerShell corrente:

```powershell
$env:EMMA_GPM_DATA_ROOT = "D:\Datasets\Deep_convection"
```

Per renderla persistente per l'utente:

```powershell
[Environment]::SetEnvironmentVariable(
  "EMMA_GPM_DATA_ROOT",
  "D:\Datasets\Deep_convection",
  "User"
)
```

Riavviare Codex o il terminale dopo l'impostazione persistente.

## 3. Ricreare l'ambiente

```powershell
cd D:\Progetti\deep_convection
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
```

Alcune dipendenze scientifiche possono essere più semplici in WSL2. Se si usa
WSL2, conservare progetto e dataset nello stesso ambiente e usare percorsi
Linux coerenti.

## 4. Contesto da dare a una nuova task

Se l'handoff nativo della chat non è disponibile, iniziare una nuova task nel
progetto clonato con questo prompt:

> Leggi README.md, HANDOFF_WINDOWS.md, REPORT.md e PLAN_COMPLETAMENTO.md. Riprendi
> il feasibility study EMMA-GPM dallo stato pubblicato. Tratta EMMA solo come
> weak supervision; non addestrare modelli finché il catalogo osservativo non è
> stato validato. Verifica prima test, percorso dati e stato Git, senza
> riscaricare file già presenti.

## 5. Handoff nativo della chat

Quando i due computer compaiono nelle connessioni Remote di Codex e il progetto
Git è salvato su entrambi, aprire questa task sul computer sorgente. Dal footer
della chat selezionare la posizione di esecuzione, scegliere il computer
Windows e quindi `Hand off`. Codex trasferisce la chat e lo stato Git verso il
progetto corrispondente sul computer di destinazione.

Non copiare alla cieca l'intera directory `.codex` tra sistemi operativi:
potrebbe includere credenziali, configurazioni locali e percorsi non portabili.
