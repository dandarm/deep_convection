# Accesso ai dati radar EUMETNET OPERA (ORD)

Questa nota permette a un'altra istanza di Codex (anche su un PC remoto) di
scaricare i compositi radar OPERA senza credenziali AWS.

## Stato verificato

La vecchia combinazione **non è valida**:

```text
s3://ord-24h-cache/
https://object-store.os-api.cci1.ecmwf.int
```

Il server risponde `NoSuchBucket`. L'infrastruttura ORD è stata spostata su
CloudFerro. I bucket pubblici corretti sono:

| Dati | Bucket |
| --- | --- |
| Ultime 24 ore | `s3://openradar-24h` |
| Archivio storico dei compositi | `s3://openradar-archive` |

Endpoint S3 comune:

```text
https://s3.waw3-1.cloudferro.com/
```

L'accesso è anonimo: usare sempre `--no-sign-request`; non servono chiavi AWS.

Verifiche eseguite il 2026-09-15:

- `openradar-24h/2026/09/15/OPERA/COMP/` conteneva file reali;
- `openradar-archive/2024/07/01/OPERA/COMP/` conteneva file reali.

## Prerequisito

Installare AWS CLI. Su Linux con Python e pip:

```bash
python3 -m pip install --user awscli
```

Se `aws` non è nel `PATH`, usare il percorso esplicito:

```bash
~/.local/bin/aws --version
```

Nei comandi sotto, sostituire `aws` con `~/.local/bin/aws` se necessario.

## Struttura delle chiavi

```text
s3://<bucket>/YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@<PRODUCT>.<ext>
```

Gli orari sono UTC. Esempio:

```text
s3://openradar-archive/2024/07/01/OPERA/COMP/OPERA@20240701T1200@0@DBZH.h5
```

## Prodotti e formati

I compositi OPERA principali sono:

- `DBZH`: riflettività massima istantanea (dBZ);
- `RATE`: intensità di precipitazione istantanea (mm/h);
- `ACRR`: accumulo di precipitazione su un'ora (mm).

I file sono disponibili in genere come:

- `.h5`: ODIM HDF5, adatto ad analisi radar scientifiche;
- `.tiff`: Cloud-Optimized GeoTIFF, più comodo per GIS/raster.

Nei dati storici possono comparire nomi/prodotti legati alla qualità, ad
esempio `DBZH_QIND`, `ACRR_QIND` e `QIND_RATE`; quindi elencare prima la
cartella e non presumere che ogni data abbia la medesima nomenclatura.

### Nota verificata sui file storici ODYSSEY

Per il 2020 i file non sono duplicati in un prodotto dati e in un prodotto di
qualità separato. `DBZH_QIND.h5` contiene due dataset ODIM nello stesso file:
`DBZH` e `QIND`; analogamente gli altri nomi composti includono la quantità
meteorologica e il relativo quality index. Il file
`OPERA@20200815T0200@0@DBZH_QIND.h5` è stato aperto e verificato: griglia
2200×1900 a 2 km, `DBZH` in dBZ e `QIND` normalizzato fra 0 e 1.

La produzione storica ODYSSEY fino al 2024 è a **15 minuti**. La cadenza di 5
minuti e i nomi semplici `DBZH.h5`, `RATE.h5` e `ACRR.h5` appartengono alla
produzione corrente CIRRUS/NIMBUS. Di conseguenza, per le date EMMA 2020 usare
gli slot `00`, `15`, `30`, `45` e cercare i file `*_QIND.h5`; non costruire
nomi a 5 minuti.

## Elencare i file di una giornata

Archivio storico:

```bash
aws s3 ls \
  s3://openradar-archive/2024/07/01/OPERA/COMP/ \
  --endpoint-url https://s3.waw3-1.cloudferro.com/ \
  --no-sign-request
```

Cache delle ultime 24 ore:

```bash
aws s3 ls \
  s3://openradar-24h/2026/09/15/OPERA/COMP/ \
  --endpoint-url https://s3.waw3-1.cloudferro.com/ \
  --no-sign-request
```

Sostituire la data. La cache non è adatta a date più vecchie di circa 24 ore.

## Scaricare un singolo file

Esempio: DBZH in HDF5 alle 12:00 UTC del 1 luglio 2024.

```bash
mkdir -p data/opera
aws s3 cp \
  s3://openradar-archive/2024/07/01/OPERA/COMP/OPERA@20240701T1200@0@DBZH.h5 \
  data/opera/ \
  --endpoint-url https://s3.waw3-1.cloudferro.com/ \
  --no-sign-request
```

## Scaricare solo un prodotto di una giornata

Questo comando scarica tutti i DBZH HDF5 della giornata; controllare prima lo
spazio disponibile e l'elenco dei file. L'ordine di `--exclude` e `--include`
è importante.

```bash
mkdir -p data/opera/2024-07-01-dbzh
aws s3 cp --recursive \
  s3://openradar-archive/2024/07/01/OPERA/COMP/ \
  data/opera/2024-07-01-dbzh/ \
  --exclude '*' \
  --include '*@DBZH.h5' \
  --endpoint-url https://s3.waw3-1.cloudferro.com/ \
  --no-sign-request
```

Per GeoTIFF sostituire il filtro con `*@DBZH.tiff`; per gli altri prodotti,
`DBZH` con `RATE` o `ACRR` dopo aver controllato i nomi effettivamente
presenti.

## API per scoprire dati e metadati

Per trovare prodotti per intervallo temporale, sito o variabile, usare la ORD
API tramite MeteoGate. L'endpoint della documentazione è:

```text
https://api.meteogate.eu/eu-eumetnet-weather-radar/docs
```

Esempio di query per i compositi OPERA (aggiornare sempre l'intervallo):

```text
https://api.meteogate.eu/eu-eumetnet-weather-radar/collections/observations/locations/0-20010-0-OPERA?datetime=2026-07-16T06%3A00Z%2F2026-07-16T06%3A30Z&f=CoverageJSON&standard_name=DBZH&format=ODIM
```

Per pochi esperimenti l'accesso anonimo è sufficiente; per uso permanente o
molte richieste, registrare una API key MeteoGate per limiti di interrogazione
più alti. Per download in blocco già identificati, preferire S3.

## Note di interpretazione

- I compositi coprono l'Europa; non sono volumi di un singolo radar.
- I compositi correnti DBZH hanno una cadenza di 5 minuti; RATE e ACRR possono
  seguire cadenze diverse. Verificare sempre la lista dei file per la data.
- Per `ACRR`, la documentazione indica che il riferimento temporale nel nome
  dell'oggetto è la **fine** dell'intervallo di accumulo (es. `T0700` indica
  l'accumulo 06:00--07:00 UTC).
- I singoli volumi radar e i compositi hanno disponibilità e licenze diverse.
  I compositi OPERA sono distribuiti con CC BY 4.0; leggere i metadati per
  eventuali eccezioni dei dati di singoli radar.

## Fonti ufficiali

- https://eumetnet.github.io/openradardata-documentation/1-ORD-API-overview/
- https://eumetnet.github.io/openradardata-documentation/2-ORD-API-discovering-and-accessing-data/
- https://api.meteogate.eu/eu-eumetnet-weather-radar/docs

