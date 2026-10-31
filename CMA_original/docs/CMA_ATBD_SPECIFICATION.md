# Specifica Ufficiale EUMETSAT NWC SAF Cloud Mask (`GEO-CMA v6.0` / `v2025`, NWC-004)

Riferimenti bibliografici ufficiali:
- **ATBD GEO:** *Algorithm Theoretical Basis Document for Cloud Product Processors of the NWC/GEO* (`NWC/CDOP2/GEO/MFL/SCI/ATBD/Cloud`, Issue 1, Rev. 1, 15 October 2016; evoluto in `NWC/CDOP3/GEO/MFL/SCI/ATBD/Cloud`, `v2025`, `GEO-CMA v6.0`, Product ID `NWC-004`, <https://www.nwcsaf.org/cma_v2025>).
- **ATBD MTG-FCI:** *Algorithm Theoretical Basis Document for Cloud Product Processors of the NWC/MTG-FCI* (`NWC/CDOP2/MTG/MFL/SCI/ATBD/Cloud`, Issue 1, Rev. 1.1, 15 April 2016).

---

## 1. Tabelle Ufficiali delle Classi di Output (`GEO-CMA`)

### 1.1 `cma` (Maschera Binaria — ATBD Tabella 4, p. 10 & Sezione 2.2.2.4, p. 59)
| Valore | Nome Simbolico | Descrizione |
| :---: | :--- | :--- |
| `0` | `CLOUD_FREE` | Cielo sereno (include superfici coperte da neve o ghiaccio prive di nubi) |
| `1` | `CLOUDY` | Pixel coperto o contaminato da nube |
| `255` | `FILL_VALUE` | Dato mancante o fuori disco |

### 1.2 `cma_cover` (Detector a 4 Livelli: 1 Classe Negativa + 3 Classi Positive — ATBD Sezione 2.2.1.2.2, p. 13)
| Valore | Nome Simbolico | Tipo | Descrizione Fisica (ATBD p. 13 & pp. 30–31) |
| :---: | :--- | :---: | :--- |
| `0` | `CLOUD_FREE` | **Negativa** | Pixel sereno privo di nubi e privo di neve/ghiaccio |
| `1` | `CLOUD_CONTAMINATED` | **Positiva 1** | Pixel parzialmente nuvoloso (sub-pixel), bordo nube, cirro sottile semitrasparente, cirro sottile su neve, o nube rilevata entro il margine di sicurezza della soglia |
| `2` | `CLOUD_FILLED` | **Positiva 2** | Pixel completamente coperto da nube opaca che supera nettamente il margine di sicurezza radiometrico in un intorno spazialmente coerente |
| `3` | `SNOW_OR_ICE` | **Positiva 3** | Pixel sereno contaminato da neve continentale (su terra) o ghiaccio marino (su acqua) |
| `255` | `FILL_VALUE` | Fill | Dato mancante |

### 1.3 `cma_cloudsnow` (Contenitore Ufficiale v2025 a 4 Livelli — ATBD Sezione 2.2.2.4, p. 59)
| Valore | Nome Simbolico | Tipo | Descrizione Ufficiale NetCDF v2025 |
| :---: | :--- | :---: | :--- |
| `0` | `CLOUD_FREE` | **Negativa** | Cloud-free |
| `1` | `CLOUD` | **Positiva 1** | Cloud (except thin ice cloud over snow) |
| `2` | `THIN_ICE_CLOUD_OVER_SNOW` | **Positiva 2** | Thin ice cloud over snow/ice |
| `3` | `SNOW_OR_ICE` | **Positiva 3** | Snow/Ice |
| `255` | `FILL_VALUE` | Fill | No data / Undetermined |

### 1.4 Flag Aerosol (`cma_dust`, `cma_volcanic`, `cma_smoke` — ATBD p. 59)
| Valore | `cma_dust` | `cma_volcanic` | `cma_smoke` |
| :---: | :--- | :--- | :--- |
| `0` | `NO_DUST` | `NO_VOLCANIC_PLUME` | `NO_SMOKE` |
| `1` | `DUST` | `VOLCANIC_PLUME` | `SMOKE` |
| `2` | `UNDEFINED` | `UNDEFINED` | `UNDEFINED` |

### 1.5 Bitmask `cma_testlist` (28 Bit — ATBD Sezione 2.2.2.4, pp. 59–60)
- **Bit 0:** Snow and Ice test
- **Bit 1:** Thin ice cloud over snow test
- **Bit 2:** $T_{10.8}$ Brightness Temperature test
- **Bit 3:** SST estimation test
- **Bit 4:** $R_{0.6}$ or $R_{0.8}$ Visible Reflectance test
- **Bit 5:** $T_{10.8} - T_{12.0}$ Split-Window test
- **Bit 6:** $T_{10.8} - T_{3.8}$ Night test
- **Bit 7:** $T_{3.8} - T_{10.8}$ Day test
- **Bit 8:** $T_{12.0} - T_{3.8}$ Night test
- **Bit 9:** $T_{8.7} - T_{10.8}$ test
- **Bit 10:** Combined $T_{10.8} - T_{12.0}$ or $T_{8.7} - T_{10.8}$ gross cloud test
- **Bit 11:** $T_{3.8} - T_{10.8}$ gross day test
- **Bit 12:** Snow false detection cancellation
- **Bit 13:** Solar contamination on $3.8\,\mu\text{m}$
- **Bit 14:** $R_{1.6}/R_{0.6}$ or $R_{3.8}/R_{0.6}$ ratio
- **Bit 15:** Local spatial texture ($3\times 3$ standard deviation) test
- **Bit 16:** $R_{1.38}$ Cirrus reflectance test
- **Bit 17:** $T_{10.8} - T_{13.4}$ Tropospheric $\text{CO}_2$ test
- **Bit 18:** $T_{7.3} - T_{10.8}$ Water Vapour test
- **Bit 19:** Sunglint restoral test
- **Bit 20:** Desert $T_{10.8} - T_{8.7}$ restoral test
- **Bit 21:** Cloud border reflectance ratio test
- **Bit 22:** $15\text{-min}$ Temporal differencing $\Delta T_{10.8}$ test
- **Bit 23:** $1\text{-hour}$ Stationary cloud temporal coherence test
- **Bit 24:** Spatial cloud expansion (region growing)
- **Bit 25:** Restoral to clear over snow/ice
- **Bit 26:** Spatial filter to clear (cold air / thermal inversion restoral)
- **Bit 27:** Isolated pixel post-filter
