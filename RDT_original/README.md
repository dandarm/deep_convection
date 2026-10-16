# `RDT_original`: Complete EUMETSAT NWC SAF GEO-RDT-CW Implementation & Proxy Fidelity Benchmark

This directory contains a self-contained, dependency-clean Python implementation of the official **EUMETSAT NWC SAF Rapidly Developing Thunderstorms – Convection Warning (`GEO-RDT-CW`)** algorithm (v4.0/v5.0), together with an automated benchmark comparing it against the repository's three proxy RDT teachers (`multispectral_utls_v0`, `refined_v2`, and `legacy_fixed_2ch`) on the exact same spatial and temporal SEVIRI domain.

---

## 1. Directory Structure

```text
RDT_original/
├── README.md                              # Architecture, algorithmic comparison & empirical results
├── .gitignore                             # Excludes PDFs, Zarr/NPY outputs, figures, and caches from Git
├── docs/
│   └── RDT_CW_ATBD_SPECIFICATION.md       # Complete mathematical & algorithmic specification from NWC SAF ATBD/UM
├── rdt_nwcsaf/
│   ├── __init__.py                        # Public package interface
│   ├── config.py                          # Official UM Table 5 parameters & 6 vertical categories (0..5)
│   ├── detection.py                       # 81-level adaptive multi-thresholding, BTLIMIT, single-tower Case A/B refinement & Lines S/S2/H/X
│   ├── tracking.py                        # 4-step tracking (forward advection, IR10.8 cross-correlation, backward advection, 1-px small-cell dilation) & common-threshold father selection
│   ├── otd.py                             # Two-step Overshooting Top Detection (16 compass rays + Conditions A & B)
│   ├── discrimination.py                  # 6-category x 5-depth-regime 5-channel IRxWVx logistic regression, convective inheritance, declassification timers, Phase (0..4), Severity (0..4) & Lagrangian forecast
│   ├── nwp_era5.py                        # API-free ERA5 reanalysis fetcher (ARCO-ERA5 Zarr): WMO tropopause, LI/SHW/KI/CAPE NWPCONV mask, 700 hPa wind & surface temperature
│   ├── ancillary.py                       # NWC SAF GEO-CMa/GEO-CT radiative cloud-type classifier & GEO-HRW multi-channel AMV first-guess motion field
│   ├── calibration.py                     # EUMETNET OPERA radar verification bridge, MAP logistic regression CAL trainer & 7-channel SEVIRI-only surrogate distillation
│   └── pipeline.py                        # Stateful NwcSafRdtTracker & run_nwcsaf_rdt_sequence()
├── scripts/
│   ├── compare_original_vs_proxies.py             # Baseline 4-way comparison on real 7-channel SEVIRI Zarr data
│   └── calibrate_and_benchmark_full_rdt.py        # Full ERA5 + OPERA calibration, SEVIRI-only distillation & 5-way benchmark
└── tests/
    └── test_rdt_nwcsaf.py                 # 10 unit & synthetic verification tests covering all NWC SAF stages
```

---

## 2. Official Reference Documents

All equations, thresholds, decision trees, and attribute definitions in `rdt_nwcsaf` are derived from the authoritative EUMETSAT NWC SAF documentation (summarized in [`docs/RDT_CW_ATBD_SPECIFICATION.md`](docs/RDT_CW_ATBD_SPECIFICATION.md)):

1. **ATBD v2.1 (`NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection`, Issue 2, Rev. 1, 15 Oct 2019)**: *Algorithm Theoretical Basis Document for Convection Products (RDT-CW and CI)* — Météo-France / EUMETSAT NWC SAF (`GEO-RDT-CW v5.0`).
2. **ATBD v1.1 (`NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection`, Issue 1, Rev. 1, 01 Feb 2016)**: *Algorithm Theoretical Basis Document for Convection Products* (`GEO-RDT-CW v4.0`, detailing single-tower `Case A`/`Case B` base refinement, complex many-to-many trajectory closure, and 16-ray OTD thresholds).
3. **User Manual v1.0 (`NWC/CDOP3/GEO/MF-PI/SCI/UM/Convection`, Issue 1, Rev. 0, 01 Dec 2021)**: *User Manual for the Convection Products*, Table 5 (pp. 30–31) operational configuration parameters.

---

## 3. Architectural Comparison: Original NWC SAF RDT-CW vs. Repository Proxy Teachers

| Pipeline Stage | Original NWC SAF GEO-RDT-CW (`rdt_nwcsaf`) | Proxy 1: `multispectral_utls_v0` (`src/emma_gpm/rdt/`) | Proxy 2: `refined_v2` (`src/emma_gpm/rdt_refined.py`) | Proxy 3: `legacy_fixed_2ch` (`src/emma_gpm/rdt_teacher.py`) |
| :--- | :--- | :--- | :--- | :--- |
| **Spectral Channels Used** | **5 core thermal channels** (`WV_062, WV_073, IR_087, IR_108, IR_120`) + `IR_097, IR_134` diagnostics | **All 7 thermal channels** (`IR_108` topology + 6 BTDs for UTLS plateau proxy) | **2 channels** (`WV_062, IR_108`) | **2 channels** (`WV_062, IR_108`) |
| **Warmest Slicing Cap ($T_{\text{warm}}$)** | **$+5^\circ\text{C}$ ($278.15\text{ K}$)** down to $-75^\circ\text{C}$ ($198.15\text{ K}$) in $1\text{ K}$ steps (81 levels) | **$245.0\text{ K}$ ($-28.15^\circ\text{C}$)** down to $195.0\text{ K}$ in $0.5\text{ K}$ steps (h-minima seeded) | **$240.0\text{ K}$ ($-33.15^\circ\text{C}$)** down to $198.0\text{ K}$ in $1.0\text{ K}$ steps | **Fixed $240.0\text{ K}$** base with $[240, 230, 220, 210]\text{ K}$ slices |
| **Min Area ($S_{\min}$) & Prominence** | $S_{\min} = \mathbf{60\text{ km}^2}$, $\Delta T_{\text{tower}} = \mathbf{6.0\text{ K}}$ | $S_{\min} = \mathbf{100\text{ km}^2}$, prominence $\mathbf{6.0\text{ K}}$ | $S_{\min} = \mathbf{60\text{ km}^2}$, $\Delta T_{\text{tower}} = \mathbf{6.0\text{ K}}$ | $S_{\min} = \mathbf{75\text{ km}^2}$ at $240\text{ K}$ |
| **Mature Base Limit (`BTLIMIT`)** | Forced warmer than $\max(\mathbf{-60^\circ\text{C} = 213.15\text{ K}}, T_{\text{tropo}})$ | Implicit via support score $\le 245\text{ K}$ | Forced $\ge \mathbf{213.0\text{ K}}$ | Fixed at $240.0\text{ K}$ |
| **Single-Tower Base Refinement** | **Case A (Budding from warm deck):** $\min(S_{n+1}/S_n) \le 0.7$ above $-35^\circ\text{C}$; **Case B (Pyramidal):** $S_{\text{BT}} \le 4 \cdot S_{\text{ST}}$ | Maximizes **support score** $\text{grad} \times \text{coherence} \times \sqrt{S}$ up to $245\text{ K}$ (no $4 \cdot S_{\text{ST}}$ cap) | **Case B only:** $S_{\text{BT}} \le 4 \cdot S_{\text{ST}}$ below $240\text{ K}$ | None (always full connected component at $240\text{ K}$) |
| **Cloud Tracking & Motion** | **4-step tree:** (1) forward advection $+\vec{V}\Delta t$, (2) local $\text{IR}_{10.8}$ cross-correlation ($\rho \ge 0.5$), (3) backward advection $-\vec{V}_{\text{est}}\Delta t$, (4) $1\text{-px}$ dilation for small cells ($\le 5\text{ px}$); $\alpha=0.5$ smoothed weighted-centroid velocity | Translation by prior velocity + $1\text{-px}$ dilated recovery; $\text{AOR} \ge 0.15$ | Domain cross-correlation guess + prior velocity + $1\text{-px}$ dilated recovery; $\text{AOR} \ge 0.15$ | Static unshifted pixel overlap ($\text{AOR} \ge 0.15$), no motion compensation |
| **Split/Merge Main Father Rule** | **Largest area at common threshold $S(T_{\text{common}})$** from `Line H`; **trajectory reset** on many-to-many `complex` links | Largest area at `max(t_edge_k)`; reset on `complex` | Largest area at `max(threshold_k)`; reset on `complex` | Strictly 1-to-1 (`len(parents)==1` and `len(children)==1`); splits/merges drop trajectory |
| **Convective Discrimination** | **6 vertical categories (`0..5`) $\times$ 5 history regimes (`0, C, M, L, VL`)**: 5-channel `IRxWVx` logistic regression ($T_{\min}$, contrast, $\text{Grad}_{P,95}$, $-\dot{T}_{\min}$, $\dot{S}/S$, $\text{BTD}_{6.2-10.8}$, $\text{WBTD}_{6.2-7.3}$, $\text{BTD}_{8.7-10.8}$, $\text{BTD}_{10.8-12.0}$) + inheritance + **30/60 min empirical declassification** | Fixed hard thresholds: $\text{contrast} \ge 10\text{ K}$, $\Delta T_{15\text{m}} \le -2.5\text{ K}$, $\Delta S_{15\text{m}} > 0$ (or inherited with $\text{contrast} \ge 10\text{ K}$) | Fixed hard thresholds: $\text{contrast} \ge 10\text{ K}$, $3\Delta T_{5\text{m}} \le -2.5\text{ K}$, $3\Delta S_{5\text{m}} > 0$ (or inherited) | Fixed hard thresholds: $\text{contrast} \ge 10\text{ K}$, $3\Delta T_{5\text{m}} \le -2.5\text{ K}$, $3\Delta S_{5\text{m}} > 0$ |
| **Overshooting Top Detection (OTD)** | **Two-step OTD:** Step 1: $T_{\min} < -50^\circ\text{C}$, $\max(\text{BTD}) > 0$, **16 compass rays** at $2L_{\text{OT}}, 4L_{\text{OT}}$ requiring $\ge 2$ anvil pixels $>6\text{ K}$ warmer, $<3\text{ K}$ extension; Step 2: $BT \le T_{\text{tropo}}-5\text{ K}$ OR $\ge 2$ of ($\text{BTD}>+1.5\text{ K}$, $\text{VIS}>70\%$, $BT \le T_{\text{mean}}-6\text{ K}$); **forces cell convective** | Relative depth $D > 1.0$ above $T_{\text{UTLS\_proxy}}$ AND $\text{WV}_{6.2}-\text{IR}_{10.8} \ge +1.0\text{ K}$ with **2-frame persistence** | Fixed $\text{IR}_{10.8} \le 215\text{ K}$ AND $\text{WV}_{6.2}-\text{IR}_{10.8} \ge +1.0\text{ K}$ inside already-convective cell | Fixed $\text{IR}_{10.8} \le 215\text{ K}$ AND $\text{WV}_{6.2}-\text{IR}_{10.8} \ge +1.0\text{ K}$ inside already-convective cell |

---

## 4. Quantitative Benchmark on Real SEVIRI Data (`2020` Archive)

Running [`scripts/compare_original_vs_proxies.py`](scripts/compare_original_vs_proxies.py) on **6 deep-convective 19-frame SEVIRI windows** (`random_092`, `random_095`, `random_096`, `random_078`, `random_066`, `random_045` — `4,816,896` evaluated 3D space-time voxels across `224 x 224` ROIs with `64-px` spatial context halo and `15-min` temporal warm-up) produces the following pooled fidelity metrics against the **Original NWC SAF GEO-RDT-CW (`nwcsaf_rdt_cw_v5.0`)** reference:

### 4.1 Pooled Voxel & Object Totals Across the 6 Windows (`4,816,896` voxels)

| Metric | Original `nwcsaf_rdt_cw_v5.0` | Proxy 1: `multispectral_utls_v0` | Proxy 2: `refined_v2` | Proxy 3: `legacy_fixed_2ch` |
| :--- | ---: | ---: | ---: | ---: |
| **Background Voxels (`class == 0`)** | `4,048,124` (`84.04%`) | `3,971,916` (`82.46%`) | `3,991,915` (`82.87%`) | `3,991,915` (`82.87%`) |
| **Total Cloud Tower Voxels (`class >= 1`)** | `768,772` (`15.96%`) | `844,980` (`17.54%`) | `824,981` (`17.13%`) | `824,981` (`17.13%`) |
| **Non-Convective Cloud Voxels (`class == 1`)** | `280,858` (`5.83%`) | `770,566` (`16.00%`) | `824,510` (`17.12%`) | `797,932` (`16.57%`) |
| **Total Convective Voxels (`class in {2,3,4}`)** | `487,914` (`10.13%`) | `74,414` (`1.54%`) | `471` (`0.01%`) | `27,049` (`0.56%`) |
| **Developing Convective Voxels (`class == 2`)** | `116,035` (`2.41%`) | `21,640` (`0.45%`) | `246` (`0.005%`) | `24,670` (`0.51%`) |
| **Mature / Decaying Convective Voxels (`class == 3`)** | `365,000` (`7.58%`) | `52,774` (`1.10%`) | `223` (`0.005%`) | `2,319` (`0.05%`) |
| **Overshooting Top Voxels (`class == 4`)** | `6,879` (`0.14%`) | `0` (`0.00%`) | `2` (`0.000%`) | `60` (`0.001%`) |
| **Mean Track Duration (frames in ROI)** | `3.60` | `4.07` | `5.58` | `2.05` |
| **Runtime per Frame (`352 x 352` window)** | `2.72 s` | `0.30 s` | `0.56 s` | `0.08 s` |

### 4.2 Fidelity Metrics vs. Original NWC SAF GEO-RDT-CW (`nwcsaf_rdt_cw_v5.0`)

| Metric vs. Original NWC SAF | `multispectral_utls_v0` (7-ch UTLS Proxy) | `refined_v2` (2-ch Adaptive Proxy) | `legacy_fixed_2ch` (2-ch Fixed Proxy) |
| :--- | ---: | ---: | ---: |
| **Overall 5-Class Pixel Accuracy** | **`86.14%`** | `77.68%` | `77.76%` |
| **Cloud Tower Mask (`class >= 1`) IoU (CSI)** | **`0.7216`** | `0.4100` | `0.4100` |
| **Cloud Tower Mask (`class >= 1`) Dice (F1)** | **`0.8383`** | `0.5815` | `0.5815` |
| **Cloud Tower Mask POD (Recall) / Precision** | **`87.99%` / `80.05%`** | `60.28%` / `56.17%` | `60.28%` / `56.17%` |
| **Cloud Tower Area Frequency Bias ($A_{\text{proxy}}/A_{\text{orig}}$)** | **`1.099x`** (`+9.9%`) | `1.073x` (`+7.3%`) | `1.073x` (`+7.3%`) |
| **Convective Mask (`class in {2,3,4}`) IoU (CSI)** | **`0.1211`** (up to **`0.254`** on `random_096`) | `0.0009` | `0.0390` |
| **Convective Mask (`class in {2,3,4}`) Dice (F1)** | **`0.2161`** (up to **`0.405`** on `random_096`) | `0.0019` | `0.0751` |
| **Convective Mask Precision (`1 - FAR`)** | **`81.64%`** (`FAR = 18.36%`) | `96.82%` (`FAR = 3.18%`) | `71.53%` (`FAR = 28.47%`) |
| **Convective Mask POD (Recall)** | **`12.45%`** | `0.09%` | `3.97%` |
| **Convective Area Frequency Bias ($A_{\text{proxy}}/A_{\text{orig}}$)** | `0.153x` | `0.001x` | `0.055x` |
| **Class 1 (Non-Convective Cloud) IoU / Dice** | **`0.2722` / `0.4279`** | `0.0522` / `0.0993` | `0.0535` / `0.1015` |
| **Class 2 (Developing Convection) IoU / Precision** | **`0.0506` / `30.63%`** | `0.0009` / `0.00%` | `0.0181` / `7.78%` |
| **Class 3 (Mature/Decaying Convection) IoU / Precision** | **`0.1000` / `71.94%`** | `0.0005` / `0.00%` | `0.0042` / `9.01%` |
| **Base Temperature ($T_{\text{edge}} - T_{\text{BT}}$) Bias / MAE** | **`+0.95 K` / `1.38 K`** | `+0.13 K` / `3.26 K` | `+7.80 K` / `9.03 K` |
| **Base Temperature Pearson Correlation ($r$)** | **`0.9876`** | `0.6119` | `N/A` (fixed `240 K`) |

---

## 5. Full Multi-Source Upgrade: ERA5 NWP + NWC SAF Ancillary + OPERA Calibration (`CAL`) & Distilled `SEVIRI-Only` RDT

To build a maximally faithful reference (`nwcsaf_rdt_full_era5_cal`) and distill from it a purely geostationary algorithm (`nwcsaf_rdt_seviri_only_cal`) requiring **zero external NWP or radar data at runtime**, [`scripts/calibrate_and_benchmark_full_rdt.py`](scripts/calibrate_and_benchmark_full_rdt.py) implements:

1. **Real ERA5 NWP Integration (`rdt_nwcsaf/nwp_era5.py`)**:
   - Direct API-key-free retrieval from Google Cloud `ARCO-ERA5` (`full_37-1h-0p25deg-chunk-1.zarr-v3`).
   - Exact WMO (1957) / Reichler et al. (2003) thermal tropopause ($T_{\text{tropo}}, P_{\text{tropo}}, Z_{\text{tropo}}$), thermodynamic instability indices ($\text{LI}, \text{SHW}, \text{KI}, \text{CAPE}$) forming the NWC SAF convective mask `NWPCONV`, $700\text{ hPa}$ steering wind $(U_{700}, V_{700})$, and surface skin temperature $T_{\text{surf}}$.
2. **NWC SAF Ancillary Products (`rdt_nwcsaf/ancillary.py`)**:
   - `GEO-CT` radiative cloud-type pre-filtering using split-window ($\text{BTD}_{10.8-12.0}$) and water-vapour ($\text{BTD}_{6.2-10.8}$, $\text{BTD}_{8.7-10.8}$) tests to mask out detached semitransparent cirrus while preserving opaque cumulonimbus anvils.
   - `GEO-HRW` multi-channel ($\text{WV}_{6.2} + \text{IR}_{10.8}$) cross-correlation Atmospheric Motion Vector (AMV) first-guess field blended with the ERA5 $700\text{ hPa}$ steering flow.
3. **EUMETNET OPERA Radar Calibration (`rdt_nwcsaf/calibration.py`)**:
   - Matches `12,838` detected RDT cloud cells across disjoint training windows against co-located 2-km European **OPERA `DBZH_QIND`** radar composites (`249` verified convective cells with $\text{DBZH} \ge 35\text{ dBZ}$ over $\ge 16\text{ km}^2$ and `12,589` verified non-convective cells).
   - Fits the 12-stratum (`6 vertical categories x 2 history regimes`) MAP logistic regression weights $\boldsymbol{\beta}^{(c,h)}$ and category thresholds (`RDT_original/results/comparison_seviri_2020/calibrated_rdt_model.json`), and distills 7-channel SEVIRI surrogates for $(T_{\text{tropo}}, \text{NWPCONV})$.

### 5.1 Fidelity to Full Multi-Source Reference (`nwcsaf_rdt_full_era5_cal`, `4,816,896` Held-Out Voxels)

| Algorithm | Needs ERA5/Radar at Runtime? | Overall 5-Class Acc | Cloud Tower (`>=1`) Dice | Convective (`{2,3,4}`) Dice | Convective IoU (CSI) | Convective POD (Recall) | Convective Precision |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`nwcsaf_rdt_seviri_only_cal` (Distilled SEVIRI-Only)** | **No (7-ch SEVIRI only)** | **`95.98%`** | **`0.9288`** | **`0.8560`** | **`0.7483`** | **`99.80%`** | **`74.94%`** |
| `multispectral_utls_v0` (Repo 7-ch Proxy) | No | `86.68%` | `0.8375` | `0.2138` | `0.1197` | `12.40%` | `75.80%` |
| `refined_v2` (Repo 2-ch Adaptive Proxy) | No | `77.38%` | `0.5703` | `0.0021` | `0.0011` | `0.10%` | `96.80%` |
| `legacy_fixed_2ch` (Repo 2-ch Fixed Proxy) | No | `77.46%` | `0.5703` | `0.0826` | `0.0431` | `4.40%` | `71.50%` |

*(Across individual windows `random_092`, `random_095`, `random_096`, `random_078`, `random_066`, `nwcsaf_rdt_seviri_only_cal` achieves **Cloud Dice = `0.9239` – `1.0000`** and **Convective Dice = `0.8879` – `0.9906`** against the full ERA5+NWC SAF reference).*

### 5.2 Independent Ground-Truth Verification vs. EUMETNET OPERA Weather Radar ($\text{DBZH} \ge 35\text{ dBZ}$)

| Algorithm | Verified Radar Cell F1 (Dice) | Verified Radar Cell POD | Verified Radar Cell Precision | Verified Radar Pixel F1 (Dice) | Verified Radar Pixel CSI (IoU) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`nwcsaf_rdt_full_era5_cal` (Full ERA5 + CT + HRW + CAL)** | **`0.3650`** | `42.8%` | **`31.8%`** | `0.7852` | `0.6464` |
| **`nwcsaf_rdt_seviri_only_cal` (Distilled 7-ch SEVIRI-Only)** | **`0.3368`** | **`45.6%`** | `26.7%` | **`0.8026`** | **`0.6703`** |
| `legacy_fixed_2ch` (Repo 2-ch Fixed Proxy) | `0.1386` | `7.9%` | `58.3%` | `0.1901` | `0.1050` |
| `multispectral_utls_v0` (Repo 7-ch Proxy) | `0.0884` | `6.1%` | `16.0%` | `0.0772` | `0.0402` |
| `refined_v2` (Repo 2-ch Adaptive Proxy) | `0.0000` | `0.0%` | `0.0%` | `0.0000` | `0.0000` |

---

## 6. Running Tests & Benchmarks

From PowerShell in the repository root:

```powershell
# 1. Run unit & synthetic verification tests for RDT_original (10 tests)
$env:PYTHONPATH = "src;RDT_original"
.\.venv\Scripts\python.exe -m pytest RDT_original/tests -q

# 2. Run the baseline 4-way SEVIRI comparison against the repository proxy teachers
.\.venv\Scripts\python.exe RDT_original/scripts/compare_original_vs_proxies.py --num-windows 6

# 3. Run the full ERA5 + OPERA calibration, SEVIRI-only distillation, and 5-way benchmark
$env:PYTHONPATH = "src;RDT_original;RDT_original/scripts"
.\.venv\Scripts\python.exe -u RDT_original/scripts/calibrate_and_benchmark_full_rdt.py
```

