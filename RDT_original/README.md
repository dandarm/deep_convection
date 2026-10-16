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
│   └── pipeline.py                        # Stateful NwcSafRdtTracker & run_nwcsaf_rdt_sequence()
├── scripts/
│   └── compare_original_vs_proxies.py     # Quantitative benchmark on real 7-channel SEVIRI Zarr data
└── tests/
    └── test_rdt_nwcsaf.py                 # Unit & synthetic verification test suite for all 6 NWC SAF stages
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

## 5. Key Diagnostic Findings: Does the Proxy Faithfully Reproduce Original RDT-CW?

1. **Cloud System Detection & Morphological Boundary (`class >= 1`): Very High Fidelity (`Dice = 0.838`, `r = 0.988`)**
   - The 7-channel **`multispectral_utls_v0`** proxy reproduces the spatial footprint and adaptive base temperature of Original NWC SAF RDT-CW cloud cells remarkably well:
     - **Cloud-tower Dice/F1 = `83.83%`** (`IoU = 0.7216`, `POD = 87.99%`, `Precision = 80.05%`, `Frequency Bias = 1.099x`).
     - Over the `676,414` jointly detected cloud voxels, the proxy's adaptive edge temperature $T_{\text{edge}}$ correlates with the NWC SAF Base of Tower temperature $T_{\text{BT}}$ at **$r = 0.9876$**, with a mean bias of only **`+0.95 K`** and **`MAE = 1.38 K`**.
     - By comparison, the 2-channel proxies (`refined_v2` and `legacy_fixed_2ch`) achieve only `IoU = 0.4100` (`Dice = 0.5815`) and $r = 0.6119$ (`MAE = 3.26 K` / `9.03 K`).

2. **Convective Discrimination (`class in {2, 3, 4}`): High Precision (`81.64%`), Low Recall (`12.45%`) Due to Strict Initiation Gate**
   - Where `multispectral_utls_v0` labels a cell as convective, it agrees with Original NWC SAF GEO-RDT-CW **`81.64%` of the time** (`Precision = 81.64%`, `FAR = 18.36%`, and `71.94%` exact class precision on Class 3 `Mature/Decaying`).
   - However, `multispectral_utls_v0` labels only **`15.3%` as many convective voxels** as Original NWC SAF (`74,414` vs. `487,914` voxels; `POD = 12.45%`).
   - **Root Cause of the Recall Gap:**
     - In `multispectral_utls_v0` (and `refined_v2`), a cloud cell can **only** become convective if it is caught during a 15-minute `continuation` interval where all three hard thresholds hold simultaneously:
       $$\text{contrast} \ge 10\text{ K} \quad\land\quad \Delta T_{\min,15\text{m}} \le -2.5\text{ K} \quad\land\quad \Delta S_{15\text{m}} > 0$$
       If a massive, already-mature Mesoscale Convective System (MCS) or supercell enters the spatial/temporal window near its tropopause equilibrium temperature ($T_{\min} \approx 209..215\text{ K}$, $\Delta T_{\min,15\text{m}} \approx -0.5..+0.5\text{ K}$), or undergoes frequent anvil splits/merges (which invalidate `growth_valid` in the proxy), the proxy leaves the entire mature cumulonimbus system in **Class 1 (`non_convective_cloud`)** (`289,154` NWC SAF Class 3 voxels are placed in Class 1 by `multispectral_utls_v0` in the confusion matrix!).
     - In **Original NWC SAF GEO-RDT-CW (`nwcsaf_rdt_cw_v5.0`)**, cells in **Vertical Categories `0` (`Mature`, $T_{\min} \le -40^\circ\text{C}$ for $\ge 30\text{ min}$) and `1` (`Mature Transition`)** are evaluated by the 5-channel `IRxWVx` statistical discrimination model (and mandatory OTD override) using their deep thermal contrast, steep peripheral gradient $\text{Grad}_{P,95}$, positive upper-tropospheric water-vapour difference ($\text{WV}_{6.2}-\text{IR}_{10.8}$), glaciation signature ($\text{IR}_{8.7}-\text{IR}_{10.8}$), and low split-window cirrus penalty ($\text{IR}_{10.8}-\text{IR}_{12.0}$), even when instantaneous cooling has plateaued at the tropopause.

3. **Overshooting Top Detection (`class == 4`):**
   - Original NWC SAF two-step OTD (16 compass rays + Condition A/B) identifies `6,879` confirmed OT voxels across the 5 severe thunderstorm windows (`random_092`, `random_095`, `random_096`, `random_078`, `random_066`).
   - In `multispectral_utls_v0`, `88%` of those NWC SAF OT pixels occur inside mature cells that the proxy left in Class 1 (`non_convective`) or where the 2-frame persistence + plateau confidence gate (`proxy_confidence >= 0.35`, `D > 1.0`, `ot_persistence_frames = 2`) suppressed the OT label.

---

## 6. Running Tests & Benchmarks

From PowerShell in the repository root:

```powershell
# 1. Run unit & synthetic verification tests for RDT_original
$env:PYTHONPATH = "src;RDT_original"
.\.venv\Scripts\python.exe -m pytest RDT_original/tests -q

# 2. Run the multi-window SEVIRI comparison against the repository proxy teachers
.\.venv\Scripts\python.exe RDT_original/scripts/compare_original_vs_proxies.py --num-windows 6
```
