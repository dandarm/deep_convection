# `CI_original`: EUMETSAT NWC SAF GEO-CI (Convection Initiation) Python Implementation

This directory contains a complete, self-contained Python implementation of the official **EUMETSAT NWC SAF Convection Initiation (`GEO-CI`)** algorithm (v2.0 / v2018 / v2025), alongside a specialized **Pure-SEVIRI IR** adaptation that runs 24/7 strictly on the 7 SEVIRI thermal and water vapor channels without external NWP, visible reflectance, or radar dependencies.

---

## 1. Directory Structure

```text
CI_original/
├── README.md                              # Technical overview, architecture & verification results
├── .gitignore                             # Excludes PDFs, intermediate datasets, and caches
├── docs/
│   ├── CI_ATBD_SPECIFICATION.md           # Exhaustive algorithmic & mathematical ATBD specification
│   ├── ATBD_PURE_SEVIRI_CI.md             # Formulation of the Pure-SEVIRI IR autonomous pipeline
│   ├── NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v2.1.pdf   # Official EUMETSAT NWC SAF ATBD (Issue 2.1)
│   └── NWC-CDOP3-GEO-MF-PI-SCI-UM-Convection_v1.0.pdf   # Official EUMETSAT NWC SAF User Manual (Issue 1.0)
├── ci_nwcsaf/
│   ├── __init__.py                        # Public package interface
│   ├── config.py                          # NWC SAF CI configuration dataclass & Table 4/5 thresholds
│   ├── candidate_selection.py             # Area of Interest (AOI), pure-IR cloud mask & eligible-CI filter
│   ├── detection_tracking.py              # Warm cloud cell multi-thresholding & 2D movement tracking
│   ├── interest_fields.py                 # Static BTDs & dynamic Lagrangian cooling rate computation
│   ├── discrimination.py                  # Table 6 empirical decision rules & monotonicity enforcement
│   ├── forward_scheme.py                  # Lagrangian forward trajectory advection (0-30', 0-60', 0-90')
│   ├── ancillary.py                       # Autonomous pure-IR AMVs & ERA5 NWP convective index bridge
│   └── pipeline.py                        # Stateful NwcSafCiTracker & run_nwcsaf_ci_sequence()
├── scripts/
│   └── benchmark_ci_on_seviri.py          # Benchmark on real SEVIRI data & RDT onset lead-time analysis
└── tests/
    └── test_ci_nwcsaf.py                  # 13 unit & synthetic verification tests
```

---

## 2. Theoretical Basis & Relationship to RDT

The Convection Initiation (`GEO-CI`) product is designed by Météo-France / EUMETSAT NWC SAF as an **early-warning precursor to Rapidly Developing Thunderstorms (`GEO-RDT-CW`)**:

| Parameter / Dimension | Convection Initiation (`GEO-CI`) | Rapidly Developing Thunderstorms (`GEO-RDT-CW`) |
| :--- | :--- | :--- |
| **Meteorological Target** | Nascent pre-convective cumulus clouds ($0–90\text{ min}$ before thunderstorm onset) | Active, developing, and mature convective cells and overshooting tops |
| **Cloud Top Temperature Window** | **Warm and developing tops**: $T_{\text{IR10.8}} \ge \mathbf{-25^\circ\text{C}}$ ($248.15\text{ K}$) | **Deep cold tops**: $T_{\text{IR10.8}} \le \mathbf{-25^\circ\text{C}}$ down to $198\text{ K}$ |
| **Detection Slicing Range** | $T_{\text{warm}} = +10^\circ\text{C}$ to $T_{\text{cold}} = -25^\circ\text{C}$ ($\Delta T = 1.0\text{ K}$) | $T_{\text{warm}} = +5^\circ\text{C}$ to $T_{\text{cold}} = -75^\circ\text{C}$ ($\Delta T = 1.0\text{ K}$) |
| **Minimum Prominence ($\Delta T_{\text{tour}}$)** | $\mathbf{3.0\text{ K}}$ (sensitive to shallow budding cumuli) | $\mathbf{6.0\text{ K}}$ (isolates deep convective cores) |
| **Output Representation** | **Pixel-level probability maps** across forecast horizons ($30', 60', 90'$) | **Object-based vector polygons**, trajectories, and severity levels |
| **Probability Classes** | 0: None, 1: 0–25%, 2: 25–50%, 3: 50–75%, 4: 75–100% | 0: Background, 1: Non-convective, 2: Developing, 3: Mature, 4: OT |

---

## 3. Pure-SEVIRI IR Formulation

To enable continuous 24/7 execution without solar geometry discontinuities or external NWP model latency:
1. **Zero External Dependencies**: Operates exclusively on SEVIRI channels `WV_062`, `WV_073`, `IR_087`, `IR_097`, `IR_108`, `IR_120`, `IR_134`.
2. **Autonomous Cloud Masking**: Detects clouds via thermal contrast and split-window differential absorption ($T_{\text{IR10.8}} - T_{\text{IR12.0}} > 0.3\text{ K}$).
3. **Autonomous Atmospheric Motion Vectors (AMV)**: Derives 2D movement fields $(\vec{u}, \vec{v})$ via local cross-correlation block matching on IR10.8 and blends with tracked warm cell centroid displacements.
4. **Authoritative Night-Time / Pure-IR Interest Field Thresholds**: Implements Météo-France's tuned Table 5 thresholds:
   - **Glaciation Family**: $T_{\text{IR10.8}} \in [-25, -20]^\circ\text{C}$, $T_{\text{FZG}} \le 30\text{ min}$, $\text{BTD}_4 (10.8 - 8.7) \in [-2.1, 0.0]^\circ\text{C}$.
   - **Height Family**: $\text{BTD} (6.2 - 10.8) \in [-34, -12]^\circ\text{C}$, $\text{BTD}_5 (12.0 - 10.8) \in [-2.2, 0.0]^\circ\text{C}$, $\text{BTD}_6 (13.4 - 10.8) \in [-17, -6]^\circ\text{C}$, $\text{WBTD} (6.2 - 7.3) \in [-17, -7]^\circ\text{C}$.
   - **Growth Family**: $\text{TxBT}_{15\text{m}} \in [-50, -4]^\circ\text{C}/15\text{m}$ (cooling $\ge 4\text{ K}/15\text{m}$), $\text{TxBTD}_{15\text{m}} > 3^\circ\text{C}/15\text{m}$, $\text{TxBTD}_{4, 15\text{m}} \in [0, 10]^\circ\text{C}/15\text{m}$.

---

## 4. Empirical Decision Rules (Table 6)

The multi-criteria count across the three physical families evaluates the final CI probability:

```text
Growth >= 2:
  Glaciation >= 3: Height >= 4 -> HIGHPROB (4, 75-100%)
  Glaciation >= 3: Height == 3 -> MODPROB  (3, 50-75%)
  Glaciation >= 3: Height <  3 -> LOWPROB  (2, 25-50%)
  Glaciation == 2: Height >= 4 -> MODPROB  (3, 50-75%)
  Glaciation == 2: Height == 3 -> LOWPROB  (2, 25-50%)
  Glaciation == 2: Height <  3 -> VLOWPROB (1, 0-25%)
Growth == 1:
  Glaciation >= 3: Height >= 4 -> MODPROB  (3, 50-75%)
  Glaciation >= 3: Height <  4 -> LOWPROB  (2, 25-50%)
  Glaciation == 2: Height >= 4 -> LOWPROB  (2, 25-50%)
  Glaciation == 2: Height == 3 -> VLOWPROB (1, 0-25%)
Growth == 0:
  Glaciation >= 3: Height >= 3 -> LOWPROB  (2, 25-50%)
  Glaciation >= 3: Height <  3 -> VLOWPROB (1, 0-25%)
Other combinations:             -> NO_PROB  (0, 0%)
```

Along the forward trajectory, probabilities are advected for $+30'$, $+60'$, and $+90'$ lead times, with cumulative probability monotonicity strictly preserved:
$$P_{30} \le P_{60} \le P_{90}$$

---

## 5. Empirical Verification on Real SEVIRI Data (`2020` Archive)

Running [`scripts/benchmark_ci_on_seviri.py`](scripts/benchmark_ci_on_seviri.py) on real SEVIRI 7-channel full-resolution scenes (`E:\Datasets\Deep_convection\processed\msg_seviri\rss_7ch_bt_emma_30N70N`) confirms the physical validity and early-warning capability of `pure_seviri_ci` against downstream convective storm development:

| Benchmark Metric | Daytime Convective Event (`random_000`, 2020-05-31 15:00 UTC) | Nighttime Event (`random_001`, 2020-08-23 00:25 UTC) |
| :--- | :---: | :---: |
| **Evaluated 3D Voxels** | `802,816` ($16 \text{ frames} \times 224 \times 224$) | `802,816` ($16 \text{ frames} \times 224 \times 224$) |
| **Lead-Time Monotonicity Violations ($P_{30} \le P_{60} \le P_{90}$)** | **`0` (`100% strict compliance`)** | **`0` (`100% strict compliance`)** |
| **Downstream Convective Onsets Detected by RDT** | `888` onsets | `0` (calm nighttime sky) |
| **CI Early Warning POD ($30\text{ min}$ Lead Time)** | **`69.9%` (`621 / 888 hits`)** | `N/A` (no convective onsets) |
| **CI Early Warning POD ($60\text{ min}$ Lead Time)** | **`76.2%` (`677 / 888 hits`)** | `N/A` (no convective onsets) |
| **CI Positive Spatial Coverage ($0–30'$)** | `45.86%` of cloud field | `7.67%` of cloud field |
| **CI Positive Spatial Coverage ($0–60'$)** | `51.45%` | `11.00%` |
| **CI Positive Spatial Coverage ($0–90'$)** | `53.39%` | `12.37%` |
| **Night-Time Capability (Zero Solar / VIS dependency)** | Verified | **Verified (`100% functional at 00:25Z`)** |

---

## 6. Verification & Tests

Run the complete test suite from PowerShell:

```powershell
$env:PYTHONPATH = ".;src"
.\.venv\Scripts\python.exe -m pytest CI_original/tests -v
```

All 13 unit and integration tests verify:
- Multi-threshold warm cell detection ($\Delta T_{\text{tour}} \ge 3\text{ K}$).
- 2D Lagrangian motion tracking & AMV cross-correlation.
- Backward Lagrangian advection & 15-minute cooling rate calculation.
- Table 6 decision matrix accuracy across all branches.
- Forecast horizon forward scheme advection & monotonicity.
- End-to-end multi-frame sequence tracking on synthetic and real satellite cases.

