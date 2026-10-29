# Pure-SEVIRI IR Convection Initiation: Algorithmic & Physical Formulation

This document defines the mathematical formulation and physical rationale of the **Pure-SEVIRI IR Convection Initiation (`pure_seviri_ci`)** pipeline. It adapts the authoritative EUMETSAT NWC SAF GEO-CI algorithm to operate **exclusively on the 7 SEVIRI infrared and water vapor channels**:

$$\mathcal{C} = \{\text{WV\_062}, \text{WV\_073}, \text{IR\_087}, \text{IR\_097}, \text{IR\_108}, \text{IR\_120}, \text{IR\_134}\}$$

All external non-IR dependencies (visible reflectance, solar geometry, microphysical CMIC retrievals, and external NWP GRIB files) are replaced by self-contained physical formulations derived directly from the geostationary observations.

---

## 1. Motivation for a Pure-IR Formulation

In operational meteorology and AI foundation model research (such as VideoMAE pretraining on SEVIRI):
1. **Diurnal Continuity (24/7)**: Visible channels (VIS 0.6 µm, VIS 0.8 µm, NIR 1.6 µm) vanish completely during nighttime. A pipeline depending on VIS or daytime CMIC suffers sharp discontinuities at sunrise and sunset.
2. **Zero External Latency / API Independence**: Operational NWP forecasts (ECMWF, ARPEGE, GFS) are often delayed by hours and may fail to resolve fine mesoscale boundaries (e.g. sea breeze convergence, topographic bubbling) where convection initiates.
3. **Representational Probing Compatibility**: VideoMAE is pretrained strictly on the 7 SEVIRI IR/WV channels. Probing VideoMAE representations on a pure-IR pseudo-ground truth ensures rigorous, unconfounded scientific evaluation.

---

## 2. Spectral Channels & Physical Roles

| Index | Channel Name | Central $\lambda$ | Physical Role in Pure-IR CI |
| :---: | :--- | :---: | :--- |
| **0** | `WV_062` | $6.2\,\mu\text{m}$ | Upper-tropospheric water vapor absorption. Forms $\text{BTD} = T_{6.2} - T_{10.8}$ to measure cloud-top penetration relative to dry upper troposphere, and $\text{WBTD} = T_{6.2} - T_{7.3}$. |
| **1** | `WV_073` | $7.3\,\mu\text{m}$ | Mid-tropospheric water vapor. Differentiates low/mid-level moist plumes from high-level moisture. |
| **2** | `IR_087` | $8.7\,\mu\text{m}$ | Cloud thermodynamic phase (ice vs. supercooled liquid water). Used in $\text{BTD}_4 = T_{10.8} - T_{8.7}$. |
| **3** | `IR_097` | $9.7\,\mu\text{m}$ | Ozone absorption / Stratospheric intrusion diagnostic. |
| **4** | `IR_108` | $10.8\,\mu\text{m}$ | Atmospheric window channel. Provides cloud-top radiometric temperature $T_{\text{IR10.8}}$, morphological object detection, and 15-min cooling rates. |
| **5** | `IR_120` | $12.0\,\mu\text{m}$ | Split-window absorption channel. Paired with $10.8\,\mu\text{m}$ for optical thickness estimation $\text{BTD}_5 = T_{12.0} - T_{10.8}$ and cloud masking. |
| **6** | `IR_134` | $13.4\,\mu\text{m}$ | $\text{CO}_2$ absorption channel. Slices the vertical cloud depth $\text{BTD}_6 = T_{13.4} - T_{10.8}$ across the lower and middle troposphere. |

---

## 3. Pure-IR Algorithmic Architecture

### 3.1 Autonomous Thermal Cloud Masking (`pure_cloud_mask`)
Without external `GEO-CMA` or visible albedo, cloud pixels are separated from warm background using a dual-test thermal filter:
1. **Split-Window Optical Contrast**:
   Thin cirrus vs. opaque low clouds are distinguished by split-window difference:
   $$\Delta T_{\text{split}} = T_{\text{IR10.8}} - T_{\text{IR12.0}}$$
2. **Thermal Depression**:
   A pixel is marked as cloudy if:
   $$T_{\text{IR10.8}} \le T_{\text{surf, clear}} - \delta T_{\text{cloud}}$$
   or if split-window / water vapor contrast satisfies:
   $$\Delta T_{\text{split}} \ge 0.5\text{ K} \quad \text{and} \quad T_{\text{IR10.8}} < 290.0\text{ K}$$

### 3.2 Pre-Convective Temperature Window (Area of Interest)
Convection initiation occurs while cloud tops are still warm or supercooled, prior to deep anvil glaciation:
$$248.15\text{ K} \le T_{\text{IR10.8}} \le 283.15\text{ K} \quad (-25.0^\circ\text{C} \le T \le +10.0^\circ\text{C})$$
- Pixels with $T_{\text{IR10.8}} < 248.15\text{ K}$ ($-25^\circ\text{C}$) are filtered out because they have already initiated and developed into deep convective systems (handled by RDT-CW).
- Pixels with $T_{\text{IR10.8}} > 283.15\text{ K}$ ($+10^\circ\text{C}$) are warm fog or boundary-layer stratus incapable of immediate deep convection.

### 3.3 Autonomous 2D Motion Tracking (`pure_motion_field`)
In the absence of external 850 hPa NWP wind files, the 2D movement field is computed autonomously:
1. **Atmospheric Motion Vectors (AMVs)**:
   A fast local block-matching cross-correlation algorithm on the $10.8\,\mu\text{m}$ and $6.2\,\mu\text{m}$ channels between slot $T-15\text{m}$ and slot $T$ computes local displacement vectors $(\Delta x, \Delta y)$.
2. **Object Centroid Tracking**:
   Tracked warm cloud cells (segmented with $T_{\text{warm}} = 283.15\text{ K}, T_{\text{cold}} = 248.15\text{ K}, \Delta T_{\text{tour}} = 3.0\text{ K}$) provide high-confidence object-level velocities that overwrite the background grid within their area of influence.

### 3.4 Night-Time / Pure-IR Interest Field Thresholds
Under the pure-IR regime, we utilize the official **NWC SAF Table 5 (Night-time / Non-CMIC) thresholds**, which were specifically tuned by Météo-France to prevent false alarms when visible microphysical data is unavailable:

1. **Glaciation Family ($N_{\text{glaciation}} \le 3$)**:
   - $T_{\text{IR10.8}} \in [248.15\text{ K}, 253.15\text{ K}]$ ($-25^\circ\text{C}$ to $-20^\circ\text{C}$) or active cooling below freezing.
   - $T_{\text{FZG}} \le 30\text{ min}$ (supercooled water transition near $273.15\text{ K}$).
   - $\text{BTD}_4 = T_{\text{IR10.8}} - T_{\text{IR8.7}} \in [-2.1\text{ K}, 0.0\text{ K}]$.
2. **Height / Vertical Depth Family ($N_{\text{height}} \le 4$)**:
   - $\text{BTD} = T_{\text{WV6.2}} - T_{\text{IR10.8}} \in [-34.0\text{ K}, -12.0\text{ K}]$.
   - $\text{BTD}_5 = T_{\text{IR12.0}} - T_{\text{IR10.8}} \in [-2.2\text{ K}, 0.0\text{ K}]$.
   - $\text{BTD}_6 = T_{\text{IR13.4}} - T_{\text{IR10.8}} \in [-17.0\text{ K}, -6.0\text{ K}]$.
   - $\text{WBTD} = T_{\text{WV6.2}} - T_{\text{WV7.3}} \in [-17.0\text{ K}, -7.0\text{ K}]$.
3. **Growth / Updraft Family ($N_{\text{growth}} \le 3$)**:
   - $\text{TxBT}_{15\text{m}} \in [-50.0\text{ K}/15\text{m}, -4.0\text{ K}/15\text{m}]$.
   - $\text{TxBTD}_{15\text{m}} > +3.0\text{ K}/15\text{m}$ or $\text{TxBTD}_{6, 15\text{m}} > +3.0\text{ K}/15\text{m}$.
   - $\text{TxBTD}_{4, 15\text{m}} \in [0.0\text{ K}/15\text{m}, 10.0\text{ K}/15\text{m}]$ or $\text{TxBTD}_{5, 15\text{m}} \in [0.0\text{ K}/15\text{m}, 10.0\text{ K}/15\text{m}]$.

---

## 4. Synthesis & Output Data Structure

The output produced by `pure_seviri_ci` adheres to the official NWC SAF container format:
1. `ci_prob30`: 2D array of discrete classes $\{0, 1, 2, 3, 4\}$ for lead time $0–30\text{ min}$.
2. `ci_prob60`: 2D array for lead time $0–60\text{ min}$, with $P_{60} \ge P_{30}$.
3. `ci_prob90`: 2D array for lead time $0–90\text{ min}$, with $P_{90} \ge P_{60}$.
4. `ci_status_flag`: Bitmask recording active channels and processing modes (e.g., bit 0 for high-res IR, bit 3 for thermal cloud mask, etc.).
