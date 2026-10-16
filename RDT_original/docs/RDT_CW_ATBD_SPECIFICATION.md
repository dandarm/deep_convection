# Exhaustive Technical & Algorithmic Specification of NWC SAF GEO-RDT-CW (v5.0 / v2018)

**Sources Analysed:**
1. **ATBD v2.1:** *Algorithm Theoretical Basis Document for the Convection Product Processors of the NWC/GEO* (`NWC/CDOP2/GEO/MFT/SCI/ATBD/Convection`, Issue 2, Rev 1, 21 January 2019; applicable to `GEO-CI v2.0 (NWC-053)` and `GEO-RDT-CW v5.0 (NWC-056)`), 62 pages.
2. **UM v1.0:** *User Manual for the Convection Product Processors of the NWC/GEO* (`NWC/CDOP3/GEO/MF-PI/SCI/UM/Convection`, Issue 1, Rev 0, 21 January 2019), 37 pages.

> **Editorial Note on ATBD v2.1 Structure:** As documented in the *Document Change Record* (ATBD p. 3), ATBD v2.1 suppressed the third "detailed mathematical level of description" that existed in CDOP/CDOP2 ATBDs (retaining *Summary + Detailed Description*, which caused the Table of Contents on pp. 4–5 and List of Figures on pp. 6–8 to retain legacy page numbers up to p. 82 while the body is 62 pages). Below, every explicit parameter, formula, diagram, table, configuration keyword, and annex field from ATBD v2.1 and UM v1.0 is extracted and cross-referenced with the legacy CDOP2/v2013/v2016 formulations where v2.1 abbreviated them.

---

## 1. Pre-Processing, Pre-Filtering & Parallax Correction

### 1.1 Stretched Pixel / Edge-of-Domain Masking & Completeness
- **Domain & Spatial Cut-off Flags (ATBD Section 3.3.3 Annex, Line `T`, p. 57; UM Section 3.1.2.1, p. 21):**
  - Only usable pixels ("non-space and available data") within the configured processing region are analysed.
  - In the NWC SAF geostationary grid processing (inherited from CDOP PGE11/PGE19), pixels near the limb of the geostationary disk where pixel distortion/stretching exceeds **$5 \times \text{nominal pixel area}$** (nominal area $\approx 9\text{ km}^2$ at nadir for MSG SEVIRI, i.e., stretched pixel area $> 45\text{ km}^2$, corresponding roughly to satellite zenith angles $\theta_{\text{sat}} \gtrsim 67^\circ\text{--}70^\circ$) or where values are incalculable are masked out.
  - Trajectory spatial cut-off flags (`ch4` of Line `T` in CTRAJ, ATBD p. 57) explicitly track:
    - `0`: No break
    - `1`: Cell touching the edge of the domain
    - `2`: Pixel(s) of incalculable value in the cell (e.g., limb/stretched or corrupted pixels)
    - `3`: Both `1 & 2`
- **Solar Zenith Angle (Day/Twilight/Night Flag):**
  - Stored per cell (`ch22` of Line `I`, ATBD p. 58): `0 = day`, `1 = twilight`, `2 = night`.
  - Used to switch the Calibrated (`CAL`) discrimination scheme from default 5-channel IR/WV mode (`IRxWVx`) to daytime VIS/NIR/IR3.8 mode (`IRxWVxJ` or `IRxWVxJ2`, ATBD Section 3.2.1.2.3.3.1, p. 40) and to enable daytime-only predictors (`COT` from CMIC, `VIS06`/`HRV` reflectance $> 70\%$ in OTD, ATBD pp. 47, 50).

### 1.2 Cloud Type (CT) / Cloud Mask (CMa) Dynamic Filtering
- **References:** ATBD Section 3.2.1.2.1.1 (p. 26), Section 2.2.1.2.1.2 (p. 14), Section 3.2.1.2.5.3 (p. 47); UM Section 3.3.1.2.4 (p. 27), Table 5 (p. 30).
- **Configuration Keywords (UM Table 5, p. 30):**
  - `CMa = 1` (default `1 = yes`, `0 = no`): Flag for using the `CT` product as a cloud mask.
  - `CT = 1` (default `1 = yes`, `0 = no`): Flag for using the `CT` product both as a mask and to assign cloud-type attributes to RDT-CW cells.
- **Algorithmic Behavior:**
  1. **When `CT` is enabled (`CMa = 1` or `CT = 1`) and available:** Prior to any cell detection, RDT-CW performs a **dynamic cloud filtering** using the NWC SAF Cloud Type (`CT`) product (preferred over `CMA` because `CT` classes are also used downstream as cell attributes). All non-cloudy (clear-sky) pixels are masked out and ignored during the adaptive $10.8\,\mu\text{m}$ brightness temperature thresholding.
  2. **Cell Attribute Assignment (`CT = 1`):** For each detected cell, `CT` is examined over all pixels within the cell's horizontal extension:
     - The **most frequent (modal) Cloud Type class** (`ch4` of Line `X`, ATBD p. 59) and its **surface proportion** (`ch5` of Line `X`, ATBD p. 59) are assigned to the cell, along with a **cumuliform vs. stratiform classification** (`ch6` of Line `X`, ATBD p. 60).
     - In tropical forced discrimination (`TROPICALDISCRI`, ATBD Section 3.2.1.2.3.8, p. 44), the cell's modal `CT` must be **Very High Opaque** or **High Opaque**.
  3. **When `CT` is disabled (`CMa = 0` and `CT = 0`) or missing:** RDT-CW operates directly on the raw $10.8\,\mu\text{m}$ brightness temperature image without clear-sky pre-masking (since $T_{\text{warm}} = +5^\circ\text{C} = 278.15\text{ K}$ combined with vertical tower criteria $\Delta T_{\text{tower}} \ge 6\text{ K}$ naturally excludes most warm clear-sky surfaces, though cold winter/high-elevation clear surfaces could otherwise enter thresholding), and the product quality indicator (`product_quality`, UM p. 21, which weights mandatory current/previous $\text{IR}_{10.8}$ at $80\%$ and optional inputs `CT, CTTH, CMIC, CRR, HRW, LGH, NWP` at $20\%$) is reduced accordingly.

### 1.3 Parallax Correction
- **References:** UM Table 5 (p. 31); ATBD Section 3.2.2.4 (p. 52).
- **Configuration Keyword:** `PARALLAX = 1` (default `1 = active`, `0 = inactive`).
- **Geometry & Equations:**
  - Because a geostationary satellite views clouds off-nadir at satellite zenith angle $\theta_{\text{sat}}$ (or local viewing elevation/zenith angle derived from pixel latitude $\phi$, longitude $\lambda$, and sub-satellite longitude $\lambda_0$), a cloud top at altitude $h$ above the Earth's surface (or above sea level, derived from `CTTH` Cloud Top Height / Pressure or from the cell's brightness temperature $T_{\text{min}}$ / $T_{\text{th}}$ mapped via the NWP temperature/geopotential profile) is displaced away from the sub-satellite point by a horizontal ground distance:
    $$\Delta d = h \tan(\theta_{\text{sat}})$$
  - More specifically, in spherical/ellipsoidal geostationary navigation with Earth local radius $R_E$ and satellite orbital radius $R_{\text{sat}} \approx 42\,164\text{ km}$, for a pixel at $(\phi, \lambda)$ with cloud top height $h$:
    - The viewing vector from the satellite to the apparent ground position $(\phi_{\text{app}}, \lambda_{\text{app}})$ intersects the altitude shell $R_E + h$ at the true geographic coordinates $(\phi_{\text{true}}, \lambda_{\text{true}})$ shifted **towards the sub-satellite point** $(0, \lambda_0)$:
      $$\Delta \phi \approx -\frac{h}{R_E} \tan(\theta_{\text{sat}}) \cos(\psi_{\text{sat}}), \qquad \Delta \lambda \approx -\frac{h}{R_E \cos\phi} \tan(\theta_{\text{sat}}) \sin(\psi_{\text{sat}})$$
      where $\theta_{\text{sat}}$ is the satellite viewing zenith angle and $\psi_{\text{sat}}$ is the satellite azimuth angle pointing away from the sub-satellite point.
  - In RDT-CW, parallax correction shifts the encoded cell contours, gravity centres, and overshooting top coordinates at the output stage when `PARALLAX = 1`.

---

## 2. Detection of Cloud Systems & Cell Identification (ATBD Section 3.2.1.2.1, pp. 26–30)

### 2.1 Exact Configuration Parameters & Thresholds (UM Table 5, p. 30)
| Parameter Keyword | Symbol in Text | Default Value (RDT-CW) | Default Value (CI, UM p. 16) | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `TCOLD` | $T_{\text{cold}}$ | **$-75.0^\circ\text{C}$ ($198.15\text{ K}$)** | $-25.0^\circ\text{C}$ ($248.15\text{ K}$) | Coldest brightness temperature threshold in adaptive multi-thresholding |
| `TWARM` | $T_{\text{warm}}$ | **$+5.0^\circ\text{C}$ ($278.15\text{ K}$)** | $+10.0^\circ\text{C}$ ($283.15\text{ K}$) | Warmest brightness temperature threshold in adaptive multi-thresholding |
| `DELTATEMPE` | $\Delta T$ | **$1.0^\circ\text{C}$ ($1.0\text{ K}$)** | $1.0^\circ\text{C}$ ($1.0\text{ K}$) | Temperature step between $T_{\text{warm}}$ and $T_{\text{cold}}$ |
| `SMIN` | $A_{\text{min}}$ ($S_{\text{min}}$) | **$60.0\text{ km}^2$** | $1.0\text{ km}^2$ | Minimum horizontal area for a valid detected cell ($\approx 6\text{--}7$ MSG pixels at nadir) |
| `SMAX` | $A_{\text{max}}$ ($S_{\text{max}}$) | **$400\,000.0\text{ km}^2$** | $10\,000.0\text{ km}^2$ | Maximum horizontal area for a detected cell |
| `BTLIMIT` | $T_{\text{BT,limit}}$ | **$-60.0^\circ\text{C}$ ($213.15\text{ K}$)** | N/A | Configurable upper (coldest) limit for the "Base of Tower" ($BT$) threshold temperature (tested between $-40^\circ\text{C}$ and $-70^\circ\text{C}$; $-60^\circ\text{C}$ recommended) |
| `LIMITTROPO` | — | **$1$ (yes)** | N/A | Flag to enforce NWP tropopause temperature $T_{\text{tropo}}$ as a non-configurable upper (coldest) limit on the $BT$ threshold |
| *(Internal)* | $\Delta T_{\text{tower}}$ | **$6.0^\circ\text{C}$ ($6.0\text{ K}$)** | $3.0^\circ\text{C}$ ($3.0\text{ K}$) | Minimum vertical temperature extent (prominence) required to define an independent thermal tower (ATBD pp. 16, 27) |

### 2.2 Step-by-Step Multi-Thresholding & Thermal Tower Identification Algorithm

#### Step 2.2.1: Slicing the $\text{IR}_{10.8}$ Image into Connected Components
1. Define the discrete sequence of temperature thresholds from warm to cold with step $\Delta T = 1\text{ K}$ (ATBD Section 3.2.1.2.1.1, pp. 26–27):
   $$T_k = T_{\text{warm}} - k\,\Delta T \quad \in \{T_{\text{warm}},\, T_{\text{warm}} - \Delta T,\, T_{\text{warm}} - 2\Delta T,\, \dots,\, T_{\text{cold}}\}$$
   i.e., from $+5^\circ\text{C}$ ($278.15\text{ K}$) down to $-75^\circ\text{C}$ ($198.15\text{ K}$) in $1\text{ K}$ steps ($81$ threshold levels).
2. At each threshold $T_k$, extract all **8-connected components** ("zones") of cloudy pixels $p$ satisfying:
   $$BT_{10.8}(p) < T_k \quad \text{and} \quad A_{\text{min}} \le \text{Area}(Z) \le A_{\text{max}}$$
   where $A_{\text{min}} = 60\text{ km}^2$ and $A_{\text{max}} = 400\,000\text{ km}^2$.
3. Because $T_{k+1} < T_k$, the connected zones across all thresholds form a **inclusion tree (component tree / contour tree)** from the warmest threshold $T_{\text{warm}}$ (root/anvil layer) to the coldest threshold $T_{\text{cold}}$ (local temperature minima / cores).

#### Step 2.2.2: Definition of a "Cloud Tower" and Separation of Adjacent Towers
- **Tower Prominence Criterion ($\Delta T_{\text{tower}} \ge 6\text{ K}$, ATBD Figure 10, p. 27 & Section 2.2.1.2.3, p. 16):**
  - A branch in the contour tree is recognized as a distinct **cloud tower** (local convective updraft / temperature minimum) if and only if its vertical thermal extension from its coldest point $T_{\text{min}}$ to the threshold where it merges with another tower (or disappears) is at least:
    $$\Delta T_{\text{tower}} = T_{\text{base,raw}} - T_{\text{min}} \ge 6\text{ K}$$
  - Any secondary local minimum whose prominence is $< 6\text{ K}$ (e.g., $\Delta T < \Delta T_{\text{tower}}$, as illustrated for the rightmost small bump in ATBD Figure 10 top panel, p. 27) is **not** separated as an independent tower; it remains absorbed inside the parent cloud system.
- **Separation of Multiple Towers (Base of Tower $T_{\text{BT}}$, ATBD p. 27, Figure 10):**
  - When a cloud system contains two or more valid towers (each having vertical prominence $\ge 6\text{ K}$), the **initial Base of Tower ($BT$) threshold** $T_{\text{BT}}$ assigned to each tower is **the warmest threshold $T_k$ that isolates that tower from all other valid nearby towers**.
  - Specifically, if two valid towers merge into a single 8-connected component at threshold $T_{\text{merge}}$ ( the "including level", which is $\Delta T = 1^\circ\text{C}$ warmer than $T_{\text{BT}}$, see ATBD Section 3.2.1.2.5.1, p. 46), then at threshold $T_{\text{BT}} = T_{\text{merge}} - \Delta T$ they are separate 8-connected components.
  - **Morphological Evolution Across Time Slots (ATBD Figure 10, p. 27):**
    - *Apparition of a secondary cloud tower (Split):* When a secondary budding inside a cell grows until its prominence reaches $\Delta T_{\text{tower}} \ge 6\text{ K}$, it splits the parent contour into two cells at a **cooler (colder)** threshold $T_{\text{BT}}$.
    - *Disappearance of a secondary cloud tower (Merge):* When a secondary tower warms or decays so that its prominence drops below $\Delta T_{\text{tower}} < 6\text{ K}$, the separation vanishes and the remaining tower drops its $T_{\text{BT}}$ back to a **warmer** threshold (merging the two contours).

#### Step 2.2.3: Definition of "Top of Tower" ($ST$) Second Contour Level
- **References:** ATBD Section 3.2.1.2.1.2.2 (p. 28), Section 3.2.1.2.5 (p. 46), Annex Line `S2` (p. 55); UM Section 3.5.1 (p. 33).
- In addition to the main **Base of Tower ($BT$)** contour at $T_{\text{BT}}$, RDT-CW defines a second, inner contour level called **Top of Tower ($ST$, *Sommet de Tour*)** to provide a 3D-like description of the cell:
  - For a tower with minimum brightness temperature $T_{\text{min}}$, the Top of Tower level $T_{\text{ST}}$ is defined near the cold cap of the tower:
    $$T_{\text{ST}} = \text{ceil}_{\Delta T}(T_{\text{min}}) + \Delta T_{\text{tower}} - \Delta T \approx T_{\text{min}} + 5\text{ K} \text{ to } 6\text{ K}$$
    (or the warmest threshold within $\Delta T_{\text{tower}}$ of $T_{\text{min}}$ that bounds the upper tower cap before significant flaring; when multiple sub-towers exist inside a mature cell due to $BT$ limitation, only the **coldest** tower contributes to the $ST$ level, ignoring warmer buddings, ATBD p. 29).
  - The $ST$ contour (`S2` line in CTRAJ, ATBD p. 55) is encoded whenever the cloud cell morphology has a sufficiently different horizontal extension at $ST$ compared to $BT$ (UM p. 33).

#### Step 2.2.4: Refinement / Correction of Excessive Base Contour for Single-Tower Growing Systems
- **References:** ATBD Section 3.2.1.2.1.2.2 (p. 28), Figures 11 (p. 28) & 15–18 (pp. 6–7).
- **When Applied:** Applied **only** when **both** conditions hold:
  1. The cloud system is defined by **only a single tower** (no sibling tower forced a cooler split threshold, so $T_{\text{BT,initial}}$ defaulted to the warmest threshold $T_{\text{warm}}$ or cloud-layer edge).
  2. The initial threshold $T_{\text{BT,initial}}$ produces a bottom contour area $S_{\text{BT}}$ that is excessively large compared to the horizontal area $S_{\text{ST}}$ of the Top of Tower ($ST$). Specifically, the condition for an excessive base (ATBD p. 28 & Figure 16/18 captions, pp. 6–7) is:
     $$\frac{S_{\text{BT}}}{S_{\text{ST}}} > 3 \quad \left(\text{equivalently, } \frac{S_{\text{ST}}}{S_{\text{BT}}} < \frac{1}{3} \text{ or } \frac{\min(S_{\text{BT}}, 3\,S_{\text{ST}})}{S_{\text{BT}}} < 1\right)$$
- **Two Morphological Correction Rules (ATBD p. 28, Figure 11):**
  - **Case A — Single Tower Emerging from a Flat Cloud Layer ("Budding rising up from a cloud layer", Figure 11 Left):**
    - Between the initial $BT$ level and the $ST$ level, let $S_n$ be the cell area at threshold level $n$ and $S_{n+1}$ be the cell area at the next colder threshold level $n+1$ ($T_{n+1} = T_n - \Delta T$, so $S_{n+1} \le S_n$).
    - Compute the **vertical area ratio between two successive temperature levels**:
      $$r_n = \frac{S_{n+1}}{S_n}$$
    - A sharp horizontal constriction (transition from the flat cloud deck $N$ to the base of the rising tower $N+1$) corresponds to the minimum of $S_{n+1}/S_n$ (maximum relative area change between consecutive $1\text{ K}$ levels).
    - If the minimum successive-level area ratio satisfies:
      $$\min_{n \in [\text{BT}, \text{ST})} \left(\frac{S_{n+1}}{S_n}\right) \le 0.7$$
      then the new Base of Tower threshold $T_{\text{BT}}$ is raised (cooled) to the level $N+1$ immediately above the maximum vertical rate of area change (the red level in ATBD Figure 11 Left, p. 28).
  - **Case B — "Flat Pyramidal" Single-Tower Cloud System (Figure 11 Right):**
    - When there is no sharp step discontinuity ($S_{n+1}/S_n > 0.7$ everywhere), or after checking Case A, if the ratio of Base of Tower area to Top of Tower area still exceeds the upper threshold of **$3$**:
      $$\frac{S_{\text{BT}}}{S_{\text{ST}}} > 3$$
      the Base of Tower level $T_{\text{BT}}$ is shifted to a colder threshold until:
      $$S_{\text{BT}} \le 3 \cdot S_{\text{ST}}$$
      (red level in ATBD Figure 11 Right, p. 28).

#### Step 2.2.5: Contour Level Limitation for Mature Cloud Systems (`LIMITTROPO` and `BTLIMIT`)
- **References:** ATBD Section 3.2.1.2.1.2.3 (pp. 28–30), Figure 12 (p. 29); UM Table 5 (p. 30).
- **Problem Addressed:** In mature convective systems containing multiple cold overshooting/updraft towers embedded in a cold anvil, the multi-thresholding rule (Step 2.2.2) would otherwise separate the individual cold cores at a very cold threshold $T_{\text{BT}}$ (e.g., $-65^\circ\text{C}$ or $-70^\circ\text{C}$), breaking up the mature anvil into small core contours (ATBD Figure 12 left column, p. 29).
- **Rule:** A warm lower bound (i.e., **coldest allowed limit**) $T_{\text{BT,min\_allowed}}$ is imposed on the Base of Tower threshold $T_{\text{BT}}$ so that $T_{\text{BT}}$ **cannot be colder** than the limit:
  1. **Tropopause Limitation (`LIMITTROPO = 1`, non-configurable threshold from NWP):**
     $$T_{\text{BT}} \ge T_{\text{tropo}}$$
     where $T_{\text{tropo}}$ is the local NWP tropopause temperature.
  2. **Absolute Configurable Limitation (`BTLIMIT = -60^\circ\text{C}`, active especially in the tropics where $T_{\text{tropo}}$ is very cold, e.g. $-75^\circ\text{C}$ to $-80^\circ\text{C}$):**
     $$T_{\text{BT}} \ge T_{\text{BT,limit}} = -60^\circ\text{C} \quad (213.15\text{ K})$$
  - Combining both when `LIMITTROPO = 1`:
    $$T_{\text{BT,final}} = \max\!\big(T_{\text{BT}},\, T_{\text{tropo}},\, T_{\text{BT,limit}}\big)$$
  - **Consequence for Multi-Tower Mature Cells (ATBD p. 29):** Because $T_{\text{BT,final}}$ is prevented from going colder than $\max(T_{\text{tropo}}, -60^\circ\text{C})$, a single mature RDT-CW cell contour can enclose several cold towers inside it. In that case, **only the coldest tower** inside the mature cell is used to define the Top of Tower ($ST$) contour level, ignoring warmer secondary buddings.

---

## 3. Tracking of Cloud Systems (ATBD Section 3.2.1.2.2, pp. 30–34)

### 3.1 Pre-Calculated 2D Guess Motion Field (HRW + NWP 700 hPa)
- **References:** ATBD Section 3.2.1.2.2 (p. 30); UM Section 3.3.1.2.1 (p. 25), Section 3.3.1.2.6 (pp. 28–29), Table 5 (pp. 30–31).
- **Configuration Keywords:** `HRW = 1` (default `1 = yes`), `NWPMVTLVL = 700` (default $700\text{ hPa}$ for RDT-CW; note `CI` uses $850\text{ hPa}$).
- **Construction of the 2D Guess Field $\vec{V}_{\text{guess}}(x,y)$:**
  1. **HRW AMV Filtering & Remapping:** Read the latest available NWC SAF High Resolution Winds (`HRW`) Atmospheric Motion Vectors. Filter observations to keep IR-channel winds matching the valid pixel's $10.8\,\mu\text{m}$ brightness temperature (retaining at most one HRW observation per pixel). Remap the sparse HRW vectors onto the satellite grid via weighted spatial interpolation using a radius of influence and cut-off distance.
  2. **NWP $700\text{ hPa}$ Wind Blending:** In grid areas where no HRW observations are within the cut-off radius, fill/blend the 2D motion field using the NWP horizontal wind field $(U_{700}, V_{700})$ at $700\text{ hPa}$ (priority is always given to HRW where available).
  3. **Usage:** Used to initialize cell velocity in "cold start" (first slot of a run) and as a fallback velocity estimate for unlinked ("orphan") cells at $t + \Delta t$ during backward-advection recovery.

### 3.2 Overlap Metric & Multi-Step Forward/Backward Advection Linking
- **References:** ATBD Section 3.2.1.2.2 (pp. 30–32), Figures 13–16 (pp. 30–32).
- **Overlap Metric Definition (ATBD Figure 13, p. 30):**
  - Let $C$ be a cell in the previous image at time $t$ with area $A(C)$ and estimated velocity $\vec{V}(C)$, and let $\tilde{C} = C + \vec{V}(C)\,\Delta t$ be the forward-advected cell at time $t + \Delta t$.
  - Let $C'$ be a cell detected in the current image at time $t + \Delta t$ with area $A(C')$.
  - The **overlapping ratio** $\text{ovl}(C, C')$ (Area Overlap Ratio, AOR) is defined as the intersection area divided by the **minimum** of the two cell areas:
    $$\text{ovl}(C, C') = \frac{A(\tilde{C} \cap C')}{\min\!\big(A(C),\, A(C')\big)}$$
  - Two cells are considered to **overlap sufficiently** if:
    $$\text{ovl}(C, C') \ge \text{ovl}_{\text{min}} \quad (\text{standard threshold } = 0.15 = 15\%)$$

- **Four-Step Linking Decision Tree for Each Current Cell $C'$ at $t + \Delta t$ (ATBD Figure 14, p. 31 & Figures 15–16, p. 32):**
  - **Step 1 (Forward Advection of Previous Cells $C$):**
    - Advect every cell $C$ from time $t$ forward by $+\vec{V}(C)\,\Delta t$ to get $\tilde{C}$.
    - If $\text{ovl}(C, C') = \frac{A(\tilde{C} \cap C')}{\min(A(C), A(C'))} \ge \text{ovl}_{\text{min}}$ for one or more cells $C$, create temporal link(s) between $C$ and $C'$, and compute $\vec{V}(C')$ from the displacement of their weighted gravity centres (Section 3.4 below).
  - **Step 2 (Velocity Estimation for Unlinked Cell $C'$):**
    - If **no link** was found for $C'$ in Step 1, estimate a provisional velocity $\vec{V}_{\text{est}}(C')$ using (depending on configuration/cost, ATBD Figure 14, p. 31):
      1. **Cross-correlation** of the $\text{IR}_{10.8}$ brightness temperature field in a window around $C'$ between $t$ and $t + \Delta t$: if the maximum correlation coefficient satisfies **$\rho_{\max} \ge 0.5$**, use the cross-correlation velocity; if **$\rho_{\max} < 0.5$**, fall back to the velocity of the nearest neighbouring tracked cloud system; *or*
      2. Directly use the **neighbouring cell speed** or the **pre-calculated 2D guess movement field** (HRW + NWP $700\text{ hPa}$).
  - **Step 3 (Backward Advection of Current Cell $C'$):**
    - Translate $C'$ backward to time $t$ by $-\vec{V}_{\text{est}}(C')\,\Delta t$ to obtain $\tilde{C}' = C' - \vec{V}_{\text{est}}(C')\,\Delta t$ (ATBD Figure 15B, p. 32).
    - Evaluate the overlap between each cell $C$ at time $t$ and $\tilde{C}'$:
      $$\text{ovl}_{\text{back}}(C, C') = \frac{A(C \cap \tilde{C}')}{\min\!\big(A(C),\, A(C')\big)}$$
    - If at least one cell $C$ overlaps sufficiently with $\tilde{C}'$, establish the link(s) and compute $\vec{V}(C')$ from the centres of gravity.
  - **Step 4 (Morphological Enlargement / Dilation of Small Backward-Advected Cell $\tilde{C}'$):**
    - If Step 3 still finds **no link** (which occurs primarily for very small cloud systems, **typically $< 5\text{ pixels}$**, ATBD pp. 30, 32), artificially enlarge the backward-advected cell $\tilde{C}'$ by **1 pixel all along its edge** (morphological dilation with a $3 \times 3$ square structuring element of "pseudo-cloudy pixels", ATBD Figure 16, p. 32).
    - Re-evaluate overlap between the enlarged $\tilde{C}'_{\text{enlarged}}$ and cells $C$ at time $t$.
    - If sufficient overlap is found, establish the link(s) and compute $\vec{V}(C')$ from the centres of gravity.
    - **Otherwise (No link after Step 4):** Identify $C'$ as the **birth (beginning) of a new trajectory** (`History = birth`, initial velocity set from cross-correlation / 2D guess field).

### 3.3 Classification of Temporal Links, Family Groups, and Primary ("Father") Trajectory Selection
- **References:** ATBD Section 3.2.1.2.2.1 (p. 32), Section 3.2.1.2.3.4 (p. 43), Annex Line `T` (`ch9`, `ch12`, p. 57).
- **Types of Temporal Links (Bipartite Graph between $t$ and $t + \Delta t$):**
  1. **1-to-1 Continuation (`Normal: 'No'`):** A single cell $C$ at $t$ links to a single cell $C'$ at $t + \Delta t$.
  2. **Split (`'s'`):** A single parent cell $C$ at $t$ links to $M \ge 2$ child cells $C'_1, \dots, C'_M$ at $t + \Delta t$.
  3. **Merge (`'g'`):** $K \ge 2$ parent cells $C_1, \dots, C_K$ at $t$ link to a single child cell $C'$ at $t + \Delta t$.
  4. **Complex (`'c'`, merge + split):** A connected bipartite component ("family") containing $K \ge 2$ cells at $t$ and $M \ge 2$ cells at $t + \Delta t$.
- **Selection of Main Link ("Father" Link) & Trajectory Continuation (ATBD Section 3.2.1.2.3.4, p. 43):**
  - When a cell $C'$ has multiple links (or when a parent $C$ splits into multiple children $C'_j$), RDT-CW identifies a **main link (named "father" link)** to represent the primary cloud system evolution and compute temporal trends/discriminating parameters.
  - **Common Temperature Threshold Surface Rule:** To compare linked cells fairly despite potentially different adaptive base thresholds $T_{\text{BT}}(C)$ and $T_{\text{BT}}(C')$, define the **common (colder) threshold temperature**:
    $$T_{\text{common}} = \min\!\big(T_{\text{BT}}(C),\, T_{\text{BT}}(C')\big)$$
    (available from each cell's vertical surface profile `Line H`, which stores the ordered surfaces $S(T)$ at every $1\text{ K}$ step from $T_{\text{BT}}$ down to $T_{\text{min}}$, ATBD p. 62).
  - The **main father** (in a merge) or **main continuation child** (in a split) is the linked cell having the **largest horizontal surface $S(T_{\text{common}})$ at the common temperature threshold** (and maximum overlap). Secondary children in a split start new secondary trajectories marked with `'s'` (`Phase = 4: Triggering from a Split`), while secondary parents in a merge terminate their trajectories marked with `'g'`.
  - **Convective Father (for Inheritance):** Independently of the geometric main father, RDT-CW also searches among **all** temporal parents at time $t$ for a **convective father** (a parent already diagnosed as convective): if *any* linked parent is convective, the child $C'$ inherits the convective diagnosis (`ch5 = 4, 5` if inherited from primary link; `ch5 = 6, 7` if inherited from secondary link, ATBD p. 58).

### 3.4 Gravity Centres, Family Motion Vector, and Temporal Velocity Smoothing ($\alpha = 0.5$)
- **References:** ATBD Section 3.2.1.2.2 (p. 31, Figure 14) & Section 3.2.1.2.2.1 (pp. 32–34, Figures 17–19); Annex Line `S` (`ch6–ch9`, `ch24–ch25`, `ch32–ch33`, p. 59).
- **Geometric vs. Weighted Gravity Centre (ATBD p. 33, Figure 17):**
  - Let $p \in C$ be the pixels belonging to cell $C$ defined at threshold $T_{\text{BT}}$, with coordinates $(\text{lat}_p, \text{lon}_p)$ (or grid coordinates $(x_p, y_p)$), pixel area $a_p$, and brightness temperature $T_p < T_{\text{BT}}$.
  1. **Geometric Gravity Centre (`ch6, ch7` of Line `S`):**
     $$\vec{G}_{\text{geom}}(C) = \frac{\sum_{p \in C} a_p\,\vec{x}_p}{\sum_{p \in C} a_p} \quad \left(\text{or simple mean } \frac{1}{N}\sum_{p \in C} \vec{x}_p\right)$$
  2. **Temperature-Weighted Gravity Centre (`ch8, ch9` of Line `S`, default for motion estimation):**
     Each pixel is weighted by its vertical thermal depth (level below the cell threshold $T_{\text{BT}}$), giving higher weight to colder temperatures (higher altitudes):
     $$w_p = T_{\text{BT}} - T_p > 0 \quad (\text{or discrete level index } n_p = \lfloor (T_{\text{BT}} - T_p)/\Delta T \rfloor + 1)$$
     $$\vec{G}_{\text{weight}}(C) = \frac{\sum_{p \in C} a_p (T_{\text{BT}} - T_p)\,\vec{x}_p}{\sum_{p \in C} a_p (T_{\text{BT}} - T_p)}$$
     As shown in ATBD Figures 17 & 18 (p. 33), using $\vec{G}_{\text{weight}}$ suppresses artificial jumps in centroid position when $T_{\text{BT}}$ changes from one slot to the next (e.g., when an anvil warms/cools or splits).

- **Family-Weighted Displacement Vector (ATBD Section 3.2.1.2.2.1, p. 32):**
  - To prevent erratic velocities during splits and merges, displacement is computed over the entire connected **"family" group** of linked cells between time $t$ (group $\{C_i\}_{i=1}^K$) and time $t + \Delta t$ (group $\{C'_j\}_{j=1}^M$), weighted by the area $A$ of each cell:
    $$\vec{G}_{\text{family}}(t) = \frac{\sum_{i=1}^K A(C_i)\,\vec{G}_{\text{weight}}(C_i)}{\sum_{i=1}^K A(C_i)}, \qquad \vec{G}_{\text{family}}(t + \Delta t) = \frac{\sum_{j=1}^M A(C'_j)\,\vec{G}_{\text{weight}}(C'_j)}{\sum_{j=1}^M A(C'_j)}$$
    $$\frac{\overrightarrow{CC'}}{\Delta t} = \frac{\vec{G}_{\text{family}}(t + \Delta t) - \vec{G}_{\text{family}}(t)}{\Delta t}$$

- **Temporal Coherence / Exponential Smoothing of Velocity (ATBD Figure 14, p. 31 & p. 33):**
  - The raw instantaneous displacement velocity $\frac{\overrightarrow{CC'}}{\Delta t}$ is averaged with the previous velocity $\vec{V}(C)$ using the exact weighting factor **$\alpha = \frac{1}{2} = 0.5$** given in the formula inside **ATBD Figure 14 (bottom-left, p. 31)**:
    $$\vec{V}(C') = \frac{1}{2}\,\frac{\overrightarrow{CC'}}{\Delta t} + \frac{1}{2}\,\vec{V}(C)$$
    (where for a merge of multiple parents, $\vec{V}(C)$ is the area-weighted mean velocity of the parent group at time $t$).

### 3.5 Computation of Temporal Trends (Cooling Rate, Expansion Rate, BTD Trends)
- **References:** ATBD Section 3.2.1.2.3.2.1 (p. 36), Section 3.2.1.2.5.10 (p. 51), Annex Line `S` (`ch26`, `ch34`, p. 59) & Line `X` (p. 60).
- **Instantaneous vs. Multi-Step History Trends ($15, 30, 45, 60, 80/90\text{ min}$):**
  - Along the main ("father") trajectory $C(t - k\Delta t) \to \dots \to C(t - \Delta t) \to C(t)$, RDT-CW stores the time series of cell attributes up to 12 past slots (ATBD p. 53).
  - **Cooling Rate (`ch26` of Line `S`, ATBD p. 59):**
    - Defined from the minimum brightness temperature $T_{\text{min}}$ (in $^\circ\text{C}/\text{h}$ or $\text{K}/\text{h}$):
      $$\text{CoolingRate}_{\text{inst}}(t) = -\frac{T_{\text{min}}(t) - T_{\text{min}}(t - \Delta t)}{\Delta t}$$
      *(Note: In CTRAJ Line `S` `ch26`, ATBD p. 59 explicitly states: "cooling rate of the cell based on the minimum temperature ($^\circ\text{C}/\text{h}$ and positive when $T_{\text{min}}$ decreases)", whereas in Section 3.2.1.2.5.10 p. 51 severity thresholds are written with negative sign $-20, -50, -100^\circ\text{C}/\text{h}$ denoting $\Delta T_{\text{min}}/\Delta t < 0$).*
  - **Horizontal Expansion Rate ($\Delta S / \Delta t$ at Common Threshold):**
    - Because $T_{\text{BT}}(t)$ and $T_{\text{BT}}(t - \Delta t)$ can differ (especially across splits, merges, or growing-tower base corrections), the relative expansion rate between $t - \Delta t$ and $t$ is evaluated **at the common (colder) threshold temperature** $T_{\text{common}} = \min\!\big(T_{\text{BT}}(t),\, T_{\text{BT}}(t - \Delta t)\big)$ using the vertical profile surfaces $S(T)$ (`Line H`, ATBD p. 62):
      $$\frac{1}{S}\frac{\Delta S}{\Delta t} = \frac{S\big(t,\, T_{\text{common}}\big) - S\big(t - \Delta t,\, T_{\text{common}}\big)}{S\big(t - \Delta t,\, T_{\text{common}}\big)\,\Delta t}$$
  - **Other Channel & BTD Trends (`Line X`, ATBD p. 60):**
    - Instantaneous trends between $t - \Delta t$ and $t$, as well as **average, standard deviation, and extrema (min/max) over the history depth** $D \in \{15, 30, 45, 60, 80/90\}\text{ min}$, are computed along the main father trajectory for:
      - $T_{\text{min}}(\text{WV}_{6.2})$, $T_{\text{min}}(\text{WV}_{7.3})$, $T_{\text{min}}(\text{IR}_{8.7})$, $T_{\text{min}}(\text{IR}_{12.0})$, $T_{\text{min}}(\text{IR}_{3.9})$
      - $\text{Refl}_{\max}(\text{VIS}_{0.6}/\text{HRV})$, $\text{Refl}_{\max}(\text{NIR}_{1.6})$, $\text{Refl}_{\max}(\text{NIR}_{2.2})$
      - Representative BTDs ($\max$, $75\text{th}$ percentile, $90\text{th}$ percentile, and spatial coherence ratio `SIG`).

---

## 4. Discrimination of Convective Objects (ATBD Section 3.2.1.2.3, pp. 34–44)

### 4.1 Step 0: NWP Convective Mask Pre-Filter
- **References:** ATBD Section 2.2.1.2.1.1 (pp. 13–14), Section 3.2.1.2.3.1 (p. 34), Figure 28 (p. 7), Annex Line `X` (`ch45–ch49`, p. 60).
- Computed at each pixel from three NWP instability indices: **Lifted Index ($\text{LI}$, $^\circ\text{C}$)**, **Showalter Index ($\text{SHW}$, $^\circ\text{C}$)**, and **K Index ($\text{KI}$, $^\circ\text{C}$)**:
  $$\text{NWP\_Mask}(p) = \begin{cases} 0 \text{ (NOCONV / Full stable)} & \text{if } (\text{LI} > 0) \;\land\; (\text{SHW} > 3) \;\land\; (\text{KI} < 20) \\ 2 \text{ (CONV / Unstable)} & \text{if } (\text{LI} < -3) \;\lor\; (\text{SHW} < -3) \;\lor\; (\text{KI} > 30) \\ 1 \text{ (NEUTRAL / Unclear)} & \text{otherwise} \end{cases}$$
- **Rule:** Cloud systems whose predominant NWP mask value over the cell (`ch45` of Line `X`) is **`0` (Full stable)** are **ignored by the statistical discrimination scheme** (remain non-convective), eliminating false alarms in stable winter/mid-season air masses.
- In addition, for cells in non-stable areas ($\text{NWP\_Mask} \in \{1, 2\}$), the cell's NWP predictors passed to discrimination (when NWP mode is active) are:
  - **Cell Tropopause Temperature $T_{\text{tropo}}$ (`ch48` of Line `X`) & Pressure $P_{\text{tropo}}$ (`ch49`):** **Median** value over the pixels of the cloud cell (ATBD Section 3.2.1.2.5.7, p. 49).
  - **Cell Lifted Index (`ch47` of Line `X`):** **Minimum** (or $10\text{th}$ percentile for $\text{LI}$/$\text{SHW}$, $90\text{th}$ percentile for $\text{KI}$) over the pixels of the cloud cell (ATBD pp. 49, 60).

### 4.2 Vertical Stratification Categories (6 Categories by $T_{\text{min}}$ and $T_{\text{BT}}$)
- **References:** ATBD Section 3.2.1.2.3.1 (p. 34), Figure 29 (p. 7), Annex Line `I` (`ch7`, p. 58).
- Every cell is stratified into one of **6 vertical categories** based on its minimum brightness temperature $T_{\text{min}}$ ($\text{IR}_{10.8}$) and its Base of Tower threshold temperature $T_{\text{BT}}$ (ATBD p. 34 & p. 58):

| Category Code (`ch7`) | Category Name | Definition in ATBD Section 3.2.1.2.3.1 (p. 34) | Exact Threshold Condition in ATBD Annex Line `I` (`ch7`, p. 58) | Model Type |
| :---: | :--- | :--- | :--- | :--- |
| **`0`** | **Mature** | $T_{\text{min}} \le -40^\circ\text{C}$ ($233.15\text{ K}$) for at least $30\text{ min}$ | $T_{\text{min}} \le -40^\circ\text{C}$ (and age in class $\ge 30\text{ min}$) | Mature model |
| **`1`** | **Mature Transition** | Crossing top temperature $-40^\circ\text{C}$ ($233.15\text{ K}$) | $T_{\text{min}} \le -40^\circ\text{C}$ (for $< 30\text{ min}$) **or** $T_{\text{BT}} \le -35^\circ\text{C}$ ($238.15\text{ K}$) | Transition model |
| **`2`** | **Cold Transition** | Crossing top temperature $-35^\circ\text{C}$ ($238.15\text{ K}$) | $-40^\circ\text{C} < T_{\text{min}} \le -35^\circ\text{C}$ **or** $T_{\text{BT}} \le -25^\circ\text{C}$ ($248.15\text{ K}$) | Transition model |
| **`3`** | **Warm Transition** (`Warm1`) | Crossing top temperature $-25^\circ\text{C}$ ($248.15\text{ K}$) | $-35^\circ\text{C} < T_{\text{min}} \le -25^\circ\text{C}$ **or** $T_{\text{BT}} \le -15^\circ\text{C}$ ($258.15\text{ K}$) | Transition model |
| **`4`** | **Warm2 Transition** (`Warm2`) | Crossing top temperature $-15^\circ\text{C}$ ($258.15\text{ K}$) | $-25^\circ\text{C} < T_{\text{min}} \le -15^\circ\text{C}$ **or** $T_{\text{BT}} \le -5^\circ\text{C}$ ($268.15\text{ K}$) | Transition model |
| **`5`** | **Warm / Hot** | $T_{\text{min}} > -15^\circ\text{C}$ ($258.15\text{ K}$) preceding Warm2 crossing | $T_{\text{min}} > -15^\circ\text{C}$ (note: if $T_{\text{BT}} > -5^\circ\text{C}$, "Hot cell", or Warm model if cooling/ascending) | Warm model |

- **Ascending/Cooling Pre-Condition (ATBD Figure 20, p. 35):**
  - Before invoking a statistical model on a currently non-convective cell, RDT-CW checks:
    $$\text{Is the cell in the Mature category (Cat 0/1), OR is it a cooling \& ascending cloud?}$$
  - If **No** (i.e., the cell is in a warmer category and is neither cooling nor ascending to a colder category), statistical discrimination is **not** called and the cell remains non-convective ("Remains not initialized").

### 4.3 Complete Set of Spatial, Radiative, Morphological, and Temporal Predictors
- **References:** ATBD Section 3.2.1.2.3.2.1 (pp. 35–36), Annex Line `S` (p. 59) & Line `X` (pp. 59–61).
- Over the cell's horizontal domain $C$ at each time step $t$, the following instantaneous attributes are extracted, and their **instantaneous values, time-depth averages, time-depth standard deviations, and time-depth extrema (min/max)** over the history window $[t - D, t]$ feed the statistical models:
  1. **Main $\text{IR}_{10.8}$ Radiative Predictors (Line `S`):**
     - $T_{\text{min}}$ (`ch12`): Minimum $\text{IR}_{10.8}$ brightness temperature of all pixels in $C$.
     - $T_{\text{mean}}$ (`ch10`): Mean $\text{IR}_{10.8}$ brightness temperature of all pixels in $C$.
     - $\sigma_T$ (`ch11`): Standard deviation of $\text{IR}_{10.8}$ brightness temperature over $C$.
     - $T_{\text{mean}} - T_{\text{min}}$ (ATBD p. 36): "Average minus minimum" temperature depth of the cell (and its maximum over the time-depth).
     - $T_{\text{BT}}$ (`ch4`): Adaptive threshold temperature defining the cell base.
  2. **$\text{IR}_{10.8}$ Morphological & Spatial Gradient Predictors (Line `S` & Line `H`):**
     - **Surface / Area $S$ (`ch13`):** Horizontal area of the cell in $\text{km}^2$ at $T_{\text{BT}}$, plus the vertical profile of surfaces $S(T)$ at every $1\text{ K}$ slice (`Line H`).
     - **Thermal Volume $V$ (ATBD p. 36):** Integral of horizontal surface over vertical temperature slices:
       $$V = \sum_{k=0}^{N-1} S(T_{\text{BT}} - k\Delta T)\,\Delta T \equiv \sum_{p \in C} a_p \big(T_{\text{BT}} - T_p\big)$$
     - **Ellipse Aspect Ratio / Eccentricity (`ch14, ch15, ch16`):** Minor axis $b$ ($\text{km}$), major axis $a$ ($\text{km}$), orientation angle $\theta$, and aspect ratio $b/a$ (or $a/b$) of the equivalent moment ellipse fitting the cell contour.
     - **Inner Temperature Gradient $\text{Grad}_{\text{int}}$ (`ch20`):** Average magnitude of the spatial temperature gradient $|\nabla T_{10.8}|$ ($^\circ\text{C}/\text{km}$) over all pixels inside the cell.
     - **Peripheral Temperature Gradient $\text{Grad}_P$ (`ch21, ch22, ch23`):** Computed over the $N_{\text{periph}}$ pixels (`ch23`) lying on the outer contour/periphery of the cell:
       - Mean peripheral gradient $\overline{\text{Grad}_P}$ (`ch21`)
       - $95\text{th}$ percentile (extreme) peripheral gradient $\text{Grad}_{P,95}$ (`ch22`)
  3. **$\text{IR}_{10.8}$ Temporal Trend Predictors:**
     - Instantaneous, mean, and standard deviation of cooling rate $\Delta T_{\text{min}} / \Delta t$ and relative surface expansion rate $\frac{1}{S}\frac{\Delta S}{\Delta t}$ (at common threshold) over the history depth $D$.
  4. **Multi-Spectral BT & BTD Predictors (5-Channel IR/WV + Daytime VIS/NIR/IR3.8, Line `X` pp. 60–61):**
     - **Single-channel extrema and trends:**
       - $T_{\text{min}}(\text{WV}_{6.2})$ and its trend $\Delta T_{\text{min}}(\text{WV}_{6.2})/\Delta t$ (`ch7, ch8`)
       - $T_{\text{min}}(\text{WV}_{7.3})$ and its trend $\Delta T_{\text{min}}(\text{WV}_{7.3})/\Delta t$ (`ch9, ch10`)
       - $T_{\text{min}}(\text{IR}_{8.7})$ and its trend $\Delta T_{\text{min}}(\text{IR}_{8.7})/\Delta t$ (`ch29, ch30`)
       - $T_{\text{min}}(\text{IR}_{12.0})$ and its trend $\Delta T_{\text{min}}(\text{IR}_{12.0})/\Delta t$ (`ch31, ch32`)
       - *Daytime mode (`IRxWVxJ` / `IRxWVxJ2`):* $T_{\text{min}}(\text{IR}_{3.8/3.9})$ & trend (`ch27, ch28`), $\text{Refl}_{\max}(\text{VIS}_{0.6}/\text{HRV})$ & trend (`ch19, ch20`), $\text{Refl}_{\max}(\text{NIR}_{1.6})$ & trend (`ch21, ch22`), $\text{Refl}_{\max}(\text{NIR}_{2.2})$ & trend (`ch57, ch58`).
     - **Brightness Temperature Differences (BTDs) & Reflectance Differences:**
       For each of the following difference fields $D(p)$ over the cell:
       1. $\text{BTD} = BT(\text{WV}_{6.2}) - BT(\text{IR}_{10.8})$ (`ch11–ch14`, upper-tropospheric moisture / cloud top height relative to tropopause)
       2. $\text{WBTD} = BT(\text{WV}_{6.2}) - BT(\text{WV}_{7.3})$ (`ch15–ch18`, mid-to-upper tropospheric water vapour differential)
       3. $\text{BTD4} = BT(\text{IR}_{8.7}) - BT(\text{IR}_{10.8})$ (or $\text{IR}_{10.8} - \text{IR}_{8.7}$, `ch37–ch40`, cloud-top glaciation / ice crystal microphysics)
       4. $\text{BTD5} = BT(\text{IR}_{12.0}) - BT(\text{IR}_{10.8})$ (or $\text{IR}_{10.8} - \text{IR}_{12.0}$, `ch41–ch44`, optical thickness / dirty-window split window)
       5. *Daytime only:* $\text{BTD3} = BT(\text{IR}_{3.9}) - BT(\text{IR}_{10.8})$ (`ch33–ch36`) and $\text{ReflDiff} = \text{Refl}(\text{NIR}_{1.6}) - \text{Refl}(\text{VIS}_{0.6})$ (`ch23–ch26`)
       - **Four summary statistics** are computed over the cell for each difference field (ATBD p. 60):
         - **Maximum value** over the cell
         - **$75\text{th}$ percentile** over the cell
         - **$90\text{th}$ percentile** over the cell
         - **Spatial coherence ratio (`SIG` ratio):**
           $$\text{Ratio}_{\text{SIG}} = \frac{\text{Number of relevant pixels }(D(p) > D_{\text{thresh}})\text{ surrounded by relevant pixels}}{\text{Total number of relevant pixels }(D(p) > D_{\text{thresh}})}$$
         plus their temporal trends over the history depth.

### 4.4 Statistical Discrimination Schemes: Generic (`GEN`) vs. Calibrated (`CAL`)
- **References:** ATBD Section 3.2.1.2.3.2 (`GEN`, pp. 35–40) and Section 3.2.1.2.3.3 (`CAL`, pp. 40–43); UM Section 3.2.1 (p. 22), Table 5 (`DISCRI`, p. 30).
- **Configuration Keyword:** `DISCRI = 3` (default `3 = use CAL if available for the satellite, otherwise fallback to GEN`; `1 = GEN only`; `2 = CAL only`; `0 = no statistical discrimination`).
- **Mathematical Form of the Statistical Models (Logistic Regression):**
  - Both `GEN` and `CAL` use **Logistic Regression** models (selected over 600-tree Random Forests for operational speed and direct probability-of-error estimation, ATBD p. 36).
  - For a given vertical category $c \in \{0, \dots, 5\}$, channel mode $m$, and history depth $d$, let $\vec{X} = (X_1, X_2, \dots, X_P)$ be the selected predictor vector, $\beta_0$ the intercept, and $\vec{\beta} = (\beta_1, \dots, \beta_P)$ the regression coefficients stored in the auxiliary file `ConvCoeffRegr*`:
    $$z = \beta_0 + \sum_{k=1}^P \beta_k X_k, \qquad P(\text{convective} \mid \vec{X}) = \frac{1}{1 + e^{-z}}$$
  - A cell is diagnosed as **convective** (`ch5 = 1`) if its logistic score $P(\text{convective} \mid \vec{X}) \ge P_{\text{thresh}}(c, m, d)$ (or equivalently $z \ge z_{\text{thresh}}(c, m, d)$), where the decision threshold is tuned on the Threat Score (TS) vs. False Alarm Ratio (FAR) curve at the inflection point maximizing $\text{TS} - \text{FAR}$ subject to $\text{FAR} \le \text{FAR}_{\max} \in [5\%, 15\%]$ (ATBD pp. 38–39).
  - The model also outputs the **likelihood of error of diagnosis (in %)** (`ch6` of Line `I`, `ch16` of Line `T`, ATBD pp. 57–58).
  - Models whose validation skill was too low (high FAR) are flagged as invalidated in `ConvCoeffRegr*_mask` (`$SAFNWC/import/Aux_data/RDT-CW/files_for_discri/`, ATBD p. 40); if the current $(c, m, d)$ model is masked out, no positive statistical decision is made at that slot.

- **Comparison of Modes and History Depths between `GEN` and `CAL`:**

| Feature | Generic (`GEN`) Scheme (ATBD pp. 35–40) | Calibrated (`CAL`) Scheme (ATBD pp. 40–43) |
| :--- | :--- | :--- |
| **Coefficient Files** | `ConvCoeffRegr` (15-min) & `ConvCoeffRegr_5` (5-min Rapid Scan) | `ConvCoeffRegr.<satellitename>.<updaterate>` (e.g. `MSG4.15`, `MSG3.5`, `MSG1.15`, `HIMA08.10`, `GOES16.15`) |
| **Training Ground Truth** | Météorage / EUCLID lightning over France/Europe (June–Aug 2008, June–Sept 2009); non-convective = $> 50\text{ pixels}$ ($\approx 150\text{--}200\text{ km}$) from flashes | Météorage (Europe), WWLLN (global), GOES-16 GLM (2018 campaigns); 80%/20% train/test split |
| **Spectral / NWP Modes** | **6 Modes (ranked by skill):**<br>1. `Classic with NWP`: $6.2 + 7.3 + 8.7 + 10.8 + 12.0\,\mu\text{m} + \text{NWP}$<br>2. `Classic`: $6.2 + 7.3 + 8.7 + 10.8 + 12.0\,\mu\text{m}$<br>3. `Limited with NWP`: $10.8 + 6.2\,\mu\text{m} + \text{NWP}$<br>4. `Limited`: $10.8 + 6.2\,\mu\text{m}$<br>5. `Mono channel with NWP`: $10.8\,\mu\text{m} + \text{NWP}$<br>6. `Mono channel`: $10.8\,\mu\text{m}$ | **4 Modes:**<br>1. `IRWV` (restricted): $\text{IR}_{10.8} + \text{WV}_{6.2}$<br>2. `IRxWVx` (default night/all-day): $\text{IR}_{10.8} + \text{WV}_{6.2} + \text{WV}_{7.3} + \text{IR}_{8.7} + \text{IR}_{12.0}$<br>3. `IRxWVxJ` (daytime MSG): `IRxWVx` $+ \text{VIS}_{0.6} + \text{NIR}_{1.6} + \text{IR}_{3.8}$<br>4. `IRxWVxJ2` (daytime Himawari/GOES): `IRxWVxJ` $+ \text{NIR}_{2.2}$ |
| **History Depth Regimes** | $15\text{ min}$, $30\text{ min}$, $45\text{ min}$, $60\text{ min}$ (selected by cell age in category, age since first detection, and history in warmer category) | **5 Depth Regimes ($6 \times 4 \times 5 = 120$ models total):**<br>• `0`: $0\text{ min}$ (triggering without history: first detection or split; **Transition categories 1–4 only**)<br>• `C` (Short): $[0, 15\text{ min}]$ or $[0, 20\text{ min}]$ (**Transition categories 1–4 only**)<br>• `M` (Medium): $[0, 30\text{ min}]$ or $[0, 40\text{ min}]$<br>• `L` (Long): $[0, 60\text{ min}]$<br>• `VL` (Very Long): $[0, 80\text{ min}]$ or $[0, 90\text{ min}]$ (**Mature category 0 only**) |

### 4.5 Convective Inheritance, Empirical Declassification Rules & Forced Diagnosis
- **References:** ATBD Sections 3.2.1.2.3.4 – 3.2.1.2.3.8 (pp. 43–44), Figure 20 (p. 35), Annex Line `I` (`ch5`, p. 58).
1. **Initial State:** At first detection (birth), a cell is initialized as non-convective ("not defined", `ch5 = 0`) before discrimination/forced rules are checked.
2. **Convective Inheritance (Section 3.2.1.2.3.4, p. 43):**
   - If a cell at time $t$ has a temporal link to *any* convective parent at time $t - \Delta t$ (either the primary father `ch5 = 4, 5` or a secondary parent `ch5 = 6, 7`), the cell **inherits the convective diagnosis** without re-running the statistical logistic regression (Figure 20, p. 35).
   - **Convective Time Counter ($t_{\text{conv}}$, `ch10` of Line `I`):**
     - If the cell ascends into a **colder vertical category** (continuous growing, e.g., from Warm1 to Cold Transition), its convective time in the category is **reset to $0$** for the new class.
     - Otherwise (stays in the same category or moves to a warmer category), its convective time is **incremented by $\Delta t$**.
3. **Empirical Declassification Rules (Section 3.2.1.2.3.5, pp. 43–44):**
   - Applied **only** to already-convective objects (`Previously Convective = Yes` in Figure 20) to diagnose decay or false alarms:
   - **Exception (Severity Protection, ATBD p. 44):** *"If the severity index of the father-cell is positive ($\text{Severity} \ge 1$), the cell is **not** declassified."*
   - **Rule for Warmest Category (Category `5` Warm):**
     - Convective diagnosis remains valid for **$30\text{ min}$** (for cooling systems only; immediate declassification if strong warming). After $30\text{ min}$ in Category `5` without ascending to a colder category, the cell is declassified (`ch5 = 3`: previously convective and declassified).
   - **Rule for Intermediate Transition Categories (Categories `1, 2, 3, 4`):**
     - Convective diagnosis remains valid for at least **$60\text{ min}$** as long as the cell does not exhibit strong warming.
     - Beyond $60\text{ min}$, the convective object is **declassified (`ch5 = 3`)** if it stays in the same category or moves into a warmer category.
   - **Rule for Coldest Mature Category (Category `0` Mature, $T_{\text{min}} \le -40^\circ\text{C}$, ATBD p. 44):**
     - Convective classification is kept for at least **$60\text{ min}$**. After $60\text{ min}$, the mature cell is **declassified (`ch5 = 3`)** if **at least one** of the following two conditions is satisfied:
       1. The cloud system warms into a warmer category, i.e.:
          $$T_{\text{min}} > -40^\circ\text{C} \quad (233.15\text{ K})$$
       2. The upper-level water-vapour difference becomes negative below $-1^\circ\text{C}$ **and** has a negative 60-minute trend:
          $$\text{BTD}_{\max}(\text{WV}_{6.2} - \text{IR}_{10.8}) < -1.0^\circ\text{C} \quad \textbf{and} \quad \frac{\Delta \text{BTD}(\text{WV}_{6.2} - \text{IR}_{10.8})}{\Delta t_{60\text{ min}}} < 0$$
4. **Forced Convective Diagnosis Overrides ("No" $\to$ "Yes" only, `ch5 = 11..15`, ATBD Sections 3.2.1.2.3.6–3.2.1.2.3.8, p. 44):**
   - **Overshooting Top Override (Section 3.2.1.2.3.6, mandatory, cannot be disabled):** If $\ge 1$ confirmed Overshooting Top (OT) is detected in the cell, its diagnosis is unconditionally forced to **convective**.
   - **Lightning Override (Section 3.2.1.2.3.7, configurable via `LGH = N > 0`, default `LGH = 1`):** If the number of lightning flashes paired with the cell (within time window $[t - \text{LGHDTANT}, t + \text{LGHDTPOST}] = [t - 900\text{ s}, t + 300\text{ s}]$ and spatial tolerance $\text{LGHTLR} = 3\text{ pixels}$, UM Table 5, p. 31) is $\ge N$ (default $\ge 1$), the cell is forced to **convective**.
   - **Tropical Minimum Temperature Override (Section 3.2.1.2.3.8, configurable via `TROPICALDISCRI`, recommended $-75^\circ\text{C}$, default disabled):** If the cell is located between $30^\circ\text{S}$ and $30^\circ\text{N}$, has $T_{\text{min}} < T_{\text{TROPICALDISCRI}}$, and its predominant `CT` is *Very High Opaque* or *High Opaque*, it is forced to **convective**.
   - *(Note: As stated in the Document Change Record on ATBD p. 3 and UM p. 28, in v2018 Convective Rain Rate `CRR`/`CRR-Ph` $\ge 30\text{ mm/h}$ no longer forces the convective diagnosis `ch5`, but sets the cell's "significant characteristic" flag `ch18 = 1` and contributes to severity/hazard).*

---

## 5. Overshooting Top Detection (OTD), Phase of Development, Severity & Forecasting

### 5.1 Overshooting Top Detection (OTD) Algorithm (ATBD Section 3.2.1.2.5.8, pp. 49–50, Figure 27)
OTD is a **two-step algorithm**: Step 1 pre-selects "OT candidates" during cell detection, and Step 2 confirms them using NWP tropopause and multi-spectral criteria.

#### Step 5.1.1: Pre-Selection of OT Candidates (Morphological & Static Analysis)
1. **Cell-Level Eligibility Pre-Conditions (ATBD p. 49):**
   - The cell's minimum brightness temperature $T_{\text{min}}$ (defining the primary pixel of interest $p_{\text{OT}}$) must be colder than `COLD_Threshold`:
     $$T_{\text{min}} < T_{\text{COLD\_Threshold}} = \begin{cases} -50^\circ\text{C} \;\;(223.15\text{ K}) & \text{in mid-latitude regions } (|\text{lat}| > 30^\circ) \\ -70^\circ\text{C} \;\;(203.15\text{ K}) & \text{in tropical regions } (|\text{lat}| \le 30^\circ) \end{cases}$$
   - The cell's maximum water-vapour minus IR window BTD must satisfy the `BTD condition`:
     $$\max_{p \in C} \big(BT_{\text{WV6.2}}(p) - BT_{\text{IR10.8}}(p)\big) > 0^\circ\text{C} \quad (0\text{ K})$$
2. **Characteristic Scales by Latitude Zone (ATBD p. 49):**
   - Characteristic OT diameter $L_{\text{OT}}$:
     $$L_{\text{OT}} = \begin{cases} 20\text{ km} & \text{mid-latitudes} \\ 50\text{ km} & \text{tropics} \end{cases}$$
   - Minimum inter-candidate separation `distance threshold` $D_{\text{sep}}$:
     $$D_{\text{sep}} = \begin{cases} 50\text{ km} & \text{mid-latitudes} \\ 200\text{ km} & \text{tropics} \end{cases}$$
3. **Surrounding Anvil Prominence Check (16 Pixels in 8 Directions, ATBD p. 49 & Figure 27 right, p. 50):**
   - Around the candidate pixel $p_{\text{OT}}$, explore **16 surrounding pixels** located along the **8 compass directions** (N, NE, E, SE, S, SW, W, NW) at **two characteristic distances**: $2 \times L_{\text{OT}}$ ($40\text{ km}$ mid-lat, $100\text{ km}$ tropics) and $4 \times L_{\text{OT}}$ ($80\text{ km}$ mid-lat, $200\text{ km}$ tropics).
   - Consider **only** those explored pixels that **belong to the same cloud cell $C$** (green arrows in Figure 27 right).
   - Identify warmer surrounding anvil pixels where the temperature difference with $p_{\text{OT}}$ exceeds the `minimum vertical extension threshold` of **$6^\circ\text{C}$ ($6\text{ K}$)**:
     $$BT_{10.8}(p_{\text{surr}}) - BT_{10.8}(p_{\text{OT}}) > 6.0^\circ\text{C}$$
     to confirm that the cold spot protrudes above surrounding warmer anvil pixels.
4. **Horizontal Extension of the OT (ATBD p. 49 & Figure 27, p. 50):**
   - Explore all pixels of the cell $C$ inside a square window of side **$2 \times L_{\text{OT}}$** ($40\text{ km} \times 40\text{ km}$ mid-lat; $100\text{ km} \times 100\text{ km}$ tropics; blue grid in Figure 27 right) centred on $p_{\text{OT}}$.
   - All neighbour pixels in that window whose temperature difference with $p_{\text{OT}}$ is below the `maximum threshold` of **$3^\circ\text{C}$ ($3\text{ K}$)**:
     $$BT_{10.8}(p) - BT_{10.8}(p_{\text{OT}}) < 3.0^\circ\text{C}$$
     are grouped as belonging to the horizontal extension of the OT (`ch22` of Line `O`, number of points belonging to OT $\ge 1$, ATBD p. 61).
5. **Additional Secondary OT Candidates in the Same Cell (ATBD p. 49):**
   - Other local extrema of `BTD` ($\text{WV}_{6.2} - \text{IR}_{10.8}$), `WBTD` ($\text{WV}_{6.2} - \text{WV}_{7.3}$), and `VIS06`/`HRV` reflectance inside cell $C$ are also tested via steps (3)–(4) as OT candidates (`ch4` of Line `O`: `"IRMin"`, `"BTDMax"`, `"ReflMax"`, ATBD p. 61), provided their distance from any previously identified OT candidate exceeds $D_{\text{sep}}$ ($50\text{ km}$ mid-lat, $200\text{ km}$ tropics).

#### Step 5.1.2: Confirmation of OT Candidates (ATBD p. 50)
An OT candidate $p_{\text{OT}}$ is **confirmed as a valid Overshooting Top** if and only if it satisfies **Condition A OR Condition B**:
- **Condition A (Strong Tropopause Penetration):**
  - NWP tropopause temperature $T_{\text{tropo}}$ is available and the OT is **at least $5^\circ\text{C}$ ($5\text{ K}$) colder** than the NWP tropopause:
    $$BT_{10.8}(p_{\text{OT}}) \le T_{\text{tropo}} - 5.0^\circ\text{C}$$
- **Condition B (Moderate Tropopause Penetration or No NWP Tropopause + Multi-Spectral/Morphological Confirmation):**
  - $\big(BT_{10.8}(p_{\text{OT}}) < T_{\text{tropo}} \text{ or } T_{\text{tropo}} \text{ unavailable}\big)$ **AND at least 2 out of the following 3 conditions hold**:
    1. **Intense WV–IR BTD:**
       $$BT_{\text{WV6.2}}(p_{\text{OT}}) - BT_{\text{IR10.8}}(p_{\text{OT}}) > +1.5^\circ\text{C} \quad (1.5\text{ K})$$
    2. **High Visible Reflectance (daytime only, when `VIS06` or `HRV` available):**
       $$\text{Reflectance}_{\text{VIS06 or HRV}}(p_{\text{OT}}) > 70\%$$
    3. **Strong Thermal Prominence Relative to Cell Mean Temperature:**
       $$BT_{10.8}(p_{\text{OT}}) \le T_{\text{mean}}(C) - 6.0^\circ\text{C}$$

---

### 5.2 Synthesis Attribute: "Phase of Development" (`0..4`, ATBD Section 3.2.1.2.5.9, pp. 50–51, Table 8)
- **Encoded Values:**
  - `0 = Triggering` (first detection / birth)
  - `1 = Growing` (developing)
  - `2 = Mature`
  - `3 = Decaying`
  - `4 = Triggering from a Split`
- **Complete Decision Tree (ATBD Table 8, p. 51):**

| History Status | Temperature Trend ($\Delta T_{\text{min}}/\Delta t$) | Vertical Category (based on $T_{\text{min}}$) | Horizontal Area Trend ($\Delta S / \Delta t$) | Activity (Convective Diagnosis, Lightning, OTD, Low Warming) | **Encoded Phase** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`birth`** (new trajectory) | Any | Any | Any | Any | **`0 = Triggering`** |
| **`Split`** (secondary child of split) | Any | Any | Any | Any | **`4 = Triggering from a Split`** |
| **`continued`** | **Cooling** ($\Delta T_{\text{min}} < 0$) | `Warm`, `Warm Transition`, `Warm2 Transition`, or `Cold Transition` | Any | Any | **`1 = Growing`** |
| **`continued`** | **Cooling** ($\Delta T_{\text{min}} < 0$) | `Mature` ($T_{\text{min}} \le -40^\circ\text{C}$) | **No expansion** ($\Delta S \le 0$) | Any | **`2 = Mature`** |
| **`continued`** | **Cooling** ($\Delta T_{\text{min}} < 0$) | `Mature` ($T_{\text{min}} \le -40^\circ\text{C}$) | **Expansion** ($\Delta S > 0$) | Any | **`1 = Growing`** |
| **`continued`** | **$\approx$ Same $T^\circ$** ($\Delta T_{\text{min}} \approx 0$) | `Warm`, `Warm Transition`, `Warm2 Transition`, or `Cold Transition` | **Expansion** ($\Delta S > 0$) **OR** ($\Delta S \le 0$ **and** $[\text{Conv} \lor \text{Lightning}]$) | Convective or Lightning active (if no expansion) | **`1 = Growing`** |
| **`continued`** | **$\approx$ Same $T^\circ$** ($\Delta T_{\text{min}} \approx 0$) | `Warm`, `Warm Transition`, `Warm2 Transition`, or `Cold Transition` | **No expansion** ($\Delta S \le 0$) | **Not Convective AND No Lightning** | **`3 = Decaying`** |
| **`continued`** | **$\approx$ Same $T^\circ$** ($\Delta T_{\text{min}} \approx 0$) | `Mature` ($T_{\text{min}} \le -40^\circ\text{C}$) | Any | Any | **`2 = Mature`** |
| **`continued`** | **Warming** ($\Delta T_{\text{min}} > 0$) | `Warm`, `Warm Transition`, `Warm2 Transition`, or `Cold Transition` | Any | **Lightning** **OR** $(\text{Convective} \land \text{low warming})$ | **`1 = Growing`** |
| **`continued`** | **Warming** ($\Delta T_{\text{min}} > 0$) | `Warm`, `Warm Transition`, `Warm2 Transition`, or `Cold Transition` | Any | `else` (No Lightning, and either not Convective or strong warming) | **`3 = Decaying`** |
| **`continued`** | **Warming** ($\Delta T_{\text{min}} > 0$) | `Mature` ($T_{\text{min}} \le -40^\circ\text{C}$) | Any | **Lightning** **OR** $(\text{Convective} \land \text{low warming})$ **OR** **OTD** | **`2 = Mature`** |
| **`continued`** | **Warming** ($\Delta T_{\text{min}} > 0$) | `Mature` ($T_{\text{min}} \le -40^\circ\text{C}$) | Any | `else` | **`3 = Decaying`** |

---

### 5.3 Synthesis Attribute: "Severity" & Hazard Type (ATBD Section 3.2.1.2.5.10, pp. 51–52; Annex Line `I` `ch19, ch20`, p. 58)
- **Encoded Severity Scale (`ch19` of Line `I`, ATBD p. 58):**
  - `0 = NIL` (no severity)
  - `1 = Light / Low`
  - `2 = Moderate`
  - `3 = High`
  - `4 = Very High`
  - `5 = Unknown`
- **Additive Severity Scoring Elements (ATBD pp. 51–52):**
  For convective cells, a 4-level severity index (`1 = Low`, `2 = Moderate`, `3 = High`, `4 = Very High`) is incremented based on the following 7 indicators:
  1. **Presence of Overshooting Top (OTD):** $\ge 1$ confirmed OT in the cell.
  2. **High Cooling Rate ($\Delta T_{\text{min}} / \Delta t$):** Stepped thresholds at **$-20^\circ\text{C}/\text{h}$**, **$-50^\circ\text{C}/\text{h}$**, and **$-100^\circ\text{C}/\text{h}$**.
  3. **Strong Horizontal Expansion Rate ($\frac{1}{S}\frac{\Delta S}{\Delta t}$):** Threshold of **$+100\% / 5\text{ min}$** (i.e., doubling of cell area in $5\text{ min}$, or $+1200\%/\text{h}$).
  4. **High Lightning Activity:** Stepped thresholds at **$20\text{ strokes per }15\text{ min}$** and **$50\text{ strokes per }15\text{ min}$**.
  5. **Occurrence of a Lightning Jump (`LJ`):** Active Lightning Jump (`LJ = 2: LJ_ON` or `LJ = 1: inherited`, diagnosed when the 2-min smoothed total flash rate exceeds $10\text{ flashes/min}$ [written $10\text{ s}^{-1}$ in ATBD p. 47] and its temporal trend exceeds $2 \times \text{RMS}$ of the preceding $\ge 12\text{ min}$ flash-rate trend history, ATBD Section 3.2.1.2.5.2, pp. 46–47).
  6. **High Convective Rain Rate (`CRR` / `CRR-Ph`):** $99\text{th}$ percentile rain rate over the cell exceeding the significance threshold (default **$30\text{ mm/h}$**, UM Section 3.3.1.2.5, p. 28).
  7. **High Altitude Ice Crystals Icing Hazard (`HIcgHzd` from `CMIC`):** Present (`HIcgHzd = 2` moderate severity / flag `1`) when cloud phase is `ice` ($> 60\%$ ice pixels), cloud top height $> 5000\text{ m}$, $\text{COT} > 40$ (daytime), and total Cloud Water Path $\text{CWP} = \text{IWP} + \text{LWP} > 0.2\text{ kg/m}^2$ (ATBD Section 3.2.1.2.5.4, pp. 47–48; UM p. 28).
- **Primary Hazard Type Encoding (`ch20` of Line `I`, ATBD pp. 52, 58):**
  - `0 = NIL`, `1 = turbulence` (when OTD is present), `2 = lightning`, `3 = icing`, `4 = high ice water content` (when `HIcgHzd` is on), `5 = hail` (when Lightning Jump is active), `6 = heavy rain rate` (when high `CRR`/`CRR-Ph`), `7 = unknown`.

---

### 5.4 Lagrangian Forecasting / Extrapolation Scheme (`+15, +30, +45, +60 min`, ATBD Section 3.2.1.2.4, pp. 44–45)
- **References:** ATBD Section 3.2.1.2.4 (pp. 44–45), Figure 25 (p. 45), Section 3.2.2.4 (p. 53); UM Table 5 (p. 31).
- **Configuration Keywords (UM Table 5, p. 31):**
  - `FCST = 60` (default maximum lead range in minutes; configurable from `0` [no forecast] up to `60` minutes).
  - `FCSTEP = 15` (default lead step in minutes; produces 4 forecast files at $+15, +30, +45, +60\text{ min}$).
  - `SMOOTHPTS = 0` (default `0 = off`; `1 = smooth advected polygon contours`, recommended for case-study mode).
  - `DILAT = 0` (default `0 = off`; `1 = inflate/contract advected polygon contours according to expansion rate and lead range`, recommended for case-study mode).
- **Algorithmic Steps:**
  1. **Lagrangian Translation of Cell Contour & Gravity Centre:**
     - For each tracked cell $C(t)$ having a valid motion vector estimate $\vec{V}(C) = (u_C, v_C)$ and each forecast lead time $\Delta t_f \in \{15, 30, 45, 60\}\text{ min}$:
     - Every vertex $\vec{x}_i(t)$ of the Base of Tower ($BT$) polygon contour and the cell gravity centre $\vec{G}(t)$ are advected along a constant-velocity Lagrangian trajectory:
       $$\vec{G}(t + \Delta t_f) = \vec{G}(t) + \vec{V}(C)\,\Delta t_f, \qquad \vec{x}_i^{\text{adv}}(t + \Delta t_f) = \vec{x}_i(t) + \vec{V}(C)\,\Delta t_f$$
     - *(Note:Forecast products contain only main cell and bottom `BT` level descriptions in bulletin format, without the `ST` top contour or raster map container, ATBD p. 53. Overlapping between neighbouring forecast cells is not merged, ATBD p. 45).*
  2. **Optional Radial Dilation / Contraction (`DILAT = 1`):**
     - When `DILAT = 1`, each advected contour vertex $\vec{x}_i^{\text{adv}}$ is scaled radially relative to the advected gravity centre $\vec{G}(t + \Delta t_f)$ using the cell's relative area expansion rate $\gamma_S = \frac{1}{S}\frac{\Delta S}{\Delta t}$ and the lead time $\Delta t_f$:
       $$\vec{x}_i^{\text{dilat}}(t + \Delta t_f) = \vec{G}(t + \Delta t_f) + \sqrt{1 + \gamma_S\,\Delta t_f}\,\Big(\vec{x}_i^{\text{adv}}(t + \Delta t_f) - \vec{G}(t + \Delta t_f)\Big)$$
       *(Wait: as noted in ATBD p. 45, for large cells with high expansion rates $\gamma_S > 0$, radial dilation can produce excessively large contours at $+45/+60\text{ min}$, which is why `DILAT = 0` by default in operational mode).*
  3. **Optional Contour Smoothing (`SMOOTHPTS = 1`):**
     - When `SMOOTHPTS = 1`, a moving-average / spline polygon vertex filter is applied along the closed sequence of contour vertices $\{\vec{x}_i\}$ to produce smooth elliptical/rounded forecast outlines (illustrated in ATBD Figure 25 Left, p. 45).
  4. **Forecast Attribute Evolution (ATBD p. 45):**
     - Most attributes remain constant from the analysis time $t$, except during the **first lead ranges up to $+30\text{ min}$**, where the cell's activity/severity index may be incremented if the cell exhibits high cooling rates, high expansion rates, or high lightning trends at time $t$.


---

## 6. Addendum from ATBD v1.1 (`GEO-RDT-CW v4.0 / v2016`, 72 pages)

Comparing **ATBD v1.1** (`NWC-CDOP2-GEO-MFT-SCI-ATBD-Convection_v1.1.pdf`, 15 October 2016, 72 pages) against **ATBD v2.1** (21 January 2019, 62 pages) provides the exact mathematical definitions and edge-case rules that were abbreviated in v2.1:

1. **Single-Tower Excessive Base Refinement ($S_{\text{BT}} < 4\,S_{\text{ST}}$, v1.1 Section 3.2.1.2.1.2, pp. 27–29, Figures 10 & 12):**
   - **Step 1 (Budding from a cloud deck, Figure 10):** If $\min_{n \in [\text{BT}, \text{ST})} (S_{n+1}/S_n) \le 0.7$, move $T_{\text{BT}}$ to the level $N+1$ of sharpest area constriction.
   - **Step 2 (Flat pyramidal single tower, Figure 12):** Using the normalized variable $Y = \min(S_{\text{BT}}, 3\,S_{\text{ST}}) / S_{\text{BT}}$ bounded below by $4/5$, the base area limit is $S_{\text{BT}} < S_{\text{ST}} \cdot 3 \cdot 5 / 4 = 3.75\,S_{\text{ST}}$, rounded to **$S_{\text{BT}} < 4 \cdot S_{\text{ST}}$**.
2. **Complex Many-to-Many Trajectory Rule (v1.1 Section 3.1.5, pp. 21–22):**
   - When $K \ge 2$ former cells match with $M \ge 2$ current cells (`complex` interaction, $< 3\%$ of trajectories), **all incoming trajectories are closed (code `'c'`) and all outgoing cells are initialized as new trajectories**.
3. **Exact 2D Vertical Stratification in $(T_{\text{min}}, T_{\text{BT}})$ Space (v1.1 p. 37 & Figure 22, p. 38):**
   - `0 (mature / DM)`: $T_{\text{min}} \le -40^\circ\text{C}$ ($233.15\text{ K}$) for at least $30\text{ min}$ (v2.1 `CAL`) or $45\text{ min}$ (v1.1 `GEN`).
   - `1 (transition_1 / DTM)`: Crossing $T_{\text{min}} = -40^\circ\text{C}$ ($233.15\text{ K}$) or $T_{\text{BT}} \le -35^\circ\text{C}$ ($238.15\text{ K}$).
   - `2 (transition_2 / Cold transition)`: Crossing $T_{\text{min}} = -35^\circ\text{C}$ ($238.15\text{ K}$) or $T_{\text{BT}} \le -25^\circ\text{C}$ ($248.15\text{ K}$).
   - `3 (transition_3 / Warm2 transition)`: Crossing $T_{\text{min}} = -25^\circ\text{C}$ ($248.15\text{ K}$) or $T_{\text{BT}} \le -15^\circ\text{C}$ ($258.15\text{ K}$).
   - `4 (transition_4 / Warm1 transition)`: Crossing $T_{\text{min}} = -15^\circ\text{C}$ ($258.15\text{ K}$) or $T_{\text{BT}} \le -5^\circ\text{C}$ ($268.15\text{ K}$).
   - `5 (chaud / Warm)`: $T_{\text{min}} > -15^\circ\text{C}$ ($258.15\text{ K}$) and $T_{\text{BT}} > -5^\circ\text{C}$ ($268.15\text{ K}$).
4. **OTD Surrounding Anvil Count Threshold (v1.1 Section 3.2.1.2.5.7, p. 47):**
   - Out of the 16 anvil pixels explored along the 8 compass directions at distances $2\,L_{\text{OT}}$ and $4\,L_{\text{OT}}$ within the same cell $C$, **at least $N_{\text{warm\_surr}} \ge 2$ pixels** must satisfy $BT_{10.8}(p_{\text{surr}}) - BT_{10.8}(p_{\text{OT}}) > 6.0^\circ\text{C}$.