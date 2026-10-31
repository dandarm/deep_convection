# Authoritative Specification: EUMETSAT NWC SAF Cloud Mask (`GEO-CMA`) & Cloud Type (`GEO-CT`) v2025

> **Reference Documents:**
> - **ATBD:** *Algorithm Theoretical Basis Document for the Cloud Product Processors of the NWC/GEO* (`NWC/CDOP2/MTG/MFL/SCI/ATBD/Cloud`, Issue 1.1.1, 30 May 2025, 124 pp., and MSG/SEVIRI `NWC/CDOP2/GEO/MFL/SCI/ATBD/Cloud`, Issue 1.1 / 2.1).
> - **UM:** *User Manual for the Cloud Product Processors of the NWC/GEO: Science Part* (`NWC/CDOP3/MTG/MFL/SCI/UM/Cloud`, Issue 1.1.1, 30 May 2025, 58 pp., applicable to **`GEO-CMA-v6.0 (NWC-004)`** and **`GEO-CT-v5.0 (NWC-008)`** in NWC/GEO release `v2025`).
> - **Official Portal:** <https://www.nwcsaf.org/cma_v2025>

---

## 1. Product Overview and Role in the NWC SAF Processing Chain

Within the EUMETSAT Nowcasting and Very Short Range Forecasting Satellite Application Facility (**NWC SAF**), the **Cloud Mask (`GEO-CMA`, PGE01 / NWC-004)** and **Cloud Type (`GEO-CT`, PGE02 / NWC-008)** processors form the foundational cloud analysis front-end at native geostationary IR horizontal resolution ($3\text{ km}$ at nadir for MSG/SEVIRI):

1. **`GEO-CMA` (Cloud Mask):** Identifies all cloud-contaminated and snow/ice-contaminated pixels, generates separate flags for mineral **dust plumes (`cma_dust`)** and **volcanic ash plumes (`cma_volcanic`)**, records a 28-bit diagnostic test bitmask (`cma_testlist`), an 11-bit processing status flag (`cma_status_flag`), and a per-pixel quality flag (`cma_quality`).
2. **`GEO-CT` (Cloud Type):** Refines every pixel analysed by `GEO-CMA` into **15 physical surface and cloud categories (`ct`)**, separating clear land/sea, snow/ice, **5 vertical levels of opaque clouds** (very low, low, mid-level, high, very high), **fractional sub-pixel clouds**, and **5 categories of high semitransparent / multilayer cirrus clouds**, plus auxiliary containers for **stratiform vs. cumuliform character (`ct_cumuliform`)** and **multilayer occurrence (`ct_multilayer`)**.
3. **Downstream Dependency:** Both **Rapidly Developing Thunderstorms (`GEO-RDT-CW`)** and **Convection Initiation (`GEO-CI`)** depend directly on `GEO-CMA` and `GEO-CT` to screen out non-convective cold semitransparent cirrus, identify developing warm/mid-level cumulus towers, and anchor clear-sky surface temperature statistics.

---

## 2. Theoretical Specification of `GEO-CMA` (Cloud Mask v6.0 / v2025)

### 2.1 Output Containers (`ATBD Section 2.2.2.4`, pp. 59–63)

| Container | Type | Categories / Bit Definitions |
| :--- | :--- | :--- |
| **`cma`** | `uint8` | `0`: Cloud-free, `1`: Cloudy, `255`: FillValue |
| **`cma_cloudsnow`** | `uint8` | `0`: Cloud-free, `1`: Cloud (except thin ice cloud over snow), `2`: Thin ice cloud over snow/ice, `3`: Snow/Ice, `255`: FillValue |
| **`cma_dust`** | `uint8` | `0`: No dust, `1`: Dust, `2`: Undefined (separability problem), `255`: FillValue |
| **`cma_volcanic`** | `uint8` | `0`: No volcanic plume, `1`: Volcanic plume, `2`: Undefined, `255`: FillValue |
| **`cma_smoke`** | `uint8` | `0`: No smoke, `1`: Smoke, `2`: Undefined (set to `2` in v2025), `255`: FillValue |
| **`cma_testlist`** | `uint32` | 28-bit bitmask of triggered cloud/snow/spatial/temporal tests (Bits `0..27`) |
| **`cma_status_flag`** | `uint16` | 11-bit bitmask of geophysical and ancillary conditions (Bits `0..10`) |
| **`cma_quality`** | `uint8` | `0`: N/A, `1`: Good, `2`: Questionable, `3`: Bad, `4`: Interpolated |

#### 28-Bit `cma_testlist` Definition (`ATBD p. 60`)
- **Bit 0:** $R_{0.6\,\mu\text{m}}$ (land) or $R_{0.8\,\mu\text{m}}$ (sea)
- **Bit 1:** $R_{1.6\,\mu\text{m}}$ (sea)
- **Bit 2:** Sunglint test using $T_{3.8\,\mu\text{m}}$
- **Bit 3:** $R_{1.38\,\mu\text{m}}$ cirrus test
- **Bit 4:** $T_{10.8\,\mu\text{m}}$ or $\text{SST}(T_{10.8}, T_{12.0})$ thermal test
- **Bit 5:** $T_{10.8\,\mu\text{m}} - T_{12.0\,\mu\text{m}}$ split-window test
- **Bit 6:** $T_{10.8\,\mu\text{m}} - T_{3.8\,\mu\text{m}}$ low water cloud test
- **Bit 7:** $T_{12.0\,\mu\text{m}} - T_{3.8\,\mu\text{m}}$ ocean low water cloud test
- **Bit 8:** $T_{3.8\,\mu\text{m}} - T_{10.8\,\mu\text{m}}$ test
- **Bit 9:** $T_{10.8\,\mu\text{m}} - T_{8.7\,\mu\text{m}}$ high-secant low cloud test
- **Bit 10:** $T_{8.7\,\mu\text{m}} - T_{10.8\,\mu\text{m}}$ thin ice cloud test
- **Bit 11:** $T_{8.7\,\mu\text{m}} - T_{3.8\,\mu\text{m}}$ desert low cloud test
- **Bits 12–14:** Snow/ice detection tests
- **Bit 15:** Local Spatial Texture test
- **Bits 16–21:** On-line RTTOV clear-sky simulation tests ($T_{10.8}$, $T_{3.8}$, $T_{8.7}-T_{3.8}$, $T_{10.8}-T_{12.0}$, $T_{10.8}-T_{8.7}$, $T_{10.8}-T_{3.8}$)
- **Bit 22:** 15-minute Temporal-differencing test ($\Delta_{15\text{min}} T_{10.8}$)
- **Bit 23:** Stationary cloud in twilight (1-hour temporal coherence)
- **Bit 24:** Spatial extension of stationary clouds (8-connected region growing)
- **Bit 25:** High resolution visible (HRV) sub-pixel test
- **Bit 26:** Spatial filtering: cloudy pixel reclassified as cloud-free
- **Bit 27:** Spatial filtering: cloud-free pixel reclassified as cloudy

---

### 2.2 Multispectral, Textural, Temporal, and Spatial Equations of `GEO-CMA`

1. **Split-Window SST Test over Ocean (`ATBD Section 2.2.1.2.2.1`, p. 14):**
   Applied over ice-free sea ($\text{SST}_{\text{clim}} > 270.15\text{ K}$):
   $$\text{SST}(T_{10.8}, T_{12.0}) = 0.977\,(T_{10.8} - 273.15) + \bigl(0.075\,(\text{SST}_{\text{clim}} - 273.15) + 1.127\,(\sec\theta_{\text{sat}} - 1)\bigr)(T_{10.8} - T_{12.0}) + 1.156 + 273.15\text{ K}$$
   A pixel is flagged cloudy if $\text{SST}(T_{10.8}, T_{12.0}) < \text{SST}_{\text{threshold}}$, where $\text{SST}_{\text{threshold}} = \text{SST}_{\text{clim}} - \Delta_{\text{sec}}(\sec\theta_{\text{sat}}) - 0.5\,\mathbb{I}_{\text{coast}}$, with $\Delta_{\text{sec}}$ varying linearly from $1.5\text{ K}$ ($\theta_{\text{sat}} \le 60^\circ$) to $2.5\text{ K}$ ($\theta_{\text{sat}} \ge 78.5^\circ$).

2. **Window Thermal Contrast Test on $T_{10.8\,\mu\text{m}}$ (`ATBD Section 2.2.1.2.2.2`, pp. 14–16):**
   $$T_{10.8} < T_{10.8,\text{threshold}} = T_{\text{surf}} - \Delta T_{\text{abs}}(\sec\theta_{\text{sat}}, \text{TCWV}) + \Delta T_{\text{elev}}$$
   Under extreme continental cooling ($z < 1500\text{ m}$ and $T_{\text{surf}} < 263\text{ K}$), $T_{10.8,\text{threshold}}$ is lowered to prevent cold clear ground from triggering false cloud alarms (`ATBD p. 16`):
   $$T_{10.8,\text{threshold}} \leftarrow \begin{cases} T_{10.8,\text{threshold}} - 5.0\text{ K}, & \text{if } T_{10.8,\text{threshold}} \ge 255\text{ K}, \\ T_{10.8,\text{threshold}} - 5.0\text{ K} - 0.4\,(255\text{ K} - T_{10.8,\text{threshold}}), & \text{if } T_{10.8,\text{threshold}} < 255\text{ K}. \end{cases}$$

3. **Split-Window Test on $T_{10.8\,\mu\text{m}} - T_{12.0\,\mu\text{m}}$ (`ATBD Section 2.2.1.2.2.3`, pp. 16–17):**
   $$T_{10.8} - T_{12.0} > T_{10.8\text{-}12.0,\text{threshold}}(\sec\theta_{\text{sat}}, \text{TCWV}) \quad \text{and (over land) } T_{10.8} < 303.15\text{ K}$$

4. **Tri-Spectral Ice Test on $T_{8.7\,\mu\text{m}} - T_{10.8\,\mu\text{m}}$ (`ATBD Section 2.2.1.2.2.4`, pp. 17–18):**
   $$T_{8.7} - T_{10.8} > T_{8.7\text{-}10.8,\text{threshold}}(\sec\theta_{\text{sat}}, \text{TCWV})$$
   With $+0.4\text{ K}$ added under extreme cooling when $T_{10.8,\text{threshold}} < 250\text{ K}$.

5. **High-Secant Low Water Cloud Test on $T_{10.8\,\mu\text{m}} - T_{8.7\,\mu\text{m}}$ (`ATBD Section 2.2.1.2.2.9`, p. 23):**
   For $\sec\theta_{\text{sat}} > 1.5$:
   $$T_{10.8} - T_{8.7} > 3.7 + 0.3\,\sec\theta_{\text{sat}}\text{ (in K)}$$

6. **Local Spatial Texture Test (`ATBD Section 2.2.1.2.2.17`, pp. 29–30):**
   Computed on a $3\times 3$ neighborhood restricted to pixels of the **same surface type** (sea or land):
   - **Over Sea:** $\text{SD}_{3\times 3}(T_{10.8}) > 0.6\text{ K}$, $\text{SD}_{3\times 3}(\text{BTD}) > 0.2\text{--}0.4\text{ K}$, and $T_{10.8} < \max_{3\times 3}(T_{10.8}) - 2\,\sigma_{\text{noise}}(T_{10.8})$.
   - **Over Land:** $\text{SD}_{3\times 3}(T_{10.8}) > 1.0\text{--}2.0\text{ K}$ and $\text{SD}_{3\times 3}(\text{BTD}) > 1.0\text{--}2.0\text{ K}$.

7. **Quality Assessment Security Margins (`ATBD Section 2.2.1.2.2.18`, p. 31):**
   A test is *really successful* (`cma_quality = GOOD`) if $|\text{feature} - \text{threshold}| > \Delta_{\text{margin}}$:
   - $\text{SST}$: $2.0\text{ K}$
   - $T_{10.8\,\mu\text{m}}$: $3.0\text{ K}$
   - $T_{10.8} - T_{12.0}$ and BTD tests: $0.5\text{ K}$
   - Local Spatial Texture & Reflectance tests: $0.2 \times \text{threshold}$.

8. **15-Minute Temporal Differencing Test (`ATBD Section 2.2.1.2.4`, pp. 37–38):**
   Applied to clear/snowy candidates using slot $t - 15\text{ min}$:
   $$\Delta_{15\text{min}}(T_{10.8}) = T_{10.8}(t) - T_{10.8}(t - 15\text{ min}) < \begin{cases} -0.6\text{ K}, & \text{over sea}, \\ \text{DT}_{10.8} \in [-3.0, -0.6]\text{ K}, & \text{over land}. \end{cases}$$
   Flagged as `QUESTIONABLE` quality (`Bit 22`).

9. **1-Hour Stationary Low/Mid Cloud Coherence & Region Growing (`ATBD Section 2.2.1.2.5`, pp. 38–41):**
   - **Step 1 (Temporal Coherence, `Bit 23`):** Restores clear pixels that were classified 1 hour earlier as low or mid-level clouds with `GOOD` quality if:
     - Over Land: $|\Delta_{1\text{h}}(T_{10.8})| < 1.0\text{ K}$ and $|\Delta_{1\text{h}}(T_{10.8} - T_{8.7})| < 0.5\text{ K}$.
     - Over Sea: $|\Delta_{1\text{h}}(T_{10.8})| < 1.0\text{ K}$ and $|\Delta_{1\text{h}}(T_{10.8} - T_{12.0})| < 0.6\text{ K}$.
   - **Step 2 (Spatial Region Growing, `Bit 24`):** Connected restored seed groups with $> 8$ pixels are expanded via 8-connectivity to neighboring clear pixels with $\overline{T}_{10.8,\text{seed}} - 5.0\text{ K} < T_{10.8}(x) < \overline{T}_{10.8,\text{seed}} + 0.5\text{ K}$ (rejected if a single seed expands to $> 10{,}000$ pixels).

10. **Final Spatial Filtering (`ATBD Section 2.2.1.2.7`, pp. 49–52):**
    - **Cold Nocturnal Inversion Restoral (`p. 50`):** Cloudy pixels detected by $T_{10.8,\text{thr}}$ where $T_{10.8,\text{thr}} < 250\text{ K}$ and $T_{7.3} - T_{10.8} > 0.5\text{ K}$ are restored to clear (`QUESTIONABLE` quality, `Bit 26`).
    - **Coastal Texture Filtering (`p. 50`):** Coastal pixels detected by texture belonging to a $3\times 3$ box $\ge 10\%$ clear and a $7\times 7$ box $\ge 50\%$ clear are restored to clear (`INTERPOLATED` quality, `Bit 26`).
    - **Snow-Edge Filtering (`p. 51`):** Outer snow-edge band pixels detected only by texture with $\ge 1$ clear neighbor and no non-texture cloudy neighbor in $3\times 3$ are restored to clear (`Bit 26`).
    - **Isolated Pixel Filtering (`p. 52`):** Isolated cloudy pixels (0 cloudy 8-neighbors) from secondary tests are restored to clear (`Bit 26`); isolated clear pixels (8 cloudy 8-neighbors) are reclassified to cloudy (`Bit 27`).

11. **Dust (`cma_dust`) and Volcanic Ash (`cma_volcanic`) Identification (`ATBD Sections 2.2.1.2.8–2.2.1.2.9`, pp. 52–55):**
    Both exploit the **reverse split-window absorption signature** of silicate particles ($T_{12.0} - T_{10.8} > 0$), combined with $T_{8.7} - T_{10.8}$, thermal range relative to $T_{\text{surf}}$, and spatial homogeneity $\text{SD}_{3\times 3}(T_{10.8}) < 0.4\text{ K}$ (sea) / $3.0\text{ K}$ (land).

---

## 3. Theoretical Specification of `GEO-CT` (Cloud Type v5.0 / v2025)

### 3.1 The 15 Official Cloud Type Categories (`ATBD Section 3.2.2.4`, p. 81)

| `ct` Value | Category Name | Physical Regime |
| :---: | :--- | :--- |
| **`0`** | `Non-processed` | No data or corrupted pixel |
| **`1`** | `Cloud-free land` | Clear continental surface (`cma_cloudsnow == 0` over land) |
| **`2`** | `Cloud-free sea` | Clear oceanic/water surface (`cma_cloudsnow == 0` over sea) |
| **`3`** | `Land contaminated by snow` | Snow-covered land (`cma_cloudsnow == 3` over land) |
| **`4`** | `Sea contaminated by ice` | Sea ice (`cma_cloudsnow == 3` over sea) |
| **`5`** | `Very low clouds` | Opaque cloud with $T_{10.8} > \text{maxT108low}$ ($P_{\text{top}} > 800\text{ hPa}$) |
| **`6`** | `Low clouds` | Opaque cloud with $\text{maxT108med} < T_{10.8} \le \text{maxT108low}$ ($650\text{--}800\text{ hPa}$), or reclassified from mid-level by $T_{10.8}-T_{7.3}$ or thermal inversion test |
| **`7`** | `Mid-level clouds` | Opaque cloud with $\text{maxT108hi} < T_{10.8} \le \text{maxT108med}$ and $T_{10.8}-T_{7.3} \le \text{T108T73thrlow}$ ($450\text{--}650\text{ hPa}$) |
| **`8`** | `High opaque clouds` | Opaque cloud with $\text{maxT108vh} < T_{10.8} \le \text{maxT108hi}$ ($300\text{--}450\text{ hPa}$) |
| **`9`** | `Very high opaque clouds` | Opaque cloud with $T_{10.8} \le \text{maxT108vh}$ ($P_{\text{top}} < 300\text{ hPa}$, deep convective cores/anvils) |
| **`10`** | `Fractional clouds` | Sub-pixel broken low clouds ($T_{10.8} > \text{maxT108med}$, $T_{10.8}-T_{12.0} > \text{T108T120opaque}$, $T_{8.7}-T_{10.8} \le \text{T87T108opaque}$) |
| **`11`** | `High semitransparent thin clouds` | Optically thin ice cirrus ($\text{maxT108med} < T_{10.8} < \text{maxT108low}+\Delta$, $T_{10.8}-T_{12.0} > \text{T108T120opaque}$, $T_{8.7}-T_{10.8} > \text{T87T108opaque}$) |
| **`12`** | `High semitransparent moderately thick clouds` | Moderately thick cirrus ($\text{maxT108hi} < T_{10.8} \le \text{maxT108med}$, $T_{10.8}-T_{12.0} > \text{T108T120thick}$) |
| **`13`** | `High semitransparent thick clouds` | Thick cirrus anvil ($T_{10.8} \le \text{maxT108hi}$, $T_{10.8}-T_{12.0} > \text{T108T120thick}$) |
| **`14`** | `High semitransparent above low or medium clouds` | Multilayer cloud system (semitransparent ice cirrus overlying a lower opaque water/mixed cloud deck) |
| **`15`** | `High semitransparent above snow/ice` | Thin ice cloud over snow/ice (`cma_cloudsnow == 2`) |

---

### 3.2 Exact Height Separation Thresholds & Guards (`ATBD Section 3.2.1.2.2.4`, pp. 76–77)

The vertical boundaries between opaque cloud classes (`Classes 5, 6, 7, 8, 9`) and semitransparent thickness classes (`Classes 11, 12, 13`) are governed by the exact NWC SAF linear combinations of atmospheric temperatures at $850\text{ hPa}$, $700\text{ hPa}$, $500\text{ hPa}$, and the tropopause (`ATBD p. 77`):

$$\begin{aligned}
\text{maxT108vh} &= \begin{cases} 0.4\,T_{500\text{hPa}} + 0.6\,T_{\text{tropo}} - 5\text{ K}, & \text{if } T_{\text{tropo}} \text{ is available}, \\ \text{maxT108hi} - 25\text{ K}, & \text{otherwise}, \end{cases} \\
\text{maxT108hi} &= 0.5\,T_{500\text{hPa}} - 0.2\,T_{700\text{hPa}} + 178\text{ K}, \\
\text{maxT108med} &= 0.8\,T_{850\text{hPa}} + 0.2\,T_{700\text{hPa}} - 8\text{ K}, \\
\text{maxT108low} &= 1.2\,T_{850\text{hPa}} - 0.2\,T_{700\text{hPa}} - 5\text{ K}, \\
\text{T108interthr} &= 0.5\,(\text{maxT108med} + \text{maxT108hi}), \\
\text{T108T73thrlow} &= 4.0\,\sec(\theta_{\text{sat}}) + 8.5\text{ K}.
\end{aligned}$$

#### Water-Vapour $T_{10.8} - T_{7.3}$ Guard (`ATBD p. 76`)
To prevent low clouds capped by cold boundary-layer air from being misclassified as medium clouds (`Class 7`), any opaque cloud with $\text{maxT108hi} < T_{10.8} \le \text{maxT108med}$ that satisfies:
$$T_{10.8} - T_{7.3} > \text{T108T73thrlow} = 4.0\,\sec(\theta_{\text{sat}}) + 8.5\text{ K}$$
is reclassified from **Mid-level (`Class 7`)** to **Low clouds (`Class 6`)**.

#### Low-Level Thermal Inversion $T_{8.7} - T_{10.8}$ Reclassification (`ATBD p. 77`)
A pixel in the medium-cloud temperature band ($\text{maxT108hi} < T_{10.8} \le \text{maxT108med}$) is also reclassified as **Low clouds (`Class 6`)** if any of the three viewing-secant-dependent tri-spectral conditions holds:
1. $T_{10.8} > \text{maxT108med} - 5\text{ K}$ and $T_{8.7} - T_{10.8} < -1.2 - (\sec\theta_{\text{sat}} - 1)\text{ K}$
2. $T_{10.8} > \text{maxT108med} - 8\text{ K}$ and $T_{8.7} - T_{10.8} < -1.7 - (\sec\theta_{\text{sat}} - 1)\text{ K}$
3. $T_{10.8} > \text{maxT108med} - 10\text{ K}$ and $T_{8.7} - T_{10.8} < -2.2 - (\sec\theta_{\text{sat}} - 1)\text{ K}$.

#### `GEO-CT` Quality Assessment (`ATBD Section 3.2.1.2.2.5`, p. 78)
A cloudy pixel is flagged as low confidence (`QUESTIONABLE`) if the absolute difference between any active spectral feature and its classification threshold is smaller than **$0.2\text{ K}$**.
