# Dataset 7: PHILSA Administrative Boundaries (ADM0–ADM4)

## Source
Philippine Space Agency (PHILSA) — Administrative Boundary Dataset (2022–2023)

## Description
This dataset provides officially recognized political and geographic boundaries of the Philippines from national to barangay levels.  
It includes hierarchical administrative codes (ADM0 to ADM4), names, area (sq km), and validity dates for temporal versioning.

This dataset is essential for geospatial mapping, LGU boundary validation, spatial joins, and regional analytics.

---

## Columns and Definitions

| Column Name | Description |
|-------------|-------------|
| **ADM4_EN** | English name of the ADM4 unit (Barangay or equivalent). |
| **ADM4_PCODE** | Official PHILSA administrative code for ADM4. |
| **ADM4_REF** | Reference code or additional metadata for ADM4 if applicable (nullable). |
| **ADM3_EN** | English name of the ADM3 unit (Municipality / City). |
| **ADM3_PCODE** | Official PHILSA administrative code for ADM3. |
| **ADM2_EN** | English name of the ADM2 unit (Province). |
| **ADM2_PCODE** | Official PHILSA administrative code for ADM2. |
| **ADM1_EN** | English name of the ADM1 unit (Region). |
| **ADM1_PCODE** | Official PHILSA administrative code for ADM1. |
| **ADM0_EN** | English name of the ADM0 unit (Country). Usually "Philippines (the)". |
| **ADM0_PCODE** | Country code (PH). |
| **date** | Date when the specific administrative boundary record was first captured or released. |
| **validOn** | Start of validity period for this boundary definition (temporal data versioning). |
| **validTo** | End of validity period; empty if still valid. |
| **AREA_SQKM** | Total area of the ADM4 geographic unit measured in square kilometers. |

---

## Notes
- PHILSA uses the **PCODE** hierarchy based on Philippine administrative structure:  
  **ADM0 → ADM1 → ADM2 → ADM3 → ADM4**  
  (Country → Region → Province → Municipality/City → Barangay)
- **Area values (AREA_SQKM)** are derived using official PHILSA geospatial boundaries.
- **validOn** and **validTo** enable time-aware analytics, useful for tracking boundary changes.
- Dataset can be used for:  
  - Choropleth maps  
  - Spatial clustering  
  - Regional demographic overlays  
  - LGU alignment for DPWH, DOH, DepEd, PSA datasets