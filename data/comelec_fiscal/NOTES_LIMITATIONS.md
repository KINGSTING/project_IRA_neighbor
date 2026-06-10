# NOTES & LIMITATIONS  
Dataset 2 — COMELEC × LGU Fiscal Panel Data

This dataset integrates COMELEC election results with BLGF (Bureau of Local Government Finance) fiscal indicators. It enables analysis of electoral behavior, fiscal governance, political competition, and local public expenditure patterns.

---

# 1. Coverage Limitations

1. **Unbalanced panel**  
   Not all LGUs have complete data for every year. Some provinces or municipalities may have gaps due to non-reporting.

2. **Barriers in fiscal disaggregation**  
   Some expenditure categories (e.g., health, education, labor) may appear as “0” because:
   - LGUs aggregated them into other categories, or  
   - BLGF did not receive complete sectoral breakdowns.

3. **Elections do not always align with fiscal years**  
   Some rows show `elecyr` different from `year`; this is normal because fiscal reporting is annual while elections occur every three years.

---

# 2. Data Quality Concerns

1. **Zeros vs Missing Values**  
   Entries such as `healthexp = 0` may mean:
   - No recorded spending, **or**  
   - The LGU did not report sectoral breakdown.  
   Caution is needed to avoid misinterpretation.

2. **Party and candidate name standardization**  
   Party labels (e.g., LP, NUP, PDPLBN) may have inconsistencies in capitalization or naming variants.

3. **Derived variables (logs & ratios)**  
   Log values may produce:
   - `NaN` or `-inf` when the raw value is zero  
   - Highly skewed ratios in small LGUs  

4. **Political competition indices**  
   `ENC_lt`, `ENC_gol`, `pi`, `p1` depend heavily on the presence of third or fourth candidates. LGUs with only two candidates may show artificially low competition.

---

# 3. Interpretation Warnings

1. **High spending is not automatically corruption**  
   Election-year spikes may reflect:
   - Legitimate infrastructure programs  
   - National transfers  
   - Disaster response spending  

   Analysts must contextualize before concluding patronage.

2. **Political monopolies**  
   High vote share (`p1`) or low effective number of candidates (`ENC_lt`) signals dominance but does **not automatically imply abuse**. It triggers further investigation.

3. **IRA dependence**  
   High IRA share may indicate:
   - Fiscal vulnerability  
   - Limited local economic activity  
   - Overreliance on central transfers

4. **Comparisons across regions**  
   BARMM often has different fiscal structures and political dynamics compared to other regions. Avoid overgeneralizing.

---

# 4. Suggested Cleaning Steps

- Convert all fiscal values to numeric (remove commas).  
- Standardize LGU names using uppercase and trimmed text.  
- Ensure vote shares recompute correctly: `pi = votes / totvot`.  
- Replace zeros in expenditure fields with NaN where appropriate.  
- Winsorize extreme outliers (top 1%).  

---

# 5. Best Uses for the Hackathon

- Election-year fiscal behavior (spending spikes).  
- IRA dependency and governance vulnerability.  
- Gender patterns in local political leadership.  
- Effect of political competition on fiscal priorities.  
- Mapping LGU clusters by governance quality.  

---

# 6. Ethical Notes

No individual-level data is included.  
All candidates listed are **public officials**, so the dataset remains compliant with ethical guidelines.  
Interpretations should avoid defamatory or accusatory language; frame findings as **patterns**, not allegations.
