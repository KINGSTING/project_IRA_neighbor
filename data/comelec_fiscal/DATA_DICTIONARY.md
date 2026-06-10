# DATASET 2 — COMELEC × LGU FISCAL PANEL DATA  
Integrated Fiscal, Electoral, and Governance Indicators  
ICARE 2025 Policy Hackathon

This dataset merges COMELEC electoral data, BLGF local fiscal data, and derived governance indicators.  
It is a **panel dataset** (LGU × Year), allowing analysis of electoral dynamics, fiscal behavior, and governance quality.

---

# 1. VARIABLE DEFINITIONS (Official Codebook)

Below is the authoritative dictionary based on the codebook provided.

| Code | Definition |
|------|------------|
| **region** | Philippine administrative region |
| **lgu** | Local Government Unit (Province / City / Municipality) name |
| **year** | Fiscal reporting year |
| **lgutype** | LGU type (Province, City, Municipality) |
| **elecyr** | Election year relevant to the row |
| **lgusincome** | LGU’s total income (in millions PHP) |
| **totlocsrc** | Total locally sourced revenue (in millions PHP) |
| **ira** | Internal Revenue Allotment (IRA) / NTA (in millions PHP) |
| **tottax** | Total tax collections (in millions PHP) |
| **totexsrc** | Total external revenue sources (in millions PHP) |
| **govexp** | Government expenditures (administrative) in millions |
| **pubwelf** | Public welfare expenditures in millions |
| **educexp** | Education expenditures in millions |
| **healthexp** | Health expenditures in millions |
| **laborexp** | Labor sector expenditures in millions |
| **housingexp** | Housing-related expenditures |
| **socservexp** | Social services expenditures |
| **econdevexp** | Economic development expenditures |
| **totexp** | Total expenditures (in millions) |
| **rid** | Region code (numeric) |
| **pid** | Province code (numeric) |
| **incumbent** | Name of incumbent governor |
| **party** | Political party of candidate |
| **sex** | Sex of the local incumbent (1 = male, coding varies) |
| **votes** | Votes obtained by the focal candidate |
| **no_cand** | Number of candidates that ran for governor |
| **totvot** | Total votes cast in the province |
| **pi** | Vote share of candidate = votes / totvot |
| **p1** | Highest vote share among all candidates |
| **sumpi2** | Sum of squared vote shares (Herfindahl index component) |
| **ENC_lt** | Effective number of candidates (Laakso-Taagepera) |
| **ENC_gol** | Effective number of candidates (Golosov index) |
| **loc_tot** | Ratio = total local sources / LGU income |
| **logpubwelf** | Log(public welfare expenditures) |
| **loggovexp** | Log(government expenditures) |
| **logira** | Log(IRA/NTA) |
| **ira_tot** | Share of IRA over total income |

---

# 2. ADDITIONAL VARIABLES FOUND IN DATA (From Your Sample Rows)

The dataset includes many additional variables not in the codebook but essential for analysis.

### 🔹 **Log and ratio variables**
| Variable | Description |
|---------|-------------|
| `logloc` | Log(totlocsrc) |
| `logtax` | Log(tottax) |
| `taxtot` | Ratio of tax to total income |
| `locira` | Locally sourced income + IRA indicator |
| `logloctot` | Log(total local revenues) |
| `logiratot` | Log(IRA ratio) |

---

### 🔹 **Model-generated variables**
| Variable | Description |
|----------|-------------|
| `_est_fe` | Fixed-effects model prediction |
| `_est_re` | Random-effects model prediction |
| `_est_mod1` to `_est_mod4` | Outputs of econometric models used by the research team |
| `ENC_r` | Additional competition index |

These should be treated as **derived**, not raw administrative variables.

---

### 🔹 **Election timing & lag variables**
| Variable | Meaning |
|----------|---------|
| `yrb4` | Indicator if the year is before an election |
| `yearb4elec` | Same as above; may differ in coding |
| `pres_elec` | 1 if it is a presidential election year |

---

### 🔹 **Averaged (smoothing) variables**
These appear to be moving averages or multi-year averages:

| Variable | Description |
|---------|-------------|
| `ave_genpub` | Average general public services expenditures |
| `logavegenpub` | Log average general public services |
| `ave_ira` | Average IRA |
| `logaveira` | Log average IRA |
| `ave_totinc` | Average total income |
| `ave_iratot` | Average IRA ratio |
| `ave_loc` | Average local revenue |
| `ave_loctot` | Average total local revenue |
| `locext` | Possibly local × external revenue index (confirm needed) |

---

### **Social services indicators**
| Variable | Meaning |
|---------|---------|
| `logsocserv` | Log(social services expenditures) |
| `soc_exp` | Social expenditure amount |
| `socserv` | Social services expenditure (duplicate naming) |
| `socserv_totexp` | Social services spending as share of total expenditures |
| `logsocserv_totexp` | Log of the above |

---

### **Government spending indicators**
| Variable | Meaning |
|----------|---------|
| `loggovt` | Log(government expenditure total) |
| `govt_totexp` | Government spending / total expenditure ratio |

---

### **Missing or blank columns (found in sample rows)**
Some rows have empty values for:
- `econdevexp`
- Some logs
- Some model estimates

These represent **missing administrative data**, not zeros.

---

# 3. KEYS AND PANEL STRUCTURE

Primary panel identifier:

Alternative identifiers:
- `pid` (province code) + `year`
- `lgu` + `year`

---

# 4. DATA TYPES SUMMARY

| Type | Variables |
|------|-----------|
| **String / Categorical** | region, lgu, lgutype, incumbent, party |
| **Integer** | year, elecyr, rid, pid, no_cand, totvot |
| **Numeric (float)** | all fiscal amounts, ratios, logs |
| **Binary/Indicators** | yearb4elec, pres_elec, sometimes sex |

---

# 5. INTERPRETATION NOTES

- **Fiscal spikes before elections** may indicate patronage or strategic spending.  
- **High IRA dependence** may relate to weaker local revenue autonomy.  
- **Vote concentration (p1, pi, ENC)** reveals local political monopolies.  
- **High administrative spending (govexp)** vs social spending (socservexp) may signal governance quality issues.

---

# 6. SUGGESTED USES FOR ICARE 2025 HACKATHON

Participants can analyze:

###  **6.1 Electoral incentives and spending**
- Do incumbents increase welfare or infrastructure spending before elections?  
- Do LGUs with higher education/health expenditures reward or punish incumbents?

###  **6.2 Political competition**
- Low `ENC_lt` and high `p1` suggest political monopolies.  
- Does political monopoly decrease local revenue effort?

###  **6.3 Fiscal governance & service delivery**
- Ratio of `socservexp / totexp` vs vote margins  
- LGUs with high administrative spending vs. low social spending  

### **6.4 Red flags & policy insights**
- Election-year overspending  
- IRA overdependence  
- Declining local revenue effort  
- Gender representation in incumbency patterns

