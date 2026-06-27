# Honours Thesis — AI and Polarization

## BuildPanel 0627

Constructs a balanced metro × year panel (2014–2024). Incoporates ACS microdata, GenAI exposure and turns microdata in metropolitan panel data. Finally merges that panel data with FHFA house prices.

**Key inputs**
- `Data/Raw/usa_00004.dat` / `.xml` — IPUMS ACS microdata (fixed-width)
- `Metropolitan AI Exposure 0624/data/raw/genaiexp_estz_occscores.csv` — GenAI scores keyed on 2010 SOC codes
- `Data/crosswalks/soc_2010_to_2018_crosswalk.xlsx` and `2018soc.xlsx`
- `Data/temp/hpi_at_metro.csv` — FHFA House Price Index by CBSA

> IPUMS encodes `OCCSOC` as 2010 SOC for 2014–2017 samples and 2018 SOC for 2018–2024 samples. Step 1 handles this vintage split.

---

**Step 0** — `00_parse_ipums.py`

Parses raw IPUMS fixed-width `.dat` file using the `.xml` codebook. Outputs `Data/Processed/usa_00004.csv`.

---

**Step 1** — `01_merge_genai.py`

Merges GenAI exposure scores onto each ACS worker row.
- Pre-2018: `OCCSOC` is 2010 SOC → matched directly.
- 2018+: `OCCSOC` is 2018 SOC → crosswalked to 2010 first.
- Aggregate codes expanded to detailed children and averaged.

Outputs `Data/Processed/usa_00004_with_genai.csv` (adds `ai_exposure`, `soc2010_mapped`).

---

**Step 2** — `02_build_panel.py`

Collapses microdata into a balanced metro × year panel (ages 25–55, non-institutional). Computes `PERWT`-weighted outcomes across migration, labor market, housing, commuting, and demographics.

Outputs `Data/Processed/panel_00.csv` — one row per metro × year.

**Step 2 diagnostic** — `02.1_compare_soc_coverage.py`: reports which SOC codes lack GenAI scores and affected worker-weight. Outputs `Data/Sumstats/02.1_soc_coverage_comparison.csv`.

---

**Step 3** — `03_merge_hpi.py`

Merges FHFA House Price Index onto the panel. MSAD sub-division CBSA codes are remapped to their parent CBSA; quarterly values are averaged to annual.

Outputs `Data/Processed/panel_01.csv` (adds `hpi_annual`).

**Step 3 diagnostic** — `03.1_compare_hpi_coverage.py`: reports HPI coverage of ACS metro codes. Outputs `Data/Sumstats/03.1_hpi_coverage.txt`.

---

## Panel Variables (`panel_01.csv`)

**Identifiers**
| Variable | Description |
|---|---|
| `MET2013` | Metro area CBSA code (panel unit) |
| `YEAR` | Survey year (2014–2024) |

**Treatment**
| Variable | Description |
|---|---|
| `mean_ai_exposure` | PERWT-weighted mean GenAI exposure score |

**Migration**
| Variable | Description |
|---|---|
| `inmig_rate` | In-migration rate (all) |
| `inmig_rate_ba` | In-migration rate, BA holders |
| `inmig_rate_no_ba` | In-migration rate, non-BA holders |
| `skilled_inmig_share` | Share of in-migrants with BA |
| `net_mig_ba_rate` | Net migration rate, BA holders |
| `net_mig_no_ba_rate` | Net migration rate, non-BA holders |
| `incumbent_share_ba` | BA share among non-migrants |
| `inmig_rate_young` | In-migration rate, ages 25–39 |
| `inmig_rate_old` | In-migration rate, ages 40–55 |

**Labor Market**
| Variable | Description |
|---|---|
| `lfp_rate` | Labor force participation rate |
| `emp_rate` | Employment rate |
| `unemp_rate` | Unemployment rate |
| `mean_wage` | Mean wage income |
| `log_mean_wage` | Log mean wage income |
| `mean_bus_inc` | Mean business income |
| `mean_uhrs` | Mean usual hours worked per week |
| `share_female_lf` | Female share of labor force |

**Education**
| Variable | Description |
|---|---|
| `share_ba` | Share of population with BA or above |
| `share_in_school` | Share currently enrolled in school |

**Housing**
| Variable | Description |
|---|---|
| `mean_rent` | Mean gross rent |
| `log_mean_rent` | Log mean gross rent |
| `mean_owncost` | Mean monthly owner cost |
| `mean_home_value` | Mean home value |
| `log_home_value` | Log mean home value |
| `renter_share` | Share renting (vs. owning) |
| `hpi_annual` | FHFA House Price Index (annual avg) |

**Demographics**
| Variable | Description |
|---|---|
| `log_pop` | Log total population |
| `mean_age` | Mean age |
| `share_white` | Share white (non-Hispanic) |
| `share_black` | Share Black |

**Commuting / Amenities**
| Variable | Description |
|---|---|
| `mean_commute` | Mean commute time (minutes) |
| `share_pub_tran` | Share using public transit |
