# Metro-Level AI Exposure Index

Constructs a metro-level AI exposure measure by mapping IPUMS ACS microdata
(2018 SOC occupation codes) to the Felten et al. (2021) AI exposure scores (2010 SOC), then aggregating to metropolitan areas using person weights.

## Directory Structure

```
.
├── README.md
├── code/
│   ├── 01_merge_felten.py      # SOC crosswalk + merge Felten scores onto ACS
│   └── 02_metro_ai_exposure.py # Aggregate to metro-level weighted average
├── data/
│   ├── raw/
│   │   ├── usa_00002.csv             # IPUMS ACS microdata extract
│   │   └── felten_ai_exposure.csv    # Felten et al. occupation-level AI exposure
│   ├── crosswalks/
│   │   ├── soc_2010_to_2018_crosswalk.xlsx  # BLS official 2010-to-2018 SOC crosswalk
│   │   └── 2018soc.xlsx                     # BLS 2018 SOC structure (Major/Minor/Broad/Detailed)
│   └── processed/
│       ├── usa_00002_with_felten.csv  # ACS microdata with AI exposure merged
│       └── metro_ai_exposure.csv      # Final metro-level AI exposure index
```

## Data Sources

| File | Source | Description |
|------|--------|-------------|
| `usa_00002.csv` | IPUMS ACS (5-year, 2015-2019) | Person-level microdata with occupation codes (OCCSOC), metro area (MET2013), person weight (PERWT), and wage income (INCWAGE). Occupation codes are in 2018 SOC. |
| `felten_ai_exposure.csv` | Felten, Raj & Seamans (2021) | AI exposure scores for 715 occupations using 2010 SOC codes. |
| `soc_2010_to_2018_crosswalk.xlsx` | Bureau of Labor Statistics | Official mapping from 2010 SOC to 2018 SOC detailed codes. |
| `2018soc.xlsx` | Bureau of Labor Statistics | Full 2018 SOC hierarchy (Major, Minor, Broad, Detailed groups). |

## Replication Steps

Run from the project root or the `code/` directory. Requires Python 3 with
`pandas` and `openpyxl`.

### Step 1: Merge Felten scores onto ACS microdata

```bash
python code/01_merge_felten.py
```

This script:
1. Loads the ACS microdata and converts raw OCCSOC codes (e.g., `212011`) to
   standard format (e.g., `21-2011`).
2. Builds a reverse crosswalk (2018 SOC -> 2010 SOC) from the BLS crosswalk.
3. For IPUMS aggregated codes (Broad/Minor/Major groups ending in `0` or `000`),
   expands them to their constituent detailed codes using the 2018 SOC structure,
   then maps each through the crosswalk.
4. Assigns Felten AI exposure scores:
   - **349 codes** map 1:1 to a single Felten score.
   - **86 codes** map to multiple 2010 codes; the score is the equal-weighted
     average of all matched Felten scores.
   - **36 codes** have no Felten match (military, unemployed, or 2010 SOC codes
     absent from Felten). These rows are kept with `NaN` exposure.
5. Outputs `data/processed/usa_00002_with_felten.csv`.

### Step 2: Aggregate to metro level

```bash
python code/02_metro_ai_exposure.py
```

This script:
1. Computes coverage statistics per metro (share of workers with a valid score).
2. Calculates the PERWT-weighted average AI exposure per MET2013 metro area.
3. Outputs `data/processed/metro_ai_exposure.csv`.

## Output Description

### `metro_ai_exposure.csv`

| Column | Description |
|--------|-------------|
| `MET2013` | Metropolitan area FIPS code (2013 delineation) |
| `ai_exposure_wtd_avg` | PERWT-weighted average Felten AI exposure score |
| `perwt_with_score` | Total person weight of workers with a matched score |
| `n_workers` | Number of microdata observations with a matched score |
| `perwt_total_all` | Total person weight of all workers (including unmatched) |
| `perwt_coverage_pct` | Percent of weighted workers with a Felten score |
| `n_workers_all` | Total microdata observations in the metro |
| `n_coverage_pct` | Percent of observations with a Felten score |

### Coverage Summary

- Overall: 91.2% of weighted workers have a Felten score.
- All 261 metros have at least 80% coverage.
- 240 metros (92%) have 90%+ coverage.
- Missing scores are concentrated in supervisor, military, and miscellaneous
  "all other" occupation categories, distributed fairly uniformly across metros.
