# OccPanel 0628 — national occupation × year panel

Asks how **rent** for high-exposed **occupations** evolved relative to low-exposed
ones around the ChatGPT release. Exposure varies natively at the occupation level,
so this design uses the variation directly rather than diluting it into a metro
average (the earlier metro panel found no effect).

**Self-contained:** the pipeline rebuilds everything from raw inputs in
`Data/Raw/` and `Data/crosswalks/` — no other folder is required.

## Run order (from `OccPanel 0628/code/`)

| Step | Script | Output |
|------|--------|--------|
| 0 | `00_parse_ipums.py` | `Data/Processed/usa_00004.csv` — raw IPUMS `.dat` → CSV |
| 1 | `01_merge_genai.py` | `Data/Processed/usa_00004_with_genai.csv` — + GenAI exposure scores (handles the 2018 OCCSOC vintage split) |
| 2 | `02_build_occ_panel.py` | `Data/Processed/occ_panel_00.csv` — occupation × year outcomes + `emp_share` |
| 3 | `03_exposure_groups.py` | `Data/Processed/occ_panel_01.csv` — adds `expo_tercile` (extreme groups) + `z_expo` (dose) |
| 4 | `04_did.py` | `figures/occ_did_pooled.csv`, `occ_did_eventstudy.csv`, `occ_rent_wfh_test.csv` + `figures/eventstudy_*.png` |

Steps run 0→1→2→3→4 in order. The WFH-mechanism test for rent (does the rent
decline just reflect the WFH/donut effect?) runs inside `04_did.py`, alongside the
`+renter_share` / `+wage` / `+log_emp` robustness specs.

## Key design choices (see each script header for detail)

- **Occupation key** = `soc2010_mapped` — stable across the 2018 OCCSOC vintage break.
- **Unscored occupations** (no GenAI match) are kept in the per-year employment
  denominator (pooled `NONE` cell) but excluded from the exposure analysis.
- **Sample** = ages 25–55, non-institutional; **no** metro restriction (national).
- **Composition/wages** are measured among the employed in each occupation.
- **Treatment** is fixed at baseline (2019); DiD `post = YEAR >= 2023`, event-study
  reference year 2022 — matching `EventStudy 0627/00_config.R`.

## Dependencies

`pandas`, `numpy` (steps 0–3); `linearmodels` (step 4, for `PanelOLS`);
`matplotlib` (optional, figures); `scipy` is pulled in by `linearmodels`.

> Note: the project `.venv` was created on another machine (Python 3.14, OneDrive
> path) and does not run here. Recreate a venv with a local interpreter, or install
> the deps into your working Python, before running.
