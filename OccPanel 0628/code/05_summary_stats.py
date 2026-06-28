"""
05_summary_stats.py

Descriptive statistics for the occupation x year panel used in the DiD/event-study
analysis. Produces the "typical research" tables:

  1. Sample construction   how many occupations / rows survive each filter
  2. Coverage by year       occupations, employment, mean exposure per year
  3. Descriptive stats       N / mean / sd / min / median / max for every outcome
  4. Baseline balance        high- vs low-exposure terciles at the baseline year,
                             with the raw gap (the pre-treatment comparison a DiD
                             reader expects to see)

Group definitions mirror 03_exposure_groups.py and 04_did.py exactly:
  - extreme terciles  expo_tercile (baseline-employment-weighted)
  - matched extremes  top-N most-exposed vs N zero-exposure occupations
  - continuous dose   z_expo

Inputs:
  ../../Data/Processed/occ_panel_00.csv   (all cells incl. unscored NONE bucket)
  ../../Data/Processed/occ_panel_01.csv   (scored occupations + treatment vars)

Outputs (all under ../output/sumstats/):
  ../output/sumstats/sumstats_coverage.csv
  ../output/sumstats/sumstats_descriptives.csv
  ../output/sumstats/sumstats_balance.csv
  ../output/sumstats/summary_stats.tex     (LaTeX fragment, \\input into main.tex)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT.parent / "Data" / "Processed"
OUT  = ROOT / "output" / "sumstats"
OUT.mkdir(parents=True, exist_ok=True)

YEARS     = list(range(2017, 2025))   # analysis window (matches 04_did.py)
BASE_YEAR = 2021                      # exposure / weights fixed here (matches 03)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
panel00 = pd.read_csv(DATA / "occ_panel_00.csv")   # everything, incl. NONE bucket
panel   = pd.read_csv(DATA / "occ_panel_01.csv")   # scored occupations + treatment

panel00 = panel00[panel00["YEAR"].isin(YEARS)]
panel   = panel[panel["YEAR"].isin(YEARS)].copy()

# Balanced analysis sample: occupations observed in every analysis year (the set
# the event study actually runs on).
yr_counts = panel.groupby("occ")["YEAR"].nunique()
balanced  = yr_counts[yr_counts == len(YEARS)].index
bal       = panel[panel["occ"].isin(balanced)].copy()

# Derived rates used in the tables
for df in (bal,):
    df["unemp_rate"] = np.where(df["in_lf"] > 0, df["unemp"] / df["in_lf"], np.nan)

# ---------------------------------------------------------------------------
# 1. Sample construction
# ---------------------------------------------------------------------------
n_all_cells   = panel00["occ"].nunique()
n_scored      = panel["occ"].nunique()
n_balanced    = bal["occ"].nunique()

# matched extremes (count-matched top-N vs zero-exposure), as in 04_did.py
occ_expo  = bal.drop_duplicates("occ")[["occ", "expo_base"]]
zero_occs = set(occ_expo.loc[occ_expo["expo_base"] == 0, "occ"])
N_MATCH   = len(zero_occs)
top_occs  = set(occ_expo.sort_values("expo_base", ascending=False)
                        .head(N_MATCH)["occ"])

print("=" * 64)
print("SAMPLE CONSTRUCTION")
print("=" * 64)
print(f"  occupation cells (incl. unscored NONE)   : {n_all_cells}")
print(f"  scored occupations (has_exposure=1)      : {n_scored}")
print(f"  balanced over {len(YEARS)} years ({YEARS[0]}-{YEARS[-1]})    : {n_balanced}")
print(f"  -> panel rows (occ x year)               : {len(bal)}")
print(f"  zero-exposure occupations                : {N_MATCH}")
print(f"  matched extremes: top {N_MATCH} vs {N_MATCH} zero-exp.")

# ---------------------------------------------------------------------------
# 2. Coverage by year
# ---------------------------------------------------------------------------
cov = bal.groupby("YEAR").agg(
    n_occ        =("occ",         "nunique"),
    employment   =("employed",    "sum"),
    mean_exposure=("ai_exposure", "mean"),
    mean_rent    =("mean_rent",   "mean"),
    mean_wage    =("mean_wage",   "mean"),
).reset_index()
cov.to_csv(OUT / "sumstats_coverage.csv", index=False)
print("\nCOVERAGE BY YEAR")
print(cov.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Descriptive statistics (over occupation-year observations, balanced sample)
# ---------------------------------------------------------------------------
# (label, column) in the order they appear in the table
VARS = [
    ("AI exposure score",         "ai_exposure"),
    ("Employment (persons)",      "employed"),
    ("Employment share",          "emp_share"),
    ("Unemployment rate",         "unemp_rate"),
    ("Mean wage (\\$)",           "mean_wage"),
    ("Mean hours/week",           "mean_uhrs"),
    ("Mean rent (\\$/mo)",        "mean_rent"),
    ("Renter share",              "renter_share"),
    ("Share with BA+",            "share_ba"),
    ("Share female",              "share_female"),
    ("Mean age",                  "mean_age"),
    ("Work-from-home share",      "share_wfh"),
    ("Migration rate",            "migr_rate"),
]

def describe(df, col):
    s = df[col].dropna()
    return dict(N=int(s.size), mean=s.mean(), sd=s.std(),
                vmin=s.min(), med=s.median(), vmax=s.max())

desc = pd.DataFrame(
    [dict(variable=lbl, **describe(bal, col)) for lbl, col in VARS]
)
desc.to_csv(OUT / "sumstats_descriptives.csv", index=False)
print("\nDESCRIPTIVE STATISTICS (balanced occupation-year sample)")
print(desc.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Baseline balance: high vs low exposure tercile at BASE_YEAR
# ---------------------------------------------------------------------------
base = bal[bal["YEAR"] == BASE_YEAR]
hi   = base[base["expo_tercile"] == "high"]
lo   = base[base["expo_tercile"] == "low"]

bal_rows = []
for lbl, col in VARS:
    if col == "ai_exposure":
        continue  # exposure is the splitting variable, not a balance check
    h, l = hi[col].dropna(), lo[col].dropna()
    bal_rows.append(dict(variable=lbl, low=l.mean(), high=h.mean(),
                         diff=h.mean() - l.mean()))
balance = pd.DataFrame(bal_rows)
balance.to_csv(OUT / "sumstats_balance.csv", index=False)
print(f"\nBASELINE BALANCE ({BASE_YEAR}): high vs low exposure tercile")
print(f"  n_occ  low={lo['occ'].nunique()}  high={hi['occ'].nunique()}")
print(balance.to_string(index=False))

# ---------------------------------------------------------------------------
# LaTeX fragment (descriptives + balance), ready to \input into main.tex
# ---------------------------------------------------------------------------
def fmt(x, col):
    """Human-readable formatting: counts with thousands sep, shares as decimals.
    Negatives are wrapped so LaTeX typesets a proper minus sign, not a hyphen."""
    if pd.isna(x):
        return "--"
    if col in ("employed", "mean_wage", "mean_rent"):
        s = f"{abs(x):,.0f}"
    elif col in ("mean_uhrs", "mean_age"):
        s = f"{abs(x):.1f}"
    else:
        s = f"{abs(x):.3f}"
    return f"$-${s}" if x < 0 else s

col_by_lbl = {lbl: col for lbl, col in VARS}

lines = []
lines.append("% Auto-generated by 05_summary_stats.py -- do not edit by hand.")
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Summary statistics for the balanced occupation$\times$year panel, "
             f"{YEARS[0]}--{YEARS[-1]}.}}")
lines.append(r"\label{tab:sumstats}")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\toprule")
lines.append(r"Variable & N & Mean & SD & Median & Max \\")
lines.append(r"\midrule")
for _, r in desc.iterrows():
    col = col_by_lbl[r["variable"]]
    lines.append(f"{r['variable']} & {int(r['N'])} & {fmt(r['mean'], col)} & "
                 f"{fmt(r['sd'], col)} & {fmt(r['med'], col)} & {fmt(r['vmax'], col)} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"")
lines.append(r"\vspace{4pt}")
lines.append(r"{\footnotesize One observation per occupation$\times$year over the "
             f"{n_balanced} occupations present in all {len(YEARS)} years. "
             r"AI exposure is fixed at its " f"{BASE_YEAR}" r" value.}")
lines.append(r"\end{table}")
lines.append(r"")

# Baseline balance table
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Baseline (" f"{BASE_YEAR}" r") characteristics of the most- vs "
             r"least-exposed occupation terciles.}")
lines.append(r"\label{tab:balance}")
lines.append(r"\begin{tabular}{lrrr}")
lines.append(r"\toprule")
lines.append(r"Variable & Low exposure & High exposure & Difference \\")
lines.append(r"\midrule")
for _, r in balance.iterrows():
    col = col_by_lbl[r["variable"]]
    lines.append(f"{r['variable']} & {fmt(r['low'], col)} & {fmt(r['high'], col)} & "
                 f"{fmt(r['diff'], col)} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"")
lines.append(r"\vspace{4pt}")
lines.append(r"{\footnotesize Group means across occupations in each tercile, "
             f"baseline year {BASE_YEAR}. "
             f"Low: {lo['occ'].nunique()} occupations; high: {hi['occ'].nunique()}.}}")
lines.append(r"\end{table}")

tex_path = OUT / "summary_stats.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote LaTeX fragment: {tex_path}")
print(f"Wrote CSVs to: {OUT}")
