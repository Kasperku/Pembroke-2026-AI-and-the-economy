"""
02.1_compare_soc_coverage.py  [diagnostic]

Reports which SOC occupation codes are missing GenAI exposure scores and how
many workers (PERWT-weighted) are affected. Ranked from most- to least-missing.

Input:
  ../../Data/Processed/usa_00004_with_genai.csv   (from 01_merge_genai.py)
  ../../Data/crosswalks/2018soc.xlsx              (BLS SOC titles)

Output:
  ../../Data/Processed/soc_coverage_comparison.csv
"""

import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).parent.parent
DATA   = ROOT.parent / "Data"
GENAI  = DATA / "Processed" / "usa_00004_with_genai.csv"
XW_SOC = DATA / "crosswalks" / "2018soc.xlsx"
OUT    = DATA / "Sumstats" / "02.1_soc_coverage_comparison.csv"

CHUNK = 500_000

# ---------------------------------------------------------------------------
# SOC titles
# ---------------------------------------------------------------------------
soc_struct = pd.read_excel(XW_SOC, header=None, skiprows=7)
soc_struct.columns = ["group", "code", "title", "definition"]
soc_struct["code"] = soc_struct["code"].astype(str).str.strip()
title_map = soc_struct.dropna(subset=["code"]).set_index("code")["title"].to_dict()

# ---------------------------------------------------------------------------
# Scan GenAI file
# ---------------------------------------------------------------------------
print("Scanning GenAI file ...")
parts   = []
n_total = 0

for chunk in pd.read_csv(GENAI, usecols=["soc2018", "ai_exposure", "PERWT"], chunksize=CHUNK):
    n_total += len(chunk)
    chunk["missing"]      = chunk["ai_exposure"].isna()
    chunk["perwt_missing"] = chunk["PERWT"] * chunk["missing"].astype(int)
    parts.append(chunk.groupby("soc2018", as_index=False).agg(
        perwt_total  = ("PERWT",         "sum"),
        perwt_missing= ("perwt_missing", "sum"),
        n_total      = ("PERWT",         "count"),
        n_missing    = ("missing",       "sum"),
    ))
    print(f"  {n_total:,} rows ...", end="\r")

print(f"\n  Done. {n_total:,} rows total.")

df = pd.concat(parts).groupby("soc2018", as_index=False).sum()
df["pct_missing"] = (df["perwt_missing"] / df["perwt_total"] * 100).round(1)
df["occ_title"]   = df["soc2018"].map(title_map).fillna("(title not found)")

# ---------------------------------------------------------------------------
# Filter to codes with any missing workers and rank
# ---------------------------------------------------------------------------
missing = (
    df[df["perwt_missing"] > 0]
    .sort_values("perwt_missing", ascending=False)
    .reset_index(drop=True)
)
missing.index += 1

cols = ["soc2018", "occ_title", "perwt_missing", "n_missing", "perwt_total", "pct_missing"]

total_miss  = missing["perwt_missing"].sum()
total_perwt = df["perwt_total"].sum()

print(f"\n{'='*70}")
print(f"SOC codes with missing GenAI score: {len(missing)}")
print(f"Missing worker-weight: {total_miss:,.0f} / {total_perwt:,.0f} ({total_miss/total_perwt*100:.1f}%)")
print(f"\n{'─'*70}")
print("TOP 20 — most workers missing a score:")
print(missing[cols].head(20).to_string())
print(f"\n{'─'*70}")
print("BOTTOM 20 — fewest workers missing a score:")
print(missing[cols].tail(20).to_string())

missing.to_csv(OUT, index=True, index_label="rank")
print(f"\nSaved to {OUT}")
