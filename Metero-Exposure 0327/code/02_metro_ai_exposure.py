"""
Step 2: Compute weighted-average AI exposure per metro area (MET2013),
using PERWT (person weight) as the weight. Also reports coverage
statistics per metro.

Input:  data/processed/usa_00002_with_felten.csv  (from 01_merge_felten.py)
Output: data/processed/metro_ai_exposure.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "processed"

df = pd.read_csv(OUT / "usa_00002_with_felten.csv")

# Compute coverage per metro BEFORE dropping NAs
df["has_score"] = df["ai_exposure"].notna()
coverage = df.groupby("MET2013").agg(
    perwt_total_all=("PERWT", "sum"),
    perwt_with_score=pd.NamedAgg(column="PERWT", aggfunc=lambda x: x[df.loc[x.index, "has_score"]].sum()),
    n_workers_all=("PERWT", "count"),
    n_with_score=("has_score", "sum"),
).reset_index()
coverage["perwt_coverage_pct"] = (coverage["perwt_with_score"] / coverage["perwt_total_all"] * 100).round(2)
coverage["n_coverage_pct"] = (coverage["n_with_score"] / coverage["n_workers_all"] * 100).round(2)

# Drop rows without an AI exposure score
df = df[df["has_score"]].copy()

# Weighted average: sum(PERWT * ai_exposure) / sum(PERWT) per metro
df["weighted_ai"] = df["PERWT"] * df["ai_exposure"]

metro = df.groupby("MET2013").agg(
    ai_exposure_wtd_avg=("weighted_ai", "sum"),
    perwt_with_score=("PERWT", "sum"),
    n_workers=("PERWT", "count"),
).reset_index()

metro["ai_exposure_wtd_avg"] = metro["ai_exposure_wtd_avg"] / metro["perwt_with_score"]

# Merge coverage info
metro = metro.merge(coverage[["MET2013", "perwt_total_all", "perwt_coverage_pct", "n_workers_all", "n_coverage_pct"]], on="MET2013")

metro = metro.sort_values("ai_exposure_wtd_avg", ascending=False).reset_index(drop=True)

out_path = OUT / "metro_ai_exposure.csv"
metro.to_csv(out_path, index=False)

print(f"Metro areas: {len(metro)}")
print(f"\nOverall PERWT coverage: {coverage['perwt_with_score'].sum() / coverage['perwt_total_all'].sum() * 100:.1f}%")
print(f"\nLowest coverage metros:")
worst = metro.nsmallest(10, "perwt_coverage_pct")[["MET2013", "perwt_coverage_pct", "n_coverage_pct", "n_workers_all", "ai_exposure_wtd_avg"]]
print(worst.to_string(index=False))
print(f"\nCoverage distribution:")
for threshold in [99, 95, 90, 85, 80]:
    count = (metro["perwt_coverage_pct"] >= threshold).sum()
    print(f"  >= {threshold}%: {count} metros ({count/len(metro)*100:.1f}%)")
print(f"\nSaved to {out_path}")
