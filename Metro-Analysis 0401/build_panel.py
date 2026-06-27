"""
Step 2: Collapse enriched microdata into a metro-year panel.

Input:  Data/Processed/usa_00004_merged.csv  (from parse_and_merge.py)
Output: Data/Processed/metro_year_panel.csv

Re-run this whenever you change year range, age restrictions,
or outcome variable definitions — without re-running parse_and_merge.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DIR    = Path(__file__).parent
DATA_DIR = Path("C:/Users/user/Desktop/Honours Thesis/Data")
TEMP_DIR = DATA_DIR / "temp"
INPUT  = TEMP_DIR / "usa_00004_merged.csv"
OUTPUT = TEMP_DIR / "metro_year_panel.csv"

# All available years
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
CHUNK = 500_000
GRP   = ["MET2013", "YEAR"]

# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------
pop_parts   = []
inmig_parts = []
outmig_parts = []
ai_rows  = {}
hpi_rows = {}  # (MET2013, YEAR) -> hpi_annual

print("Reading and filtering chunks ...")
n_total = 0
n_kept  = 0

reader = pd.read_csv(INPUT, chunksize=CHUNK)
for chunk in reader:
    n_total += len(chunk)

    # --- Filters ---
    chunk = chunk[chunk["YEAR"].isin(YEARS)]
    chunk = chunk[chunk["MET2013"] != 0]
    chunk = chunk[~chunk["GQ"].isin([3, 4])]        # drop institutional group quarters
    chunk = chunk[(chunk["AGE"] >= 25) & (chunk["AGE"] <= 55)]  # working-age adults
    n_kept += len(chunk)

    if len(chunk) == 0:
        continue

    # --- Derived columns ---
    chunk["w"]          = chunk["PERWT"]
    chunk["ba_plus"]    = (chunk["EDUCD"] >= 101).astype(int)
    chunk["metro_mover"] = (
        chunk["MIGRATE1"].isin([2, 3, 4])
        & (chunk["MIGMET131"] != 0)
        & (chunk["MIGMET131"] != chunk["MET2013"])
    )
    chunk["valid_wage"] = (chunk["INCWAGE"] > 0) & (chunk["INCWAGE"] < 999999)

    # --- Population stats ---
    chunk["w_ba"]       = chunk["w"] * chunk["ba_plus"]
    chunk["w_female_lf"] = chunk["w"] * ((chunk["SEX"] == 2) & (chunk["LABFORCE"] == 2)).astype(int)
    chunk["w_in_lf"]    = chunk["w"] * (chunk["LABFORCE"] == 2).astype(int)
    chunk["w_unemp"]    = chunk["w"] * (chunk["EMPSTAT"] == 2).astype(int)
    chunk["w_age"]      = chunk["w"] * chunk["AGE"]
    chunk["w_wage"]     = np.where(chunk["valid_wage"], chunk["w"] * chunk["INCWAGE"], 0)
    chunk["w_wage_n"]   = np.where(chunk["valid_wage"], chunk["w"], 0)
    chunk["w_ba_all"]   = chunk["w"] * chunk["ba_plus"]

    pop_agg = chunk.groupby(GRP, as_index=False).agg(
        pop        =("w",          "sum"),
        pop_ba     =("w_ba",       "sum"),
        female_lf  =("w_female_lf","sum"),
        in_lf      =("w_in_lf",    "sum"),
        unemployed =("w_unemp",    "sum"),
        age_wtd    =("w_age",      "sum"),
        wage_wtd   =("w_wage",     "sum"),
        wage_n     =("w_wage_n",   "sum"),
        ba_wtd_all =("w_ba_all",   "sum"),
    )
    pop_parts.append(pop_agg)

    # --- In-migrants: moved INTO metro m from a different metro ---
    movers = chunk[chunk["metro_mover"]].copy()
    if len(movers) > 0:
        movers["w_ba_mig"]    = movers["w"] * movers["ba_plus"]
        movers["w_no_ba_mig"] = movers["w"] * (1 - movers["ba_plus"])
        movers["w_young"]     = movers["w"] * ((movers["AGE"] >= 25) & (movers["AGE"] <= 35)).astype(int)
        movers["w_old"]       = movers["w"] * ((movers["AGE"] >= 45) & (movers["AGE"] <= 55)).astype(int)

        inmig_agg = movers.groupby(GRP, as_index=False).agg(
            inmig       =("w",          "sum"),
            inmig_ba    =("w_ba_mig",   "sum"),
            inmig_no_ba =("w_no_ba_mig","sum"),
            inmig_young =("w_young",    "sum"),
            inmig_old   =("w_old",      "sum"),
        )
        inmig_parts.append(inmig_agg)

        # --- Out-migrants: these same movers LEFT their origin metro ---
        movers["w_ba_out"]    = movers["w"] * movers["ba_plus"]
        movers["w_no_ba_out"] = movers["w"] * (1 - movers["ba_plus"])

        outmig_agg = movers.groupby(["MIGMET131", "YEAR"], as_index=False).agg(
            outmig       =("w",           "sum"),
            outmig_ba    =("w_ba_out",    "sum"),
            outmig_no_ba =("w_no_ba_out", "sum"),
        )
        outmig_agg.rename(columns={"MIGMET131": "MET2013"}, inplace=True)
        outmig_parts.append(outmig_agg)

    # --- AI exposure (constant per metro, grab first valid) ---
    if "ai_exposure_wtd_avg" in chunk.columns:
        valid = chunk[chunk["ai_exposure_wtd_avg"].notna()][["MET2013", "ai_exposure_wtd_avg"]].drop_duplicates("MET2013")
        for _, r in valid.iterrows():
            m = int(r["MET2013"])
            if m not in ai_rows:
                ai_rows[m] = r["ai_exposure_wtd_avg"]

    # --- HPI (per metro-year, grab first valid) ---
    if "hpi_annual" in chunk.columns:
        chunk["_hpi_num"] = pd.to_numeric(chunk["hpi_annual"], errors="coerce")
        valid_hpi = chunk[chunk["_hpi_num"].notna()][["MET2013", "YEAR", "_hpi_num"]].drop_duplicates(GRP)
        for _, r in valid_hpi.iterrows():
            key = (int(r["MET2013"]), int(r["YEAR"]))
            if key not in hpi_rows:
                hpi_rows[key] = r["_hpi_num"]

    print(f"  {n_total:,} read, {n_kept:,} kept ...", end="\r")

print(f"\n  Done reading. {n_total:,} total, {n_kept:,} after filters.")

# ---------------------------------------------------------------------------
# Combine chunks
# ---------------------------------------------------------------------------
print("Aggregating ...")
pop_df = pd.concat(pop_parts).groupby(GRP, as_index=False).sum()

inmig_df = (
    pd.concat(inmig_parts).groupby(GRP, as_index=False).sum()
    if inmig_parts else pd.DataFrame(columns=GRP)
)
outmig_df = (
    pd.concat(outmig_parts).groupby(GRP, as_index=False).sum()
    if outmig_parts else pd.DataFrame(columns=GRP)
)

panel = pop_df.merge(inmig_df,  on=GRP, how="left")
panel = panel.merge(outmig_df, on=GRP, how="left")

for col in ["inmig", "inmig_ba", "inmig_no_ba", "inmig_young", "inmig_old",
            "outmig", "outmig_ba", "outmig_no_ba"]:
    if col in panel.columns:
        panel[col] = panel[col].fillna(0)

# ---------------------------------------------------------------------------
# Balanced panel: keep only metros present in all years
# ---------------------------------------------------------------------------
metro_counts   = panel.groupby("MET2013")["YEAR"].nunique()
balanced_metros = metro_counts[metro_counts == len(YEARS)].index
panel = panel[panel["MET2013"].isin(balanced_metros)].copy()
print(f"Balanced panel: {len(balanced_metros)} metros × {len(YEARS)} years = {len(panel)} cells")

# ---------------------------------------------------------------------------
# Compute outcome and control variables
# ---------------------------------------------------------------------------
p = panel

# Outcomes
p["inmig_rate"]       = p["inmig"]    / p["pop"]
p["inmig_rate_ba"]    = p["inmig_ba"] / p["pop"]
p["inmig_rate_no_ba"] = p["inmig_no_ba"] / p["pop"]
p["skilled_inmig_share"] = np.where(p["inmig"] > 0, p["inmig_ba"] / p["inmig"], np.nan)
p["net_mig_ba_rate"]    = (p["inmig_ba"]    - p["outmig_ba"])    / p["pop"]
p["net_mig_no_ba_rate"] = (p["inmig_no_ba"] - p["outmig_no_ba"]) / p["pop"]

# Share BA: incumbents (non-movers)
inc_ba  = p["ba_wtd_all"] - p["inmig_ba"]
inc_pop = p["pop"] - p["inmig"]
p["incumbent_share_ba"] = np.where(inc_pop > 0, inc_ba / inc_pop, np.nan)

p["inmig_rate_young"] = p["inmig_young"] / p["pop"]
p["inmig_rate_old"]   = p["inmig_old"]   / p["pop"]

# Controls
p["share_ba"]        = p["pop_ba"] / p["pop"]
p["log_pop"]         = np.log(p["pop"])
p["mean_age"]        = p["age_wtd"] / p["pop"]
p["share_female_lf"] = np.where(p["in_lf"] > 0, p["female_lf"] / p["in_lf"], np.nan)
p["unemp_rate"]      = np.where(p["in_lf"] > 0, p["unemployed"] / p["in_lf"], np.nan)
p["mean_wage"]       = np.where(p["wage_n"] > 0, p["wage_wtd"] / p["wage_n"], np.nan)

# AI exposure
ai_df = pd.DataFrame(list(ai_rows.items()), columns=["MET2013", "ai_exposure"])
p = p.merge(ai_df, on="MET2013", how="left")

# HPI (collected during first pass)
hpi_df = pd.DataFrame(
    [(m, y, v) for (m, y), v in hpi_rows.items()],
    columns=["MET2013", "YEAR", "hpi_annual"],
)
p = p.merge(hpi_df, on=GRP, how="left")

# ---------------------------------------------------------------------------
# Select and rename final columns
# ---------------------------------------------------------------------------
final_cols = [
    "MET2013", "YEAR", "ai_exposure", "hpi_annual",
    # Outcomes
    "pop", "inmig_rate", "inmig_rate_ba", "inmig_rate_no_ba",
    "skilled_inmig_share", "net_mig_ba_rate", "net_mig_no_ba_rate",
    "incumbent_share_ba", "inmig_rate_young", "inmig_rate_old",
    # Controls
    "share_ba", "log_pop", "mean_age", "share_female_lf", "unemp_rate", "mean_wage",
]
p = p[[c for c in final_cols if c in p.columns]]
p.rename(columns={"MET2013": "met2013", "YEAR": "year"}, inplace=True)
p.sort_values(["met2013", "year"], inplace=True)
p.to_csv(OUTPUT, index=False)

print(f"\nPanel saved to {OUTPUT}")
print(f"Shape: {p.shape}")
print(f"Metros with AI exposure:    {p['ai_exposure'].notna().sum() // len(YEARS)}")
print(f"Metros missing AI exposure: {p['ai_exposure'].isna().sum() // len(YEARS)}")
print(f"\nSample:")
print(p.head(5).to_string())
