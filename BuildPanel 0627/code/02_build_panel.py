"""
02_build_panel.py

Collapses ACS microdata into a balanced metro x year panel covering migration,
labor market, housing, commuting, demographics, and individual-level AI exposure.

Inputs:
  ../../Data/Processed/usa_00004_with_genai.csv  (IPUMS microdata + AI exposure, from 01)

Output:
  ../../Data/Processed/panel_00.csv               (one row per metro x year)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT  = Path(__file__).parent.parent
DATA  = ROOT.parent / "Data" / "Processed"
YEARS = list(range(2014, 2025))
CHUNK = 500_000
GRP   = ["MET2013", "YEAR"]

# IPUMS missing-value sentinels
OWNCOST_TOP  = 99999    # OWNCOST >= 99999 → N/A
VALUEH_TOP   = 9999999  # VALUEH >= 9999999 → N/A
INCWAGE_TOP  = 999999   # INCWAGE >= 999999 → N/A
INCBUS_TOP   = 9999999  # INCBUS00 >= 9999999 → N/A

# TRANWORK codes for public transit (bus, streetcar, subway, elevated/light rail, railroad, ferry)
TRANSIT_CODES = {30, 31, 32, 33, 34, 35, 36}

# ---------------------------------------------------------------------------
# Stream microdata and accumulate weighted aggregates by metro x year
# ---------------------------------------------------------------------------
pop_parts, inmig_parts, outmig_parts = [], [], []

for i, chunk in enumerate(pd.read_csv(DATA / "usa_00004_with_genai.csv", chunksize=CHUNK)):
    chunk = chunk[chunk["YEAR"].isin(YEARS)]
    chunk = chunk[chunk["MET2013"] != 0]
    chunk = chunk[~chunk["GQ"].isin([3, 4])]       # drop institutional group quarters
    chunk = chunk[(chunk["AGE"] >= 25) & (chunk["AGE"] <= 55)]
    if chunk.empty:
        continue

    # --- binary indicators ---
    chunk["ba"]          = (chunk["EDUCD"] >= 101).astype(int)
    chunk["in_lf"]       = (chunk["LABFORCE"] == 2).astype(int)
    chunk["employed"]    = (chunk["EMPSTAT"] == 1).astype(int)
    chunk["unemp"]       = (chunk["EMPSTAT"] == 2).astype(int)
    chunk["female_lf"]   = ((chunk["SEX"] == 2) & (chunk["LABFORCE"] == 2)).astype(int)
    chunk["in_school"]   = (chunk["SCHOOL"] == 2).astype(int)
    chunk["white"]       = (chunk["RACE"] == 1).astype(int)
    chunk["black"]       = (chunk["RACE"] == 2).astype(int)
    chunk["pub_transit"] = chunk["TRANWORK"].isin(TRANSIT_CODES).astype(int)

    # valid-observation masks (exclude coded missings)
    chunk["valid_wage"]  = (chunk["INCWAGE"]  > 0) & (chunk["INCWAGE"]  < INCWAGE_TOP)
    chunk["valid_bus"]   = (chunk["INCBUS00"] > 0) & (chunk["INCBUS00"] < INCBUS_TOP)
    chunk["valid_rent"]  = chunk["RENT"] > 0          # RENT == 0 → owner-occupied
    chunk["valid_own"]   = (chunk["OWNCOST"] > 0) & (chunk["OWNCOST"] < OWNCOST_TOP)
    chunk["valid_val"]   = (chunk["VALUEH"]  > 0) & (chunk["VALUEH"]  < VALUEH_TOP)
    chunk["valid_tran"]  = (chunk["TRANTIME"] > 0) & ~chunk["TRANWORK"].isin({0, 70})  # not N/A or WFH
    chunk["valid_hrs"]   = (chunk["UHRSWORK"] > 0) & (chunk["employed"] == 1)
    chunk["valid_ai"]    = chunk["ai_exposure"].notna()

    chunk["metro_mover"] = (
        chunk["MIGRATE1"].isin([2, 3, 4])
        & (chunk["MIGMET131"] != 0)
        & (chunk["MIGMET131"] != chunk["MET2013"])
    )

    w = chunk["PERWT"]

    # --- weighted sums ---
    chunk["w_ba"]        = w * chunk["ba"]
    chunk["w_in_lf"]     = w * chunk["in_lf"]
    chunk["w_employed"]  = w * chunk["employed"]
    chunk["w_unemp"]     = w * chunk["unemp"]
    chunk["w_female_lf"] = w * chunk["female_lf"]
    chunk["w_age"]       = w * chunk["AGE"]
    chunk["w_school"]    = w * chunk["in_school"]
    chunk["w_white"]     = w * chunk["white"]
    chunk["w_black"]     = w * chunk["black"]
    chunk["w_pub_tran"]  = w * chunk["pub_transit"]
    chunk["w_wage"]      = w * chunk["INCWAGE"].where(chunk["valid_wage"], 0)
    chunk["w_wage_n"]    = w.where(chunk["valid_wage"], 0)
    chunk["w_bus"]       = w * chunk["INCBUS00"].where(chunk["valid_bus"], 0)
    chunk["w_bus_n"]     = w.where(chunk["valid_bus"], 0)
    chunk["w_rent"]      = w * chunk["RENT"].where(chunk["valid_rent"], 0)
    chunk["w_rent_n"]    = w.where(chunk["valid_rent"], 0)
    chunk["w_own"]       = w * chunk["OWNCOST"].where(chunk["valid_own"], 0)
    chunk["w_own_n"]     = w.where(chunk["valid_own"], 0)
    chunk["w_val"]       = w * chunk["VALUEH"].where(chunk["valid_val"], 0)
    chunk["w_val_n"]     = w.where(chunk["valid_val"], 0)
    chunk["w_tran"]      = w * chunk["TRANTIME"].where(chunk["valid_tran"], 0)
    chunk["w_tran_n"]    = w.where(chunk["valid_tran"], 0)
    chunk["w_hrs"]       = w * chunk["UHRSWORK"].where(chunk["valid_hrs"], 0)
    chunk["w_hrs_n"]     = w.where(chunk["valid_hrs"], 0)
    chunk["w_ai"]        = w * chunk["ai_exposure"].where(chunk["valid_ai"], 0)
    chunk["w_ai_n"]      = w.where(chunk["valid_ai"], 0)

    pop_parts.append(chunk.groupby(GRP, as_index=False).agg(
        pop         = ("PERWT",       "sum"),
        pop_ba      = ("w_ba",        "sum"),
        in_lf       = ("w_in_lf",     "sum"),
        employed    = ("w_employed",  "sum"),
        unemp       = ("w_unemp",     "sum"),
        female_lf   = ("w_female_lf", "sum"),
        age_sum     = ("w_age",       "sum"),
        school_sum  = ("w_school",    "sum"),
        white_sum   = ("w_white",     "sum"),
        black_sum   = ("w_black",     "sum"),
        pub_tran    = ("w_pub_tran",  "sum"),
        wage_sum    = ("w_wage",      "sum"),
        wage_n      = ("w_wage_n",    "sum"),
        bus_sum     = ("w_bus",       "sum"),
        bus_n       = ("w_bus_n",     "sum"),
        rent_sum    = ("w_rent",      "sum"),
        rent_n      = ("w_rent_n",    "sum"),
        own_sum     = ("w_own",       "sum"),
        own_n       = ("w_own_n",     "sum"),
        val_sum     = ("w_val",       "sum"),
        val_n       = ("w_val_n",     "sum"),
        tran_sum    = ("w_tran",      "sum"),
        tran_n      = ("w_tran_n",    "sum"),
        hrs_sum     = ("w_hrs",       "sum"),
        hrs_n       = ("w_hrs_n",     "sum"),
        ai_sum      = ("w_ai",        "sum"),
        ai_n        = ("w_ai_n",      "sum"),
    ))

    movers = chunk[chunk["metro_mover"]].copy()
    if not movers.empty:
        wm = movers["PERWT"]
        movers["w_ba"]    = wm * movers["ba"]
        movers["w_no_ba"] = wm * (1 - movers["ba"])
        movers["w_young"] = wm * ((movers["AGE"] >= 25) & (movers["AGE"] <= 35)).astype(int)
        movers["w_old"]   = wm * ((movers["AGE"] >= 45) & (movers["AGE"] <= 55)).astype(int)

        inmig_parts.append(movers.groupby(GRP, as_index=False).agg(
            inmig       = ("PERWT",   "sum"),
            inmig_ba    = ("w_ba",    "sum"),
            inmig_no_ba = ("w_no_ba", "sum"),
            inmig_young = ("w_young", "sum"),
            inmig_old   = ("w_old",   "sum"),
        ))
        outmig_parts.append(movers.groupby(["MIGMET131", "YEAR"], as_index=False).agg(
            outmig       = ("PERWT",   "sum"),
            outmig_ba    = ("w_ba",    "sum"),
            outmig_no_ba = ("w_no_ba", "sum"),
        ).rename(columns={"MIGMET131": "MET2013"}))

    print(f"  chunk {i+1} done ...", end="\r")

print()

# ---------------------------------------------------------------------------
# Combine chunks
# ---------------------------------------------------------------------------
pop    = pd.concat(pop_parts).groupby(GRP, as_index=False).sum()
inmig  = pd.concat(inmig_parts).groupby(GRP, as_index=False).sum()  if inmig_parts  else pd.DataFrame(columns=GRP)
outmig = pd.concat(outmig_parts).groupby(GRP, as_index=False).sum() if outmig_parts else pd.DataFrame(columns=GRP)

panel = pop.merge(inmig, on=GRP, how="left").merge(outmig, on=GRP, how="left")
for col in ["inmig", "inmig_ba", "inmig_no_ba", "inmig_young", "inmig_old",
            "outmig", "outmig_ba", "outmig_no_ba"]:
    panel[col] = panel[col].fillna(0)

# Retain only metros observed in every year (balanced panel)
year_counts = panel.groupby("MET2013")["YEAR"].nunique()
panel = panel[panel["MET2013"].isin(year_counts[year_counts == len(YEARS)].index)].copy()

# ---------------------------------------------------------------------------
# Compute outcomes and controls
# ---------------------------------------------------------------------------
p = panel

# Migration
p["inmig_rate"]          = p["inmig"]       / p["pop"]
p["inmig_rate_ba"]       = p["inmig_ba"]    / p["pop"]
p["inmig_rate_no_ba"]    = p["inmig_no_ba"] / p["pop"]
p["skilled_inmig_share"] = np.where(p["inmig"] > 0, p["inmig_ba"] / p["inmig"], np.nan)
p["net_mig_ba_rate"]     = (p["inmig_ba"]    - p["outmig_ba"])    / p["pop"]
p["net_mig_no_ba_rate"]  = (p["inmig_no_ba"] - p["outmig_no_ba"]) / p["pop"]
inc_ba  = p["pop_ba"] - p["inmig_ba"]
inc_pop = p["pop"]    - p["inmig"]
p["incumbent_share_ba"]  = np.where(inc_pop > 0, inc_ba / inc_pop, np.nan)
p["inmig_rate_young"]    = p["inmig_young"] / p["pop"]
p["inmig_rate_old"]      = p["inmig_old"]   / p["pop"]

# Labor market
p["lfp_rate"]        = p["in_lf"]     / p["pop"]
p["unemp_rate"]      = np.where(p["in_lf"] > 0, p["unemp"]     / p["in_lf"],  np.nan)
p["emp_rate"]        = p["employed"]  / p["pop"]
p["share_female_lf"] = np.where(p["in_lf"] > 0, p["female_lf"] / p["in_lf"],  np.nan)
p["mean_wage"]       = np.where(p["wage_n"] > 0, p["wage_sum"]  / p["wage_n"], np.nan)
p["log_mean_wage"]   = np.log(p["mean_wage"])
p["mean_bus_inc"]    = np.where(p["bus_n"]  > 0, p["bus_sum"]   / p["bus_n"],  np.nan)
p["mean_uhrs"]       = np.where(p["hrs_n"]  > 0, p["hrs_sum"]   / p["hrs_n"],  np.nan)

# Demographics
p["share_ba"]        = p["pop_ba"]     / p["pop"]
p["log_pop"]         = np.log(p["pop"])
p["mean_age"]        = p["age_sum"]    / p["pop"]
p["share_in_school"] = p["school_sum"] / p["pop"]
p["share_white"]     = p["white_sum"]  / p["pop"]
p["share_black"]     = p["black_sum"]  / p["pop"]

# Housing
p["mean_rent"]       = np.where(p["rent_n"] > 0, p["rent_sum"] / p["rent_n"], np.nan)
p["log_mean_rent"]   = np.log(p["mean_rent"])
p["mean_owncost"]    = np.where(p["own_n"]  > 0, p["own_sum"]  / p["own_n"],  np.nan)
p["mean_home_value"] = np.where(p["val_n"]  > 0, p["val_sum"]  / p["val_n"],  np.nan)
p["log_home_value"]  = np.log(p["mean_home_value"])
p["renter_share"]    = np.where(
    p["rent_n"] + p["own_n"] > 0,
    p["rent_n"] / (p["rent_n"] + p["own_n"]), np.nan
)

# Commuting
p["mean_commute"]   = np.where(p["tran_n"] > 0, p["tran_sum"] / p["tran_n"], np.nan)
p["share_pub_tran"] = np.where(p["in_lf"]  > 0, p["pub_tran"] / p["in_lf"],  np.nan)

# AI exposure (population-weighted mean of individual-level scores)
p["mean_ai_exposure"] = np.where(p["ai_n"] > 0, p["ai_sum"] / p["ai_n"], np.nan)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
final_cols = [
    "MET2013", "YEAR",
    # AI exposure
    "mean_ai_exposure",
    # Migration outcomes
    "inmig_rate", "inmig_rate_ba", "inmig_rate_no_ba",
    "skilled_inmig_share", "net_mig_ba_rate", "net_mig_no_ba_rate",
    "incumbent_share_ba", "inmig_rate_young", "inmig_rate_old",
    # Labor market
    "lfp_rate", "emp_rate", "unemp_rate", "share_female_lf",
    "mean_wage", "log_mean_wage", "mean_bus_inc", "mean_uhrs",
    # Demographics
    "share_ba", "log_pop", "mean_age", "share_in_school",
    "share_white", "share_black",
    # Housing
    "mean_rent", "log_mean_rent", "mean_owncost",
    "mean_home_value", "log_home_value", "renter_share",
    # Commuting
    "mean_commute", "share_pub_tran",
]

p[final_cols].sort_values(GRP).to_csv(DATA / "panel_00.csv", index=False)

n_metros = p["MET2013"].nunique()
n_ai     = p["mean_ai_exposure"].notna().sum() // len(YEARS)
print(f"Panel saved: {n_metros} metros x {len(YEARS)} years = {len(p)} rows")
print(f"AI exposure matched: {n_ai} / {n_metros} metros")
