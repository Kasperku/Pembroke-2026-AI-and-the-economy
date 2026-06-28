"""
02_build_occ_panel.py

Collapses ACS microdata into a national occupation x year panel: labor market
outcomes (employment, wages, hours, composition) and each occupation's share of
total employment. 

Occupation key is `soc2010_mapped` (from 01_merge_genai.py): the GenAI-matched
2010 SOC code(s). It is stable across the 2018 OCCSOC vintage break because the
matched-set -> string mapping is a bijection with the ai_exposure score, so the
same occupation carries the same key in every year. Rows with no GenAI match
(blank soc2010_mapped / NaN ai_exposure) are pooled into a single "NONE" cell
and KEPT, so the per-year employment-share denominator stays complete; they are
flagged has_exposure=0 and dropped by the downstream analysis.

Sample filters mirror the metro panel (ages 25-55, drop institutional group
quarters) EXCEPT we do NOT restrict to MET2013 != 0: this is a national
occupation analysis, so non-metro residents are kept for fuller occupation counts.

Inputs:
  ../../Data/Processed/usa_00004_with_genai.csv   (IPUMS microdata + AI exposure, from 01)

Output:
  ../../Data/Processed/occ_panel_00.csv            (one row per occupation x year)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT  = Path(__file__).parent.parent
DATA  = ROOT.parent / "Data" / "Processed"
YEARS = list(range(2014, 2025))
CHUNK = 500_000
GRP   = ["occ", "YEAR"]

# IPUMS missing-value sentinels (verified against Data/Raw/usa_00004.xml codebook)
INCWAGE_TOP = 999998   # INCWAGE: 999999=N/A, 999998=Missing -> drop both (strict <)
OWNCOST_TOP = 99999    # OWNCOST == 99999 -> not-in-universe

# TRANWORK public-transit codes (per codebook): 31 bus..37 train, 39 ferry.
# Excludes 38 taxi/ride-hail and 80 worked-at-home.
TRANSIT_CODES = {31, 32, 33, 34, 35, 36, 37, 39}

# Only the columns we need (the source file is ~3 GB; usecols keeps memory sane).
USECOLS = ["YEAR", "AGE", "GQ", "PERWT", "EMPSTAT", "LABFORCE", "EDUCD",
           "SEX", "UHRSWORK", "INCWAGE", "ai_exposure", "soc2010_mapped",
           "CLASSWKR", "RENT", "OWNCOST", "TRANWORK", "TRANTIME",
           "MIGRATE1", "MIGRATE1D"]

# ---------------------------------------------------------------------------
# Stream microdata and accumulate weighted aggregates by occupation x year
# ---------------------------------------------------------------------------
parts = []
ret_parts = []   # retiree-placebo rent aggregates (captured before the 25-55 filter)

for i, chunk in enumerate(pd.read_csv(DATA / "usa_00004_with_genai.csv",
                                      usecols=USECOLS, chunksize=CHUNK)):
    chunk = chunk[chunk["YEAR"].isin(YEARS)]
    chunk = chunk[~chunk["GQ"].isin([3, 4])]            # drop institutional group quarters

    # Retiree placebo: out of the labour force and 60+, captured BEFORE the 25-55
    # working-age filter below. Rent among cash-renters whose former occupation is X
    # -- people whose careers are over, so AI uncertainty can't touch them.
    ret = chunk[(chunk["LABFORCE"] == 1) & (chunk["AGE"] >= 60)].copy()
    if not ret.empty:
        ret["occ"] = ret["soc2010_mapped"].fillna("NONE")
        vr = ret["RENT"] > 0
        ret["w_rent_ret"]   = ret["PERWT"] * ret["RENT"].where(vr, 0)
        ret["w_rent_ret_n"] = ret["PERWT"].where(vr, 0)
        ret_parts.append(ret.groupby(GRP, as_index=False).agg(
            rent_ret_sum=("w_rent_ret",   "sum"),
            rent_ret_n  =("w_rent_ret_n", "sum"),
        ))

    chunk = chunk[(chunk["AGE"] >= 25) & (chunk["AGE"] <= 55)]
    if chunk.empty:
        continue

    # blank soc2010_mapped reads back from CSV as NaN -> pool into one "NONE" cell
    chunk["occ"] = chunk["soc2010_mapped"].fillna("NONE")

    # --- binary indicators ---
    chunk["employed"] = (chunk["EMPSTAT"] == 1).astype(int)
    chunk["in_lf"]    = (chunk["LABFORCE"] == 2).astype(int)
    chunk["unemp"]    = (chunk["EMPSTAT"] == 2).astype(int)
    # BA+ = bachelor's, master's, professional, doctoral. Explicit set, not a
    # range: legacy EDUCD codes 110-113 ("5-8 years of college") are unpopulated
    # in the 2014-2024 ACS (degree-coded), but they are NOT degrees, so a
    # between(101,116) range would silently miscount them if they ever appeared.
    chunk["ba"]       = chunk["EDUCD"].isin([101, 114, 115, 116]).astype(int)
    chunk["female"]   = (chunk["SEX"] == 2).astype(int)
    chunk["fulltime"] = ((chunk["UHRSWORK"] >= 35) & (chunk["employed"] == 1)).astype(int)
    chunk["self_emp"] = ((chunk["CLASSWKR"] == 1) & (chunk["employed"] == 1)).astype(int)  # self-employed
    chunk["pub_tran"] = chunk["TRANWORK"].isin(TRANSIT_CODES).astype(int)
    chunk["wfh"]      = (chunk["TRANWORK"] == 80).astype(int)                # worked at home (TRANWORK==80)  -> share_wfh below
    chunk["moved"]    = chunk["MIGRATE1"].isin([2, 3, 4]).astype(int)        # moved in last year
    # Interstate-mobility move types: total between-states (MIGRATE1==3), then the
    # MIGRATE1D contiguous/non-contiguous split (code 30 "different state general"
    # is empty in the ACS, so bstate ~= contig + noncontig).
    chunk["mv_bstate"]    = (chunk["MIGRATE1"]  == 3).astype(int)            # moved between states
    chunk["mv_contig"]    = (chunk["MIGRATE1D"] == 31).astype(int)           # between contiguous states
    chunk["mv_noncontig"] = (chunk["MIGRATE1D"] == 32).astype(int)          # between non-contiguous states

    emp = chunk["employed"] == 1   # composition/wages are measured among the employed in the occ

    # valid-observation masks (exclude coded missings); all conditioned on employed
    chunk["valid_wage"] = (chunk["INCWAGE"] > 0) & (chunk["INCWAGE"] < INCWAGE_TOP) & emp
    chunk["valid_hrs"]  = (chunk["UHRSWORK"] > 0) & emp
    chunk["valid_ai"]   = chunk["ai_exposure"].notna()
    chunk["valid_rent"] = (chunk["RENT"] > 0) & emp                          # RENT==0 -> owner-occupied
    chunk["valid_rent_all"] = chunk["RENT"] > 0                              # all cash-renters in the occ, regardless of employment
    chunk["valid_own"]  = (chunk["OWNCOST"] > 0) & (chunk["OWNCOST"] < OWNCOST_TOP) & emp
    chunk["valid_tran"] = ((chunk["TRANTIME"] > 0) & (chunk["TRANTIME"] != 888)
                           & ~chunk["TRANWORK"].isin([0, 80]) & emp)         # exclude N/A, suppressed, WFH
    chunk["commuter"]   = (chunk["TRANWORK"] != 0) & emp                     # reported journey-to-work
    chunk["valid_migr"] = chunk["MIGRATE1"].isin([1, 2, 3, 4]) & emp         # known migration status

    w = chunk["PERWT"]

    # --- weighted sums (composition numerators are conditioned on employed) ---
    chunk["w_emp"]        = w * chunk["employed"]
    chunk["w_in_lf"]      = w * chunk["in_lf"]
    chunk["w_unemp"]      = w * chunk["unemp"]
    chunk["w_emp_ba"]     = w * chunk["ba"]     * chunk["employed"]
    chunk["w_emp_female"] = w * chunk["female"] * chunk["employed"]
    chunk["w_fulltime"]   = w * chunk["fulltime"]
    chunk["w_self_emp"]   = w * chunk["self_emp"]
    chunk["w_emp_age"]    = w * chunk["AGE"]    * chunk["employed"]
    chunk["w_wage"]       = w * chunk["INCWAGE"].where(chunk["valid_wage"], 0)
    chunk["w_wage_n"]     = w.where(chunk["valid_wage"], 0)
    chunk["w_hrs"]        = w * chunk["UHRSWORK"].where(chunk["valid_hrs"], 0)
    chunk["w_hrs_n"]      = w.where(chunk["valid_hrs"], 0)
    chunk["w_ai"]         = w * chunk["ai_exposure"].where(chunk["valid_ai"], 0)
    chunk["w_ai_n"]       = w.where(chunk["valid_ai"], 0)
    # housing
    chunk["w_rent"]       = w * chunk["RENT"].where(chunk["valid_rent"], 0)
    chunk["w_rent_n"]     = w.where(chunk["valid_rent"], 0)
    chunk["w_rent_all"]   = w * chunk["RENT"].where(chunk["valid_rent_all"], 0)
    chunk["w_rent_all_n"] = w.where(chunk["valid_rent_all"], 0)
    chunk["w_own_n"]      = w.where(chunk["valid_own"], 0)
    # commuting
    chunk["w_tran"]       = w * chunk["TRANTIME"].where(chunk["valid_tran"], 0)
    chunk["w_tran_n"]     = w.where(chunk["valid_tran"], 0)
    chunk["w_pub_tran"]   = w * (chunk["pub_tran"] * chunk["commuter"])
    chunk["w_commuter"]   = w.where(chunk["commuter"], 0)
    chunk["w_wfh"]        = w * (chunk["wfh"] * chunk["employed"])           # WFH numerator; commuter_n is the denominator
    # mobility (moved in last year): overall any-move rate, plus interstate
    # move-types (total / contiguous / non-contiguous) split by education.
    chunk["w_moved"]      = w * (chunk["moved"] * chunk["valid_migr"])
    chunk["w_migr_n"]     = w.where(chunk["valid_migr"], 0)
    chunk["w_migr_ba_n"]  = w.where(chunk["valid_migr"] & (chunk["ba"] == 1), 0)
    chunk["w_migr_noba_n"] = w.where(chunk["valid_migr"] & (chunk["ba"] == 0), 0)
    for mv in ["bstate", "contig", "noncontig"]:
        m = chunk[f"mv_{mv}"] * chunk["valid_migr"]
        chunk[f"w_{mv}_ba"]   = w * (m * chunk["ba"])
        chunk[f"w_{mv}_noba"] = w * (m * (1 - chunk["ba"]))

    parts.append(chunk.groupby(GRP, as_index=False).agg(
        pop          = ("PERWT",        "sum"),
        employed     = ("w_emp",        "sum"),
        in_lf        = ("w_in_lf",      "sum"),
        unemp        = ("w_unemp",      "sum"),
        emp_ba       = ("w_emp_ba",     "sum"),
        emp_female   = ("w_emp_female", "sum"),
        fulltime_sum = ("w_fulltime",   "sum"),
        self_emp_sum = ("w_self_emp",   "sum"),
        emp_age_sum  = ("w_emp_age",    "sum"),
        wage_sum     = ("w_wage",       "sum"),
        wage_n       = ("w_wage_n",     "sum"),
        hrs_sum      = ("w_hrs",        "sum"),
        hrs_n        = ("w_hrs_n",      "sum"),
        ai_sum       = ("w_ai",         "sum"),
        ai_n         = ("w_ai_n",       "sum"),
        rent_sum     = ("w_rent",       "sum"),
        rent_n       = ("w_rent_n",     "sum"),
        rent_all_sum = ("w_rent_all",   "sum"),
        rent_all_n   = ("w_rent_all_n", "sum"),
        own_n        = ("w_own_n",      "sum"),
        tran_sum     = ("w_tran",       "sum"),
        tran_n       = ("w_tran_n",     "sum"),
        pub_tran_sum = ("w_pub_tran",   "sum"),
        commuter_n   = ("w_commuter",   "sum"),
        wfh_sum      = ("w_wfh",        "sum"),
        moved_sum    = ("w_moved",      "sum"),
        migr_n       = ("w_migr_n",     "sum"),
        migr_ba_n    = ("w_migr_ba_n",  "sum"),
        migr_noba_n  = ("w_migr_noba_n", "sum"),
        bstate_ba    = ("w_bstate_ba",    "sum"),
        contig_ba    = ("w_contig_ba",    "sum"),
        noncontig_ba = ("w_noncontig_ba", "sum"),
        bstate_noba  = ("w_bstate_noba",    "sum"),
        contig_noba  = ("w_contig_noba",    "sum"),
        noncontig_noba = ("w_noncontig_noba", "sum"),
    ))

    print(f"  chunk {i+1} done ...", end="\r")

print()

# ---------------------------------------------------------------------------
# Combine chunks
# ---------------------------------------------------------------------------
panel = pd.concat(parts).groupby(GRP, as_index=False).sum()

# Merge the retiree-placebo rent aggregates (separate older-age population)
if ret_parts:
    ret_panel = pd.concat(ret_parts).groupby(GRP, as_index=False).sum()
    panel = panel.merge(ret_panel, on=GRP, how="left")
else:
    panel["rent_ret_sum"] = 0.0
    panel["rent_ret_n"]   = 0.0

# Per-year total employment denominator (sums over ALL cells, incl. NONE bucket)
total_emp = panel.groupby("YEAR", as_index=False)["employed"].sum().rename(
    columns={"employed": "total_emp_year"})
panel = panel.merge(total_emp, on="YEAR", how="left")

# ---------------------------------------------------------------------------
# Compute outcomes
# ---------------------------------------------------------------------------
p = panel
p["ai_exposure"]    = np.where(p["ai_n"] > 0, p["ai_sum"] / p["ai_n"], np.nan)
p["has_exposure"]   = (p["ai_n"] > 0).astype(int)
p["emp_share"]      = p["employed"] / p["total_emp_year"]
p["log_emp"]        = np.where(p["employed"] > 0, np.log(p["employed"]), np.nan)
p["mean_wage"]      = np.where(p["wage_n"] > 0, p["wage_sum"] / p["wage_n"], np.nan)
p["log_mean_wage"]  = np.log(p["mean_wage"])
p["mean_uhrs"]      = np.where(p["hrs_n"]  > 0, p["hrs_sum"] / p["hrs_n"],  np.nan)
p["share_ba"]       = np.where(p["employed"] > 0, p["emp_ba"]     / p["employed"], np.nan)
p["share_female"]   = np.where(p["employed"] > 0, p["emp_female"] / p["employed"], np.nan)
p["share_fulltime"] = np.where(p["employed"] > 0, p["fulltime_sum"] / p["employed"], np.nan)
p["self_employed_share"] = np.where(p["employed"] > 0, p["self_emp_sum"] / p["employed"], np.nan)
p["mean_age"]       = np.where(p["employed"] > 0, p["emp_age_sum"] / p["employed"], np.nan)

# Housing (workers in the occupation)
p["mean_rent"]      = np.where(p["rent_n"] > 0, p["rent_sum"] / p["rent_n"], np.nan)
p["log_mean_rent"]  = np.log(p["mean_rent"])
# Same rent average but over ALL cash-renters whose (most-recent) occupation is this
# one, employed or not -- so it includes people not currently working. Compare to
# log_mean_rent (employed only).
p["mean_rent_all"]     = np.where(p["rent_all_n"] > 0, p["rent_all_sum"] / p["rent_all_n"], np.nan)
p["log_mean_rent_all"] = np.log(p["mean_rent_all"])
# Retiree placebo: rent among out-of-labour-force renters aged 60+ whose former
# occupation is this one. Should NOT fall after ChatGPT if the rent decline is
# about labour-market uncertainty (retirees face none).
p["mean_rent_retired"]     = np.where(p["rent_ret_n"] > 0, p["rent_ret_sum"] / p["rent_ret_n"], np.nan)
p["log_mean_rent_retired"] = np.log(p["mean_rent_retired"])
p["renter_share"]   = np.where(p["rent_n"] + p["own_n"] > 0,
                               p["rent_n"] / (p["rent_n"] + p["own_n"]), np.nan)

# Commuting (employed commuters in the occupation)
p["mean_commute"]   = np.where(p["tran_n"] > 0, p["tran_sum"] / p["tran_n"], np.nan)
p["share_pub_tran"] = np.where(p["commuter_n"] > 0, p["pub_tran_sum"] / p["commuter_n"], np.nan)
# WFH share: share of the occupation's workers (reported journey-to-work) who
# worked at home. Used in 03 as a baseline WFH-propensity index, not as an outcome.
p["share_wfh"]      = np.where(p["commuter_n"] > 0, p["wfh_sum"] / p["commuter_n"], np.nan)

# Mobility: overall any-move rate, plus interstate move-types by education.
# Denominator is the group's valid-migration weight (all workers with known
# migration status), so each rate is the share making that specific move.
p["migr_rate"] = np.where(p["migr_n"] > 0, p["moved_sum"] / p["migr_n"], np.nan)
for mv in ["bstate", "contig", "noncontig"]:
    p[f"migr_rate_{mv}_ba"]    = np.where(p["migr_ba_n"]   > 0, p[f"{mv}_ba"]   / p["migr_ba_n"],   np.nan)
    p[f"migr_rate_{mv}_no_ba"] = np.where(p["migr_noba_n"] > 0, p[f"{mv}_noba"] / p["migr_noba_n"], np.nan)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
final_cols = [
    "occ", "YEAR", "ai_exposure", "has_exposure",
    # employment / structure
    "employed", "emp_share", "log_emp", "pop", "in_lf", "unemp",
    # wages / hours
    "mean_wage", "log_mean_wage", "mean_uhrs",
    # composition (among the employed in the occupation)
    "share_ba", "share_female", "share_fulltime", "self_employed_share", "mean_age",
    # housing
    "mean_rent", "log_mean_rent", "mean_rent_all", "log_mean_rent_all",
    "mean_rent_retired", "log_mean_rent_retired", "renter_share",
    # commuting
    "mean_commute", "share_pub_tran", "share_wfh",
    # mobility (overall any-move, plus interstate move-types by education)
    "migr_rate",
    "migr_rate_bstate_ba", "migr_rate_contig_ba", "migr_rate_noncontig_ba",
    "migr_rate_bstate_no_ba", "migr_rate_contig_no_ba", "migr_rate_noncontig_no_ba",
]
p[final_cols].sort_values(GRP).to_csv(DATA / "occ_panel_00.csv", index=False)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
scored = p[p["has_exposure"] == 1]
share_sums = p.groupby("YEAR")["emp_share"].sum()
print(f"Occupation panel saved: {p['occ'].nunique()} occ x {p['YEAR'].nunique()} years = {len(p)} rows")
print(f"  scored occupations (has_exposure=1): {scored['occ'].nunique()}")
print(f"  unscored 'NONE' bucket employment share, 2019: "
      f"{p[(p['occ']=='NONE') & (p['YEAR']==2019)]['emp_share'].iloc[0]:.3f}")
print(f"  emp_share sums per year (should be ~1.0): "
      f"min={share_sums.min():.6f} max={share_sums.max():.6f}")

# Migration consistency: code 30 (different state, general) is empty in the ACS,
# so weighted between-states movers should equal contiguous + non-contiguous.
mv_tot   = (p["bstate_ba"] + p["bstate_noba"]).sum()
mv_split = (p["contig_ba"] + p["contig_noba"] + p["noncontig_ba"] + p["noncontig_noba"]).sum()
print(f"  interstate movers (weighted): bstate={mv_tot:,.0f}  contig+noncontig={mv_split:,.0f}  "
      f"(diff {mv_tot - mv_split:,.0f} = residual code-30)")
print(f"  share_ba range (sanity vs. between(101,116)): "
      f"{scored['share_ba'].min():.3f}-{scored['share_ba'].max():.3f}")
