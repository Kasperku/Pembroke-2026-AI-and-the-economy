"""
03_exposure_groups.py

Assigns each occupation a fixed (time-invariant) GenAI-exposure treatment, in two
forms used as co-equal designs downstream (04_did.py):

  expo_tercile   baseline-employment-weighted terciles of ai_exposure
                 (labels low/mid/high) -> extreme-group DiD compares high vs low
  z_expo         z-scored ai_exposure across occupations -> continuous-dose DiD

Baseline employment weight is 2019 employment (predetermined, pre-ChatGPT), the
same baseline year the metro event study uses (EventStudy 0627/00_config.R).

Sanity check: the high-tercile cutpoint should land near the metro panel's
HIGH_EXP_OCC = 0.361 cutoff, which is the 2014-2019 PERWT-weighted 
top-tercile of individual ai_exposure.

Input:
  ../../Data/Processed/occ_panel_00.csv     (from 02_build_occ_panel.py)

Output:
  ../../Data/Processed/occ_panel_01.csv      (scored occupations + treatment vars)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT.parent / "Data" / "Processed"

BASE_YEAR     = 2021
WFH_BASE_YEAR = 2021   # revealed teleworkability: post-COVID, pre-ChatGPT WFH share
METRO_CUTOFF  = 0.361  # high-exposure cutoff from the metro panel, for sanity comparison


def weighted_quantile(values, quantiles, weights):
    """Weighted quantiles (values need not be sorted)."""
    order  = np.argsort(values)
    v, w   = np.asarray(values)[order], np.asarray(weights)[order]
    cum    = np.cumsum(w) - 0.5 * w
    cum   /= np.sum(w)
    return np.interp(quantiles, cum, v)


panel = pd.read_csv(DATA / "occ_panel_00.csv")

# Keep only scored occupations (drop the pooled NONE bucket / NaN exposure).
panel = panel[panel["has_exposure"] == 1].copy()

# ── year-over-year growth (computed on the full 2014-2024 series, so a 2019 row
#    still gets its 2018 lag; the analysis window is applied later in 03) ───────
panel = panel.sort_values(["occ", "YEAR"])
panel["emp_growth"]  = panel.groupby("occ")["log_emp"].diff()    # Δ log employment
panel["d_emp_share"] = panel.groupby("occ")["emp_share"].diff()  # Δ employment share
yr_gap = panel.groupby("occ")["YEAR"].diff()
panel.loc[yr_gap != 1, ["emp_growth", "d_emp_share"]] = np.nan   # null across missing-year gaps

# ── time-invariant occupation attributes ─────────────────────────────────────
# ai_exposure is constant within an occupation; take its first value to be safe.
expo = panel.groupby("occ", as_index=False)["ai_exposure"].first().rename(
    columns={"ai_exposure": "expo_base"})

base_emp = (
    panel[panel["YEAR"] == BASE_YEAR][["occ", "employed"]]
    .rename(columns={"employed": "base_emp"})
)
occ_attr = expo.merge(base_emp, on="occ", how="left")
# occupations absent in the baseline year get zero weight (excluded from terciles)
occ_attr["base_emp"] = occ_attr["base_emp"].fillna(0)

# ── baseline-employment-weighted terciles ────────────────────────────────────
mask = occ_attr["base_emp"] > 0
q33, q67 = weighted_quantile(
    occ_attr.loc[mask, "expo_base"], [1/3, 2/3], occ_attr.loc[mask, "base_emp"])

occ_attr["expo_tercile"] = pd.cut(
    occ_attr["expo_base"], bins=[-np.inf, q33, q67, np.inf],
    labels=["low", "mid", "high"])

# ── z-scored continuous dose ─────────────────────────────────────────────────
mu, sd = occ_attr["expo_base"].mean(), occ_attr["expo_base"].std()
occ_attr["z_expo"] = (occ_attr["expo_base"] - mu) / sd

# ── baseline WFH propensity (revealed teleworkability) ───────────────────────
# Each occupation's 2021 WFH share: a fixed, predetermined remote-ability index
# (post-COVID peak, before ChatGPT). Used in 03 to test whether the rent result
# is just the WFH/donut effect — as a stratifier (wfh_tercile) and continuous
# control (z_wfh). NOT a time-varying outcome (its variation is the COVID shock).
wfh_base = (
    panel[panel["YEAR"] == WFH_BASE_YEAR][["occ", "share_wfh"]]
    .rename(columns={"share_wfh": "wfh_base"})
)
occ_attr = occ_attr.merge(wfh_base, on="occ", how="left")

wmask = mask & occ_attr["wfh_base"].notna()
wq33, wq67 = weighted_quantile(
    occ_attr.loc[wmask, "wfh_base"], [1/3, 2/3], occ_attr.loc[wmask, "base_emp"])
occ_attr["wfh_tercile"] = pd.cut(
    occ_attr["wfh_base"], bins=[-np.inf, wq33, wq67, np.inf],
    labels=["low", "mid", "high"])

wmu, wsd = occ_attr["wfh_base"].mean(), occ_attr["wfh_base"].std()
occ_attr["z_wfh"] = (occ_attr["wfh_base"] - wmu) / wsd

# ── merge attributes back onto the panel ─────────────────────────────────────
out = panel.merge(
    occ_attr[["occ", "expo_base", "base_emp", "expo_tercile", "z_expo",
              "wfh_base", "wfh_tercile", "z_wfh"]],
    on="occ", how="left")
out.to_csv(DATA / "occ_panel_01.csv", index=False)

# ── diagnostics ──────────────────────────────────────────────────────────────
print(f"Scored occupation panel saved: {out['occ'].nunique()} occ x "
      f"{out['YEAR'].nunique()} years = {len(out)} rows")
print(f"Tercile cutpoints (base-emp weighted): low|mid = {q33:.3f}, mid|high = {q67:.3f}")
print(f"  metro panel high cutoff for comparison: {METRO_CUTOFF:.3f} "
      f"({'within' if q33 <= METRO_CUTOFF <= q67 or q67 <= METRO_CUTOFF else 'near'} the high band)")
grp = occ_attr[mask].groupby("expo_tercile", observed=True).agg(
    n_occ=("occ", "nunique"), base_emp=("base_emp", "sum"),
    mean_expo=("expo_base", "mean"))
print("\nTercile composition (baseline year employment):")
print(grp.to_string())

# Exposure x WFH cross-tab: surfaces the support/collinearity problem for the
# stratified rent test (high-exposure occupations cluster in high-WFH cells).
print(f"\nBaseline WFH (share_wfh @ {WFH_BASE_YEAR}) terciles: "
      f"low|mid = {wq33:.3f}, mid|high = {wq67:.3f}")
ct = occ_attr[wmask]
xt_n   = pd.crosstab(ct["expo_tercile"], ct["wfh_tercile"])
xt_emp = pd.crosstab(ct["expo_tercile"], ct["wfh_tercile"],
                     values=ct["base_emp"], aggfunc="sum")
print("\nExposure tercile (rows) x WFH tercile (cols) — occupation counts:")
print(xt_n.to_string())
print("\n  ... weighted by baseline employment:")
print(xt_emp.to_string())
print(f"\ncor(z_expo, z_wfh) across occupations = "
      f"{occ_attr[['z_expo','z_wfh']].corr().iloc[0,1]:.3f}")
