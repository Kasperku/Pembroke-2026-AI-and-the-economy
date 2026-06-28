"""
04_did.py

Difference-in-differences around the ChatGPT release (Nov 2022), asking whether
high-exposed occupations evolved differently from low-exposed ones. Three co-equal
designs:

  Extreme-group : top vs bottom baseline-exposure tercile; did = high x post
  Matched       : top-N most-exposed vs the N zero-exposure occupations that
                  survive the balanced panel (count-matched); did = high_m x post
  Continuous    : z-scored baseline exposure; did = z_expo x post

Both use occupation fixed effects + year fixed effects, are weighted by baseline
(2019) employment, and cluster standard errors by occupation. For each design a
pooled DiD (post = YEAR >= 2023) and an event study (year interactions, reference
year 2022, matching EventStudy 0627/00_config.R) are estimated.

Outcomes: log_mean_rent, log_mean_wage, and three job-loss measures (log_emp,
emp_share, unemp_rate). The job-loss outcomes test whether the rent decline is
just people losing their jobs; they are estimated as outcomes, NOT used as
controls (employment is post-treatment, so controlling for it is a bad control).

The rent spec is estimated three ways: a base version, a +renter_share version,
and a +log_emp version (the wage/job-loss specs are base only). Both controls are
composition-robustness checks, not headline estimates:
  +renter_share guards against a tenure artifact: if exposed occupations' renters
  converted to homeowners, the *remaining* renter pool changes and mean rent could
  fall mechanically. Conditioning on renter_share holds the rent/own mix fixed.
  +log_emp guards against a job-loss artifact: rent is already measured among the
  employed (valid_rent = RENT>0 & emp), so holding the occupation's employment
  count fixed checks that the rent fall isn't driven by who stays employed.
Both controls are themselves plausibly affected by exposure (bad controls), so
read the controlled estimates as robustness checks, not causal headlines.

A WFH-mechanism test for rent (does the exposure rent decline just reflect the
WFH/donut effect?) is run at the end, alongside the +control robustness specs.

Input:
  ../../Data/Processed/occ_panel_01.csv     (from 03_exposure_groups.py)

Output (all under ../output/ — results no longer written to Data/Processed):
  ../output/tables/occ_did_pooled.csv             (pooled DiD coefficient table)
  ../output/tables/occ_did_eventstudy.csv         (event-study coefficients)
  ../output/figures/eventstudy_<outcome>.png      (event-study plots, if matplotlib)
  ../output/tables/occ_rent_wfh_test.csv          (WFH-mechanism test for rent)
"""

import pandas as pd
from pathlib import Path
from linearmodels.panel import PanelOLS

ROOT   = Path(__file__).parent.parent
DATA   = ROOT.parent / "Data" / "Processed"
OUTPUT = ROOT / "output"
FIGS   = OUTPUT / "figures"   # event-study plots (.png)
TABLES = OUTPUT / "tables"    # coefficient / result tables (.csv)

YEARS    = [2020, 2017, 2018, 2019, 2021, 2022, 2023, 2024]   # years considered in analysis
REF_YEAR = 2022   # event-study reference (last pre-ChatGPT year)
POST_FROM = 2023  # ChatGPT effect first observable in 2023 ACS

# Outcomes, at the occupation level: housing rent and wages.
# (share_wfh is built in 00 but NOT used as an outcome: its variation is the
#  2020-21 COVID reshuffle, which dwarfs any ChatGPT signal and destroys parallel
#  trends. It is retained in the panel for use as a baseline WFH-propensity
#  stratifier/control instead — see the WFH-mechanism test at the end.)
OUTCOMES = ["log_mean_rent", "log_mean_rent_retired",
            "log_mean_wage", "log_emp", "emp_share", "unemp_rate"]

# Composition-robustness controls, per outcome. The rent spec is run several ways
# so one figure shows whether the rent effect survives each control and all of them
# together. Each control is itself plausibly affected by exposure (a "bad control"),
# so these are robustness checks that the result PERSISTS, not headline estimates:
#   +renter_share  holds the rent/own mix fixed (tenure-composition artifact)
#   +wage          holds occupation wages fixed (is rent falling beyond income?)
#   +log_emp       holds the employed headcount fixed (job-loss/composition artifact)
#   +all           all three at once
# Non-rent outcomes get the base spec only.
# WFH is deliberately NOT a control here: baseline WFH correlates ~0.6 with exposure,
# so adding did_w (=z_wfh x post) inflates the SEs via collinearity. The WFH confound
# is handled cleanly by the stratified WFH-mechanism test at the end of the file.
SPECS_BY_OUTCOME = {
    "log_mean_rent": [("base", []),
                      ("+renter_share", ["renter_share"]),
                      ("+wage",         ["log_mean_wage"]),
                      ("+log_emp",      ["log_emp"]),
                      ("+all",          ["renter_share", "log_mean_wage", "log_emp"])],
    "log_mean_rent_retired": [("base", [])],  # placebo: rent of retirees (NILF 60+) from this occ
    "log_mean_wage": [("base", [])],
    # Job-loss check: estimated as their own outcomes (NOT as controls — employment
    # is a post-treatment consequence of exposure, so conditioning on it would be a
    # bad control). If these are flat post-release, job loss isn't driving the rent fall.
    "log_emp":       [("base", [])],
    "emp_share":     [("base", [])],
    "unemp_rate":    [("base", [])],
}

# Economic-event lag (years): the gap between the ACS survey year and when the
# measured event actually happened.  RENT is the current contract rent at
# interview (contemporaneous, lag 0).  Everything downstream is aligned on
# effective_year = YEAR - lag, so post = effective_year >= 2023 and the
# event-study reference is effective_year = 2022.
LAG = {
    "log_mean_rent": 0,
    "log_mean_rent_retired": 0,
    "log_mean_wage": 0,
    "log_emp":       0,   # employment status is measured at the survey (contemporaneous)
    "emp_share":     0,
    "unemp_rate":    0,
    "share_wfh":     0,
}

# ---------------------------------------------------------------------------
# Load, restrict to the window, and balance (occupations observed every year)
# ---------------------------------------------------------------------------
panel = pd.read_csv(DATA / "occ_panel_01.csv")
# Unemployment rate within the occupation (not precomputed in the panel): unemployed
# over labour-force participants. NaN when no one is in the labour force (dropped by fit).
panel["unemp_rate"] = (panel["unemp"] / panel["in_lf"]).where(panel["in_lf"] > 0)
panel = panel[panel["YEAR"].isin(YEARS)].copy()
yr_counts = panel.groupby("occ")["YEAR"].nunique()
balanced  = yr_counts[yr_counts == len(YEARS)].index
panel = panel[panel["occ"].isin(balanced)].copy()
panel = panel[panel["base_emp"] > 0]   # need a baseline weight

panel["high"]   = (panel["expo_tercile"] == "high").astype(int)

# ---------------------------------------------------------------------------
# Count-matched extreme design: the zero-exposure occupations that survive the
# balanced panel vs an EQUAL NUMBER of the most-exposed occupations. A symmetric,
# like-for-like alternative to the tercile split (which lumps every zero-exposure
# occupation into a much larger "low" bin). N is derived from the data, not
# hard-coded, so it tracks the panel: N = #(surviving zero-exposure occupations).
#
# NOTE on balancing: groups are matched on the NUMBER OF OCCUPATIONS, not on the
# number of workers. The DiD is weighted by 2019 employment, so the two groups
# can still carry very different total employment (47 exposed office occupations
# vs 47 zero-exposure manual ones). This is an occupation-count match, read it as
# such; an employment-matched control would be a separate, more involved design.
occ_expo  = panel.drop_duplicates("occ")[["occ", "expo_base"]]
zero_occs = set(occ_expo.loc[occ_expo["expo_base"] == 0, "occ"])
N_EXTREME = len(zero_occs)
top_occs  = set(occ_expo.sort_values("expo_base", ascending=False)
                        .head(N_EXTREME)["occ"])
panel["high_m"]      = panel["occ"].isin(top_occs).astype(int)   # 1=top-N, 0=zero-exposure
panel["matched_grp"] = panel["occ"].isin(zero_occs | top_occs)   # membership in the matched sample


def with_treatment(df, lag):
    """Add lag-aware treatment columns: effective_year, post, did_x/_xm/_z/_w."""
    d = df.copy()
    d["eff_year"] = d["YEAR"] - lag
    d["post"]   = (d["eff_year"] >= POST_FROM).astype(int)
    d["did_x"]  = d["high"]   * d["post"]      # extreme-group (tercile) treatment
    if "high_m" in d.columns:                  # count-matched extreme treatment
        d["did_xm"] = d["high_m"] * d["post"]  #   (absent in the WFH-test panel, which
                                               #    doesn't use the matched design)
    d["did_z"]  = d["z_expo"] * d["post"]      # continuous-dose treatment
    d["did_w"]  = d["z_wfh"]  * d["post"]      # WFH-propensity x post (mechanism control)
    return d

print(f"Balanced panel: {panel['occ'].nunique()} occupations x {len(YEARS)} years "
      f"= {len(panel)} rows")
print(f"  extreme-group sample (high+low terciles): "
      f"{panel[panel['expo_tercile'].isin(['high','low'])]['occ'].nunique()} occupations")
print(f"  count-matched extremes: top {N_EXTREME} exposed vs {N_EXTREME} zero-exposure "
      f"occupations\n")


def fit_panel(df, outcome, treat_cols):
    """PanelOLS with occ + year FE, weighted by base_emp, clustered by occ.

    Drops rows missing the outcome OR any regressor (renter_share can be NaN in
    cells with no housing observations)."""
    d = df.dropna(subset=[outcome] + list(treat_cols)).copy()
    d = d.set_index(["occ", "YEAR"])
    mod = PanelOLS(d[outcome], d[treat_cols], weights=d["base_emp"],
                   entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def event_terms(df, treat):
    """Build effective-year interaction columns (ref year omitted); needs eff_year."""
    cols = []
    for y in sorted(df["eff_year"].unique()):
        if y == REF_YEAR:
            continue
        c = f"{treat}_{int(y)}"
        df[c] = df[treat] * (df["eff_year"] == y).astype(int)
        cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Pooled DiD (both designs, both specs, all outcomes)
# ---------------------------------------------------------------------------
extreme   = panel[panel["expo_tercile"].isin(["high", "low"])].copy()
extreme_m = panel[panel["matched_grp"]].copy()   # count-matched top-N vs zero-exposure


def pooled_did(extreme_df, matched_df, panel_df):
    """Pooled tercile (extreme), count-matched (matched), and per-SD (dose) DiD."""
    rows = []
    for outcome in OUTCOMES:
        lag = LAG[outcome]
        ex = with_treatment(extreme_df, lag)
        em = with_treatment(matched_df, lag)
        pa = with_treatment(panel_df,   lag)
        for label, ctrl in SPECS_BY_OUTCOME[outcome]:
            try:
                rx = fit_panel(ex, outcome, ["did_x"]  + ctrl)
                rm = fit_panel(em, outcome, ["did_xm"] + ctrl)
                rz = fit_panel(pa, outcome, ["did_z"]  + ctrl)
                rows.append(dict(
                    outcome=outcome, spec=label, lag=lag,
                    extreme_coef=rx.params["did_x"],  extreme_se=rx.std_errors["did_x"],
                    extreme_p=rx.pvalues["did_x"],
                    matched_coef=rm.params["did_xm"], matched_se=rm.std_errors["did_xm"],
                    matched_p=rm.pvalues["did_xm"],
                    dose_coef=rz.params["did_z"],     dose_se=rz.std_errors["did_z"],
                    dose_p=rz.pvalues["did_z"],
                ))
            except Exception as e:   # sparse outcome (e.g. unemployed-only rent) -> warn, keep going
                print(f"  [skip pooled] {outcome}/{label}: {e}")
                nan = float("nan")
                rows.append(dict(
                    outcome=outcome, spec=label, lag=lag,
                    extreme_coef=nan, extreme_se=nan, extreme_p=nan,
                    matched_coef=nan, matched_se=nan, matched_p=nan,
                    dose_coef=nan,    dose_se=nan,    dose_p=nan,
                ))
    return pd.DataFrame(rows)


def print_pooled(df, title):
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'outcome':<15}{'spec':<14}{'tercile hi-lo':>20}"
          f"{'matched top-bot':>22}{'continuous dose':>22}")
    for _, r in df.iterrows():
        xs = "*" if r["extreme_p"] < .05 else " "
        ms = "*" if r["matched_p"] < .05 else " "
        zs = "*" if r["dose_p"]    < .05 else " "
        print(f"{r['outcome']:<15}{r['spec']:<14}"
              f"{r['extreme_coef']:>11.4f} [{r['extreme_se']:.4f}]{xs}"
              f"{r['matched_coef']:>11.4f} [{r['matched_se']:.4f}]{ms}"
              f"{r['dose_coef']:>11.4f} [{r['dose_se']:.4f}]{zs}")
    print("=" * 100)


pooled = pooled_did(extreme, extreme_m, panel)
pooled["window"] = f"{YEARS[0]}-{YEARS[-1]}"
TABLES.mkdir(parents=True, exist_ok=True)
pooled.to_csv(TABLES / "occ_did_pooled.csv", index=False)

print_pooled(pooled, f"POOLED DiD  {YEARS[0]}-{YEARS[-1]}  (coef [se], * p<.05)  "
                     "high-vs-low ChatGPT effect, occ+year FE, base-emp wt")

# ---------------------------------------------------------------------------
# Event study (both designs, both specs, all outcomes)
# ---------------------------------------------------------------------------
es_rows = []
es_curves = {}   # (outcome, design, spec) -> DataFrame(year, coef, lo, hi)
for outcome in OUTCOMES:
    lag = LAG[outcome]
    for design, df0, treat in [("extreme", extreme,   "high"),
                               ("matched", extreme_m, "high_m"),
                               ("dose",    panel,     "z_expo")]:
        df = with_treatment(df0, lag)
        cols = event_terms(df, treat)
        for label, ctrl in SPECS_BY_OUTCOME[outcome]:
            try:
                res = fit_panel(df, outcome, cols + ctrl)
                recs = []
                for c in cols:
                    y = int(c.split("_")[-1])
                    recs.append(dict(year=y, coef=res.params[c], se=res.std_errors[c]))
            except Exception as e:   # sparse outcome -> warn and skip this curve
                print(f"  [skip event-study] {outcome}/{design}/{label}: {e}")
                continue
            recs.append(dict(year=REF_YEAR, coef=0.0, se=0.0))   # reference normalized to 0
            curve = pd.DataFrame(recs).sort_values("year").reset_index(drop=True)
            curve["lo"] = curve["coef"] - 1.96 * curve["se"]
            curve["hi"] = curve["coef"] + 1.96 * curve["se"]
            es_curves[(outcome, design, label)] = curve
            for _, rr in curve.iterrows():
                es_rows.append(dict(outcome=outcome, design=design, spec=label,
                                    lag=lag, **rr.to_dict()))

es_df = pd.DataFrame(es_rows)
es_df.to_csv(TABLES / "occ_did_eventstudy.csv", index=False)
print(f"\nEvent-study coefficients saved: {TABLES / 'occ_did_eventstudy.csv'}")

# ---------------------------------------------------------------------------
# Event-study figures: one per outcome, all control specs overlaid on the same axes
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    styles = {"base":          dict(marker="o", capsize=2, ls="-",  markersize=5),
              "+renter_share": dict(marker="s", capsize=2, ls="--", markersize=4),
              "+wage":         dict(marker="D", capsize=2, ls=":",  markersize=4),
              "+log_emp":      dict(marker="^", capsize=2, ls="-.", markersize=4),
              "+all":          dict(marker="v", capsize=2, ls="-",  markersize=4)}
    for outcome in OUTCOMES:
        if not any((outcome, d, l) in es_curves
                   for d in ("extreme", "dose") for l, _ in SPECS_BY_OUTCOME[outcome]):
            print(f"  [skip figure] {outcome}: no curves fit (too sparse)")
            continue
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharex=True)
        for ax, design, title in [
                (axes[0], "extreme", "Extreme group (high vs low tercile)"),
                (axes[1], "matched", f"Matched extremes (top {N_EXTREME} vs {N_EXTREME} zero-exp.)"),
                (axes[2], "dose",    "Continuous dose (per 1 SD)")]:
            ax.axhline(0, color="grey", lw=.8)
            ax.axvline(REF_YEAR, ls=":", color="grey", lw=.8)
            for label, _ in SPECS_BY_OUTCOME[outcome]:
                c = es_curves.get((outcome, design, label))
                if c is None:
                    continue
                ax.errorbar(c["year"], c["coef"], yerr=1.96 * c["se"],
                            label=label, elinewidth=0.8, alpha=0.85, **styles[label])
            xlab = "Event year (move year)" if LAG[outcome] else "Year"
            ax.set_title(title); ax.set_xlabel(xlab)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        axes[0].set_ylabel(f"{outcome}  (rel. to {REF_YEAR})")
        fig.suptitle(f"Event study: {outcome}  (lag {LAG[outcome]}y)")
        fig.tight_layout()
        fig.savefig(FIGS / f"eventstudy_{outcome}.png", dpi=150)
        plt.close(fig)
    print(f"Event-study figures saved to {FIGS}")
except ImportError:
    print("(matplotlib not available - skipped figures)")

# ===========================================================================
# WFH-mechanism test for rent
# ===========================================================================
# Is the AI-exposure rent decline just the WFH / donut effect?
#   (a) Stratify by baseline (2021) WFH propensity and re-estimate the exposure
#       rent DiD within the LOW-WFH and HIGH-WFH strata. If the rent effect is
#       WFH-driven it should vanish where WFH can't operate (low-WFH stratum).
#   (b) Control: add z_wfh x post next to z_expo x post in the pooled dose DiD.
#       If the exposure coefficient survives, rent is not merely WFH-ability.
# z_expo and z_wfh correlate ~0.6, so the control spec (b) has inflated SEs by
# construction (collinearity) — read the stratified test (a) as the more robust one.
WFH_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]  # one extra pre-year vs the main window; 2020 dropped
WFH_RENT  = "log_mean_rent"
WFH_LAG   = LAG[WFH_RENT]   # rent is contemporaneous (lag 0)

# Re-load with the WFH window and re-balance (occupations observed every year here).
wfh_panel = pd.read_csv(DATA / "occ_panel_01.csv")
wfh_panel = wfh_panel[wfh_panel["YEAR"].isin(WFH_YEARS)].copy()
_wyrc = wfh_panel.groupby("occ")["YEAR"].nunique()
wfh_panel = wfh_panel[wfh_panel["occ"].isin(_wyrc[_wyrc == len(WFH_YEARS)].index)].copy()
wfh_panel = wfh_panel[wfh_panel["base_emp"] > 0]
wfh_panel["high"] = (wfh_panel["expo_tercile"] == "high").astype(int)

NA = dict(coef=float("nan"), se=float("nan"), p=float("nan"))


def _coef(res, name):
    return dict(coef=res.params[name], se=res.std_errors[name], p=res.pvalues[name])


def _safe(df, outcome, name):
    """Fit and pull one coefficient; return NaNs if the cell is degenerate."""
    try:
        return _coef(fit_panel(df, outcome, [name]), name)
    except Exception:
        return dict(NA)


rent_rows = []

# (a) Stratified exposure DiD within WFH terciles
for stratum in ["low", "high"]:
    sub = wfh_panel[wfh_panel["wfh_tercile"] == stratum]
    sx  = with_treatment(sub[sub["expo_tercile"].isin(["high", "low"])], WFH_LAG)
    sd_ = with_treatment(sub, WFH_LAG)
    rent_rows.append(dict(
        spec=f"stratum WFH={stratum}", n_occ=sd_["occ"].nunique(),
        extreme=_safe(sx, WFH_RENT, "did_x"), dose=_safe(sd_, WFH_RENT, "did_z")))

# (b) Pooled dose DiD, exposure alone vs exposure + WFH control
d_all  = with_treatment(wfh_panel, WFH_LAG)
r_base = fit_panel(d_all, WFH_RENT, ["did_z"])
r_ctrl = fit_panel(d_all, WFH_RENT, ["did_z", "did_w"])

print("\n" + "=" * 78)
print("WFH-MECHANISM TEST for rent  (log_mean_rent, lag 0)")
print("=" * 78)
print("(a) exposure rent DiD within WFH strata  (coef [se], * p<.05)")
for r in rent_rows:
    xs = "*" if r["extreme"]["p"] < .05 else " "
    zs = "*" if r["dose"]["p"]    < .05 else " "
    print(f"   {r['spec']:<16} n_occ={r['n_occ']:>3}   "
          f"extreme {r['extreme']['coef']:>8.4f} [{r['extreme']['se']:.4f}]{xs}   "
          f"dose {r['dose']['coef']:>8.4f} [{r['dose']['se']:.4f}]{zs}")
print("\n(b) pooled dose DiD, exposure alone vs + WFH-propensity control")
print(f"   exposure only      did_z = {r_base.params['did_z']:>8.4f} "
      f"[{r_base.std_errors['did_z']:.4f}]  p={r_base.pvalues['did_z']:.3f}")
print(f"   + z_wfh x post      did_z = {r_ctrl.params['did_z']:>8.4f} "
      f"[{r_ctrl.std_errors['did_z']:.4f}]  p={r_ctrl.pvalues['did_z']:.3f}")
print(f"                       did_w = {r_ctrl.params['did_w']:>8.4f} "
      f"[{r_ctrl.std_errors['did_w']:.4f}]  p={r_ctrl.pvalues['did_w']:.3f}")
print("=" * 78)

# Tidy CSV of the same numbers
wfh_out = []
for r in rent_rows:
    for design in ["extreme", "dose"]:
        wfh_out.append(dict(spec=r["spec"], n_occ=r["n_occ"], design=design, **r[design]))
wfh_out += [
    dict(spec="pooled exposure-only", n_occ=d_all["occ"].nunique(), design="dose",
         **_coef(r_base, "did_z")),
    dict(spec="pooled +wfh-control",  n_occ=d_all["occ"].nunique(), design="dose_z_expo",
         **_coef(r_ctrl, "did_z")),
    dict(spec="pooled +wfh-control",  n_occ=d_all["occ"].nunique(), design="dose_z_wfh",
         **_coef(r_ctrl, "did_w")),
]
pd.DataFrame(wfh_out).to_csv(TABLES / "occ_rent_wfh_test.csv", index=False)
print(f"WFH-mechanism test saved: {TABLES / 'occ_rent_wfh_test.csv'}")
