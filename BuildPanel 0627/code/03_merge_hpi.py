"""
03_merge_hpi.py

Merges FHFA House Price Index onto the metro x year panel.
MSAD sub-division CBSA codes are remapped to their parent CBSA before merging.
Quarterly HPI values are averaged to an annual figure per metro x year.

Inputs:
  ../../Data/Processed/panel_00.csv           (from 02_build_panel.py)
  ../../Data/Raw/hpi_at_metro.csv             (FHFA HPI, no header)

Output:
  ../../Data/Processed/panel_01.csv           (panel with hpi_annual added)
"""

import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA     = ROOT.parent / "Data"
PANEL_IN  = DATA / "Processed" / "panel_00.csv"
PANEL_OUT = DATA / "Processed" / "panel_01.csv"
HPI_PATH = DATA / "Raw" / "hpi_at_metro.csv"

# MSAD division -> parent CBSA crosswalk (Census Delineation 2013)
# Download link: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/historical-delineation-files.html
division_to_parent_2013 = {
    14454: 14460, 15764: 14460, 40484: 14460,                # Boston
    16974: 16980, 20994: 16980, 23844: 16980, 29404: 16980,  # Chicago
    19124: 19100, 23104: 19100,                               # Dallas
    19804: 19820, 47664: 19820,                               # Detroit
    11244: 31080, 31084: 31080,                               # Los Angeles
    22744: 33100, 33124: 33100, 48424: 33100,                 # Miami
    20524: 35620, 35004: 35620, 35084: 35620, 35614: 35620,   # New York
    15804: 37980, 33874: 37980, 37964: 37980, 48864: 37980,   # Philadelphia
    36084: 41860, 41884: 41860, 42034: 41860,                 # San Francisco
    42644: 42660, 45104: 42660,                               # Seattle
    43524: 47900, 47894: 47900,                               # Washington
}

# ---------------------------------------------------------------------------
# Load and prepare HPI
# ---------------------------------------------------------------------------
hpi = pd.read_csv(
    HPI_PATH,
    header=None,
    names=["metro_name", "cbsa", "year", "quarter", "hpi", "hpi_se"],
)

hpi["hpi"] = pd.to_numeric(hpi["hpi"], errors="coerce")
hpi["cbsa"] = hpi["cbsa"].map(division_to_parent_2013).fillna(hpi["cbsa"]).astype(int)

# Average quarters (and remapped MSAD divisions) into one annual value per metro x year
hpi_annual = (
    hpi.groupby(["cbsa", "year"], as_index=False)
    .agg(hpi_annual=("hpi", "mean"))
)

# ---------------------------------------------------------------------------
# Merge onto panel
# ---------------------------------------------------------------------------
panel = pd.read_csv(PANEL_IN)

panel = panel.merge(
    hpi_annual,
    left_on=["MET2013", "YEAR"],
    right_on=["cbsa", "year"],
    how="left",
).drop(columns=["cbsa", "year"])

panel.to_csv(PANEL_OUT, index=False)

n_total   = len(panel)
n_matched = panel["hpi_annual"].notna().sum()
n_metros  = panel["MET2013"].nunique()
n_hpi_metros = panel.groupby("MET2013")["hpi_annual"].any().sum()
print(f"Panel rows:           {n_total}")
print(f"HPI matched rows:     {n_matched} / {n_total} ({n_matched/n_total*100:.1f}%)")
print(f"Metros with HPI:      {n_hpi_metros} / {n_metros}")
