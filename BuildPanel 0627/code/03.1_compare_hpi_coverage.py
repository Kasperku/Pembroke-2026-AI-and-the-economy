"""
03.1_compare_hpi_coverage.py  [diagnostic]

Checks how well the FHFA HPI data covers the ACS metro areas. Reports both
unique-code coverage and row-level coverage after remapping MSAD sub-divisions
to their parent CBSA (same crosswalk used in 03_merge_hpi.py).

Inputs:
  ../../Data/Processed/usa_00004_with_genai.csv  (from 01_merge_genai.py)
  ../../Data/Raw/hpi_at_metro.csv                (FHFA HPI, no header)

Output:
  ../../Data/Sumstats/03.1_hpi_coverage.txt
"""

import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA     = ROOT.parent / "Data"
ACS_PATH = DATA / "Processed" / "usa_00004_with_genai.csv"
HPI_PATH = DATA / "Raw" / "hpi_at_metro.csv"
OUT_PATH = DATA / "Sumstats" / "03.1_hpi_coverage.txt"

# MSAD division -> parent CBSA crosswalk (mirrors 03_merge_hpi.py)
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
# Load HPI
# ---------------------------------------------------------------------------
hpi = pd.read_csv(
    HPI_PATH,
    header=None,
    names=["metro_name", "cbsa", "year", "quarter", "hpi", "hpi_se"],
)

msad_mask    = hpi["metro_name"].str.strip().str.endswith("(MSAD)")
n_msad_rows  = msad_mask.sum()

hpi["cbsa"]  = hpi["cbsa"].map(division_to_parent_2013).fillna(hpi["cbsa"]).astype(int)
hpi_cbsa_codes = set(hpi["cbsa"].unique())

# ---------------------------------------------------------------------------
# Load ACS metro codes only
# ---------------------------------------------------------------------------
acs = pd.read_csv(ACS_PATH, usecols=["MET2013"])
acs_with_metro  = acs[acs["MET2013"] != 0]
acs_metro_codes = set(acs_with_metro["MET2013"].unique())

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
matched_codes   = acs_metro_codes & hpi_cbsa_codes
unmatched_codes = acs_metro_codes - hpi_cbsa_codes

n_acs_metros   = len(acs_metro_codes)
n_matched      = len(matched_codes)
n_unmatched    = len(unmatched_codes)
coverage_pct   = n_matched / n_acs_metros * 100 if n_acs_metros > 0 else 0.0

n_rows_with_metro = len(acs_with_metro)
n_rows_covered    = acs_with_metro["MET2013"].isin(hpi_cbsa_codes).sum()
row_coverage_pct  = n_rows_covered / n_rows_with_metro * 100 if n_rows_with_metro > 0 else 0.0

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
lines = [
    "=" * 60,
    "HPI COVERAGE OF ACS MICRODATA (MET2013)",
    "=" * 60,
    f"\nHPI file: {HPI_PATH.name}",
    f"  MSAD rows remapped        : {n_msad_rows:,}",
    f"  Unique CBSA codes in HPI  : {len(hpi_cbsa_codes):,}",
    f"\nACS file: {ACS_PATH.name}",
    f"  Total rows                : {len(acs):,}",
    f"  Rows with MET2013 == 0    : {(acs['MET2013'] == 0).sum():,}",
    f"  Rows with a metro code    : {n_rows_with_metro:,}",
    f"  Unique MET2013 codes      : {n_acs_metros:,}",
    f"\nCoverage (unique code level):",
    f"  MET2013 codes also in HPI : {n_matched:,} / {n_acs_metros:,}  ({coverage_pct:.1f}%)",
    f"  MET2013 codes NOT in HPI  : {n_unmatched:,}",
    f"\nCoverage (row level):",
    f"  Rows whose metro is in HPI: {n_rows_covered:,} / {n_rows_with_metro:,}  ({row_coverage_pct:.1f}%)",
]

if unmatched_codes:
    counts = (
        acs_with_metro[acs_with_metro["MET2013"].isin(unmatched_codes)]
        ["MET2013"]
        .value_counts()
        .rename_axis("MET2013")
        .reset_index(name="acs_rows")
    )
    lines.append(f"\nUnmatched MET2013 codes ({n_unmatched}):")
    lines.append(counts.to_string(index=False))

report = "\n".join(lines)
print(report)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(report, encoding="utf-8")
print(f"\nReport saved: {OUT_PATH}")
