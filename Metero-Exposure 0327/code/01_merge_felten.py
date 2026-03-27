"""
Step 1: Convert ACS microdata 2018 SOC codes to 2010 SOC, then merge
Felten AI exposure scores.

Mapping logic:
  1. Direct crosswalk hit  -> use 2010 code(s)
  2. Broad/Minor/Major code -> expand to detailed 2018 via SOC structure,
     then crosswalk each to 2010

Score assignment:
  - Auto-mapped (1 Felten code)  -> that score
  - Ambiguous   (N Felten codes) -> equal-weighted average of N scores
  - No match / no crosswalk      -> NaN (rows kept but no score)

Input:
  data/raw/usa_00002.csv
  data/raw/felten_ai_exposure.csv
  data/crosswalks/soc_2010_to_2018_crosswalk.xlsx
  data/crosswalks/2018soc.xlsx

Output:
  data/processed/usa_00002_with_felten.csv
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
XW = ROOT / "data" / "crosswalks"
OUT = ROOT / "data" / "processed"

# ── Load source data ────────────────────────────────────────────
acs = pd.read_csv(RAW / "usa_00002.csv")
acs = acs.dropna(subset=["OCCSOC"])
acs["soc_raw"] = acs["OCCSOC"].astype(int).astype(str)
acs["soc2018"] = acs["soc_raw"].apply(
    lambda x: f"{x[:2]}-{x[2:]}" if len(x) == 6 else x
)

felten = pd.read_csv(RAW / "felten_ai_exposure.csv")
felten["SOC Code"] = felten["SOC Code"].str.strip()
felten_scores = felten.set_index("SOC Code")["AI Exposure Score"].to_dict()
felten_codes = set(felten_scores.keys())

# ── Load BLS 2010-to-2018 crosswalk ─────────────────────────────
xw = pd.read_excel(XW / "soc_2010_to_2018_crosswalk.xlsx", header=None, skiprows=8)
xw.columns = ["soc2010", "title2010", "soc2018", "title2018"]
xw["soc2010"] = xw["soc2010"].astype(str).str.strip()
xw["soc2018"] = xw["soc2018"].astype(str).str.strip()

map_2018_to_2010 = defaultdict(set)
for _, row in xw.iterrows():
    map_2018_to_2010[row["soc2018"]].add(row["soc2010"])

# ── Load 2018 SOC structure (hierarchy) ──────────────────────────
soc2018 = pd.read_excel(XW / "2018soc.xlsx", header=None, skiprows=7)
soc2018.columns = ["group", "code", "title", "definition"]
soc2018["group"] = soc2018["group"].astype(str).str.strip()
soc2018["code"] = soc2018["code"].astype(str).str.strip()

broad_to_detailed = defaultdict(list)
minor_to_detailed = defaultdict(list)
major_to_detailed = defaultdict(list)
current_broad = current_minor = current_major = None
for _, row in soc2018.iterrows():
    if row["group"] == "Major":
        current_major = row["code"]
        current_minor = current_broad = None
    elif row["group"] == "Minor":
        current_minor = row["code"]
        current_broad = None
    elif row["group"] == "Broad":
        current_broad = row["code"]
    elif row["group"] == "Detailed":
        if current_broad is not None:
            broad_to_detailed[current_broad].append(row["code"])
        if current_minor is not None:
            minor_to_detailed[current_minor].append(row["code"])
        if current_major is not None:
            major_to_detailed[current_major].append(row["code"])

# ── Build 2018 -> Felten score lookup ────────────────────────────
def get_felten_2010_codes(code_2018):
    """Return set of 2010 SOC codes in Felten for a given 2018 code."""
    candidates = map_2018_to_2010.get(code_2018)
    if candidates is not None:
        return candidates & felten_codes

    detail_codes = None
    if code_2018 in broad_to_detailed:
        detail_codes = broad_to_detailed[code_2018]
    elif code_2018 in minor_to_detailed:
        detail_codes = minor_to_detailed[code_2018]
    elif code_2018 in major_to_detailed:
        detail_codes = major_to_detailed[code_2018]

    if detail_codes is not None:
        all_2010 = set()
        for dc in detail_codes:
            dc_2010 = map_2018_to_2010.get(dc, set())
            all_2010.update(dc_2010 & felten_codes)
        return all_2010

    return set()

# Pre-compute score for each unique 2018 code
unique_2018 = acs["soc2018"].unique()
score_lookup = {}
mapped_2010_lookup = {}

for code in unique_2018:
    matched_2010 = get_felten_2010_codes(code)
    if len(matched_2010) == 0:
        score_lookup[code] = None
        mapped_2010_lookup[code] = ""
    else:
        scores = [felten_scores[c] for c in matched_2010]
        score_lookup[code] = sum(scores) / len(scores)
        mapped_2010_lookup[code] = "; ".join(sorted(matched_2010))

# ── Merge onto ACS ──────────────────────────────────────────────
acs["soc2010_mapped"] = acs["soc2018"].map(mapped_2010_lookup)
acs["ai_exposure"] = acs["soc2018"].map(score_lookup)
acs = acs.drop(columns=["soc_raw"])

# ── Summary ─────────────────────────────────────────────────────
total = len(acs)
with_score = acs["ai_exposure"].notna().sum()
without_score = acs["ai_exposure"].isna().sum()
print(f"Total rows:          {total:,}")
print(f"With AI exposure:    {with_score:,} ({with_score/total*100:.1f}%)")
print(f"Without AI exposure: {without_score:,} ({without_score/total*100:.1f}%)")
print()

n_auto = sum(1 for c in unique_2018 if len(get_felten_2010_codes(c)) == 1)
n_ambig = sum(1 for c in unique_2018 if len(get_felten_2010_codes(c)) > 1)
n_none = sum(1 for c in unique_2018 if len(get_felten_2010_codes(c)) == 0)
print(f"Unique 2018 codes: {len(unique_2018)}")
print(f"  Auto-mapped (1:1):     {n_auto}")
print(f"  Ambiguous (averaged):  {n_ambig}")
print(f"  No Felten match:       {n_none}")

# ── Save ────────────────────────────────────────────────────────
out_path = OUT / "usa_00002_with_felten.csv"
acs.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
