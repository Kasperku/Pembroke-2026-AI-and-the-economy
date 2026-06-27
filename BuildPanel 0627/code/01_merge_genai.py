"""
Step 1: Merge GenAI exposure scores onto ACS microdata, accounting for
the IPUMS OCCSOC vintage split:
  - 2014-2017 ACS samples use 2010 SOC codes  -> match GenAI scores directly
  - 2018-2024 ACS samples use 2018 SOC codes  -> crosswalk 2018->2010 first

Score assignment:
  - Exact match (1 GenAI code)  -> that score
  - Ambiguous   (N GenAI codes) -> equal-weighted average of N scores
  - No match                    -> NaN (rows kept but no score)

Input:
  data/raw/genaiexp_estz_occscores.csv         (this folder)
  ../../Data/Processed/usa_00004.csv
  ../../Data/crosswalks/soc_2010_to_2018_crosswalk.xlsx
  ../../Data/crosswalks/2018soc.xlsx

Output:
  ../../Data/Processed/usa_00004_with_genai.csv
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
DATA = ROOT.parent / "Data"
XW   = DATA / "crosswalks"
OUT  = DATA / "Processed"

# ── GenAI scores (keyed on 2010 SOC codes) ──────────────────────────────────
genai = pd.read_csv(RAW / "genaiexp_estz_occscores.csv")
genai["soc2010"] = genai["soc2010"].astype(str).str.strip()
genai_scores = genai.set_index("soc2010")["genaiexp_estz_total"].to_dict()
genai_codes  = set(genai_scores.keys())

# ── 2010->2018 crosswalk ─────────────────────────────────────────────────────
xw = pd.read_excel(XW / "soc_2010_to_2018_crosswalk.xlsx", header=None, skiprows=8)
xw.columns = ["soc2010", "title2010", "soc2018_xw", "title2018"]
xw["soc2010"]    = xw["soc2010"].astype(str).str.strip()
xw["soc2018_xw"] = xw["soc2018_xw"].astype(str).str.strip()
xw = xw[xw["soc2010"] != "2010 SOC Code"]  # drop header row if present

map_2018_to_2010 = defaultdict(set)
for _, row in xw.iterrows():
    map_2018_to_2010[row["soc2018_xw"]].add(row["soc2010"])

# ── 2010 SOC hierarchy (derived from crosswalk's detailed codes) ─────────────
# All codes in the crosswalk are detailed 2010 SOC codes.
# Derive broad/minor/major groupings from code structure (XX-XXXX):
#   broad = code[:-1] + "0"     e.g. 31-1011 -> 31-1010
#   minor = code[:4]  + "000"   e.g. 31-1011 -> 31-1000
#   major = code[:2]  + "-0000" e.g. 31-1011 -> 31-0000
all_2010_detailed = {c for c in xw["soc2010"].unique() if len(c) == 7 and c[2] == "-"}

broad_to_det_2010 = defaultdict(list)
minor_to_det_2010 = defaultdict(list)
major_to_det_2010 = defaultdict(list)
for code in all_2010_detailed:
    broad_to_det_2010[code[:-1] + "0"].append(code)
    minor_to_det_2010[code[:4]  + "000"].append(code)
    major_to_det_2010[code[:2]  + "-0000"].append(code)

# ── 2018 SOC hierarchy (from BLS structure file) ─────────────────────────────
soc2018_struct = pd.read_excel(XW / "2018soc.xlsx", header=None, skiprows=7)
soc2018_struct.columns = ["group", "code", "title", "definition"]
soc2018_struct["group"] = soc2018_struct["group"].astype(str).str.strip()
soc2018_struct["code"]  = soc2018_struct["code"].astype(str).str.strip()

broad_to_detailed = defaultdict(list)
minor_to_detailed = defaultdict(list)
major_to_detailed = defaultdict(list)
current_broad = current_minor = current_major = None
for _, row in soc2018_struct.iterrows():
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

# ── Lookup functions ─────────────────────────────────────────────────────────
def get_genai_for_2010_code(code):
    """Return GenAI-matched 2010 SOC codes for a given 2010 SOC code (handles aggregates)."""
    if code in genai_codes:
        return {code}
    for lookup in (broad_to_det_2010, minor_to_det_2010, major_to_det_2010):
        if code in lookup:
            matched = {c for c in lookup[code] if c in genai_codes}
            if matched:
                return matched
    return set()

def get_genai_for_2018_code(code):
    """Return GenAI-matched 2010 SOC codes for a given 2018 SOC code (handles aggregates)."""
    candidates = map_2018_to_2010.get(code)
    if candidates is not None:
        return candidates & genai_codes

    detail_codes = None
    if code in broad_to_detailed:
        detail_codes = broad_to_detailed[code]
    elif code in minor_to_detailed:
        detail_codes = minor_to_detailed[code]
    elif code in major_to_detailed:
        detail_codes = major_to_detailed[code]

    if detail_codes is not None:
        all_2010 = set()
        for dc in detail_codes:
            all_2010.update(map_2018_to_2010.get(dc, set()) & genai_codes)
        return all_2010

    return set()

def scores_for_matched(matched):
    """Convert a set of matched 2010 SOC codes to (avg_score, joined_codes_str)."""
    if not matched:
        return float("nan"), ""
    scores = [genai_scores[c] for c in matched]
    return sum(scores) / len(scores), "; ".join(sorted(matched))

# ── Load ACS data ────────────────────────────────────────────────────────────
acs = pd.read_csv(DATA / "Processed" / "usa_00004.csv")
acs = acs[acs["OCCSOC"].notna()
          & (acs["OCCSOC"].astype(float) != 0)
          & (acs["OCCSOC"].astype(float) != 999920)]
acs["soc_raw"] = acs["OCCSOC"].astype(int).astype(str)
# Note: for YEAR < 2018 this column holds 2010 SOC codes formatted as XX-XXXX;
#       for YEAR >= 2018 it holds 2018 SOC codes in the same format.
acs["soc2018"] = acs["soc_raw"].apply(
    lambda x: f"{x[:2]}-{x[2:]}" if len(x) == 6 else x
)

# ── Build score lookups, split by vintage ────────────────────────────────────
pre_mask  = acs["YEAR"] < 2018   # OCCSOC = 2010 SOC codes
post_mask = ~pre_mask             # OCCSOC = 2018 SOC codes

unique_pre  = acs.loc[pre_mask,  "soc2018"].unique()
unique_post = acs.loc[post_mask, "soc2018"].unique()

score_pre,  mapped_pre  = {}, {}
score_post, mapped_post = {}, {}

for code in unique_pre:
    score_pre[code], mapped_pre[code] = scores_for_matched(get_genai_for_2010_code(code))

for code in unique_post:
    score_post[code], mapped_post[code] = scores_for_matched(get_genai_for_2018_code(code))

# ── Apply lookups ────────────────────────────────────────────────────────────
acs["soc2010_mapped"] = ""
acs["ai_exposure"]    = float("nan")

acs.loc[pre_mask,  "soc2010_mapped"] = acs.loc[pre_mask,  "soc2018"].map(mapped_pre)
acs.loc[pre_mask,  "ai_exposure"]    = acs.loc[pre_mask,  "soc2018"].map(score_pre).astype(float)
acs.loc[post_mask, "soc2010_mapped"] = acs.loc[post_mask, "soc2018"].map(mapped_post)
acs.loc[post_mask, "ai_exposure"]    = acs.loc[post_mask, "soc2018"].map(score_post).astype(float)

acs = acs.drop(columns=["soc_raw"])

# ── Summary ──────────────────────────────────────────────────────────────────
total      = len(acs)
with_score = acs["ai_exposure"].notna().sum()
without    = acs["ai_exposure"].isna().sum()
print(f"Total rows:          {total:,}")
print(f"With AI exposure:    {with_score:,} ({with_score/total*100:.1f}%)")
print(f"Without AI exposure: {without:,} ({without/total*100:.1f}%)")
print()

n_auto  = (sum(1 for c in unique_pre  if len(get_genai_for_2010_code(c)) == 1)
         + sum(1 for c in unique_post if len(get_genai_for_2018_code(c)) == 1))
n_ambig = (sum(1 for c in unique_pre  if len(get_genai_for_2010_code(c)) > 1)
         + sum(1 for c in unique_post if len(get_genai_for_2018_code(c)) > 1))
n_none  = (sum(1 for c in unique_pre  if len(get_genai_for_2010_code(c)) == 0)
         + sum(1 for c in unique_post if len(get_genai_for_2018_code(c)) == 0))

print(f"Unique SOC codes — pre-2018 (2010 vintage): {len(unique_pre)}")
print(f"Unique SOC codes — 2018+   (2018 vintage):  {len(unique_post)}")
print(f"  Auto-mapped (1:1):     {n_auto}")
print(f"  Ambiguous (averaged):  {n_ambig}")
print(f"  No GenAI match:        {n_none}")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT / "usa_00004_with_genai.csv"
acs.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
