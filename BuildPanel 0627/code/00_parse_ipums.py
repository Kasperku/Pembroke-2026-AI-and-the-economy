"""
Step 0: Parse raw IPUMS fixed-width .dat file into a CSV.

Input:
  C:/Users/user/Desktop/Honours Thesis/Data/Raw/usa_00004.dat
  C:/Users/user/Desktop/Honours Thesis/Data/Raw/usa_00004.xml

Output:
  C:/Users/user/Desktop/Honours Thesis/Data/Processed/usa_00004.csv
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

RAW  = Path("C:/Users/user/Desktop/Honours Thesis/Data/Raw")
PROC = Path("C:/Users/user/Desktop/Honours Thesis/Data/Processed")
PROC.mkdir(parents=True, exist_ok=True)

DAT_PATH = RAW / "usa_00004.dat"
XML_PATH = RAW / "usa_00004.xml"
OUT_PATH = PROC / "usa_00004.csv"

CHUNK = 500_000

ns   = {"ddi": "ddi:codebook:2_5"}
tree = ET.parse(XML_PATH)
root = tree.getroot()

colspecs = []
names    = []
decimals = {}

for var in root.findall(".//ddi:var", ns):
    name = var.attrib["name"]
    dcml = int(var.attrib.get("dcml", 0))
    loc  = var.find("ddi:location", ns)
    colspecs.append((int(loc.attrib["StartPos"]) - 1, int(loc.attrib["EndPos"])))
    names.append(name)
    if dcml > 0:
        decimals[name] = dcml

print(f"Variables: {', '.join(names)}")

first_chunk = True
total_rows  = 0

for chunk in pd.read_fwf(DAT_PATH, colspecs=colspecs, names=names, dtype=str, chunksize=CHUNK, encoding="latin-1"):
    for col, places in decimals.items():
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce") / (10 ** places)
    for col in chunk.columns:
        if col not in decimals:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk.to_csv(OUT_PATH, mode="w" if first_chunk else "a", index=False, header=first_chunk)
    first_chunk = False
    total_rows += len(chunk)
    print(f"  {total_rows:,} rows written ...", end="\r")

print(f"\nDone. {total_rows:,} rows -> {OUT_PATH}")
