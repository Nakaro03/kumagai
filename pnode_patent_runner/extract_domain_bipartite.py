"""
Extract domain-specific bipartite events from PatentsView bulk TSV files.
Output format: ts,u,i  (same as bipartite_events.csv)
  ts = patent_date (YYYY-MM-DD)
  u  = disambig inventor_id
  i  = cpc_group (e.g. "A61K31/00")

Domains:
  pharma       : A61, C12
  energy       : Y02
  semiconductor: H01L, H04
  agrifood     : A01, A23
  computing    : G06
  construction : E04, E02, E03, E21
"""

import argparse
import os

import pandas as pd

BASE = "/home/nakamuraroi/kumagai/notebooks/work/dataset/PatentsViewBulkData"
OUT_DIR = "/home/nakamuraroi/kumagai/data/processed"

DOMAINS = {
    "pharma":         ["A61", "C12"],
    "energy":         ["Y02"],
    "semiconductor":  ["H01L", "H04"],
    "agrifood":       ["A01", "A23"],
    "computing":      ["G06"],
    "construction":   ["E04", "E02", "E03", "E21"],
}

YEAR_START = 2000
YEAR_END   = 2021

def domain_prefixes_match(group: pd.Series, prefixes) -> pd.Series:
    """True if cpc_group starts with any of prefixes."""
    mask = pd.Series(False, index=group.index)
    for p in prefixes:
        mask |= group.str.startswith(p)
    return mask


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract domain-specific bipartite events from PatentsView bulk TSV files."
    )
    parser.add_argument("domain", nargs="?", help="Optional domain to extract (default: all domains)")
    parser.add_argument(
        "--year-end",
        type=int,
        default=YEAR_END,
        help=f"Last patent year to include (default: {YEAR_END})",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    target_domain = args.domain
    year_end = args.year_end
    domains_to_run = {k: v for k, v in DOMAINS.items()
                      if target_domain is None or k == target_domain}

    print("Loading g_patent.tsv (date column only)...", flush=True)
    patent = pd.read_csv(
        f"{BASE}/g_patent.tsv", sep="\t", quotechar='"',
        usecols=["patent_id", "patent_date"],
        dtype={"patent_id": str, "patent_date": str},
    )
    patent["patent_date"] = pd.to_datetime(patent["patent_date"], errors="coerce")
    patent = patent.dropna(subset=["patent_date"])
    patent["year"] = patent["patent_date"].dt.year
    patent = patent[(patent["year"] >= YEAR_START) & (patent["year"] <= year_end)]
    patent_ids = set(patent["patent_id"].values)
    print(f"  Patents in {YEAR_START}-{year_end}: {len(patent_ids):,}", flush=True)

    print("Loading g_inventor_disambiguated.tsv ...", flush=True)
    inv = pd.read_csv(
        f"{BASE}/g_inventor_disambiguated.tsv", sep="\t", quotechar='"',
        usecols=["patent_id", "inventor_id"],
        dtype={"patent_id": str, "inventor_id": str},
    )
    inv = inv[inv["patent_id"].isin(patent_ids)]
    print(f"  Inventor records in window: {len(inv):,}", flush=True)

    print("Loading g_cpc_current.tsv (inventional only) ...", flush=True)
    # Read in chunks since it's 57M rows
    chunks = []
    chunksize = 2_000_000
    reader = pd.read_csv(
        f"{BASE}/g_cpc_current.tsv", sep="\t", quotechar='"',
        usecols=["patent_id", "cpc_group", "cpc_type"],
        dtype={"patent_id": str, "cpc_group": str, "cpc_type": str},
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk[
            (chunk["cpc_type"].isin(["inventional", "additional"])) &
            (chunk["patent_id"].isin(patent_ids))
        ]
        if len(chunk):
            chunks.append(chunk[["patent_id", "cpc_group"]])
    cpc = pd.concat(chunks, ignore_index=True)
    print(f"  CPC records in window: {len(cpc):,}", flush=True)

    # Build merged: patent_id -> date, inventor, cpc_group
    merged = (
        cpc
        .merge(patent[["patent_id", "patent_date"]], on="patent_id", how="inner")
        .merge(inv, on="patent_id", how="inner")
    )
    print(f"  Merged records: {len(merged):,}", flush=True)

    for domain, prefixes in domains_to_run.items():
        mask = domain_prefixes_match(merged["cpc_group"], prefixes)
        df = merged[mask][["patent_date", "inventor_id", "cpc_group"]].copy()
        df.columns = ["ts", "u", "i"]
        df["ts"] = df["ts"].dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates()
        out_path = f"{OUT_DIR}/bipartite_{domain}.csv"
        df.to_csv(out_path, index=False)
        print(f"[{domain}] rows={len(df):,}  unique_u={df['u'].nunique():,}  "
              f"unique_i={df['i'].nunique():,}  -> {out_path}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
