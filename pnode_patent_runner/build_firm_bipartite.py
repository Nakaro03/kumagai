"""build_firm_bipartite.py — firm(assignee) x CPC bipartite over time.

The recommender's decision unit should be the FIRM, not the inventor. We have no
firm column in bipartite_*.csv, so we build it from PatentsView bulk by joining
on patent_id:
  g_cpc_current.tsv     patent_id -> cpc_group   (filter to the domain code set)
  g_assignee_*.tsv      patent_id -> assignee_id, organization (firms only)
  g_application.tsv     patent_id -> filing_date

Output: data/processed/bipartite_{domain}_firm.csv  (ts, u=assignee_id, i=cpc)
plus a {domain}_firm_names.csv (assignee_id -> organization) for readability.

Run:  python pnode_patent_runner/build_firm_bipartite.py --domain construction
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path("/home/nakamuraroi/kumagai")
BULK = ROOT / "notebooks/work/dataset/PatentsViewBulkData"
CSV = {"energy": "data/processed/bipartite_energy.csv",
       "construction": "data/processed/bipartite_construction.csv",
       "computing": "data/processed/bipartite_computing.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    args = ap.parse_args()

    codes = sorted(pd.read_csv(ROOT / CSV[args.domain], usecols=["i"])["i"].unique())
    print(f"domain={args.domain} target CPC codes={len(codes)}")
    tmp = tempfile.mkdtemp()
    codes_f = os.path.join(tmp, "codes.txt"); Path(codes_f).write_text("\n".join(codes) + "\n")
    pc_f = os.path.join(tmp, "pid_cpc.tsv")
    pid_f = os.path.join(tmp, "pids.txt")
    asg_f = os.path.join(tmp, "pid_asg.tsv")
    dat_f = os.path.join(tmp, "pid_date.tsv")

    # 1) ALL (patent_id, cpc) rows for the domain code set
    print("scanning g_cpc_current.tsv ...")
    subprocess.run(["awk", "-F", "\t",
        'FNR==NR{want[$1]=1; next}{p=$1;g=$6;gsub(/"/,"",p);gsub(/"/,"",g);'
        'if(g in want)print p"\t"g}', codes_f, str(BULK / "g_cpc_current.tsv")],
        stdout=open(pc_f, "w"), check=True)
    pc = pd.read_csv(pc_f, sep="\t", names=["pid", "cpc"], dtype=str)
    print(f"  (patent,cpc) rows={len(pc)} unique patents={pc.pid.nunique()}")
    pc[["pid"]].drop_duplicates().to_csv(pid_f, index=False, header=False)

    # 2) assignee (firms only: organization non-empty) for those patents
    print("scanning g_assignee_disambiguated.tsv ...")
    subprocess.run(["awk", "-F", "\t",
        'FNR==NR{want[$1]=1; next}{p=$1;a=$3;o=$6;gsub(/"/,"",p);gsub(/"/,"",a);gsub(/"/,"",o);'
        'if((p in want) && o!="")print p"\t"a"\t"o}', pid_f, str(BULK / "g_assignee_disambiguated.tsv")],
        stdout=open(asg_f, "w"), check=True)
    asg = pd.read_csv(asg_f, sep="\t", names=["pid", "assignee_id", "org"], dtype=str)
    print(f"  patent-assignee rows={len(asg)} unique firms={asg.assignee_id.nunique()}")

    # 3) filing date for those patents
    print("scanning g_application.tsv ...")
    subprocess.run(["awk", "-F", "\t",
        'FNR==NR{want[$1]=1; next}{p=$2;d=$4;gsub(/"/,"",p);gsub(/"/,"",d);'
        'if(p in want)print p"\t"d}', pid_f, str(BULK / "g_application.tsv")],
        stdout=open(dat_f, "w"), check=True)
    dat = pd.read_csv(dat_f, sep="\t", names=["pid", "ts"], dtype=str).dropna()
    print(f"  patent-date rows={len(dat)}")

    # join -> (ts, assignee_id, cpc)
    df = pc.merge(asg, on="pid").merge(dat, on="pid")
    out = df[["ts", "assignee_id", "cpc"]].rename(columns={"assignee_id": "u", "cpc": "i"})
    out = out.dropna()
    out = out[out.ts.str.len() >= 7]
    out_path = ROOT / f"data/processed/bipartite_{args.domain}_firm.csv"
    out.to_csv(out_path, index=False)
    names = df[["assignee_id", "org"]].drop_duplicates().rename(columns={"assignee_id": "u"})
    names.to_csv(ROOT / f"data/processed/{args.domain}_firm_names.csv", index=False)
    yr = pd.to_datetime(out.ts, errors="coerce").dt.year
    print(f"\nwrote {out_path}")
    print(f"  rows={len(out)} firms={out.u.nunique()} CPC={out.i.nunique()} "
          f"years={int(yr.min())}-{int(yr.max())}")


if __name__ == "__main__":
    main()
