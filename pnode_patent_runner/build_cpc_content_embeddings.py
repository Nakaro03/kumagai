"""build_cpc_content_embeddings.py — content (semantic) vector per CPC code.

For the human-aware emergence diagnostic we need a SEMANTIC representation of
each technology that is INDEPENDENT of co-occurrence structure (so it can carry
signal Adamic-Adar cannot). We build it from patent TITLE text:

  content_emb(cpc) = mean over a sample of patents in that CPC of
                     MiniLM(patent_title)

Sources (PatentsView bulk, local):
  g_cpc_current.tsv : patent_id -> cpc_group   (3.1G, field 1 and 6)
  g_patent.tsv      : patent_id -> patent_title (1.1G, field 1 and 4)

Output: data/processed/cpc_content_{domain}.npz  (codes[], emb[n,d])

Run (main env, all-MiniLM-L6-v2 cached, offline):
  python pnode_patent_runner/build_cpc_content_embeddings.py --domain energy
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path("/home/nakamuraroi/kumagai")
BULK = ROOT / "notebooks/work/dataset/PatentsViewBulkData"
CSV = {
    "energy": "data/processed/bipartite_energy.csv",
    "construction": "data/processed/bipartite_construction.csv",
    "computing": "data/processed/bipartite_computing.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="energy")
    ap.add_argument("--per-code", type=int, default=30, help="patent titles sampled per CPC")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    codes = sorted(pd.read_csv(ROOT / CSV[args.domain], usecols=["i"])["i"].unique())
    print(f"domain={args.domain}  unique CPC={len(codes)}")

    tmp = tempfile.mkdtemp()
    codes_f = os.path.join(tmp, "codes.txt")
    Path(codes_f).write_text("\n".join(codes) + "\n")
    pid_cpc_f = os.path.join(tmp, "pid_cpc.tsv")
    pid_title_f = os.path.join(tmp, "pid_title.tsv")

    # 1) up to per-code patent_ids per target CPC, single stream over g_cpc_current
    print("scanning g_cpc_current.tsv (patent_id -> cpc) ...")
    subprocess.run(
        ["awk", "-F", "\t", "-v", f"K={args.per_code}",
         'FNR==NR{want[$1]=1; next}'
         '{p=$1; g=$6; gsub(/"/,"",p); gsub(/"/,"",g);'
         ' if(g in want && cnt[g]<K){print p"\t"g; cnt[g]++}}',
         codes_f, str(BULK / "g_cpc_current.tsv")],
        stdout=open(pid_cpc_f, "w"), check=True)
    pc = pd.read_csv(pid_cpc_f, sep="\t", names=["pid", "cpc"], dtype=str)
    print(f"  collected {len(pc)} patent-cpc rows for {pc.cpc.nunique()} codes")

    # 2) titles for those patent_ids, single stream over g_patent
    pids_f = os.path.join(tmp, "pids.txt")
    pc[["pid"]].drop_duplicates().to_csv(pids_f, index=False, header=False)
    print("scanning g_patent.tsv (patent_id -> title) ...")
    subprocess.run(
        ["awk", "-F", "\t",
         'FNR==NR{want[$1]=1; next}'
         '{p=$1; gsub(/"/,"",p); if(p in want){t=$4; gsub(/"/,"",t); print p"\t"t}}',
         pids_f, str(BULK / "g_patent.tsv")],
        stdout=open(pid_title_f, "w"), check=True)
    tt = pd.read_csv(pid_title_f, sep="\t", names=["pid", "title"], dtype=str).dropna()
    print(f"  got {len(tt)} titles")

    df = pc.merge(tt, on="pid")
    titles = df["title"].tolist()

    print(f"embedding {len(titles)} titles with {args.model} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    vecs = model.encode(titles, batch_size=256, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    df = df.assign(_row=range(len(df)))

    d = vecs.shape[1]
    emb = np.zeros((len(codes), d), dtype=np.float32)
    cidx = {c: k for k, c in enumerate(codes)}
    n_have = 0
    for c, grp in df.groupby("cpc"):
        rows = grp["_row"].to_numpy()
        emb[cidx[c]] = vecs[rows].mean(0)
        n_have += 1
    print(f"  built content vectors for {n_have}/{len(codes)} codes "
          f"({len(codes)-n_have} had no patent title -> zero vector)")

    out = ROOT / f"data/processed/cpc_content_{args.domain}.npz"
    np.savez(out, codes=np.array(codes), emb=emb)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
