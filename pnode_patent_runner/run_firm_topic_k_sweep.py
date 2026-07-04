"""Firm-topic K sweep for P-NODE validation.

This script converts firm x CPC events into firm x topic events by clustering CPC
content embeddings, then optionally runs the existing benchmark comparison for
each K.

Example:
  python -m pnode_patent_runner.run_firm_topic_k_sweep --run --epochs 5
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "pnode_patent_runner" / "outputs" / "firm_topic_k_sweep"
DEFAULT_KS = (10, 30, 50, 100, 300, 500, 1000)


def parse_ks(s: str) -> List[int]:
    if not s.strip():
        return list(DEFAULT_KS)
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_topic_csvs(
    ks: Iterable[int],
    *,
    firm_csv: Path,
    cpc_emb_npz: Path,
    outdir: Path,
    seed: int,
    force: bool,
) -> List[Path]:
    from sklearn.cluster import KMeans, MiniBatchKMeans

    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(firm_csv, usecols=["ts", "u", "i"], dtype=str)
    df = df.dropna(subset=["ts", "u", "i"]).reset_index(drop=True)

    z = np.load(cpc_emb_npz, allow_pickle=True)
    codes = [str(c) for c in z["codes"]]
    emb = np.asarray(z["emb"], dtype=np.float32)
    code_to_row = {c: i for i, c in enumerate(codes)}

    active_codes = sorted(df["i"].dropna().unique().tolist())
    missing = [c for c in active_codes if c not in code_to_row]
    if missing:
        # Deterministic zero vectors for codes that are in the firm graph but not
        # in the inventor-derived CPC embedding cache.
        emb_dim = int(emb.shape[1])
        active_emb = np.zeros((len(active_codes), emb_dim), dtype=np.float32)
        for j, c in enumerate(active_codes):
            idx = code_to_row.get(c)
            if idx is not None:
                active_emb[j] = emb[idx]
    else:
        active_emb = emb[[code_to_row[c] for c in active_codes]]

    out_paths: List[Path] = []
    for k in ks:
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        if k > len(active_codes):
            raise ValueError(f"K={k} exceeds active CPC count {len(active_codes)}")

        out_csv = outdir / f"firm_topic_construction_K{k}.csv"
        out_map = outdir / f"firm_topic_construction_K{k}_cpc_map.csv"
        if out_csv.exists() and out_map.exists() and not force:
            out_paths.append(out_csv)
            print(f"[K={k}] exists: {out_csv.relative_to(ROOT)}")
            continue

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=2048,
            n_init=5,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        labels = km.fit_predict(active_emb)
        if len(np.unique(labels)) < k:
            print(
                f"[K={k}] MiniBatchKMeans produced {len(np.unique(labels))} non-empty "
                "clusters; retrying with full KMeans."
            )
            km_full = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=3,
                max_iter=300,
                algorithm="lloyd",
            )
            labels = km_full.fit_predict(active_emb)
        topic_names = [f"T{int(x):04d}" for x in labels]
        cpc_to_topic = dict(zip(active_codes, topic_names))

        out = df.copy()
        out["i"] = out["i"].map(cpc_to_topic)
        out = out.dropna(subset=["i"])
        out = out.drop_duplicates(subset=["ts", "u", "i"]).reset_index(drop=True)
        out.to_csv(out_csv, index=False)

        pd.DataFrame({"cpc": active_codes, "topic": topic_names}).to_csv(out_map, index=False)
        out_paths.append(out_csv)
        print(
            f"[K={k}] saved {out_csv.relative_to(ROOT)} "
            f"rows={len(out):,} firms={out['u'].nunique():,} topics={out['i'].nunique():,}"
        )

    return out_paths


def benchmark_one(
    csv_path: Path,
    *,
    k: int,
    outdir: Path,
    seed: int,
    epochs: int,
    year_start: int,
    year_end: int,
    holdout: int,
    min_events: int,
    methods: str,
    hidden_dim: int,
    latent_dim: int,
    num_neg_recon: int,
    num_neg_future: int,
) -> Path:
    out_json = outdir / f"benchmark_firm_topic_K{k}_seed{seed}.json"
    cmd = [
        sys.executable,
        "-m",
        "pnode_patent_runner.run_benchmark_comparison",
        "--data-domain",
        "construction",
        "--data",
        str(csv_path),
        "--year-range",
        str(year_start),
        str(year_end),
        "--holdout-test-year",
        str(holdout),
        "--min-patents",
        str(min_events),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--methods",
        methods,
        "--hidden-dim",
        str(hidden_dim),
        "--latent-dim",
        str(latent_dim),
        "--num-neg-recon",
        str(num_neg_recon),
        "--num-neg-future",
        str(num_neg_future),
        "--pnode-ode-method",
        "rk4",
        "--output-json",
        str(out_json),
    ]
    print(f"\n[K={k}] running benchmark -> {out_json.relative_to(ROOT)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_json


def discover_ks(outdir: Path, seed: int) -> List[int]:
    """Find every K that has a benchmark JSON on disk for this seed.

    The per-run ``--ks`` list does not include Ks from earlier runs (e.g. K=10
    benchmarked in a separate invocation), so summarising must scan the output
    directory rather than trust the CLI list.
    """
    pat = re.compile(rf"benchmark_firm_topic_K(\d+)_seed{seed}\.json$")
    found = []
    for p in outdir.glob(f"benchmark_firm_topic_K*_seed{seed}.json"):
        m = pat.search(p.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(set(found))


def summarize(outdir: Path, ks: Iterable[int], seed: int) -> Path:
    # Always aggregate every K present on disk, not just this run's --ks, so the
    # summary CSV is rebuilt complete (incl. K=10 from earlier runs).
    disk_ks = discover_ks(outdir, seed)
    all_ks = sorted(set(disk_ks) | {int(k) for k in ks})

    rows = []
    for k in all_ks:
        p = outdir / f"benchmark_firm_topic_K{k}_seed{seed}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d.get("results", []):
            rows.append(
                {
                    "K": k,
                    "method": r.get("key"),
                    "label": r.get("label"),
                    "final_auc": r.get("final_val_auc"),
                    "final_ap": r.get("final_val_ap"),
                    "final_ece": r.get("final_val_ece"),
                    "train_auc": r.get("train_split_val_auc"),
                    "train_ap": r.get("train_split_val_ap"),
                    "best_val_auc": r.get("best_val_auc"),
                }
            )
    summary_csv = outdir / f"summary_seed{seed}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_csv, index=False)
    print(f"\nsummary -> {summary_csv.relative_to(ROOT)} (Ks: {all_ks})")
    if not rows:
        return summary_csv

    auc = df.pivot(index="K", columns="method", values="final_auc")
    ap = df.pivot(index="K", columns="method", values="final_ap")
    print("\nfinal AUC (holdout)")
    print(auc.round(4).to_string())
    print("\nfinal AP (holdout)")
    print(ap.round(4).to_string())

    # Direct answer to "does finer granularity let P-NODE beat Static?":
    # per-K best method and P-NODE's margin over Static on holdout AUC/AP.
    cmp_csv = outdir / f"summary_seed{seed}_vs_static.csv"
    cmp_rows = []
    for k in auc.index:
        row = {"K": int(k)}
        auc_k, ap_k = auc.loc[k], ap.loc[k]
        if "static" in auc_k and "pnode" in auc_k:
            row["pnode_minus_static_auc"] = auc_k["pnode"] - auc_k["static"]
        if "static" in ap_k and "pnode" in ap_k:
            row["pnode_minus_static_ap"] = ap_k["pnode"] - ap_k["static"]
        row["best_method_auc"] = auc_k.idxmax() if auc_k.notna().any() else None
        row["best_auc"] = auc_k.max()
        cmp_rows.append(row)
    cmp = pd.DataFrame(cmp_rows).set_index("K")
    cmp.to_csv(cmp_csv)
    print("\nP-NODE margin vs Static (positive = P-NODE wins) + best method per K")
    print(cmp.round(4).to_string())
    wins = cmp["best_method_auc"].eq("pnode").sum() if "best_method_auc" in cmp else 0
    print(f"\nP-NODE is best at {wins}/{len(cmp)} K values (holdout AUC).")
    return summary_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--firm-csv", default="data/processed/bipartite_construction_firm.csv")
    ap.add_argument("--cpc-emb", default="data/processed/cpc_content_construction.npz")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--year-start", type=int, default=2010)
    ap.add_argument("--year-end", type=int, default=2021)
    ap.add_argument("--holdout", type=int, default=2021)
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--methods", default="static,neural_ode,pnode")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--num-neg-recon", type=int, default=400)
    ap.add_argument("--num-neg-future", type=int, default=200)
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--run", action="store_true", help="Run benchmarks after building CSVs.")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    ks = parse_ks(args.ks)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir

    if args.summarize_only:
        summarize(outdir, ks, args.seed)
        return

    csvs = build_topic_csvs(
        ks,
        firm_csv=ROOT / args.firm_csv,
        cpc_emb_npz=ROOT / args.cpc_emb,
        outdir=outdir,
        seed=args.seed,
        force=args.force_build,
    )

    if args.run:
        for k, csv_path in zip(ks, csvs):
            benchmark_one(
                csv_path,
                k=k,
                outdir=outdir,
                seed=args.seed,
                epochs=args.epochs,
                year_start=args.year_start,
                year_end=args.year_end,
                holdout=args.holdout,
                min_events=args.min_events,
                methods=args.methods,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                num_neg_recon=args.num_neg_recon,
                num_neg_future=args.num_neg_future,
            )
        summarize(outdir, ks, args.seed)


if __name__ == "__main__":
    main()
