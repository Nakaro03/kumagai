#!/usr/bin/env python3
"""
訓練不要の「予測可能性の天井」計測（予測可能性マップ論文の Table 1 候補）。

各ドメイン・各年遷移 t→t+1 について、技術カテゴリ j 単位で:
  - M_j(t)      : 質量（イベント数）
  - I_j(t+1)    : 新規ペア流入（それまで観測されていない (актор, カテゴリ) ペア数）
  - D_j(t)      : トレンド = M_j(t) − M_j(t−1)
を作り、次の Spearman 相関を遷移ごとに計測して平均±SD を報告する:
  - popularity  : ρ( M_j(t),  I_j(t+1) )   … 人気度天井（レベルがレベルを予測）
  - persistence : ρ( I_j(t),  I_j(t+1) )   … 流入の持続性
  - trend_level : ρ( D_j(t),  I_j(t+1) )   … トレンド→レベル
  - trend_partial: rank偏相関 ρ( D_j(t), I_j(t+1) | M_j(t) ) … 人気度を統制した増分
  - change_on_change: ρ( D_j(t), ΔI_j(t+1) ) … 変化が変化を予測するか（ハードテスト）

特許 6 ドメイン（ts,u,i の PatentsView 派生 CSV; カテゴリ=CPC メイングループ or サブクラス）
と arXiv 著者–トピックを同一手続きで処理する。学習は一切行わない。

例:
  python -m pnode_patent_runner.predictability_ceilings \\
    --output-json pnode_patent_runner/outputs/predictability_map/ceilings.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PATENT_DOMAINS = ("agrifood", "construction", "energy", "semiconductor", "pharma", "computing")


def _cpc_level(code: str, level: str) -> str:
    if level == "subclass":
        return code[:4]
    if level == "maingroup":
        return code.split("/")[0]
    if level == "subgroup":
        return code  # フルコード（①のDual-Force/TAP-NODE実験と同じ粒度）
    raise ValueError(level)


def _rank_partial(x, y, z) -> float:
    """rank 化した偏相関 ρ(x, y | z)。"""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    cxy = np.corrcoef(rx, ry)[0, 1]
    cxz = np.corrcoef(rx, rz)[0, 1]
    cyz = np.corrcoef(ry, rz)[0, 1]
    den = np.sqrt((1 - cxz**2) * (1 - cyz**2))
    return float((cxy - cxz * cyz) / den) if den > 1e-12 else float("nan")


def compute_ceilings(
    pairs: pd.DataFrame,  # columns: year, actor, cat
    transitions: list,  # [(t, t+1), ...]
    history_start: int,
) -> dict:
    """遷移ごとの相関を返す。pairs は (year, actor, cat) のイベント表（重複可）。

    大規模ドメイン向けに (actor, cat) を int64 キーへ factorize し、
    新規ペア判定は np.isin のベクトル演算で行う。
    """
    per_trans = {k: [] for k in ("popularity", "persistence", "trend_level", "trend_partial", "change_on_change")}
    cat_codes, cats_all = pd.factorize(pairs["cat"], sort=True)
    actor_codes, _ = pd.factorize(pairs["actor"])
    n_cats = len(cats_all)
    years = pairs["year"].to_numpy()
    keys = actor_codes.astype(np.int64) * n_cats + cat_codes.astype(np.int64)

    mass_df = pd.DataFrame({"year": years, "cat": cat_codes}).groupby(["year", "cat"]).size()

    def year_unique_keys(y: int) -> np.ndarray:
        return np.unique(keys[years == y])

    # 履歴を年順に積み上げつつ、各年の「新規ペア」流入をカテゴリ別に数える
    years_needed = sorted({y for tr in transitions for y in tr})
    hist = np.unique(keys[(years >= history_start) & (years < years_needed[0])])
    inflow_by_year = {}
    for y in years_needed:
        ky = year_unique_keys(y)
        fresh = ky[~np.isin(ky, hist, kind="sort")]
        cnt = np.bincount((fresh % n_cats).astype(np.int64), minlength=n_cats).astype(float)
        inflow_by_year[y] = cnt
        hist = np.union1d(hist, ky)

    def mvec(y):
        v = np.zeros(n_cats)
        if y in mass_df.index.get_level_values(0):
            s = mass_df.loc[y]
            v[s.index.to_numpy()] = s.values
        return v

    for (t, t1) in transitions:
        M_t, M_tm1 = mvec(t), mvec(t - 1)
        D_t = M_t - M_tm1
        I_t, I_t1 = inflow_by_year[t], inflow_by_year[t1]
        dI = I_t1 - I_t
        # 全期間ゼロのカテゴリはノイズなので除外
        keep = (M_t + I_t + I_t1) > 0
        if keep.sum() < 10:
            continue
        sp = lambda a, b: float(stats.spearmanr(a[keep], b[keep]).statistic)
        per_trans["popularity"].append(sp(M_t, I_t1))
        per_trans["persistence"].append(sp(I_t, I_t1))
        per_trans["trend_level"].append(sp(D_t, I_t1))
        per_trans["trend_partial"].append(_rank_partial(D_t[keep], I_t1[keep], M_t[keep]))
        per_trans["change_on_change"].append(sp(D_t, dI))

    out = {}
    for k, v in per_trans.items():
        v = np.array(v, dtype=float)
        out[k] = {
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v)),
            "per_transition": [round(float(x), 4) for x in v],
        }
    out["n_transitions"] = len(per_trans["popularity"])
    out["n_categories_total"] = len(cats_all)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="training-free predictability ceilings")
    p.add_argument("--year-min", type=int, default=2016, help="最初の遷移元年")
    p.add_argument("--year-max", type=int, default=2021, help="最後の遷移先年")
    p.add_argument("--history-start", type=int, default=2010, help="新規判定に使う履歴の開始年")
    p.add_argument("--cpc-levels", type=str, nargs="+", default=["maingroup", "subclass"])
    p.add_argument("--domains", type=str, nargs="+", default=list(PATENT_DOMAINS) + ["author_topic"])
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    transitions = [(t, t + 1) for t in range(args.year_min, args.year_max)]
    results = {}

    for dom in args.domains:
        if dom == "author_topic":
            from pnode_patent_runner.data_arxiv import preprocess_author_topic_data

            df = preprocess_author_topic_data(
                "data/processed/arxiv_cs_embedded_2020-2026_full.csv", topic_column="topic"
            )
            # authors_list を展開して (year, author, topic) イベント表にする
            df = df.explode("authors_list").dropna(subset=["authors_list"])
            pairs = pd.DataFrame(
                {"year": df["year"], "actor": df["authors_list"], "cat": df["topic"]}
            )
            at_trans = [(t, t + 1) for t in range(2021, 2025)]
            results["author_topic"] = compute_ceilings(pairs, at_trans, history_start=2020)
            results["author_topic"]["transitions"] = [list(t) for t in at_trans]
            print(f"author_topic done: {results['author_topic']['n_transitions']} transitions")
            continue

        raw = pd.read_csv(f"data/processed/bipartite_{dom}.csv")
        raw["year"] = raw["ts"].str[:4].astype(int)
        raw = raw[(raw["year"] >= args.history_start) & (raw["year"] <= args.year_max)]
        for lvl in args.cpc_levels:
            pairs = pd.DataFrame(
                {
                    "year": raw["year"],
                    "actor": raw["u"],
                    "cat": raw["i"].map(lambda c: _cpc_level(c, lvl)),
                }
            )
            key = f"{dom}:{lvl}"
            results[key] = compute_ceilings(pairs, transitions, history_start=args.history_start)
            results[key]["transitions"] = [list(t) for t in transitions]
            print(f"{key} done: n_cats={results[key]['n_categories_total']}")

    out = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "inflow": "new (actor, cat) pairs not seen since history_start",
            "year_min": args.year_min,
            "year_max": args.year_max,
            "history_start": args.history_start,
        },
        "results": results,
    }
    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    with open(oj, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {oj}")
    return 0


if __name__ == "__main__":
    main()
