#!/usr/bin/env python3
"""
Gate 0: バーストレジームの実時間検出可能性テスト（訓練不要）。
`docs/FILING_COUNT_FORECAST_DESIGN.md` §2, §5 の実装。

問い: `outputs/predictability_map/RESULTS.md`（検証B）で見つかった computing ドメインの
change-on-change 正の効果（momentum が翌年の momentum を正に予測する、+0.34±0.17,
5遷移すべて正）は、他ドメイン・他時期でも再現するリアルタイム検出可能なレジームか、
それとも「AI/MLブームだった」という後知恵でしか説明できないものか。

手続き（カテゴリ j、年 t）:
  M_j(t)          : そのカテゴリの年 t 出願件数（総出願数）
  mom_j(t)        = log1p(M_j(t)) - log1p(M_j(t-1))                      対数モメンタム
  burst_j(t)      = 1[ mom_j(t) >= その年の正のmom_j(t)分布の80パーセンタイル ]
                    （`dual_force_data_patent.py` の `attach_topic_dynamics` と同一定義。
                    t 以前の情報のみで構築 → リークなし）
  next_mom_j(t+1) = log1p(M_j(t+1)) - log1p(M_j(t))                      翌年のモメンタム（正解）

回帰（プールされた全 (カテゴリ, 遷移) ペア上、OLS）:
  next_mom ~ mom + burst + mom:burst

判定: mom:burst（交互作用項）の係数が、**computing 以外の**独立したドメイン・時期で
有意に正なら Gate 0 通過（Phase 1 へ進む正当性あり）。computing でしか再現しないなら
不通過（②はここで打ち止め、predictability map の1セルとして確定）。

例:
  python -m pnode_patent_runner.gate0_regime_detectability \\
    --output-json pnode_patent_runner/outputs/predictability_map/gate0_results.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

BURST_PERCENTILE = 80.0
PATENT_DOMAINS_EX_COMPUTING = ("agrifood", "construction", "energy", "semiconductor", "pharma")


def _cpc_level(code: str, level: str) -> str:
    if level == "subclass":
        return code[:4]
    if level == "maingroup":
        return code.split("/")[0]
    raise ValueError(level)


def _mass_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs: columns year, actor, cat → 各 (year, cat) の出願件数 M_j(t) の表 (index=year, columns=cat)。"""
    m = pairs.groupby(["year", "cat"]).size().unstack(fill_value=0)
    return m


def _build_observations(
    mass: pd.DataFrame, transitions: List[Tuple[int, int, int]]
) -> pd.DataFrame:
    """
    transitions: [(t-1, t, t+1), ...]。各カテゴリ×遷移で mom_j(t), burst_j(t), next_mom_j(t+1) を作る。
    burst は年 t 内の正の mom 分布の80パーセンタイルで判定（リークフリー、t以前の情報のみ）。
    """
    rows = []
    for (tm1, t, t1) in transitions:
        if tm1 not in mass.index or t not in mass.index or t1 not in mass.index:
            continue
        M_tm1 = mass.loc[tm1]
        M_t = mass.loc[t]
        M_t1 = mass.loc[t1]
        cats = M_t.index
        mom_t = np.log1p(M_t.reindex(cats, fill_value=0)) - np.log1p(M_tm1.reindex(cats, fill_value=0))
        next_mom = np.log1p(M_t1.reindex(cats, fill_value=0)) - np.log1p(M_t.reindex(cats, fill_value=0))
        pos_mom = mom_t[mom_t > 0]
        thr = np.percentile(pos_mom, BURST_PERCENTILE) if len(pos_mom) else np.inf
        burst = (mom_t >= thr).astype(float)
        keep = (M_tm1.reindex(cats, fill_value=0) + M_t.reindex(cats, fill_value=0) + M_t1.reindex(cats, fill_value=0)) > 0
        df = pd.DataFrame({
            "cat": cats, "t": t, "mom": mom_t.values, "burst": burst.values, "next_mom": next_mom.values,
        })
        rows.append(df[keep.values])
    if not rows:
        return pd.DataFrame(columns=["cat", "t", "mom", "burst", "next_mom"])
    return pd.concat(rows, ignore_index=True)


def _fit_interaction(obs: pd.DataFrame) -> Dict:
    """next_mom ~ mom + burst + mom:burst を OLS で当てはめ、交互作用項の統計量を返す。

    同一カテゴリが複数の年遷移にまたがって登場する（プールされた観測はカテゴリ内で
    系列相関しうる）ため、HC1（不均一分散頑健）だけでなく、カテゴリ単位のクラスタ頑健
    標準誤差でも有意性を確認する。厳しい方（p値が大きい方）を採用し、過小評価された
    有意性で誤ってGate 0を通過させないようにする。
    """
    if len(obs) < 20 or obs["burst"].sum() < 5:
        return {"n": int(len(obs)), "n_burst": int(obs["burst"].sum()), "status": "insufficient_data"}
    X = pd.DataFrame({
        "mom": obs["mom"],
        "burst": obs["burst"],
        "mom_burst": obs["mom"] * obs["burst"],
    })
    X = sm.add_constant(X)
    y = obs["next_mom"]
    model = sm.OLS(y, X).fit(cov_type="HC1")  # heteroskedasticity-robust SE
    model_cluster = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": obs["cat"].to_numpy()}
    )
    p_hc1 = float(model.pvalues["mom_burst"])
    p_cluster = float(model_cluster.pvalues["mom_burst"])
    p_conservative = max(p_hc1, p_cluster)
    return {
        "n": int(len(obs)),
        "n_burst": int(obs["burst"].sum()),
        "n_categories": int(obs["cat"].nunique()),
        "coef_mom_burst": float(model.params["mom_burst"]),
        "se_mom_burst_hc1": float(model.bse["mom_burst"]),
        "p_mom_burst_hc1": p_hc1,
        "p_mom_burst_cluster": p_cluster,
        "p_mom_burst": p_conservative,
        "coef_mom": float(model.params["mom"]),
        "p_mom": float(model.pvalues["mom"]),
        "r2": float(model.rsquared),
        "status": "ok",
    }


def load_domain_pairs(dom: str, cpc_level: str, year_min: int, year_max: int) -> pd.DataFrame:
    raw = pd.read_csv(f"data/processed/bipartite_{dom}.csv")
    raw["year"] = raw["ts"].str[:4].astype(int)
    raw = raw[(raw["year"] >= year_min) & (raw["year"] <= year_max)]
    return pd.DataFrame({
        "year": raw["year"], "actor": raw["u"], "cat": raw["i"].map(lambda c: _cpc_level(c, cpc_level)),
    })


def load_author_topic_pairs(year_min: int, year_max: int) -> pd.DataFrame:
    from pnode_patent_runner.data_arxiv import preprocess_author_topic_data

    df = preprocess_author_topic_data(
        "data/processed/arxiv_cs_embedded_2020-2026_full.csv", topic_column="topic"
    )
    df = df.explode("authors_list").dropna(subset=["authors_list"])
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    return pd.DataFrame({"year": df["year"], "actor": df["authors_list"], "cat": df["topic"]})


def main() -> int:
    p = argparse.ArgumentParser(description="Gate 0: burst-regime real-time detectability test")
    p.add_argument("--year-min", type=int, default=2010, help="質量系列の開始年（momの安定のため遷移年より前を含める）")
    p.add_argument("--year-max", type=int, default=2021)
    p.add_argument("--cpc-level", type=str, default="maingroup", choices=("maingroup", "subclass"))
    p.add_argument("--domains", type=str, nargs="+", default=list(PATENT_DOMAINS_EX_COMPUTING) + ["author_topic"])
    p.add_argument("--computing-rolling-window", type=int, default=5, help="computing内部のローリング検証の窓幅（年）")
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    results: Dict[str, Dict] = {}

    # ── 主検定: computing以外の各ドメインで mom×burst 係数が有意に正か ──────────
    for dom in args.domains:
        if dom == "author_topic":
            pairs = load_author_topic_pairs(2020, 2025)
            transitions = [(t - 1, t, t + 1) for t in range(2022, 2025)]
        else:
            pairs = load_domain_pairs(dom, args.cpc_level, args.year_min, args.year_max)
            transitions = [(t - 1, t, t + 1) for t in range(args.year_min + 1, args.year_max)]
        mass = _mass_table(pairs)
        obs = _build_observations(mass, transitions)
        fit = _fit_interaction(obs)
        results[dom] = fit
        print(f"[{dom}] n={fit.get('n')} n_burst={fit.get('n_burst')} "
              f"coef(mom×burst)={fit.get('coef_mom_burst')} "
              f"p_hc1={fit.get('p_mom_burst_hc1')} p_cluster={fit.get('p_mom_burst_cluster')}")

    # ── computing 内部のローリング窓検証: ブーム期以外のサブ期間でも再現するか ──
    comp_pairs = load_domain_pairs("computing", args.cpc_level, args.year_min, args.year_max)
    comp_mass = _mass_table(comp_pairs)
    all_years = sorted(comp_mass.index)
    w = args.computing_rolling_window
    rolling: Dict[str, Dict] = {}
    for start in range(all_years[0] + 1, all_years[-1] - w + 1):
        end = start + w
        if end > all_years[-1]:
            break
        trans = [(t - 1, t, t + 1) for t in range(start, end)]
        obs = _build_observations(comp_mass, trans)
        fit = _fit_interaction(obs)
        key = f"{start}-{end}"
        rolling[key] = fit
        print(f"[computing rolling {key}] n={fit.get('n')} coef(mom×burst)={fit.get('coef_mom_burst')} "
              f"p={fit.get('p_mom_burst')}")
    results["computing_rolling"] = rolling

    # ── Gate 0 判定 ──────────────────────────────────────────────────────────
    non_computing_domains = [d for d in args.domains if d != "computing"]
    passing = [
        d for d in non_computing_domains
        if results[d].get("status") == "ok"
        and results[d]["coef_mom_burst"] > 0
        and results[d]["p_mom_burst"] < 0.05
    ]
    gate0_pass = len(passing) > 0

    out = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "question": "Is the computing-domain momentum×burst positive effect a real-time-detectable "
                         "regime, or hindsight bias unique to the AI/ML boom?",
            "regression": "next_mom ~ mom + burst + mom:burst (OLS, HC1 robust SE), pooled across transitions",
            "decision_rule": "PASS if >=1 domain OTHER than computing shows coef(mom x burst) > 0 with p<0.05",
            "burst_definition": "leak-free, t-or-earlier info only, 80th percentile of positive momentum within year",
        },
        "results": results,
        "gate0_pass": gate0_pass,
        "passing_domains": passing,
    }
    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    with open(oj, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nGate 0 判定: {'PASS' if gate0_pass else 'NOT PASS'}  (通過ドメイン: {passing})")
    print(f"Wrote: {oj}")
    return 0


if __name__ == "__main__":
    main()
