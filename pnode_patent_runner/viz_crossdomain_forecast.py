"""viz_crossdomain_forecast.py — cross-domain (patent vs paper) forecasting comparison.

Apply the SAME 3 forecasting methods (Persistence, Linear, Holt-Winters) to:
  - Patent domain: construction firm-CPC bipartite (per-CPC activity, 2001-2018 backtest)
  - Paper domain: arxiv CS papers (per-topic count, 2020-2025 backtest)

Compare MAPE distributions across both. Confirms whether the "aggregate
forecasting works" finding extends from technology to science.

Run:  python pnode_patent_runner/viz_crossdomain_forecast.py
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from diagnose_convergence_signal import ROOT
import recommender_firm as R

warnings.filterwarnings("ignore")
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.linewidth": 0.8, "axes.edgecolor": "#222",
    "axes.spines.right": False, "axes.spines.top": False,
})


def load_patent():
    df = pd.read_csv(ROOT / "data/processed/bipartite_construction_firm.csv")
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)
    df = df[(df.year >= 2001) & (df.year <= 2018)]
    df["i"] = df["i"].map(R.coarsen); df = df.dropna(subset=["i"]).drop_duplicates(["u", "i", "year"])
    # per-CPC activity = unique firms per year
    pivot = (df.groupby(["i", "year"])["u"].nunique()
               .unstack(fill_value=0)
               .reindex(columns=range(2001, 2019), fill_value=0))
    total = pivot.loc[:, 2001:2015].sum(1)
    items = total[total >= 50].index.tolist()
    return pivot.loc[items], 2001, 2015, 2018


def load_paper():
    df = pd.read_csv(ROOT / "data/processed/arxiv_cs_embedded_2020-2026.csv",
                     usecols=["topic", "year"], low_memory=False)
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)
    df = df[(df.year >= 2020) & (df.year <= 2025)]
    # per-topic activity = paper count per year
    pivot = (df.groupby(["topic", "year"]).size()
               .unstack(fill_value=0)
               .reindex(columns=range(2020, 2026), fill_value=0))
    total = pivot.loc[:, 2020:2023].sum(1)
    items = total[total >= 10].index.tolist()
    return pivot.loc[items], 2020, 2023, 2025


def forecast_persist(series, h):
    return np.full(h, float(series.iloc[-1]))


def forecast_linear(series, h, window=3):
    y = series.iloc[-window:].to_numpy(float); x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    fx = np.arange(len(y), len(y) + h)
    return np.clip(a * fx + b, 0, None)


def forecast_hw(series, h):
    try:
        m = ExponentialSmoothing(series.values, trend="add",
                                 initialization_method="estimated")
        fit = m.fit(disp=False)
        return np.clip(np.asarray(fit.forecast(h)), 0, None)
    except Exception:
        return forecast_linear(series, h)


def mape(actual, pred):
    actual = np.asarray(actual, float); pred = np.asarray(pred, float)
    m = actual > 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[m] - pred[m]) / actual[m]))


def evaluate(pivot, y0, hist_end, fc_end):
    H = fc_end - hist_end
    history_years = list(range(y0, hist_end + 1))
    future_years = list(range(hist_end + 1, fc_end + 1))
    per_mape = {}
    for item in pivot.index:
        s_h = pivot.loc[item, y0:hist_end]
        s_f = pivot.loc[item, future_years[0]:future_years[-1]]
        if s_f.sum() == 0 or s_h.iloc[-1] == 0:
            continue
        fps = {"Persistence": forecast_persist(s_h, H),
               "Linear": forecast_linear(s_h, H),
               "Holt-Winters": forecast_hw(s_h, H)}
        per_mape[item] = {nm: mape(s_f.values, f) for nm, f in fps.items()}
    return per_mape, history_years, future_years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="viz_crossdomain_forecast.png")
    args = ap.parse_args(); R.LEVEL = "group"

    print("=== PATENT (construction firm × CPC) ===")
    pat_pivot, pat_y0, pat_he, pat_fe = load_patent()
    pat_mape, pat_h, pat_f = evaluate(pat_pivot, pat_y0, pat_he, pat_fe)
    print(f"  {len(pat_pivot)} active CPCs, evaluated {len(pat_mape)}")

    print("\n=== PAPER (arxiv CS topic) ===")
    pap_pivot, pap_y0, pap_he, pap_fe = load_paper()
    pap_mape, pap_h, pap_f = evaluate(pap_pivot, pap_y0, pap_he, pap_fe)
    print(f"  {len(pap_pivot)} active topics, evaluated {len(pap_mape)}")

    methods = ["Persistence", "Linear", "Holt-Winters"]
    def stats(mp_dict):
        return {nm: [v[nm] for v in mp_dict.values() if not np.isnan(v[nm])]
                for nm in methods}
    pat_stats = stats(pat_mape); pap_stats = stats(pap_mape)

    print("\nFORECAST MAPE COMPARISON:")
    print(f"{'method':14s}  {'patent med':>12s}  {'patent p25-p75':>16s}  "
          f"{'paper med':>11s}  {'paper p25-p75':>16s}")
    for nm in methods:
        pmed = np.median(pat_stats[nm]); pp25, pp75 = np.percentile(pat_stats[nm], [25, 75])
        smed = np.median(pap_stats[nm]); sp25, sp75 = np.percentile(pap_stats[nm], [25, 75])
        print(f"  {nm:12s}  {pmed:12.3f}  {pp25:.3f}-{pp75:.3f}    "
              f"{smed:11.3f}  {sp25:.3f}-{sp75:.3f}")

    # pick top predictable for each domain to show
    pat_best = sorted(pat_mape.items(), key=lambda x: min(x[1].values()))[:4]
    pap_best = sorted(pap_mape.items(), key=lambda x: min(x[1].values()))[:4]

    # ---- figure ----
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], wspace=0.28, hspace=0.55,
                          left=0.05, right=0.97, top=0.92, bottom=0.06)

    # Row 1: patent examples
    fig.text(0.5, 0.945, "Cross-domain forecasting backtest: same methods, "
             "patents (technology) vs arxiv CS (science)",
             ha="center", fontsize=13, weight="bold")
    fig.text(0.02, 0.89, "(a)  PATENT: construction firm × CPC  "
             f"[history {pat_y0}-{pat_he}, forecast {pat_f[0]}-{pat_f[-1]}]",
             fontsize=11, weight="bold", color="#1F4E79")
    for k, (item, _) in enumerate(pat_best):
        ax = fig.add_subplot(gs[0, k])
        s_h = pat_pivot.loc[item, pat_y0:pat_he]
        s_f = pat_pivot.loc[item, pat_f[0]:pat_f[-1]]
        H = len(pat_f)
        fps = {"Persistence": forecast_persist(s_h, H),
               "Linear": forecast_linear(s_h, H),
               "Holt-Winters": forecast_hw(s_h, H)}
        ax.plot(pat_h, s_h.values, "-", color="#1F4E79", lw=1.8, label="actual")
        ax.plot(pat_f, s_f.values, "--", color="#1F4E79", lw=1.8, alpha=0.85, label="actual (future)")
        cols = {"Persistence": "#888", "Linear": "#D62728", "Holt-Winters": "#2CA02C"}
        for nm, f in fps.items():
            mv = mape(s_f.values, f)
            ax.plot(pat_f, f, "-", color=cols[nm], lw=1.3, marker="o", markersize=4,
                    alpha=0.85, label=f"{nm} ({mv:.2f})")
        ax.axvline(pat_he + 0.5, color="grey", linestyle=":", alpha=0.4)
        ax.set_title(f"{item}", fontsize=10)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.text(0.02, 0.595, "(b)  PAPER: arxiv CS topic  "
             f"[history {pap_y0}-{pap_he}, forecast {pap_f[0]}-{pap_f[-1]}]",
             fontsize=11, weight="bold", color="#7B241C")
    for k, (item, _) in enumerate(pap_best):
        ax = fig.add_subplot(gs[1, k])
        s_h = pap_pivot.loc[item, pap_y0:pap_he]
        s_f = pap_pivot.loc[item, pap_f[0]:pap_f[-1]]
        H = len(pap_f)
        fps = {"Persistence": forecast_persist(s_h, H),
               "Linear": forecast_linear(s_h, H),
               "Holt-Winters": forecast_hw(s_h, H)}
        ax.plot(pap_h, s_h.values, "-", color="#7B241C", lw=1.8, label="actual")
        ax.plot(pap_f, s_f.values, "--", color="#7B241C", lw=1.8, alpha=0.85, label="actual (future)")
        cols = {"Persistence": "#888", "Linear": "#D62728", "Holt-Winters": "#2CA02C"}
        for nm, f in fps.items():
            mv = mape(s_f.values, f)
            ax.plot(pap_f, f, "-", color=cols[nm], lw=1.3, marker="o", markersize=4,
                    alpha=0.85, label=f"{nm} ({mv:.2f})")
        ax.axvline(pap_he + 0.5, color="grey", linestyle=":", alpha=0.4)
        ax.set_title(f"{item}", fontsize=10)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)

    # Row 3: domain-side-by-side MAPE comparison
    fig.text(0.02, 0.30, "(c)  Aggregate MAPE comparison across domains",
             fontsize=11, weight="bold")
    ax_p = fig.add_subplot(gs[2, 0:2])
    ax_s = fig.add_subplot(gs[2, 2:4])

    positions_pat = [1, 2, 3]; positions_pap = [1, 2, 3]
    pat_data = [pat_stats[nm] for nm in methods]
    pap_data = [pap_stats[nm] for nm in methods]
    bp1 = ax_p.boxplot(pat_data, positions=positions_pat, widths=0.55,
                       patch_artist=True, showfliers=False)
    for patch, c in zip(bp1["boxes"], ["#bbb", "#D62728", "#2CA02C"]):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax_p.set_xticks(positions_pat); ax_p.set_xticklabels(methods, fontsize=9)
    ax_p.set_ylabel("MAPE"); ax_p.set_title(f"Patent CPCs (n={len(pat_mape)})", fontsize=10)
    for p, vs in zip(positions_pat, pat_data):
        ax_p.text(p, np.median(vs), f" med={np.median(vs):.2f}", va="center", fontsize=8.5)
    ax_p.set_ylim(0, np.percentile([v for vs in pat_data for v in vs], 95) * 1.15)

    bp2 = ax_s.boxplot(pap_data, positions=positions_pap, widths=0.55,
                       patch_artist=True, showfliers=False)
    for patch, c in zip(bp2["boxes"], ["#bbb", "#D62728", "#2CA02C"]):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax_s.set_xticks(positions_pap); ax_s.set_xticklabels(methods, fontsize=9)
    ax_s.set_ylabel("MAPE"); ax_s.set_title(f"arxiv CS topics (n={len(pap_mape)})", fontsize=10)
    for p, vs in zip(positions_pap, pap_data):
        ax_s.text(p, np.median(vs), f" med={np.median(vs):.2f}", va="center", fontsize=8.5)
    ax_s.set_ylim(0, np.percentile([v for vs in pap_data for v in vs], 95) * 1.15)

    fig.text(0.5, 0.015,
             "Same 3 forecasting methods (Persistence/Linear/Holt-Winters) applied to two domains. "
             "Median MAPE values printed inside each box. Lower = better forecast accuracy.",
             ha="center", va="bottom", fontsize=9, color="#333")
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
