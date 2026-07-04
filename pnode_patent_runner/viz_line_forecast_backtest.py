"""viz_line_forecast_backtest.py — backtest of per-CPC activity forecasting.

History: 2001-2015 (data complete). Test: 2016-2018 (data still complete).
Forecast 3 years ahead with:
  - Persistence (last year continues)
  - Linear (5y window extrapolation)
  - Holt-Winters exponential smoothing (with prediction interval)

Reports: per-method per-CPC MAPE, ranking of predictable vs unpredictable CPCs,
publication-quality multi-panel figure.

Run:  python pnode_patent_runner/viz_line_forecast_backtest.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.linewidth": 0.8, "axes.edgecolor": "#222",
    "axes.spines.right": False, "axes.spines.top": False,
})


def load(domain):
    df = pd.read_csv(ROOT / f"data/processed/bipartite_{domain}_firm.csv")
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)
    df = df[(df.year >= 1990) & (df.year <= 2020)]
    df["i"] = df["i"].map(R.coarsen); df = df.dropna(subset=["i"]).drop_duplicates(["u", "i", "year"])
    return df


def forecast_persist(series, h):
    return np.full(h, float(series.iloc[-1]))


def forecast_linear(series, h, window=5):
    y = series.iloc[-window:].to_numpy(float)
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    fx = np.arange(len(y), len(y) + h)
    return np.clip(a * fx + b, 0, None)


def forecast_hw(series, h):
    try:
        m = ExponentialSmoothing(series.values, trend="add",
                                 initialization_method="estimated")
        fit = m.fit(disp=False)
        return np.clip(np.asarray(fit.forecast(h)), 0, None), fit
    except Exception:
        return forecast_linear(series, h), None


def mape(actual, pred):
    actual = np.asarray(actual, float); pred = np.asarray(pred, float)
    m = actual > 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[m] - pred[m]) / actual[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--y0", type=int, default=2001)
    ap.add_argument("--history-end", type=int, default=2015)
    ap.add_argument("--forecast-end", type=int, default=2018)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--out", default="viz_line_forecast_backtest.png")
    args = ap.parse_args()

    df = load(args.domain); R.LEVEL = "group"
    print(f"firms={df.u.nunique()} CPC-groups={df.i.nunique()}")

    pivot = (df.groupby(["i", "year"])["u"].nunique()
               .unstack(fill_value=0)
               .reindex(columns=range(args.y0, args.forecast_end + 1), fill_value=0))
    total = pivot.loc[:, args.y0:args.history_end].sum(1)
    H = args.forecast_end - args.history_end
    history_years = list(range(args.y0, args.history_end + 1))
    future_years = list(range(args.history_end + 1, args.forecast_end + 1))

    # compute MAPE for ALL CPCs with reasonable activity
    active_cpcs = total[total >= 50].index.tolist()
    print(f"evaluating {len(active_cpcs)} active CPCs (>=50 firm-years cumulative)")
    per_cpc_mape = {}
    for cpc in active_cpcs:
        s_h = pivot.loc[cpc, args.y0:args.history_end]
        s_f = pivot.loc[cpc, future_years[0]:future_years[-1]]
        if s_f.sum() == 0 or s_h.iloc[-1] == 0:
            continue
        fps = {"Persistence": forecast_persist(s_h, H),
               "Linear": forecast_linear(s_h, H),
               "Holt-Winters": forecast_hw(s_h, H)[0]}
        per_cpc_mape[cpc] = {nm: mape(s_f.values, f) for nm, f in fps.items()}

    # aggregate
    methods = ["Persistence", "Linear", "Holt-Winters"]
    mapes_by_method = {nm: [v[nm] for v in per_cpc_mape.values() if not np.isnan(v[nm])]
                       for nm in methods}
    print(f"\nForecast MAPE across {len(per_cpc_mape)} CPCs (lower=better):")
    for nm in methods:
        v = mapes_by_method[nm]
        print(f"  {nm:14s}  mean={np.mean(v):.3f}  median={np.median(v):.3f}  "
              f"p25={np.percentile(v, 25):.3f}  p75={np.percentile(v, 75):.3f}")

    # rank CPCs by predictability (using best-of-3 MAPE)
    best_mape = {cpc: min(per_cpc_mape[cpc][nm] for nm in methods
                          if not np.isnan(per_cpc_mape[cpc][nm]))
                 for cpc in per_cpc_mape}
    sorted_cpcs = sorted(best_mape.items(), key=lambda x: x[1])
    print(f"\nTop-5 most predictable CPCs:")
    for c, m in sorted_cpcs[:5]:
        print(f"  {c}: best MAPE = {m:.3f}")
    print(f"\nTop-5 least predictable CPCs:")
    for c, m in sorted_cpcs[-5:]:
        print(f"  {c}: best MAPE = {m:.3f}")

    # ---- figure ----
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9],
                          left=0.04, right=0.97, top=0.93, bottom=0.06,
                          wspace=0.22, hspace=0.40)

    # top 8 most predictable
    top_show = [c for c, _ in sorted_cpcs[:args.top_n]]
    for k, cpc in enumerate(top_show):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        s_h = pivot.loc[cpc, args.y0:args.history_end]
        s_f = pivot.loc[cpc, future_years[0]:future_years[-1]]
        fps = {"Persistence": forecast_persist(s_h, H),
               "Linear": forecast_linear(s_h, H),
               "Holt-Winters": forecast_hw(s_h, H)[0]}
        ax.plot(history_years, s_h.values, "-", color="#1F4E79", lw=2,
                label="actual (history)")
        ax.plot(future_years, s_f.values, "--", color="#1F4E79", lw=2, alpha=0.85,
                label="actual (future)")
        cols = {"Persistence": "#888", "Linear": "#D62728", "Holt-Winters": "#2CA02C"}
        for nm, f in fps.items():
            mape_val = mape(s_f.values, f)
            ax.plot(future_years, f, "-", color=cols[nm], lw=1.5,
                    marker="o", markersize=4.5, alpha=0.9,
                    label=f"{nm} (MAPE {mape_val:.2f})")
        ax.axvline(args.history_end + 0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(f"{cpc}  (best MAPE = {best_mape[cpc]:.2f})", fontsize=10.5)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7.0, ncol=1)

    # summary row: box plot + scatter showing easiness
    ax_box = fig.add_subplot(gs[2, 0:2])
    data = [mapes_by_method[nm] for nm in methods]
    bp = ax_box.boxplot(data, positions=range(len(methods)), widths=0.55,
                        patch_artist=True, showfliers=False)
    cols_b = ["#bbb", "#D62728", "#2CA02C"]
    for patch, c in zip(bp["boxes"], cols_b):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax_box.set_xticks(range(len(methods))); ax_box.set_xticklabels(methods, fontsize=10)
    ax_box.set_ylabel("MAPE (forecast error)")
    for k, nm in enumerate(methods):
        ax_box.text(k, np.median(mapes_by_method[nm]),
                    f"  med={np.median(mapes_by_method[nm]):.2f}",
                    va="center", fontsize=9, color="#000", weight="bold")
    ax_box.set_title(f"Forecast accuracy across {len(per_cpc_mape)} CPCs "
                     f"(test {future_years[0]}-{future_years[-1]}, lower=better)", fontsize=11)
    ax_box.set_ylim(0, np.percentile([v for vs in data for v in vs], 95) * 1.1)

    # predictable vs unpredictable scatter
    ax_sc = fig.add_subplot(gs[2, 2:4])
    activity_levels = [total[c] for c in per_cpc_mape.keys()]
    best_mapes = [best_mape[c] for c in per_cpc_mape.keys()]
    ax_sc.scatter(activity_levels, best_mapes, s=20, alpha=0.5, color="#444")
    # mark top predictable in green, least in red
    for c, m in sorted_cpcs[:5]:
        ax_sc.scatter([total[c]], [m], color="#138D75", s=80, zorder=5,
                      edgecolors="white", linewidths=1.5)
        ax_sc.annotate(c, (total[c], m), fontsize=8, color="#0E6655",
                       xytext=(5, 5), textcoords="offset points")
    for c, m in sorted_cpcs[-5:]:
        ax_sc.scatter([total[c]], [m], color="#B03A2E", s=80, zorder=5,
                      edgecolors="white", linewidths=1.5)
        ax_sc.annotate(c, (total[c], m), fontsize=8, color="#7B241C",
                       xytext=(5, 5), textcoords="offset points")
    ax_sc.set_xscale("log"); ax_sc.set_xlabel("cumulative activity (history, log)")
    ax_sc.set_ylabel("best MAPE (lower=more predictable)")
    ax_sc.set_title("Predictability vs activity volume\n"
                    "(green = most predictable, red = least)", fontsize=11)
    ax_sc.set_ylim(0, np.percentile(best_mapes, 95) * 1.2)

    fig.suptitle(f"Per-CPC line forecasting (backtest) — {args.domain}\n"
                 f"history {args.y0}-{args.history_end}, "
                 f"forecast {future_years[0]}-{future_years[-1]}", fontsize=13, y=0.985)
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
