"""viz_line_forecast.py — per-CPC activity time-series forecasting.

The project's "prediction is hard" finding was about RANKING (which firm
enters which CPC). At the AGGREGATE COUNT level (how many firms file in a CPC
each year) simple time-series methods MAY work because momentum carries.

This script:
  - For top-N CPCs by activity, plots historical activity (2001-2020)
  - Forecasts 2021-2023 with 3 simple methods:
      * Persistence (last year continues)
      * Linear extrapolation (5y window)
      * Holt-Winters exponential smoothing
  - Compares forecast vs ACTUAL (data has 2021-2024)
  - Reports MAPE per method per CPC
  - Bottom panel: forecast accuracy summary across all CPCs

Run:  python pnode_patent_runner/viz_line_forecast.py --domain construction
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
    df = df[(df.year >= 1990) & (df.year <= 2024)]
    df["i"] = df["i"].map(R.coarsen); df = df.dropna(subset=["i"]).drop_duplicates(["u", "i", "year"])
    return df


def forecast_persist(series, h):
    last = series.iloc[-1]
    return np.full(h, last, dtype=float)


def forecast_linear(series, h, window=5):
    y = series.iloc[-window:].to_numpy(float)
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + h)
    return np.clip(a * future_x + b, 0, None)


def forecast_hw(series, h):
    try:
        model = ExponentialSmoothing(series.values, trend="add",
                                     initialization_method="estimated")
        fit = model.fit(disp=False)
        return np.clip(np.asarray(fit.forecast(h)), 0, None)
    except Exception:
        return forecast_linear(series, h)


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
    ap.add_argument("--history-end", type=int, default=2020)
    ap.add_argument("--forecast-end", type=int, default=2024)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--out", default="viz_line_forecast.png")
    args = ap.parse_args()

    df = load(args.domain)
    R.LEVEL = "group"
    print(f"firms={df.u.nunique()} CPC-groups={df.i.nunique()}")

    # activity per CPC per year (unique firms)
    pivot = (df.groupby(["i", "year"])["u"].nunique()
               .unstack(fill_value=0)
               .reindex(columns=range(args.y0, args.forecast_end + 1), fill_value=0))
    # top CPCs by total activity over history
    total = pivot.loc[:, args.y0:args.history_end].sum(1)
    top = total.sort_values(ascending=False).head(args.top_n).index.tolist()
    print(f"top {len(top)} CPCs: {top[:5]}...")

    H = args.forecast_end - args.history_end
    history_years = list(range(args.y0, args.history_end + 1))
    future_years = list(range(args.history_end + 1, args.forecast_end + 1))

    # ---- figure: top-N CPC panels + summary ----
    cols = 4
    rows = int(np.ceil(len(top) / cols)) + 1     # +1 for summary
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.0 * rows))
    axes = axes.flatten()

    all_mape = defaultdict(list)
    for k, cpc in enumerate(top):
        ax = axes[k]
        s_hist = pivot.loc[cpc, args.y0:args.history_end]
        s_actual_future = pivot.loc[cpc, future_years[0]:future_years[-1]]
        forecasts = {
            "Persistence": forecast_persist(s_hist, H),
            "Linear (5y)": forecast_linear(s_hist, H),
            "Holt-Winters": forecast_hw(s_hist, H),
        }
        # plot history
        ax.plot(history_years, s_hist.values, "-", color="#1F4E79",
                lw=2.0, label="actual (history)")
        # plot actual future as dashed
        if s_actual_future.sum() > 0:
            ax.plot(future_years, s_actual_future.values, "--", color="#1F4E79",
                    lw=2.0, alpha=0.7, label="actual (future)")
        # plot forecasts
        colors = {"Persistence": "#888", "Linear (5y)": "#D62728", "Holt-Winters": "#2CA02C"}
        for nm, f in forecasts.items():
            ax.plot(future_years, f, "-", color=colors[nm], lw=1.5,
                    marker="o", markersize=4, alpha=0.85, label=nm)
            if s_actual_future.sum() > 0:
                all_mape[nm].append(mape(s_actual_future.values, f))
        ax.axvline(args.history_end + 0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(cpc, fontsize=10.5)
        ax.set_xlim(args.y0, args.forecast_end)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7.5)

    # hide unused panels in CPC row
    for k in range(len(top), rows * cols - cols):
        axes[k].axis("off")

    # summary panel: MAPE distribution across CPCs
    ax_s = axes[-cols]
    ax_s.axis("off")
    summary_ax = fig.add_subplot(rows, 1, rows)
    method_names = list(all_mape.keys())
    if all_mape and all(len(v) > 0 for v in all_mape.values()):
        positions = range(1, len(method_names) + 1)
        data = [all_mape[nm] for nm in method_names]
        bp = summary_ax.boxplot(data, positions=positions, widths=0.6,
                                patch_artist=True, showfliers=False)
        for patch, nm in zip(bp["boxes"], method_names):
            patch.set_facecolor({"Persistence": "#bbb", "Linear (5y)": "#D62728",
                                 "Holt-Winters": "#2CA02C"}[nm])
            patch.set_alpha(0.7)
        summary_ax.set_xticks(positions); summary_ax.set_xticklabels(method_names, fontsize=10)
        summary_ax.set_ylabel("MAPE (forecast error)")
        means = [np.nanmean(v) for v in data]
        for p, m in zip(positions, means):
            summary_ax.text(p, m, f"  mean={m:.2f}", va="center", fontsize=9, color="#222")
        summary_ax.set_title(f"Forecast accuracy across top-{len(top)} CPCs "
                             f"(MAPE on {future_years[0]}-{future_years[-1]} vs actual)", fontsize=11)
    # hide last row's other axes
    for k in range(rows * cols - cols + 1, rows * cols):
        axes[k].axis("off")
    axes[-cols].axis("off")  # ensure leftmost of last row is the summary subplot only

    fig.suptitle(f"Per-CPC activity forecasting — {args.domain}\n"
                 f"history {args.y0}-{args.history_end}, forecast {future_years[0]}-{future_years[-1]}",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.005,
             "Solid blue = historical activity (unique firms/yr). Dashed blue = actual future. "
             "Grey/Red/Green = 3 forecasting methods. Bottom = MAPE distribution across CPCs.",
             ha="center", va="bottom", fontsize=9, color="#333")
    plt.tight_layout(rect=[0, 0.03, 1, 0.965])
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")

    # print MAPE summary
    print("\nForecast accuracy summary (MAPE, lower = better):")
    for nm in method_names:
        vals = all_mape[nm]
        print(f"  {nm:14s}  mean={np.nanmean(vals):.3f}  median={np.nanmedian(vals):.3f}  "
              f"({len(vals)} CPCs)")


if __name__ == "__main__":
    main()
