"""
EDA: Statistical properties of the 6 PatentsView domain datasets.

Outputs:
  eda_domain_stats.pdf  (publication-quality multi-panel figure)
  eda_domain_stats.png  (same, 300 dpi)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── config ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR  = Path(__file__).parent

DOMAINS = ["pharma", "semiconductor", "computing", "energy", "construction", "agrifood"]

COLORS = {
    "pharma":        "#e6194b",
    "semiconductor": "#3cb44b",
    "computing":     "#4363d8",
    "energy":        "#f58231",
    "construction":  "#911eb4",
    "agrifood":      "#42d4f4",
}

YEAR_MIN, YEAR_MAX = 2000, 2021

CHUNKSIZE = 1_000_000   # rows per chunk for large CSVs

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "figure.dpi":     150,
})


# ── helpers ────────────────────────────────────────────────────────────────────

def _subclass(code: str) -> str:
    return code[:4] if len(code) >= 4 else code


def load_domain(domain: str) -> dict:
    """
    Load a bipartite_*.csv and compute per-year statistics.
    Returns a stats dict.
    """
    path = DATA_DIR / f"bipartite_{domain}.csv"
    print(f"  Loading {domain} ...", flush=True)

    rows = []
    for chunk in pd.read_csv(path, usecols=["ts", "u", "i"], dtype=str,
                              chunksize=CHUNKSIZE):
        chunk["ts"] = pd.to_datetime(chunk["ts"], errors="coerce")
        chunk = chunk.dropna(subset=["ts"])
        chunk["year"] = chunk["ts"].dt.year.astype(int)
        chunk = chunk[(chunk["year"] >= YEAR_MIN) & (chunk["year"] <= YEAR_MAX)]
        chunk = chunk[["year", "u", "i"]].drop_duplicates()
        rows.append(chunk)

    df = pd.concat(rows, ignore_index=True).drop_duplicates()

    # drop low-activity inventors (< 2 events over full period)
    counts = df["u"].value_counts()
    active = set(counts[counts >= 2].index)
    df = df[df["u"].isin(active)]

    total_events   = len(df)
    total_inventors = df["u"].nunique()
    total_ipc       = df["i"].nunique()
    total_subclass  = df["i"].map(_subclass).nunique()

    # inventor event count distribution (for CCDF)
    inv_counts = df["u"].value_counts().values

    # per-year stats
    year_stats: dict[int, dict] = {}
    seen_inventors: set = set()
    for yr in range(YEAR_MIN, YEAR_MAX + 1):
        sub = df[df["year"] == yr]
        if sub.empty:
            year_stats[yr] = dict(events=0, inventors=0, ipc_codes=0,
                                  subclasses=0, new_inventors=0)
            continue
        inv_yr  = set(sub["u"].unique())
        new_inv = inv_yr - seen_inventors
        seen_inventors |= inv_yr
        year_stats[yr] = dict(
            events      = len(sub),
            inventors   = len(inv_yr),
            ipc_codes   = sub["i"].nunique(),
            subclasses  = sub["i"].map(_subclass).nunique(),
            new_inventors = len(new_inv),
        )

    # top-10 CPC subclasses
    sub_counts = df["i"].map(_subclass).value_counts().head(10)

    return dict(
        domain          = domain,
        df              = df,
        total_events    = total_events,
        total_inventors = total_inventors,
        total_ipc       = total_ipc,
        total_subclass  = total_subclass,
        inv_counts      = inv_counts,
        year_stats      = year_stats,
        top_subclasses  = sub_counts,
    )


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_all(stats_list: list[dict]):
    years = list(range(YEAR_MIN, YEAR_MAX + 1))

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "PatentsView Domain Dataset — Statistical Properties (2000–2021)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])   # total events bar
    ax2 = fig.add_subplot(gs[0, 1])   # annual events line
    ax3 = fig.add_subplot(gs[0, 2])   # annual active inventors
    ax4 = fig.add_subplot(gs[1, 0])   # inventor CCDF (log-log)
    ax5 = fig.add_subplot(gs[1, 1])   # annual new-inventor ratio
    ax6 = fig.add_subplot(gs[1, 2])   # annual CPC subclass count
    ax7 = fig.add_subplot(gs[2, :2])  # top-15 subclass heatmap
    ax8 = fig.add_subplot(gs[2, 2])   # summary table

    # ── (1) total events bar ────────────────────────────────────────────────────
    totals = [s["total_events"] for s in stats_list]
    bars   = ax1.barh(DOMAINS, totals,
                      color=[COLORS[d] for d in DOMAINS], edgecolor="white", height=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel("Total events (log scale)")
    ax1.set_title("(a) Total events per domain")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    for bar, val in zip(bars, totals):
        ax1.text(val * 1.05, bar.get_y() + bar.get_height() / 2,
                 f"{val/1e6:.2f}M" if val >= 1e6 else f"{val/1e3:.0f}K",
                 va="center", fontsize=7.5)
    ax1.invert_yaxis()

    # ── (2) annual event counts ─────────────────────────────────────────────────
    for s in stats_list:
        d = s["domain"]
        vals = [s["year_stats"][y]["events"] for y in years]
        ax2.plot(years, vals, color=COLORS[d], label=d, linewidth=1.5)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Events")
    ax2.set_title("(b) Annual event counts")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax2.legend(ncol=2, loc="upper left")

    # ── (3) annual active inventors ─────────────────────────────────────────────
    for s in stats_list:
        d = s["domain"]
        vals = [s["year_stats"][y]["inventors"] for y in years]
        ax3.plot(years, vals, color=COLORS[d], label=d, linewidth=1.5)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Active inventors")
    ax3.set_title("(c) Annual active inventors")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}K"))
    ax3.legend(ncol=2, loc="upper left")

    # ── (4) inventor activity CCDF (log-log) ────────────────────────────────────
    for s in stats_list:
        d = s["domain"]
        cnt = np.sort(s["inv_counts"])[::-1]
        rank = np.arange(1, len(cnt) + 1)
        ax4.loglog(cnt, rank / rank[-1], color=COLORS[d], label=d,
                   linewidth=1.4, alpha=0.85)
    ax4.set_xlabel("Events per inventor")
    ax4.set_ylabel("CCDF  P(X ≥ x)")
    ax4.set_title("(d) Inventor activity CCDF (log-log)")
    ax4.legend(ncol=2, loc="lower left")

    # ── (5) new-inventor ratio ──────────────────────────────────────────────────
    for s in stats_list:
        d = s["domain"]
        ratio = []
        for y in years:
            ys = s["year_stats"][y]
            if ys["inventors"] > 0:
                ratio.append(ys["new_inventors"] / ys["inventors"])
            else:
                ratio.append(0.0)
        ax5.plot(years, ratio, color=COLORS[d], label=d, linewidth=1.5)
    ax5.set_xlabel("Year")
    ax5.set_ylabel("New-inventor ratio")
    ax5.set_title("(e) New inventor ratio per year")
    ax5.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax5.legend(ncol=2, loc="upper right")

    # ── (6) annual CPC subclass count ───────────────────────────────────────────
    for s in stats_list:
        d = s["domain"]
        vals = [s["year_stats"][y]["subclasses"] for y in years]
        ax6.plot(years, vals, color=COLORS[d], label=d, linewidth=1.5)
    ax6.set_xlabel("Year")
    ax6.set_ylabel("Unique CPC subclasses")
    ax6.set_title("(f) Annual CPC subclass diversity")
    ax6.legend(ncol=2, loc="upper left")

    # ── (7) top CPC subclasses heatmap ──────────────────────────────────────────
    # collect global top-15 subclasses across all domains
    global_sub_cnt: dict[str, int] = defaultdict(int)
    for s in stats_list:
        for sc, cnt in s["top_subclasses"].items():
            global_sub_cnt[sc] += cnt
    top15 = sorted(global_sub_cnt, key=lambda k: -global_sub_cnt[k])[:15]

    heatmap = np.zeros((len(DOMAINS), len(top15)))
    for i, s in enumerate(stats_list):
        sub_counts = s["df"]["i"].map(_subclass).value_counts()
        total = sub_counts.sum()
        for j, sc in enumerate(top15):
            heatmap[i, j] = sub_counts.get(sc, 0) / total if total > 0 else 0

    im = ax7.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0)
    ax7.set_xticks(range(len(top15)))
    ax7.set_xticklabels(top15, rotation=40, ha="right", fontsize=7.5)
    ax7.set_yticks(range(len(DOMAINS)))
    ax7.set_yticklabels(DOMAINS)
    ax7.set_title("(g) Top-15 CPC subclass share per domain")
    fig.colorbar(im, ax=ax7, label="Share of events", shrink=0.8)

    # ── (8) summary table ───────────────────────────────────────────────────────
    ax8.axis("off")
    col_labels = ["Domain", "Events", "Inventors", "IPC codes", "Subclasses"]
    rows_data  = []
    for s in stats_list:
        def _fmt(n):
            return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.1f}K"
        rows_data.append([
            s["domain"],
            _fmt(s["total_events"]),
            _fmt(s["total_inventors"]),
            _fmt(s["total_ipc"]),
            str(s["total_subclass"]),
        ])
    tbl = ax8.table(
        cellText  = rows_data,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dddddd")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    ax8.set_title("(h) Summary statistics", pad=8)

    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading datasets ...", flush=True)
    stats_list = [load_domain(d) for d in DOMAINS]

    print("Plotting ...", flush=True)
    fig = plot_all(stats_list)

    out_png = OUT_DIR / "eda_domain_stats.png"
    out_pdf = OUT_DIR / "eda_domain_stats.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved → {out_png}")
    print(f"Saved → {out_pdf}")


if __name__ == "__main__":
    main()
