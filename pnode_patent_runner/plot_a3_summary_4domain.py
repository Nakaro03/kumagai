"""
A3 leaveout summary across 4 domains: X1 PI-SDE vs baselines.
論文 Fig.6 想定。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

DOMAINS = {
    "Paper\n(arXiv CS, t=3)":              {"root": "PNode_Paper_X1",                  "last_t": 3},
    "Patent Energy\n(CPC Y02, t=11)":      {"root": "PNode_Patent_Energy_X1_top50",    "last_t": 11},
    "arXiv Construction\n(legacy, t=10)":  {"root": "PNode_ArXiv_Construction_X1_v2",  "last_t": 10},
    "JP Construction\n(J-STAGE, t=10)":    {"root": "PNode_JP_Construction_X1",        "last_t": 10},
}
DOMAIN_KEY_FOR_BL = {  # baselines/<key>/baselines_seed*.json
    "Paper\n(arXiv CS, t=3)":              "paper",
    "Patent Energy\n(CPC Y02, t=11)":      "patent_energy_top50",
    "arXiv Construction\n(legacy, t=10)":  "arxiv_construction",
    "JP Construction\n(J-STAGE, t=10)":    "jp_construction",
}
SEEDS = [0, 1, 42, 123, 999]
BASELINES = ["Naive_lastg", "Linear", "ARIMA", "LSTM", "Transformer"]


def gather_baseline(domain_key, method, metric):
    vals = []
    if domain_key is None:
        return vals
    for s in SEEDS:
        p = Path(f"RESULTS/baselines/{domain_key}/baselines_seed{s}.json")
        if not p.exists(): continue
        d = json.load(p.open())
        if method in d and metric in d[method]:
            v = d[method][metric]
            if v == v:
                vals.append(v)
    return vals


def gather_x1_leaveout(root, last_t):
    """X1 PI-SDE leaveout=last_t over seeds."""
    sp, nd, p10 = [], [], []
    for s in SEEDS:
        for p in Path(f"RESULTS/{root}").rglob("evaluation_x1.json"):
            tag = p.parents[2].name if len(p.parents) >= 3 else ""
            if not tag.endswith("-x1_v1.0_g0.1_b0.01"): continue
            if f"seed_{s}" not in str(p): continue
            if f"/leaveout{last_t}/" not in str(p): continue
            d = json.load(p.open())
            r = next((r for r in d["results"] if r["t"] == last_t and r.get("split") == "test"), None)
            if r is None: continue
            sp.append(r["spearman_r"])
            nd.append(r["ndcg"])
            p10.append(r["prec_at_10"])
            break
    return np.array(sp), np.array(nd), np.array(p10)


def main():
    # Build summary table
    summary = []
    for dname, cfg in DOMAINS.items():
        domain_key = DOMAIN_KEY_FOR_BL[dname]
        sp_x1, nd_x1, p10_x1 = gather_x1_leaveout(cfg["root"], cfg["last_t"])
        row = {"domain": dname, "X1": (sp_x1, nd_x1, p10_x1)}
        for m in BASELINES:
            sps = gather_baseline(domain_key, m, "spearman_r")
            nds = gather_baseline(domain_key, m, "ndcg_at_10")
            p10s = gather_baseline(domain_key, m, "prec_at_10")
            row[m] = (np.array(sps), np.array(nds), np.array(p10s))
        summary.append(row)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ["Spearman", "NDCG@10", "P@10"]
    metric_idx = [0, 1, 2]

    method_order = BASELINES + ["X1"]
    method_colors = {
        "Naive_lastg": "#888",
        "Linear":      "#9ec1d6",
        "ARIMA":       "#5b95b9",
        "LSTM":        "#f29e4c",
        "Transformer": "#e85d75",
        "X1":          "#2c7a2c",
    }

    x_domains = np.arange(len(DOMAINS))
    n_methods = len(method_order)
    bar_w = 0.13

    for ax_idx, (ax, mname, midx) in enumerate(zip(axes, metrics, metric_idx)):
        for mi, m in enumerate(method_order):
            means, stds = [], []
            for row in summary:
                vals = row[m][midx] if m in row else np.array([])
                if len(vals) > 0:
                    means.append(vals.mean()); stds.append(vals.std())
                else:
                    means.append(np.nan); stds.append(0)
            means = np.array(means); stds = np.array(stds)
            offset = (mi - n_methods / 2 + 0.5) * bar_w
            ec = "black" if m == "X1" else "none"
            lw = 1.2 if m == "X1" else 0
            ax.bar(x_domains + offset, means, bar_w, yerr=stds,
                   label=("X1 PI-SDE (本研究)" if m == "X1" else m),
                   color=method_colors[m], edgecolor=ec, linewidth=lw,
                   capsize=2, error_kw={"linewidth": 0.5})

        ax.set_xticks(x_domains)
        ax.set_xticklabels(list(DOMAINS.keys()), fontsize=9)
        ax.set_title(f"{mname}  (5 seed mean ± std)", fontsize=12, fontweight="bold")
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Spearman r\n(low / negative = better)", fontsize=10)
            ax.legend(fontsize=8, ncol=2, loc="lower right")
        elif ax_idx == 1:
            ax.set_ylabel("NDCG@10  (higher = better)", fontsize=10)
        else:
            ax.set_ylabel("Precision@10  (higher = better)", fontsize=10)

    plt.suptitle("A3 Leaveout: X1 PI-SDE vs Baselines  (true future prediction across 4 domains)",
                 fontsize=14, fontweight="bold", y=1.02)
    out_p = Path("RESULTS/fig6_a3_summary_4domain.png")
    plt.savefig(out_p, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_p}")


if __name__ == "__main__":
    main()
