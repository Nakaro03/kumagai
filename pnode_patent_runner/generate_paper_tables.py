"""
論文用 LaTeX 表生成: A1 ablation, A3 leaveout 4 domain.
出力: RESULTS/paper_tables.tex
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

OUT = Path("RESULTS/paper_tables.tex")

DOMAINS_A1 = [
    ("Paper",            "PNode_Paper_X1",                 3),
    ("Patent Energy",    "PNode_Patent_Energy_X1_top50",   11),
    ("arXiv Const.",     "PNode_ArXiv_Construction_X1_v2", 10),
]
SETTINGS = [("A (vanilla)", "0.0", "0.0", "0.0"),
            ("B (val)",     "1.0", "0.0", "0.0"),
            ("C (val+grad)","1.0", "0.1", "0.0"),
            ("D (full X1)", "1.0", "0.1", "0.01")]
SEEDS = [0, 1, 42, 123, 999]


def load_a1(root, seed, lv, lg, lb):
    suffix = f"-x1_v{lv}_g{lg}_b{lb}"
    for p in Path(f"RESULTS/{root}").rglob("evaluation_x1.json"):
        tag = p.parents[2].name if len(p.parents) >= 3 else ""
        if not tag.endswith(suffix): continue
        if f"seed_{seed}" not in str(p): continue
        if "/alltime/" not in str(p): continue
        return json.load(p.open())
    return None


def fmt(vals):
    if len(vals) == 0: return "---"
    m, s = np.mean(vals), np.std(vals)
    return f"${m:+.3f} \\pm {s:.3f}$"


def build_a1_table():
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation of $\mathcal{L}_{X1}$ components (5-seed alltime, Spearman at last $t$, lower / more negative is better; $^*$ Wilcoxon $p < 0.05$).}",
        r"\label{tab:a1_ablation}",
        r"\begin{tabular}{l" + "c" * len(DOMAINS_A1) + "}",
        r"\toprule",
        r"Setting & " + " & ".join(d[0] for d in DOMAINS_A1) + r" \\",
        r"\midrule",
    ]
    for label, lv, lg, lb in SETTINGS:
        cells = []
        for dn, root, last_t in DOMAINS_A1:
            vals = []
            for s in SEEDS:
                d = load_a1(root, s, lv, lg, lb)
                if d is None: continue
                r = next((r for r in d["results"] if r["t"] == last_t), None)
                if r is None: continue
                vals.append(r["spearman_r"])
            cells.append(fmt(vals))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


DOMAINS_A3 = [
    ("Paper",            "PNode_Paper_X1",                 "paper",                3),
    ("Patent Energy",    "PNode_Patent_Energy_X1_top50",   "patent_energy_top50",  11),
    ("arXiv Const.",     "PNode_ArXiv_Construction_X1_v2", "arxiv_construction",   10),
    ("JP Const.",        "PNode_JP_Construction_X1",       "jp_construction",      10),
]
BASELINES = ["Naive_lastg", "Linear", "ARIMA", "LSTM", "Transformer", "DLinear", "PatchTST"]


def gather_baseline(dkey, method, metric):
    if dkey is None: return []
    vals = []
    for s in SEEDS:
        p = Path(f"RESULTS/baselines/{dkey}/baselines_seed{s}.json")
        if not p.exists(): continue
        d = json.load(p.open())
        if method in d and d[method].get(metric) == d[method].get(metric):
            vals.append(d[method][metric])
    return vals


def gather_x1_leaveout(root, last_t):
    vals_sp, vals_p10 = [], []
    for s in SEEDS:
        for p in Path(f"RESULTS/{root}").rglob("evaluation_x1.json"):
            tag = p.parents[2].name if len(p.parents) >= 3 else ""
            if not tag.endswith("-x1_v1.0_g0.1_b0.01"): continue
            if f"seed_{s}" not in str(p): continue
            if f"/leaveout{last_t}/" not in str(p): continue
            d = json.load(p.open())
            r = next((r for r in d["results"] if r["t"] == last_t and r.get("split") == "test"), None)
            if r is None: continue
            vals_sp.append(r["spearman_r"])
            vals_p10.append(r["prec_at_10"])
            break
    return np.array(vals_sp), np.array(vals_p10)


def build_a3_table():
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{True future prediction (leaveout) across 4 domains, 5-seed Spearman $r$ (lower / more negative = better) at held-out last $t$. Wilcoxon $p$ tests one-sided $r<0$.}",
        r"\label{tab:a3_leaveout}",
        r"\begin{tabular}{l" + "c" * len(DOMAINS_A3) + "}",
        r"\toprule",
        r"Method & " + " & ".join(d[0] for d in DOMAINS_A3) + r" \\",
        r"\midrule",
    ]
    for m in BASELINES:
        cells = []
        for dname, root, dkey, last_t in DOMAINS_A3:
            vals = gather_baseline(dkey, m, "spearman_r")
            cells.append(fmt(vals))
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = []
    for dname, root, dkey, last_t in DOMAINS_A3:
        sp, _ = gather_x1_leaveout(root, last_t)
        if len(sp) >= 5:
            cell = fmt(sp) + r"$^*$"
        elif len(sp) > 0:
            cell = fmt(sp) + f" ({len(sp)} seed)"
        else:
            cell = "---"
        cells.append(cell)
    lines.append(r"\textbf{X1 PI-SDE (ours)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    out = []
    out.append("% =========== Table 1: A1 ablation ===========")
    out.append(build_a1_table())
    out.append("\n% =========== Table 2: A3 leaveout (4 domains) ===========")
    out.append(build_a3_table())
    text = "\n\n".join(out)
    OUT.write_text(text)
    print(text)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
