#!/usr/bin/env python3
"""
仮説 H2–H5 の検定を一括実行する。

H1（PNODE vs Neural ODE, k=2）は `aggregate_benchmark_seeds --paired-pnode-vs neural_ode`
で得られるため、ここでは **ペア差の二次処理が必要な H2–H5** を扱う。

入力: `run_benchmark_comparison --eval-horizon-gaps 1,2,3` で得た 5 シード分の JSON。

例:
  python -m pnode_patent_runner.evaluate_hypothesis_h2_h5 \
    --glob "pnode_patent_runner/outputs/hypothesis_long_horizon/benchmark_author_topic_seed*.json" \
    --markdown

検証計画: docs/HYPOTHESIS_LONG_HORIZON_VERIFICATION.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from glob import glob as globfn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Tuple[int, List[Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    seed = int(raw.get("seed", -1))
    rows = raw.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"results が無い: {path}")
    return seed, rows


def _horizon_auc(row: Dict[str, Any], k: int, split: str = "final") -> float:
    block_key = (
        "final_metrics_by_horizon_gap"
        if split == "final"
        else "train_split_metrics_by_horizon_gap"
    )
    block = row.get(block_key)
    if not isinstance(block, dict):
        return float("nan")
    cell = block.get(str(k))
    if not isinstance(cell, dict):
        return float("nan")
    v = cell.get("auc")
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return float("nan")


def _extract_method_horizon(
    paths: List[Path], method: str, k: int, split: str = "final",
) -> List[float]:
    vals: List[float] = []
    for p in paths:
        _, rows = _load_json(p)
        found = False
        for row in rows:
            if str(row.get("key", "")) == method:
                vals.append(_horizon_auc(row, k, split))
                found = True
                break
        if not found:
            vals.append(float("nan"))
    return vals


def _wilcoxon_one_sided(
    a: List[float], b: List[float], alt: str = "greater",
) -> Tuple[Optional[float], str]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None, "scipy 未インストール"
    aa = [x for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    bb = [y for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    n = len(aa)
    if n < 3:
        return None, f"有効ペア n={n} < 3"
    res = wilcoxon(aa, bb, alternative=alt, zero_method="wilcox")
    return float(res.pvalue), ""


def _sign_count(vals: List[float]) -> Tuple[int, int, int]:
    pos = sum(1 for v in vals if not math.isnan(v) and v > 0)
    neg = sum(1 for v in vals if not math.isnan(v) and v < 0)
    zero = sum(1 for v in vals if not math.isnan(v) and v == 0)
    return pos, neg, zero


def _median(vals: List[float]) -> float:
    clean = sorted(v for v in vals if not math.isnan(v))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2 == 0:
        return (clean[mid - 1] + clean[mid]) / 2
    return clean[mid]


def _mean(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _print_hypothesis(
    name: str,
    desc: str,
    diffs: List[float],
    direction: str,
    p_val: Optional[float],
    err: str,
    threshold: float,
    md: bool,
) -> str:
    """判定を表示し、 'PASS' / 'FAIL' / 'INCONCLUSIVE' を返す。"""
    med = _median(diffs)
    avg = _mean(diffs)
    pos, neg, zero = _sign_count(diffs)
    n_valid = pos + neg + zero

    verdict = "INCONCLUSIVE"
    if p_val is not None and p_val < 0.05 and abs(med) >= threshold:
        if direction == "greater" and pos > n_valid / 2:
            verdict = "PASS"
        elif direction == "less" and neg > n_valid / 2:
            verdict = "PASS"
        else:
            verdict = "FAIL"
    elif p_val is not None and p_val >= 0.05:
        verdict = "FAIL"
    elif err:
        verdict = "INCONCLUSIVE"

    tag = {"PASS": "**PASS**", "FAIL": "**FAIL**", "INCONCLUSIVE": "INCONCLUSIVE"}[verdict]

    if md:
        print(f"\n### {name}: {desc}\n")
        print(f"| 指標 | 値 |")
        print(f"|------|----|")
        print(f"| ペア差 (各シード) | {[round(d, 4) for d in diffs]} |")
        print(f"| 中央値 | {med:.4f} |")
        print(f"| 平均 | {avg:.4f} |")
        print(f"| 符号 (+/−/0) | {pos}/{neg}/{zero} |")
        if p_val is not None:
            print(f"| Wilcoxon p ({direction}) | {p_val:.6f} |")
        else:
            print(f"| Wilcoxon | {err} |")
        print(f"| 閾値 | {threshold} |")
        print(f"| **判定** | {tag} |")
    else:
        print(f"\n[{name}] {desc}")
        print(f"  diffs={[round(d, 4) for d in diffs]}")
        print(f"  median={med:.4f}  mean={avg:.4f}  sign=+{pos}/-{neg}/0:{zero}")
        if p_val is not None:
            print(f"  Wilcoxon p ({direction}) = {p_val:.6f}")
        else:
            print(f"  Wilcoxon: {err}")
        print(f"  verdict: {verdict}")

    return verdict


def main() -> None:
    p = argparse.ArgumentParser(
        description="H2–H5 仮説検定（docs/HYPOTHESIS_LONG_HORIZON_VERIFICATION.md）",
    )
    p.add_argument("json_paths", nargs="*", type=Path)
    p.add_argument("--glob", type=str, default="")
    p.add_argument("--horizon-split", type=str, default="final",
                    choices=("final", "train"))
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    paths: List[Path] = list(args.json_paths or [])
    if args.glob.strip():
        for m in sorted(globfn(args.glob.strip())):
            pp = Path(m)
            if pp not in paths:
                paths.append(pp)
    if not paths:
        raise SystemExit("JSON が 0 本。--glob またはパスを指定してください。")
    paths.sort()
    print(f"JSON: {len(paths)} 本  split={args.horizon_split}")
    for pp in paths:
        print(f"  {pp}")

    md = args.markdown
    sp = args.horizon_split

    pnode_k1 = _extract_method_horizon(paths, "pnode", 1, sp)
    pnode_k2 = _extract_method_horizon(paths, "pnode", 2, sp)
    node_k1 = _extract_method_horizon(paths, "neural_ode", 1, sp)
    node_k2 = _extract_method_horizon(paths, "neural_ode", 2, sp)
    rnn_k1 = _extract_method_horizon(paths, "rnn", 1, sp)
    rnn_k2 = _extract_method_horizon(paths, "rnn", 2, sp)
    static_k1 = _extract_method_horizon(paths, "static", 1, sp)
    static_k2 = _extract_method_horizon(paths, "static", 2, sp)

    if md:
        print("\n## 仮説検定結果\n")

    verdicts: Dict[str, str] = {}

    # ----------------------------------------------------------------
    # H2: 劣化率 R = AUC(k=1) - AUC(k=2) について R_pnode < R_neural_ode
    # ----------------------------------------------------------------
    r_pnode = [a - b for a, b in zip(pnode_k1, pnode_k2)]
    r_node = [a - b for a, b in zip(node_k1, node_k2)]
    h2_diff = [rp - rn for rp, rn in zip(r_pnode, r_node)]
    pv, err = _wilcoxon_one_sided(r_node, r_pnode, alt="greater")
    verdicts["H2"] = _print_hypothesis(
        "H2", "劣化率: R_pnode − R_neural_ode < 0",
        h2_diff, "less", pv, err, threshold=0.0, md=md,
    )

    # ----------------------------------------------------------------
    # H3a: 潜在 MSE  (aggregate_benchmark_seeds では扱えないため参考表示)
    # ----------------------------------------------------------------
    if md:
        print("\n### H3a / H3b: 潜在ロールアウト MSE・方向一致率\n")
        print("> `final_latent_metrics_by_horizon_gap` の値を JSON から直接確認してください。")
        print("> 本スクリプトでは future-link AUC ベースの H2/H4/H5 のみ自動検定します。\n")

    # ----------------------------------------------------------------
    # H4: Gap(k=2) > Gap(k=1)  where Gap = AUC_pnode - AUC_rnn
    # ----------------------------------------------------------------
    gap_k1 = [p - r for p, r in zip(pnode_k1, rnn_k1)]
    gap_k2 = [p - r for p, r in zip(pnode_k2, rnn_k2)]
    h4_diff = [g2 - g1 for g2, g1 in zip(gap_k2, gap_k1)]
    pv4, err4 = _wilcoxon_one_sided(gap_k2, gap_k1, alt="greater")
    verdicts["H4"] = _print_hypothesis(
        "H4", "RNNとの差が k 増加で改善: Gap(k=2) − Gap(k=1) > 0",
        h4_diff, "greater", pv4, err4, threshold=0.0, md=md,
    )

    # ----------------------------------------------------------------
    # H5: Margin(k=2) > Margin(k=1)  where Margin = AUC_pnode - AUC_static
    # ----------------------------------------------------------------
    margin_k1 = [p - s for p, s in zip(pnode_k1, static_k1)]
    margin_k2 = [p - s for p, s in zip(pnode_k2, static_k2)]
    h5_diff = [m2 - m1 for m2, m1 in zip(margin_k2, margin_k1)]
    pv5, err5 = _wilcoxon_one_sided(margin_k2, margin_k1, alt="greater")
    verdicts["H5"] = _print_hypothesis(
        "H5", "Static とのマージン拡大: Margin(k=2) − Margin(k=1) > 0",
        h5_diff, "greater", pv5, err5, threshold=0.0, md=md,
    )

    # ----------------------------------------------------------------
    # サマリ
    # ----------------------------------------------------------------
    if md:
        print("\n## サマリ\n")
        print("| 仮説 | 判定 |")
        print("|------|------|")
        for h in ("H2", "H4", "H5"):
            print(f"| {h} | {verdicts[h]} |")
        print()
        print("> H1 は `aggregate_benchmark_seeds --paired-pnode-vs neural_ode` で確認。")
        print("> H3 は `final_latent_metrics_by_horizon_gap` を JSON から直接確認。")
    else:
        print("\n--- サマリ ---")
        for h, v in verdicts.items():
            print(f"  {h}: {v}")
        print("  H1: aggregate_benchmark_seeds --paired-pnode-vs neural_ode で確認")
        print("  H3: final_latent_metrics_by_horizon_gap を JSON で確認")


if __name__ == "__main__":
    main()
