"""diagnose_exit_hazard.py — Gate 0: is firm-CPC EXIT (mirror of entry) predictable at all?

Every entry/growth/novelty framing tried in this project (static/RNN/NeuralODE/PNODE/
Dual-Force/TAP-NODE/Jump-ODE-TPP, WHERE/HOW-MUCH/WHEN/WHAT-NEW) converges on the same
wall: structural/momentum baselines dominate, nothing beyond them survives honest
evaluation. A crude proxy of the mirror-image question ("does an actor stop appearing
in a technology it was in") already exists as Exit-AUC in trend_evaluation.py (author_topic
domain, 1-year-absence definition, near chance: static .489/NeuralODE .495/PNODE .516).
This script runs the PROPER version — firm-level, established tenure (K consecutive
years), H-year lapse, construction domain — as a training-free-only closing diagnostic
before any neural-model investment is considered. No GPU, no learned model.

Design (per Codex MCP Gate-0 sketch, 2026-08-21):
  - established pair (u,c): u filed in maingroup c in EVERY year of [Y-K+1, Y]
  - EXIT=1 iff u does NOT file in c in ANY year of (Y, Y+H]
  - domain_silent flag: u has ZERO filings **within this domain's CPC scope**
    anywhere in (Y, Y+H]. NOTE (corrected 2026-08-21 per Codex review): this is
    NOT true cross-domain firm death — the source CSV is already filtered to this
    domain's CPC codes (build_firm_bipartite.py), so a firm silent here may still
    be active in other technology areas entirely outside this dataset. Reported
    separately as a domain-specific competing-risk flag, not firm-wide silence.
  - baselines (all computed from data <= Y only): base rate, streak length
    (consecutive years active in c ending at Y), recent activity count (years
    active in c within [Y-4,Y]), firm-wide portfolio-size trend, global CPC-level
    momentum (trend in distinct-firm count active in c). NOTE: streak_length and
    recent_activity are highly correlated (Spearman ~0.9, printed at runtime) —
    treat as ONE "relationship persistence/tenure" signal family, not two
    independent discoveries, when counting how many things cleared the gate.
  - stop rule (pre-registered): no baseline clears AUC>=0.60 or PR-lift>=1.25x
    consistently across cutoffs -> record as closed, do not invest further.
  - CIs are firm-clustered bootstrap (resample firms with replacement, keep all
    their pairs) per Codex review 2026-08-21 — a per-pair bootstrap is optimistic
    since firms contribute ~2-3 correlated pairs each on average.

Run: python -m pnode_patent_runner.diagnose_exit_hazard --domain construction
     python -m pnode_patent_runner.diagnose_exit_hazard --domain agrifood   # cross-domain check (inventor-level)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def coarsen(code: str) -> str:
    return str(code).split("/")[0]


def load_firm(domain: str, y0: int, y1: int) -> tuple[pd.DataFrame, str]:
    """Prefer the firm-level file; fall back to the base (inventor-level) file for
    domains that don't have one, so cross-domain replication is still possible."""
    firm_path = ROOT / f"data/processed/bipartite_{domain}_firm.csv"
    base_path = ROOT / f"data/processed/bipartite_{domain}.csv"
    path, node_type = (firm_path, "firm") if firm_path.exists() else (base_path, "inventor")
    df = pd.read_csv(path, dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df.year >= y0) & (df.year <= y1)]
    df["i"] = df["i"].map(coarsen)
    return df[["year", "u", "i"]].dropna(subset=["i"]).drop_duplicates(), node_type


def bootstrap_ci_clustered(u, y_true, score, metric_fn, n_boot=300, seed=0):
    """Firm(entity)-clustered bootstrap: resample entities with replacement, keep
    all of each resampled entity's pairs. Per-pair bootstrap is optimistic since
    one entity contributes multiple correlated pairs."""
    rng = np.random.default_rng(seed)
    u = np.asarray(u)
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    unique_u = np.unique(u)
    idx_by_u = {uu: np.where(u == uu)[0] for uu in unique_u}
    n_entities = len(unique_u)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_u, n_entities, replace=True)
        idx = np.concatenate([idx_by_u[uu] for uu in sampled])
        yt, sc = y_true[idx], score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(metric_fn(yt, sc))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def run_cutoff(df, Y, K, H, min_lookback):
    """One rolling cutoff. Returns dict of baseline -> (auc, auc_lo, auc_hi, pr_lift)."""
    fy = defaultdict(dict)  # fy[u][year] = set(maingroups)
    for (u, y), g in df[df.year <= Y].groupby(["u", "year"]):
        fy[u][y] = set(g.i)

    firm_active_count = defaultdict(dict)  # firm_active_count[u][year] = n distinct maingroups
    for u, d in fy.items():
        for y, s in d.items():
            firm_active_count[u][y] = len(s)

    global_firms_per_c = defaultdict(dict)  # global_firms_per_c[c][year] = n distinct firms
    for y in range(Y - min_lookback + 1, Y + 1):
        yr_df = df[df.year == y]
        counts = yr_df.groupby("i")["u"].nunique()
        for c, n in counts.items():
            global_firms_per_c[c][y] = int(n)

    lookback_years = list(range(Y - K + 1, Y + 1))
    established = []
    for u, d in fy.items():
        active_years = set(d.keys())
        for c in d.get(Y, set()):
            if all((yr in active_years and c in d[yr]) for yr in lookback_years):
                established.append((u, c))

    future = df[(df.year > Y) & (df.year <= Y + H)]
    still_here = defaultdict(set)  # (u -> set of c filed again in horizon)
    active_in_domain = set()  # u with ANY filing in horizon, WITHIN this domain's CPC scope
    for u, g in future.groupby("u"):
        still_here[u] = set(g.i)
        active_in_domain.add(u)

    rows = []
    for u, c in established:
        exit_label = 0 if c in still_here.get(u, set()) else 1
        domain_silent = 0 if u in active_in_domain else 1

        streak = 0
        for yr in range(Y, Y - min_lookback, -1):
            if c in fy.get(u, {}).get(yr, set()):
                streak += 1
            else:
                break

        recent5 = [1 for yr in range(max(Y - 4, Y - min_lookback + 1), Y + 1)
                   if c in fy.get(u, {}).get(yr, set())]
        recent_activity = sum(recent5)

        sizes = [firm_active_count.get(u, {}).get(yr, np.nan) for yr in lookback_years]
        sizes = [s for s in sizes if not np.isnan(s)]
        firm_trend = (sizes[-1] - sizes[0]) if len(sizes) >= 2 else 0.0

        gcounts = [global_firms_per_c.get(c, {}).get(yr, np.nan)
                   for yr in range(Y - min_lookback + 1, Y + 1)]
        gcounts = [g for g in gcounts if not np.isnan(g)]
        global_momentum = (gcounts[-1] - gcounts[0]) if len(gcounts) >= 2 else 0.0

        rows.append(dict(u=u, c=c, exit=exit_label, domain_silent=domain_silent,
                          streak=streak, recent_activity=recent_activity,
                          firm_trend=firm_trend, global_momentum=global_momentum))

    out = pd.DataFrame(rows)
    return out


def calibration_table(sub: pd.DataFrame, col: str, max_bins: int = 8) -> list[dict]:
    """Observed exit rate per bin of a discrete/ordinal signal — a model-free
    'calibration' view (AUC/PR-lift show ranking, not this)."""
    vals = sub[col].to_numpy()
    y = sub["exit"].to_numpy()
    uniq = np.unique(vals)
    if len(uniq) > max_bins:
        # collapse the long tail into an open-ended top bin
        cut = uniq[max_bins - 1]
        binned = np.minimum(vals, cut)
        uniq = np.unique(binned)
    else:
        binned = vals
    rows = []
    for v in sorted(uniq):
        mask = binned == v
        n = int(mask.sum())
        if n == 0:
            continue
        label = f">={int(v)}" if len(np.unique(vals)) > max_bins and v == uniq[-1] else str(int(v))
        rows.append({col: label, "n": n, "exit_rate": float(y[mask].mean())})
    return rows


def evaluate(out: pd.DataFrame, exclude_domain_silent: bool):
    sub = out[out.domain_silent == 0] if exclude_domain_silent else out
    y = sub["exit"].to_numpy()
    n_pos = int(y.sum())
    if n_pos < 10 or n_pos > len(y) - 10:
        return None
    prevalence = y.mean()
    u = sub["u"].to_numpy()
    results = {"n": int(len(sub)), "n_exit": n_pos, "prevalence": float(prevalence)}
    corr = float(sub["streak"].corr(sub["recent_activity"], method="spearman"))
    results["streak_recent_activity_spearman"] = corr
    results["calibration_streak"] = calibration_table(sub, "streak")
    baselines = {
        "streak_length_inv": -sub["streak"].to_numpy(),       # longer streak -> lower exit risk expected
        "recent_activity_inv": -sub["recent_activity"].to_numpy(),
        "firm_trend_inv": -sub["firm_trend"].to_numpy(),      # shrinking firm portfolio -> higher exit risk
        "global_momentum_inv": -sub["global_momentum"].to_numpy(),  # declining field -> higher exit risk
    }
    for name, score in baselines.items():
        try:
            auc = roc_auc_score(y, score)
        except ValueError:
            continue
        ap = average_precision_score(y, score)
        pr_lift = ap / prevalence if prevalence > 0 else float("nan")
        lo, hi = bootstrap_ci_clustered(u, y, score, roc_auc_score)
        results[name] = {"auc": float(auc), "auc_ci95": [lo, hi], "pr_lift": float(pr_lift)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--y0", type=int, default=2005)
    ap.add_argument("--y1", type=int, default=2021)  # avoid publication-lag-contaminated 2022-2025 tail
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[2015, 2016, 2017])
    ap.add_argument("--min-lookback", type=int, default=6)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    df, node_type = load_firm(args.domain, args.y0, args.y1)
    print(f"[{args.domain}] node_type={node_type} rows={len(df)} entities={df.u.nunique()} "
          f"maingroups={df.i.nunique()} years={df.year.min()}-{df.year.max()}")

    all_results = {}
    for H in args.horizons:
        for Y in args.cutoffs:
            if Y + H > args.y1:
                continue
            out = run_cutoff(df, Y, args.K, H, args.min_lookback)
            key = f"H{H}_Y{Y}"
            r_incl = evaluate(out, exclude_domain_silent=False)
            r_excl = evaluate(out, exclude_domain_silent=True)
            all_results[key] = {"including_domain_silent": r_incl, "excluding_domain_silent": r_excl}
            print(f"\n=== K={args.K} H={H} Y={Y} ===")
            for label, r in [("incl. domain-silent", r_incl), ("excl. domain-silent", r_excl)]:
                if r is None:
                    print(f"  [{label}] insufficient positives/negatives, skipped")
                    continue
                print(f"  [{label}] n={r['n']} n_exit={r['n_exit']} prevalence={r['prevalence']:.3f} "
                      f"streak~recent_activity Spearman={r['streak_recent_activity_spearman']:.3f}")
                for name, m in r.items():
                    if not isinstance(m, dict):
                        continue
                    print(f"    {name:<22} AUC={m['auc']:.3f} (95% CI {m['auc_ci95'][0]:.3f}-{m['auc_ci95'][1]:.3f})  "
                          f"PR-lift={m['pr_lift']:.2f}x")
                if label == "incl. domain-silent":
                    print("    calibration (streak -> observed exit rate): " +
                          ", ".join(f"{row['streak']}:{row['exit_rate']:.1%}(n={row['n']})"
                                    for row in r["calibration_streak"]))

    # streak_length_inv and recent_activity_inv are ~one signal family (Spearman ~0.9);
    # dedupe them when counting how many INDEPENDENT things cleared the gate.
    signal_family = {"streak_length_inv": "persistence_family", "recent_activity_inv": "persistence_family"}
    clears = []
    for key, res in all_results.items():
        for scope in ("including_domain_silent", "excluding_domain_silent"):
            r = res[scope]
            if r is None:
                continue
            for name, m in r.items():
                if not isinstance(m, dict):
                    continue
                if m["auc"] >= 0.60 or m["pr_lift"] >= 1.25:
                    clears.append((key, scope, signal_family.get(name, name), m["auc"], m["pr_lift"]))
    independent_families_cleared = {c[2] for c in clears}

    print("\n=== PRE-REGISTERED STOP-RULE VERDICT ===")
    if clears:
        print(f"  {len(clears)} (cutoff, baseline) combinations cleared AUC>=0.60 or PR-lift>=1.25x, "
              f"spanning {len(independent_families_cleared)} independent signal familie(s): "
              f"{sorted(independent_families_cleared)}")
        for c in clears:
            print(f"    {c}")
        print("  => GATE PASSES for at least some cells; worth a closer look before writing off.")
    else:
        print("  No baseline cleared AUC>=0.60 or PR-lift>=1.25x in any cutoff/scope.")
        print("  => STOP. Firm-CPC exit is not predictable from this bipartite structure alone "
              "at this K/H; record as a closed cell in the predictability map (mirrors Exit-AUC "
              "near-chance result in trend_evaluation.py, now confirmed with a proper firm-level, "
              "established-tenure, multi-baseline test).")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"domain": args.domain, "node_type": node_type, "K": args.K, "results": all_results,
                       "stop_rule_clears": clears,
                       "independent_signal_families_cleared": sorted(independent_families_cleared)},
                      f, ensure_ascii=False, indent=2)
        print(f"\nWrote: {args.output_json}")


if __name__ == "__main__":
    main()
