"""
Chronos-Bolt zero-shot baseline for topic growth forecasting.

Same protocol as X4-predict (run_pisde_x4_predict.py):
  - train_t = first floor(T_TRAIN_FRAC * T) timepoints (used only as context)
  - test_t  = remaining timepoints, predicted 1..K steps ahead
  - K=50 topic-wise univariate series, length T (=12 for patent_energy)
  - Metrics per test_t: Spearman ρ vs growth_raw, NDCG@10, MSE/MAE

Also computes 3 naive baselines for context:
  - persistence: y_pred(t) = growth(t-1)
  - mean:        y_pred(t) = mean(growth[train_t])
  - linear:      OLS linear fit on train_t, extrapolate

Run with the tsfm venv:
  .tsfm_venv/bin/python baseline_chronos.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

# ── Paths / Hyperparameters ─────────────────────────────────────────
ROOT = Path("/home/nakamuraroi/kumagai")
os.chdir(ROOT)

DOMAIN = os.environ.get("PNODE_DOMAIN_TARGET", "patent_energy_top50")
DOMAIN_MAP = {
    "paper":              ("PNode_Paper_X1",                  "data/PNode_Paper_X1"),
    "patent_energy_top50":("PNode_Patent_Energy_X1_top50",    "data/PNode_Patent_Energy_X1_top50"),
    "arxiv_construction": ("PNode_ArXiv_Construction_X1_v2",  "data/PNode_ArXiv_Construction_X1_v2"),
    "jp_construction":    ("PNode_JP_Construction_X1",        "data/PNode_JP_Construction_X1"),
}
DATA_NAME, DATA_DIR = DOMAIN_MAP[DOMAIN]

T_TRAIN_FRAC = float(os.environ.get("PNODE_T_TRAIN_FRAC", 0.7))
MODEL_NAME   = os.environ.get("CHRONOS_MODEL", "amazon/chronos-bolt-small")
NUM_SAMPLES  = int(os.environ.get("CHRONOS_NUM_SAMPLES", 20))
SEED         = int(os.environ.get("PNODE_SEED", 42))

OUT_DIR = ROOT / f"RESULTS_TSFM_BASELINE/{DATA_NAME}/chronos_bolt/split{int(T_TRAIN_FRAC*100)}/seed_{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────
def load_growth_matrix():
    data = torch.load(f"{DATA_DIR}/alltime/fate_train.pt", weights_only=False)
    g_raw  = torch.stack([data["growth"][t]      for t in range(len(data["growth"]))])
    g_norm = torch.stack([data["growth_norm"][t] for t in range(len(data["growth_norm"]))])
    # First timepoint t=0 is the "before any change" anchor (all zeros). Drop it
    # so the actual series runs from t=1 to t=T_data; this matches X4 (train_t starts at 1).
    return g_raw.numpy(), g_norm.numpy(), data.get("topic_names")


def make_temporal_split(T_data: int, frac: float):
    T_train = max(1, int(round(frac * T_data)))
    T_train = min(T_train, T_data - 1)
    train_t = list(range(1, T_train + 1))
    test_t = list(range(T_train + 1, T_data + 1))
    return train_t, test_t


# ── Metrics ────────────────────────────────────────────────────────
def metrics_one_t(g_pred: np.ndarray, g_true_raw: np.ndarray, g_true_norm: np.ndarray, k_ndcg: int = 10):
    """Spearman ρ, NDCG@K, prec@K (raw scale) + MSE/MAE on both scales."""
    r, p = stats.spearmanr(g_pred, g_true_raw)
    K = min(k_ndcg, len(g_true_raw))
    rel = np.maximum(g_true_raw, 0.0)
    ideal = np.argsort(-rel)
    idcg = sum(rel[ideal[k]] / np.log2(k + 2) for k in range(K))
    order_pred = np.argsort(-g_pred)
    ndcg = sum(rel[order_pred[k]] / np.log2(k + 2) for k in range(K)) / (idcg + 1e-10)
    prec = (g_true_raw[order_pred[:K]] > 0).mean()
    return {
        "spearman_r":    float(r),
        "spearman_p":    float(p),
        "ndcg":          float(ndcg),
        "prec_at_10":    float(prec),
        "mse_raw":       float(((g_pred - g_true_raw) ** 2).mean()),
        "mae_raw":       float(np.abs(g_pred - g_true_raw).mean()),
        "mse_norm":      float(((g_pred - g_true_norm) ** 2).mean()),
        "mae_norm":      float(np.abs(g_pred - g_true_norm).mean()),
    }


# ── Naive baselines ────────────────────────────────────────────────
def predict_persistence(g_raw, train_t, test_t):
    """y_pred(t) = y(t-1).  For t=test_t[0], uses last train; subsequent uses ground-truth previous (oracle-ish)
    To be a strict forecaster, use the last train value for ALL test t. Provide both."""
    last_train = g_raw[train_t[-1] - 1]  # train_t are 1-indexed in our convention but g_raw is 0-indexed
    # Wait — the y index: y = [0,1,...,T_data], and g_raw has T_data+1 rows.
    # In our case, g_raw[t] corresponds to time index t.  train_t = [1..8] are time indices, so g_raw[8] is the last train.
    preds = {t: g_raw[train_t[-1]] for t in test_t}
    return preds


def predict_mean(g_raw, train_t, test_t):
    mean_g = g_raw[train_t].mean(axis=0)   # (K,)
    return {t: mean_g for t in test_t}


def predict_linear(g_raw, train_t, test_t):
    """OLS linear fit on train_t (per topic), extrapolate to test_t."""
    train_arr = np.array(train_t, dtype=np.float64)
    K = g_raw.shape[1]
    preds = {}
    for t in test_t:
        out = np.zeros(K)
        for k in range(K):
            y = g_raw[train_t, k]
            slope, intercept, *_ = stats.linregress(train_arr, y)
            out[k] = slope * float(t) + intercept
        preds[t] = out
    return preds


# ── Chronos zero-shot ──────────────────────────────────────────────
def predict_chronos(g_raw, train_t, test_t, model_name=MODEL_NAME, num_samples=NUM_SAMPLES):
    """Univariate per-topic zero-shot. Context = g_raw[train_t], horizon = max(test_t) - train_t[-1].
    Returns: {t_test: (K,)} median forecast.
    """
    from chronos import ChronosBoltPipeline

    print(f"  loading {model_name} ...")
    pipe = ChronosBoltPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    horizon = max(test_t) - train_t[-1]  # e.g., test=[9,10,11], train_last=8 → horizon=3
    K = g_raw.shape[1]
    contexts = [torch.tensor(g_raw[train_t, k], dtype=torch.float32) for k in range(K)]  # K x context_len

    print(f"  predicting K={K} series, horizon={horizon} ...")
    # Bolt's predict_quantiles returns quantiles directly (deterministic for a given input)
    quantiles, mean = pipe.predict_quantiles(
        inputs=contexts,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    # quantiles: (K, horizon, 3), mean: (K, horizon)
    median = quantiles[:, :, 1].cpu().numpy()  # (K, horizon)

    preds = {}
    for i, t in enumerate(test_t):
        step = t - train_t[-1] - 1  # 0-indexed offset into horizon
        preds[t] = median[:, step]
    return preds


# ── Main ───────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)

    g_raw, g_norm, topic_names = load_growth_matrix()
    T_data = g_raw.shape[0] - 1   # convention: y has T_data+1 entries, t=0 anchor
    train_t, test_t = make_temporal_split(T_data, T_TRAIN_FRAC)

    print("=" * 78)
    print(f"  Chronos-Bolt baseline — {DATA_NAME}")
    print(f"  T={T_data+1} timepoints, K={g_raw.shape[1]} topics")
    print(f"  train_t = {train_t}")
    print(f"  test_t  = {test_t}")
    print(f"  model   = {MODEL_NAME}")
    print(f"  out_dir = {OUT_DIR}")
    print("=" * 78)

    methods = {
        "persistence": predict_persistence(g_raw, train_t, test_t),
        "mean":        predict_mean(g_raw, train_t, test_t),
        "linear":      predict_linear(g_raw, train_t, test_t),
        "chronos":     predict_chronos(g_raw, train_t, test_t),
    }

    all_results = {}
    print(f"\n  {'method':<12} {'t':<3} {'K':<3} {'Sp ρ':<14} {'NDCG':<8} {'P@10':<6} "
          f"{'MSE(raw)':<10} {'MSE(norm)':<10}")
    print("  " + "-" * 78)
    for name, preds in methods.items():
        all_results[name] = []
        for t_eval in test_t:
            K_ahead = t_eval - train_t[-1]
            m = metrics_one_t(preds[t_eval], g_raw[t_eval], g_norm[t_eval])
            m.update({"t_eval": int(t_eval), "K_ahead": int(K_ahead)})
            all_results[name].append(m)
            sig = "*" if m["spearman_p"] < 0.05 else " "
            print(f"  {name:<12} {t_eval:<3} {K_ahead:<3} "
                  f"{m['spearman_r']:+.3f}{sig:<9} {m['ndcg']:<8.3f} {m['prec_at_10']:<6.2f} "
                  f"{m['mse_raw']:<10.4f} {m['mse_norm']:<10.4f}")
        # mean across test_t
        mr = np.mean([r["spearman_r"] for r in all_results[name]])
        mn = np.mean([r["ndcg"] for r in all_results[name]])
        print(f"  {name:<12} mean ρ={mr:+.3f}  mean NDCG={mn:.3f}")
        print()

    out_file = OUT_DIR / "evaluation.json"
    json.dump({
        "domain": DOMAIN, "data_name": DATA_NAME,
        "train_t": train_t, "test_t": test_t,
        "seed": SEED, "model_name": MODEL_NAME,
        "results": all_results,
    }, out_file.open("w"), indent=2)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
