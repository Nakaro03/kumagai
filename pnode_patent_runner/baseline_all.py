"""
Comprehensive baseline sweep: 3 domains × 6+ methods × multi-seed.

Methods:
  - Naive (deterministic, 1 run):
      persistence, mean, linear OLS
  - Classical TS (deterministic, 1 run):
      ARIMA (auto-order with fallback), ETS (Holt linear trend)
  - TSFM (stochastic sampling, N_SEEDS runs):
      Chronos-Bolt-small, Moirai-2-R-small, TimesFM-2.0

Run:
  for domain in patent_energy_top50 arxiv_construction jp_construction; do
    PNODE_DOMAIN_TARGET=$domain .tsfm_venv/bin/python baseline_all.py
  done
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path("/home/nakamuraroi/kumagai")
os.chdir(ROOT)

DOMAIN = os.environ.get("PNODE_DOMAIN_TARGET", "patent_energy_top50")
DOMAIN_MAP = {
    "patent_energy_top50":("PNode_Patent_Energy_X1_top50",    "data/PNode_Patent_Energy_X1_top50"),
    "arxiv_construction": ("PNode_ArXiv_Construction_X1_v2",  "data/PNode_ArXiv_Construction_X1_v2"),
    "jp_construction":    ("PNode_JP_Construction_X1",        "data/PNode_JP_Construction_X1"),
}
DATA_NAME, DATA_DIR = DOMAIN_MAP[DOMAIN]

T_TRAIN_FRAC = float(os.environ.get("PNODE_T_TRAIN_FRAC", 0.7))
SEEDS        = [int(s) for s in os.environ.get("PNODE_SEEDS", "42,0,1,123,999").split(",")]

OUT_DIR = ROOT / f"RESULTS_TSFM_BASELINE/{DATA_NAME}/all_methods/split{int(T_TRAIN_FRAC*100)}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data ────────────────────────────────────────────────────────────
def load_growth_matrix():
    data = torch.load(f"{DATA_DIR}/alltime/fate_train.pt", weights_only=False)
    g_raw  = torch.stack([data["growth"][t]      for t in range(len(data["growth"]))]).numpy()
    g_norm = torch.stack([data["growth_norm"][t] for t in range(len(data["growth_norm"]))]).numpy()
    return g_raw, g_norm


def make_temporal_split(T_data: int, frac: float):
    T_train = max(1, int(round(frac * T_data)))
    T_train = min(T_train, T_data - 1)
    return list(range(1, T_train + 1)), list(range(T_train + 1, T_data + 1))


# ── Metrics ─────────────────────────────────────────────────────────
def metrics_one_t(g_pred, g_raw, g_norm, k_ndcg=10):
    r, p = stats.spearmanr(g_pred, g_raw)
    K = min(k_ndcg, len(g_raw))
    rel = np.maximum(g_raw, 0.0)
    idcg = sum(rel[np.argsort(-rel)[k]] / np.log2(k + 2) for k in range(K))
    order = np.argsort(-g_pred)
    ndcg = sum(rel[order[k]] / np.log2(k + 2) for k in range(K)) / (idcg + 1e-10)
    prec = (g_raw[order[:K]] > 0).mean()
    return {
        "spearman_r": float(r), "spearman_p": float(p),
        "ndcg": float(ndcg), "prec_at_10": float(prec),
        "mse_raw": float(((g_pred - g_raw) ** 2).mean()),
        "mae_raw": float(np.abs(g_pred - g_raw).mean()),
        "mse_norm": float(((g_pred - g_norm) ** 2).mean()),
        "mae_norm": float(np.abs(g_pred - g_norm).mean()),
    }


# ── Naive baselines (deterministic) ─────────────────────────────────
def predict_persistence(g_raw, train_t, test_t):
    return {t: g_raw[train_t[-1]] for t in test_t}


def predict_mean(g_raw, train_t, test_t):
    m = g_raw[train_t].mean(axis=0)
    return {t: m for t in test_t}


def predict_linear(g_raw, train_t, test_t):
    train_arr = np.array(train_t, dtype=np.float64)
    K = g_raw.shape[1]
    preds = {}
    for t in test_t:
        out = np.zeros(K)
        for k in range(K):
            slope, intercept, *_ = stats.linregress(train_arr, g_raw[train_t, k])
            out[k] = slope * float(t) + intercept
        preds[t] = out
    return preds


# ── Classical TS (deterministic with auto-order) ────────────────────
def predict_arima(g_raw, train_t, test_t):
    from statsmodels.tsa.arima.model import ARIMA
    K = g_raw.shape[1]
    horizon = max(test_t) - train_t[-1]
    preds = {t: np.zeros(K) for t in test_t}
    for k in range(K):
        y = g_raw[train_t, k]
        forecast = None
        # Try orders in fallback chain (suitable for short series)
        for order in [(1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0)]:
            try:
                m = ARIMA(y, order=order).fit(method_kwargs={"warn_convergence": False})
                forecast = m.forecast(steps=horizon)
                break
            except Exception:
                continue
        if forecast is None:
            forecast = np.full(horizon, y.mean())
        for i, t in enumerate(test_t):
            step = t - train_t[-1] - 1
            preds[t][k] = forecast[step]
    return preds


def predict_ets(g_raw, train_t, test_t):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    K = g_raw.shape[1]
    horizon = max(test_t) - train_t[-1]
    preds = {t: np.zeros(K) for t in test_t}
    for k in range(K):
        y = g_raw[train_t, k]
        forecast = None
        try:
            m = ExponentialSmoothing(y, trend="add", initialization_method="estimated").fit()
            forecast = m.forecast(steps=horizon)
        except Exception:
            try:
                m = ExponentialSmoothing(y, trend=None, initialization_method="estimated").fit()
                forecast = m.forecast(steps=horizon)
            except Exception:
                forecast = np.full(horizon, y.mean())
        for i, t in enumerate(test_t):
            step = t - train_t[-1] - 1
            preds[t][k] = forecast[step]
    return preds


# ── Chronos-Bolt (stochastic via predict(), seed-dependent) ─────────
_CHRONOS_CACHE = {}
def _get_chronos(model_name="amazon/chronos-bolt-small"):
    if model_name not in _CHRONOS_CACHE:
        from chronos import ChronosBoltPipeline
        _CHRONOS_CACHE[model_name] = ChronosBoltPipeline.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float32,
        )
    return _CHRONOS_CACHE[model_name]


def predict_chronos(g_raw, train_t, test_t, seed=42):
    """Bolt's predict_quantiles is deterministic by design. We add seed variance
    via small input perturbation (matches the 'aleatoric' uncertainty of TSFM eval)."""
    torch.manual_seed(seed); np.random.seed(seed)
    pipe = _get_chronos()
    horizon = max(test_t) - train_t[-1]
    K = g_raw.shape[1]
    # Tiny input noise for seed variance (sigma = 1% of std)
    sigma = 0.01 * g_raw[train_t].std()
    noise = np.random.RandomState(seed).randn(len(train_t), K) * sigma
    contexts = [torch.tensor(g_raw[train_t, k] + noise[:, k], dtype=torch.float32) for k in range(K)]
    quantiles, _ = pipe.predict_quantiles(
        inputs=contexts, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
    )
    median = quantiles[:, :, 1].cpu().numpy()
    preds = {}
    for t in test_t:
        step = t - train_t[-1] - 1
        preds[t] = median[:, step]
    return preds


# ── Moirai-2 (zero-shot, quantile output — pseudo-stochastic via input perturb) ─
_MOIRAI_CACHE = {}
def _get_moirai(ctx_len, horizon, ckpt="Salesforce/moirai-2.0-R-small"):
    key = (ctx_len, horizon, ckpt)
    if key not in _MOIRAI_CACHE:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        module = Moirai2Module.from_pretrained(ckpt)
        model = Moirai2Forecast(
            prediction_length=horizon,
            context_length=ctx_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            module=module,
        ).eval()
        _MOIRAI_CACHE[key] = model
    return _MOIRAI_CACHE[key]


def predict_moirai(g_raw, train_t, test_t, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    K = g_raw.shape[1]
    ctx_len = len(train_t)
    horizon = max(test_t) - train_t[-1]
    model = _get_moirai(ctx_len, horizon)
    # Tiny input noise for seed variance
    sigma = 0.01 * g_raw[train_t].std()
    noise = np.random.RandomState(seed).randn(*g_raw[train_t].shape) * sigma
    past_target = torch.tensor((g_raw[train_t] + noise).T[:, :, None], dtype=torch.float32)  # (K, ctx, 1)
    past_observed = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros((K, ctx_len), dtype=torch.bool)
    with torch.no_grad():
        out = model(past_target=past_target, past_observed_target=past_observed,
                    past_is_pad=past_is_pad)
    # out: (batch, num_quantiles, future_time, *tgt)  — quantiles directly
    if out.ndim == 4:
        out = out[..., 0]
    # take median quantile (index = len/2)
    median_idx = out.shape[1] // 2
    median = out[:, median_idx, :].cpu().numpy()  # (K, future_time)
    preds = {}
    for t in test_t:
        step = t - train_t[-1] - 1
        preds[t] = median[:, step]
    return preds


# ── TimesFM (zero-shot, deterministic per repo but supports seed) ───
_TIMESFM_CACHE = {}
def _get_timesfm(ctx_len, horizon):
    key = (ctx_len, horizon)
    if key not in _TIMESFM_CACHE:
        import timesfm
        # 2.0-500m hparams (from HF model card)
        ctx_padded = max(32, ((ctx_len + 31) // 32) * 32)
        hparams = timesfm.TimesFmHparams(
            backend="cpu", per_core_batch_size=32,
            horizon_len=128, context_len=ctx_padded,
            input_patch_len=32, output_patch_len=128,
            num_layers=50, model_dims=1280,
            use_positional_embedding=False,
        )
        ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
        m = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
        _TIMESFM_CACHE[key] = (m, ctx_padded)
    return _TIMESFM_CACHE[key]


def predict_timesfm(g_raw, train_t, test_t, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    K = g_raw.shape[1]
    ctx_len = len(train_t)
    horizon = max(test_t) - train_t[-1]
    model, ctx_padded = _get_timesfm(ctx_len, horizon)
    inputs = [g_raw[train_t, k].astype(np.float32) for k in range(K)]  # K series
    freq = [0] * K   # high frequency
    point_forecast, _ = model.forecast(inputs=inputs, freq=freq)
    # point_forecast: (K, horizon)
    pf = np.asarray(point_forecast)
    preds = {}
    for t in test_t:
        step = t - train_t[-1] - 1
        preds[t] = pf[:, step]
    return preds


# ── Sweep ───────────────────────────────────────────────────────────
DETERMINISTIC = {"persistence", "mean", "linear", "arima", "ets"}
STOCHASTIC    = {"chronos", "moirai", "timesfm"}

METHOD_FNS = {
    "persistence": predict_persistence,
    "mean":        predict_mean,
    "linear":      predict_linear,
    "arima":       predict_arima,
    "ets":         predict_ets,
    "chronos":     predict_chronos,
    "moirai":      predict_moirai,
    "timesfm":     predict_timesfm,
}


def run_method(name, g_raw, g_norm, train_t, test_t, seed):
    fn = METHOD_FNS[name]
    if name in DETERMINISTIC:
        preds = fn(g_raw, train_t, test_t)
    else:
        preds = fn(g_raw, train_t, test_t, seed=seed)
    out = []
    for t in test_t:
        m = metrics_one_t(preds[t], g_raw[t], g_norm[t])
        m["t_eval"] = int(t); m["K_ahead"] = int(t - train_t[-1])
        out.append(m)
    return out


def main():
    g_raw, g_norm = load_growth_matrix()
    T_data = g_raw.shape[0] - 1
    train_t, test_t = make_temporal_split(T_data, T_TRAIN_FRAC)
    print(f"=== {DATA_NAME}  T={T_data+1}, K={g_raw.shape[1]}, train={train_t}, test={test_t} ===")

    enabled = os.environ.get("PNODE_METHODS", "all")
    methods = list(METHOD_FNS.keys()) if enabled == "all" else enabled.split(",")
    print(f"  methods: {methods}")
    print(f"  seeds (for stochastic): {SEEDS}")

    all_results = {}
    for name in methods:
        seeds = SEEDS if name in STOCHASTIC else [SEEDS[0]]
        runs = []
        for seed in seeds:
            try:
                res = run_method(name, g_raw, g_norm, train_t, test_t, seed)
                runs.append({"seed": seed, "per_t": res})
                rho_mean = np.mean([r["spearman_r"] for r in res])
                print(f"  [{name:<12} seed={seed}] mean ρ={rho_mean:+.3f}")
            except Exception as e:
                print(f"  [{name:<12} seed={seed}] ERROR: {type(e).__name__}: {e}")
        all_results[name] = {
            "deterministic": name in DETERMINISTIC,
            "runs": runs,
        }

    out_file = OUT_DIR / "evaluation_all.json"
    json.dump({
        "domain": DOMAIN, "data_name": DATA_NAME,
        "train_t": train_t, "test_t": test_t,
        "seeds": SEEDS, "T_train_frac": T_TRAIN_FRAC,
        "results": all_results,
    }, out_file.open("w"), indent=2)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
