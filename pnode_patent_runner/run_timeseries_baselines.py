"""
時系列ベースライン (ARIMA / LSTM / Transformer / Naive) によるトピック成長率予測。

各トピックの年次論文数 |N_j^t| を時系列入力として、
g_j^{T+1} = (|N_j^{T+1}| - |N_j^T|) / (|N_j^T| + 1) を予測する。

PC-PNODE との直接比較のため:
  - 同じデータ (ArXiv CS 2022-2025)
  - 同じ評価指標 (MSE, MAE, DirAcc, Spearman, NDCG@10)

出力: pnode_patent_runner/outputs/trend_benchmark/timeseries_baselines.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

DATA_CSV    = "data/processed/arxiv_cs_embedded_2020-2026_full.csv"
TOPIC_COL   = "topic"
YEAR_RANGE  = (2020, 2025)        # 学習に使える年範囲
EVAL_T      = 2024                # T = この年までで学習
EVAL_TARGET = 2025                # T+1 を予測
SEED        = int(os.environ.get("PNODE_SEED", 42))
OUT_JSON    = Path(f"pnode_patent_runner/outputs/trend_benchmark/timeseries_baselines_seed{SEED}.json")


# ─────────────────────────────────────────────────────────────────────────────
# データ抽出: トピック × 年 の論文数時系列
# ─────────────────────────────────────────────────────────────────────────────

def load_topic_timeseries(csv_path: str, year_range: Tuple[int, int]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Returns
    -------
    counts : (T, n_topics)  — 年 × トピック の論文数
    topics : List[str]      — トピック名リスト
    years  : (T,)           — 年配列
    """
    df = pd.read_csv(csv_path, usecols=[TOPIC_COL, "year"])
    df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    df = df.dropna(subset=[TOPIC_COL])

    pivot = df.groupby(["year", TOPIC_COL]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()

    # min_papers ≥ 5 のトピックだけ残す（PNODE 側と同じフィルタ）
    keep = pivot.sum(axis=0) >= 5
    pivot = pivot.loc[:, keep]

    years   = pivot.index.values.astype(int)
    topics  = list(pivot.columns)
    counts  = pivot.values.astype(np.float32)
    return counts, topics, years


def compute_growth(counts_t: np.ndarray, counts_tp1: np.ndarray) -> np.ndarray:
    """g = (c_{t+1} - c_t) / (c_t + 1)"""
    return (counts_tp1 - counts_t) / (counts_t + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ベースライン①: Naive (g_hat = g_T)
# ─────────────────────────────────────────────────────────────────────────────

def predict_naive(counts: np.ndarray, eval_t_idx: int) -> np.ndarray:
    """前年の成長率をそのまま使う"""
    g_prev = compute_growth(counts[eval_t_idx - 1], counts[eval_t_idx])
    return g_prev  # 同じ g を T+1 でも使う


# ─────────────────────────────────────────────────────────────────────────────
# ベースライン②: ARIMA (各トピック独立)
# ─────────────────────────────────────────────────────────────────────────────

def predict_arima(counts: np.ndarray, eval_t_idx: int) -> np.ndarray:
    """各トピックに ARIMA(1,1,1) を fit して T+1 を予測"""
    from statsmodels.tsa.arima.model import ARIMA
    n_topics = counts.shape[1]
    pred_count = np.zeros(n_topics, dtype=np.float32)
    train_series = counts[: eval_t_idx + 1, :]   # 0..T
    actual_T     = counts[eval_t_idx, :]
    for j in range(n_topics):
        try:
            ts = train_series[:, j]
            if ts.sum() < 5 or len(ts) < 3:
                pred_count[j] = ts[-1]
                continue
            model = ARIMA(ts, order=(1, 1, 1))
            fit = model.fit()
            forecast = fit.forecast(steps=1)
            pred_count[j] = max(forecast[0], 0.0)
        except Exception:
            pred_count[j] = train_series[-1, j]
    g_hat = (pred_count - actual_T) / (actual_T + 1.0)
    return g_hat


# ─────────────────────────────────────────────────────────────────────────────
# ベースライン③: LSTM (全トピック共有モデル)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):                  # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _train_neural_forecaster(model, X_train, y_train, epochs=200, lr=1e-3):
    """X_train: (n, T, 1)  y_train: (n,)  L2 損失"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, patience = float("inf"), 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if loss.item() < best - 1e-5:
            best = loss.item(); patience = 0
        else:
            patience += 1
            if patience >= 30:
                break
    return model


def _make_lagged_dataset(counts: np.ndarray, eval_t_idx: int, window: int = 3):
    """
    lag-window 形式の (X, y) サンプルを作る。
    n_samples = (eval_t_idx - window) × n_topics
    X: (n, window, 1)  y: (n,)
    予測対象は次の年のカウント。
    """
    n_topics = counts.shape[1]
    X_list, y_list = [], []
    for t in range(window, eval_t_idx):     # t は予測元の最後のインデックス
        for j in range(n_topics):
            x = counts[t - window:t, j]
            y = counts[t, j]
            X_list.append(x.reshape(window, 1))
            y_list.append(y)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    # 予測時用の入力 (T 直前の window 年)
    X_test = np.stack([counts[eval_t_idx - window + 1:eval_t_idx + 1, j].reshape(window, 1)
                       for j in range(n_topics)]).astype(np.float32)
    return X, y, X_test


def predict_lstm(counts: np.ndarray, eval_t_idx: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    window = 3
    X, y, X_test = _make_lagged_dataset(counts, eval_t_idx, window=window)

    # 標準化
    mean, std = y.mean(), y.std() + 1e-6
    y_n = (y - mean) / std
    X_n = (X - mean) / std
    X_test_n = (X_test - mean) / std

    model = LSTMForecaster(hidden_dim=64, num_layers=2)
    model = _train_neural_forecaster(model, X_n, y_n, epochs=300, lr=1e-3)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pred_n = model(torch.tensor(X_test_n, device=device)).cpu().numpy()
    pred_count = pred_n * std + mean
    pred_count = np.maximum(pred_count, 0.0)
    actual_T = counts[eval_t_idx, :]
    return (pred_count - actual_T) / (actual_T + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ベースライン④: Transformer (全トピック共有モデル)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerForecaster(nn.Module):
    def __init__(self, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos   = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=d_model*2,
                                           dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                    # x: (B, T, 1)
        T = x.size(1)
        h = self.embed(x) + self.pos[:, :T]
        h = self.enc(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def predict_transformer(counts: np.ndarray, eval_t_idx: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    window = 3
    X, y, X_test = _make_lagged_dataset(counts, eval_t_idx, window=window)
    mean, std = y.mean(), y.std() + 1e-6
    y_n = (y - mean) / std
    X_n = (X - mean) / std
    X_test_n = (X_test - mean) / std

    model = TransformerForecaster(d_model=64, num_heads=4, num_layers=2)
    model = _train_neural_forecaster(model, X_n, y_n, epochs=300, lr=1e-3)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pred_n = model(torch.tensor(X_test_n, device=device)).cpu().numpy()
    pred_count = pred_n * std + mean
    pred_count = np.maximum(pred_count, 0.0)
    actual_T = counts[eval_t_idx, :]
    return (pred_count - actual_T) / (actual_T + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 評価指標
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(g_hat: np.ndarray, g_true: np.ndarray, eps: float = 0.05) -> Dict[str, float]:
    mask = np.isfinite(g_hat) & np.isfinite(g_true)
    g_hat = g_hat[mask]; g_true = g_true[mask]
    if len(g_hat) < 5:
        return {k: float("nan") for k in ["mse","mae","dir_acc","spearman_r","spearman_p","ndcg_at_10"]}

    mse = float(np.mean((g_hat - g_true) ** 2))
    mae = float(np.mean(np.abs(g_hat - g_true)))

    # Direction agreement (3-class with ε tolerance)
    sign_h = np.where(np.abs(g_hat) < eps, 0, np.sign(g_hat))
    sign_t = np.where(np.abs(g_true) < eps, 0, np.sign(g_true))
    dir_acc = float(np.mean(sign_h == sign_t))

    # Spearman (high g_hat → high g_true: 正の相関を期待。NDCG はこの仮定で計算)
    r, p = stats.spearmanr(g_hat, g_true)

    # NDCG@10 (g_hat 降順 = 注目予測, ターゲット = max(g_true, 0))
    K = 10
    rel = np.maximum(g_true, 0.0)
    if rel.sum() < 1e-8:
        ndcg = float("nan")
    else:
        order = np.argsort(-g_hat)
        ideal = np.argsort(-rel)
        dcg  = sum(rel[order[r]] / np.log2(r + 2) for r in range(min(K, len(rel))))
        idcg = sum(rel[ideal[r]] / np.log2(r + 2) for r in range(min(K, len(rel))))
        ndcg = float(dcg / idcg) if idcg > 0 else float("nan")

    return {
        "mse": mse, "mae": mae, "dir_acc": dir_acc,
        "spearman_r": float(r), "spearman_p": float(p),
        "ndcg_at_10": ndcg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading time series from {DATA_CSV}...")
    counts, topics, years = load_topic_timeseries(DATA_CSV, YEAR_RANGE)
    print(f"  shape: counts={counts.shape}, topics={len(topics)}, years={list(years)}")

    eval_t_idx     = int(np.where(years == EVAL_T)[0][0])
    eval_target_idx = int(np.where(years == EVAL_TARGET)[0][0])

    # 真値
    g_true = compute_growth(counts[eval_t_idx], counts[eval_target_idx])
    print(f"\n真値 g_{{{EVAL_T}→{EVAL_TARGET}}}: range [{g_true.min():+.3f}, {g_true.max():+.3f}]  mean={g_true.mean():+.3f}\n")

    methods = {
        "naive":        lambda: predict_naive(counts, eval_t_idx),
        "arima":        lambda: predict_arima(counts, eval_t_idx),
        "lstm":         lambda: predict_lstm(counts, eval_t_idx, SEED),
        "transformer":  lambda: predict_transformer(counts, eval_t_idx, SEED),
    }

    results = []
    for name, fn in methods.items():
        print(f"  [{name}] 予測中...")
        try:
            g_hat = fn()
            m = evaluate(g_hat, g_true)
            m["method"] = name
            results.append(m)
            sig = "*" if m["spearman_p"] < 0.05 else ""
            print(f"    MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  DirAcc={m['dir_acc']:.3f}  "
                  f"Spearman={m['spearman_r']:+.4f}{sig}  NDCG@10={m['ndcg_at_10']:.3f}")
        except Exception as e:
            print(f"    エラー: {e}")
            results.append({"method": name, "error": str(e)})

    print("\n" + "=" * 75)
    print(f"  Time-Series Baselines (seed={SEED}, eval={EVAL_T}→{EVAL_TARGET})")
    print("=" * 75)
    print(f"  {'手法':<14} {'MSE':>8} {'MAE':>8} {'DirAcc':>8} {'Spearman':>10} {'NDCG@10':>9}")
    print("  " + "-" * 60)
    for r in results:
        if "error" in r: continue
        print(f"  {r['method']:<14} {r['mse']:>8.4f} {r['mae']:>8.4f} "
              f"{r['dir_acc']:>8.3f} {r['spearman_r']:>+10.4f} {r['ndcg_at_10']:>9.3f}")

    out = {"results": results, "settings": {
        "data": str(DATA_CSV), "year_range": list(YEAR_RANGE),
        "eval_t": EVAL_T, "eval_target": EVAL_TARGET,
        "n_topics": len(topics), "seed": SEED,
    }}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
