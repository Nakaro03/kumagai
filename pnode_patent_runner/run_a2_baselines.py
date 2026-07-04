"""
A2: 時系列ベースライン (ARIMA / LSTM / Transformer / Linear) を実装し
PI-SDE + X1 と同じ評価指標で比較。

タスク: トピック j の最終時点 t=T での成長率 ĝ_j を予測する。

入力 per topic: 過去の paper count シーケンス {N_j^0, ..., N_j^{T-1}}
出力 per topic: 予測成長率 ĝ_j = (N_j^T_pred - N_j^{T-1}) / (N_j^{T-1} + 1)

評価:
  MSE / MAE          (regression on g)
  Spearman(ĝ, g)     (rank correlation)
  NDCG@10            (top-K ranking)
  P@10               (precision at top-K)

各 baseline を 5 seed で実行 (LSTM/Transformer; ARIMA/Linear は決定的)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

# ─────────────────────────────────────────────────────────────────
# ドメイン設定
# ─────────────────────────────────────────────────────────────────
DOMAINS = {
    "paper": {
        "data_path": "data/PNode_Paper_X1/alltime/fate_train.pt",
        "name": "Paper (ArXiv CS)",
    },
    "patent_energy_top50": {
        "data_path": "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt",
        "name": "Patent Energy",
    },
    "arxiv_construction": {
        "data_path": "data/PNode_ArXiv_Construction_X1_v2/alltime/fate_train.pt",
        "name": "arXiv Construction",
    },
    "jp_construction": {
        "data_path": "data/PNode_JP_Construction_X1/alltime/fate_train.pt",
        "name": "JP Construction",
    },
}

DOMAIN  = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
SEED    = int(os.environ.get("PNODE_SEED", 42))
EPOCHS  = int(os.environ.get("PNODE_EPOCHS", 200))


def load_timeseries(data_path: str):
    """
    データセットから per-topic 時系列 (count) を抽出。

    重要: 予測対象は g_true = growth[T-1] = (count[T-1] - count[T-2]) / (count[T-2] + 1)
         予測時の入力は counts[0..T-2] のみ (count[T-1] は使わない、データリーク防止)
    """
    data = torch.load(data_path, weights_only=False)
    xp = data["xp"]
    topics = data["topics"]
    n_topics = data["n_topics"]
    T = len(xp)
    counts = np.zeros((T, n_topics), dtype=np.float32)
    for t in range(T):
        topics_t = topics[t].numpy()
        for j in range(n_topics):
            counts[t, j] = (topics_t == j).sum()
    g_true = data["growth"][T-1].numpy()
    return counts, g_true, n_topics, T


# ─────────────────────────────────────────────────────────────────
# Baseline 1: ARIMA
# ─────────────────────────────────────────────────────────────────
def predict_arima(counts: np.ndarray, T: int):
    """
    ARIMA(1,1,1) per topic.
    input: counts[0..T-2] のみ使用 (count[T-1] はリーク)
    output: g_pred = (predicted_count[T-1] - counts[T-2]) / (counts[T-2] + 1)
    """
    from statsmodels.tsa.arima.model import ARIMA
    n_topics = counts.shape[1]
    preds = np.zeros(n_topics)
    last_in = T - 1     # 入力範囲: 0..T-2 (= T-1 個の値)
    for j in range(n_topics):
        ts = counts[:last_in, j]
        if ts.sum() < 3:
            preds[j] = ts[-1] if len(ts) > 0 else 0
            continue
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            fit = model.fit()
            pred = fit.forecast(steps=1)
            preds[j] = max(0, pred[0])
        except Exception:
            preds[j] = ts[-1]
    prev = counts[T-2]   # T-2 番目 (T-2 + 1 = T-1 が目標)
    g_pred = (preds - prev) / (prev + 1.0)
    return g_pred


# ─────────────────────────────────────────────────────────────────
# Baseline 2: LSTM
# ─────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def predict_lstm(counts: np.ndarray, T: int, seed: int = 42):
    """
    LSTM. 入力: counts[0..T-2], 予測: count[T-1].
    """
    torch.manual_seed(seed)
    n_topics = counts.shape[1]
    last_in = T - 1
    # 学習: 過去シーケンス → 次の値 (autoregressive)
    X_train, y_train = [], []
    for j in range(n_topics):
        ts = counts[:last_in, j]
        for t in range(1, last_in):
            X_train.append(ts[:t].reshape(-1, 1))
            y_train.append(ts[t])
    if not X_train:
        return np.zeros(n_topics)
    max_len = max(last_in - 1, 1)
    X_pad = np.zeros((len(X_train), max_len, 1), dtype=np.float32)
    for i, x in enumerate(X_train):
        X_pad[i, -len(x):] = x
    y_arr = np.array(y_train, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X_pad, device=device)
    yt = torch.tensor(y_arr, device=device)

    model = LSTMModel(hidden=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(Xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    preds = np.zeros(n_topics)
    test_len = last_in   # counts[0..T-2] の長さ
    with torch.no_grad():
        for j in range(n_topics):
            ts = counts[:last_in, j]
            X_test = np.zeros((1, max(max_len, test_len), 1), dtype=np.float32)
            X_test[0, -len(ts):] = ts.reshape(-1, 1)
            X_test = X_test[:, :max_len if max_len >= test_len else test_len, :]
            X_test_t = torch.tensor(X_test, device=device)
            pred = model(X_test_t).cpu().numpy()[0, 0]
            preds[j] = max(0, pred)
    prev = counts[T-2]
    g_pred = (preds - prev) / (prev + 1.0)
    return g_pred


# ─────────────────────────────────────────────────────────────────
# Baseline 3: Transformer
# ─────────────────────────────────────────────────────────────────
class TransformerModel(nn.Module):
    def __init__(self, d_model=16, nhead=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, 1)
        h = self.embed(x)
        h = self.transformer(h)
        return self.fc(h[:, -1, :])


def predict_transformer(counts: np.ndarray, T: int, seed: int = 42):
    torch.manual_seed(seed)
    n_topics = counts.shape[1]
    last_in = T - 1
    X_train, y_train = [], []
    for j in range(n_topics):
        ts = counts[:last_in, j]
        for t in range(1, last_in):
            X_train.append(ts[:t].reshape(-1, 1))
            y_train.append(ts[t])
    if not X_train:
        return np.zeros(n_topics)
    max_len = max(last_in - 1, 1)
    test_len = last_in
    pad_to = max(max_len, test_len)
    X_pad = np.zeros((len(X_train), pad_to, 1), dtype=np.float32)
    for i, x in enumerate(X_train):
        X_pad[i, -len(x):] = x
    y_arr = np.array(y_train, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X_pad, device=device)
    yt = torch.tensor(y_arr, device=device)

    model = TransformerModel(d_model=16, nhead=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(Xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    preds = np.zeros(n_topics)
    with torch.no_grad():
        for j in range(n_topics):
            ts = counts[:last_in, j]
            X_test = np.zeros((1, pad_to, 1), dtype=np.float32)
            X_test[0, -len(ts):] = ts.reshape(-1, 1)
            X_test_t = torch.tensor(X_test, device=device)
            pred = model(X_test_t).cpu().numpy()[0, 0]
            preds[j] = max(0, pred)
    prev = counts[T-2]
    g_pred = (preds - prev) / (prev + 1.0)
    return g_pred


# ─────────────────────────────────────────────────────────────────
# Baseline 4: Linear (lag-based regression)
# ─────────────────────────────────────────────────────────────────
def predict_linear(counts: np.ndarray, T: int):
    """Linear regression with lag features. 入力: counts[0..T-2]"""
    from sklearn.linear_model import LinearRegression
    n_topics = counts.shape[1]
    last_in = T - 1
    preds = np.zeros(n_topics)
    for j in range(n_topics):
        ts = counts[:last_in, j]
        if len(ts) < 3:
            preds[j] = ts[-1] if len(ts) > 0 else 0
            continue
        X = np.column_stack([ts[:-2], ts[1:-1]])
        y = ts[2:]
        try:
            lr = LinearRegression().fit(X, y)
            pred = lr.predict([[ts[-2], ts[-1]]])[0]
            preds[j] = max(0, pred)
        except Exception:
            preds[j] = ts[-1]
    prev = counts[T-2]
    g_pred = (preds - prev) / (prev + 1.0)
    return g_pred


# ─────────────────────────────────────────────────────────────────
# Naive baselines (data leak 防止: counts[0..T-2] のみ使用)
# ─────────────────────────────────────────────────────────────────
def predict_naive_lastgrowth(counts: np.ndarray, T: int):
    """ĝ = 過去年の成長率 (=growth[T-2]). counts[T-1] は使わない"""
    if T < 3:
        return np.zeros(counts.shape[1])
    # growth[T-2] = (counts[T-2] - counts[T-3]) / (counts[T-3] + 1)
    return (counts[T-2] - counts[T-3]) / (counts[T-3] + 1.0)


def predict_naive_zero(counts: np.ndarray, T: int):
    """ĝ = 0 (no change)"""
    return np.zeros(counts.shape[1])


# ─────────────────────────────────────────────────────────────────
# Baseline: DLinear (ICLR 2023 — trend + residual)
# ─────────────────────────────────────────────────────────────────
class DLinearModel(nn.Module):
    """Channel-independent trend + residual linear model."""
    def __init__(self, seq_len, pred_len=1, kernel=3):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = max(2, min(kernel, seq_len))
        # use 1D average pool for trend (moving average)
        self.trend_pool = nn.AvgPool1d(self.kernel, stride=1,
                                       padding=(self.kernel - 1) // 2,
                                       count_include_pad=False)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_residual = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (batch, seq_len)
        x_p = x.unsqueeze(1)                          # (batch, 1, seq_len)
        trend = self.trend_pool(x_p).squeeze(1)       # (batch, seq_len)
        if trend.shape[1] != self.seq_len:
            # pad / crop to align
            if trend.shape[1] > self.seq_len:
                trend = trend[:, :self.seq_len]
            else:
                pad = self.seq_len - trend.shape[1]
                trend = torch.cat([trend, trend[:, -1:].repeat(1, pad)], dim=1)
        residual = x - trend
        out_t = self.linear_trend(trend)
        out_r = self.linear_residual(residual)
        return out_t + out_r                          # (batch, pred_len)


def predict_dlinear(counts: np.ndarray, T: int, seed: int = 42):
    """DLinear forecast (no leak): input counts[0..T-2], predict count[T-1]."""
    torch.manual_seed(seed); np.random.seed(seed)
    last_in = T - 1
    if last_in < 2:
        return np.zeros(counts.shape[1])
    X = counts[:last_in].T                            # (n_topics, seq_len)
    X_t = torch.tensor(X, dtype=torch.float32)
    # target = next count for fitting
    if last_in < 3:
        # not enough data for sliding window, just fit on last point as target
        return predict_linear(counts, T)
    n_topics, seq_len = X_t.shape
    # Build training pairs: predict counts[1:last_in] from counts[0:last_in-1] shifted
    train_x_list, train_y_list = [], []
    win = min(seq_len - 1, seq_len)
    for t in range(2, last_in):
        # input window: counts[0..t-1], target: counts[t]
        x_win = X_t[:, :t]
        # pad to seq_len for fixed input size
        if x_win.shape[1] < seq_len:
            pad = seq_len - x_win.shape[1]
            x_win = torch.cat([x_win[:, :1].repeat(1, pad), x_win], dim=1)
        train_x_list.append(x_win)
        train_y_list.append(counts[t])
    if not train_x_list:
        return predict_linear(counts, T)
    train_x = torch.stack(train_x_list).reshape(-1, seq_len)            # (N*B, seq_len)
    train_y = torch.tensor(np.stack(train_y_list), dtype=torch.float32).flatten()  # (N*B,)

    model = DLinearModel(seq_len=seq_len, pred_len=1, kernel=max(2, seq_len // 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for ep in range(300):
        model.train()
        pred = model(train_x).squeeze(-1)
        loss = ((pred - train_y) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Inference on X_t to predict next counts[last_in]
    model.eval()
    with torch.no_grad():
        pred_next = model(X_t).squeeze(-1).numpy()    # (n_topics,)
    pred_next = np.maximum(pred_next, 0)
    prev = counts[T - 2]
    g_pred = (pred_next - prev) / (prev + 1.0)
    return g_pred


# ─────────────────────────────────────────────────────────────────
# Baseline: PatchTST (ICLR 2023 — patch + transformer encoder, channel-independent)
# ─────────────────────────────────────────────────────────────────
class PatchTSTModel(nn.Module):
    def __init__(self, seq_len, patch_len=2, d_model=16, n_heads=2,
                 n_layers=2, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        n_patches = max(1, seq_len - patch_len + 1)
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=2 * d_model,
                                               batch_first=True,
                                               dropout=0.1)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model, pred_len)

    def forward(self, x):
        # x: (batch, seq_len)
        # build patches with stride 1
        patches = []
        for i in range(self.n_patches):
            patches.append(x[:, i:i+self.patch_len])
        p = torch.stack(patches, dim=1)              # (batch, n_patches, patch_len)
        p = self.patch_embed(p) + self.pos_embed     # (batch, n_patches, d_model)
        h = self.enc(p)                              # (batch, n_patches, d_model)
        h_flat = h.flatten(start_dim=1)
        out = self.head(h_flat)
        return out                                   # (batch, pred_len)


def predict_patchtst(counts: np.ndarray, T: int, seed: int = 42):
    torch.manual_seed(seed); np.random.seed(seed)
    last_in = T - 1
    if last_in < 3:
        return predict_linear(counts, T)
    X = counts[:last_in].T                            # (n_topics, seq_len)
    X_t = torch.tensor(X, dtype=torch.float32)
    n_topics, seq_len = X_t.shape
    patch_len = max(2, min(2, seq_len - 1))           # very small patches for short series

    # Build training pairs
    train_x_list, train_y_list = [], []
    for t in range(2, last_in):
        x_win = X_t[:, :t]
        if x_win.shape[1] < seq_len:
            pad = seq_len - x_win.shape[1]
            x_win = torch.cat([x_win[:, :1].repeat(1, pad), x_win], dim=1)
        train_x_list.append(x_win)
        train_y_list.append(counts[t])
    if not train_x_list:
        return predict_linear(counts, T)
    train_x = torch.stack(train_x_list).reshape(-1, seq_len)
    train_y = torch.tensor(np.stack(train_y_list), dtype=torch.float32).flatten()

    model = PatchTSTModel(seq_len=seq_len, patch_len=patch_len,
                          d_model=16, n_heads=2, n_layers=2, pred_len=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    for ep in range(300):
        model.train()
        pred = model(train_x).squeeze(-1)
        loss = ((pred - train_y) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_next = model(X_t).squeeze(-1).numpy()
    pred_next = np.maximum(pred_next, 0)
    prev = counts[T - 2]
    g_pred = (pred_next - prev) / (prev + 1.0)
    return g_pred


def predict_naive_mean(counts: np.ndarray, T: int):
    """ĝ = counts[0..T-2] における平均成長率"""
    g_history = []
    for t in range(1, T-1):     # t=1..T-2 の growth
        g_t = (counts[t] - counts[t-1]) / (counts[t-1] + 1.0)
        g_history.append(g_t)
    return np.mean(g_history, axis=0) if g_history else np.zeros(counts.shape[1])


# ─────────────────────────────────────────────────────────────────
# 評価
# ─────────────────────────────────────────────────────────────────
def evaluate(g_pred: np.ndarray, g_true: np.ndarray):
    mask = np.isfinite(g_pred) & np.isfinite(g_true)
    p, t = g_pred[mask], g_true[mask]
    if len(p) < 2:
        return {}

    mse = float(((p - t) ** 2).mean())
    mae = float(np.abs(p - t).mean())
    try:
        sp_r, sp_p = stats.spearmanr(p, t)
    except Exception:
        sp_r, sp_p = float("nan"), 1.0

    # NDCG@10 (predicted から見て、 g_true が高い順)
    K = min(10, len(t))
    order_pred = np.argsort(-p)        # 降順 (高い ĝ から)
    rel = np.maximum(t, 0.0)
    dcg = sum(rel[order_pred[k]] / np.log2(k + 2) for k in range(K))
    ideal_order = np.argsort(-rel)
    idcg = sum(rel[ideal_order[k]] / np.log2(k + 2) for k in range(K))
    ndcg = float(dcg / (idcg + 1e-10)) if idcg > 0 else float("nan")

    # P@10
    top_k = order_pred[:K]
    prec = float((t[top_k] > 0).mean())

    return {
        "mse": mse, "mae": mae,
        "spearman_r": float(sp_r), "spearman_p": float(sp_p),
        "ndcg_at_10": ndcg, "prec_at_10": prec,
    }


def main():
    cfg = DOMAINS[DOMAIN]
    print("=" * 70)
    print(f"  A2 Baselines on {cfg['name']}  seed={SEED}")
    print("=" * 70)
    counts, g_true, n_topics, T = load_timeseries(cfg["data_path"])
    print(f"  T={T}, n_topics={n_topics}")
    print(f"  Counts shape: {counts.shape}")
    print(f"  g_true range: [{g_true.min():+.3f}, {g_true.max():+.3f}]")

    methods = {
        "Naive_zero":    lambda: predict_naive_zero(counts, T),
        "Naive_mean":    lambda: predict_naive_mean(counts, T),
        "Naive_lastg":   lambda: predict_naive_lastgrowth(counts, T),
        "Linear":        lambda: predict_linear(counts, T),
        "ARIMA":         lambda: predict_arima(counts, T),
        "LSTM":          lambda: predict_lstm(counts, T, seed=SEED),
        "Transformer":   lambda: predict_transformer(counts, T, seed=SEED),
        "DLinear":       lambda: predict_dlinear(counts, T, seed=SEED),
        "PatchTST":      lambda: predict_patchtst(counts, T, seed=SEED),
    }

    results = {}
    for name, fn in methods.items():
        try:
            g_pred = fn()
            m = evaluate(g_pred, g_true)
            results[name] = m
            sig = "*" if m["spearman_p"] < 0.05 else " "
            print(f"  {name:<14} MSE={m['mse']:.3f}  MAE={m['mae']:.3f}  "
                  f"Sp={m['spearman_r']:+.3f}{sig}  NDCG={m['ndcg_at_10']:.3f}  "
                  f"P@10={m['prec_at_10']:.2f}")
        except Exception as e:
            print(f"  {name:<14} FAILED: {e}")
            results[name] = {"error": str(e)}

    # 保存
    out_dir = Path(f"RESULTS/baselines/{DOMAIN}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"baselines_seed{SEED}.json"
    json.dump(results, out_file.open("w"), indent=2)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
