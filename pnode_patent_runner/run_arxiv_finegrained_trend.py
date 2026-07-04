"""
ArXiv embedding を K-means クラスタリングで細粒度トピック化 → trend benchmark。

目的:
  従来の cs.X (n=33) では小さすぎる検出力を、K=250 細粒度クラスタで補強。
  Reviewer #2 R5「データ規模が論文水準でない」への対応。

手順:
  1. 26K papers × 1024-dim embedding を MiniBatchKMeans で K クラスタに分割
  2. cluster_id を新たな "topic" として CSV に保存 (キャッシュ)
  3. load_author_topic_graph_bundle(topic_column="fine_topic") で読み込み
  4. PC-PNODE A+BEF で 5 seed 実行

使用:
  PNODE_K=250 PNODE_SEED=42 python -m pnode_patent_runner.run_arxiv_finegrained_trend
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE
from pnode_patent_runner.cope_experiment import load_author_topic_graph_bundle
from pnode_patent_runner.unified_training import (
    evaluate_val_future_link_metrics_for_years,
    train_model_improved,
)
from pnode_patent_runner.trend_evaluation import evaluate_trend_metrics

# ── 設定 ──────────────────────────────────────────────────────────────────────
SRC_CSV    = "data/processed/arxiv_cs_embedded_2020-2026_full.csv"
K          = int(os.environ.get("PNODE_K", 250))
KMEANS_SEED = 42        # クラスタリングは固定 seed (再現性)
CACHED_CSV = f"data/processed/arxiv_cs_finegrained_K{K}.csv"
SEED       = int(os.environ.get("PNODE_SEED", 42))
EPOCHS     = int(os.environ.get("PNODE_EPOCHS", 30))
YEAR_RANGE = (2022, 2025)
HOLDOUT    = 2025
MIN_PAPERS = int(os.environ.get("PNODE_MIN_PAPERS", 5))
USE_A      = bool(int(os.environ.get("PNODE_USE_A", "1")))

OUTDIR = Path("pnode_patent_runner/outputs/arxiv_finegrained_trend")
TAG    = "_a" if USE_A else ""
OUT_JSON = OUTDIR / f"trend_K{K}{TAG}_seed{SEED}.json"

METHODS = {
    "static":     dict(variant="static",     trend_weight=0.0),
    "neural_ode": dict(variant="neural_ode", trend_weight=0.0),
    "pnode":      dict(variant="pnode",      trend_weight=0.0),
    "pnode_pc":   dict(variant="pnode_pc",   trend_weight=1.0),
}

TRAIN_KWARGS = dict(
    num_epochs=EPOCHS, lr=1e-3, beta=0.01, pos_weight=5.0,
    latent_pred_weight=1.0, future_link_weight=10.0,
    potential_weight=0.1, trajectory_weight=0.1,
    num_neg_recon=800, num_neg_future=400,
    density_align_weight=0.05,
    loss_aux_warmup_epochs=5,
)


def parse_embedding(s):
    """文字列 '0.1 0.2 ...' (スペース区切り) や '[0.1, 0.2, ...]' を numpy array にパース"""
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=np.float32)
    s = str(s).strip()
    if not s or s == "nan":
        return None
    try:
        # まず brackets を除去
        s_clean = s.strip("[]")
        # スペース区切りを優先 (このデータセットの形式)
        arr = np.fromstring(s_clean, sep=" ", dtype=np.float32)
        if arr.size <= 1:
            # フォールバック: コンマ区切り
            arr = np.fromstring(s_clean, sep=",", dtype=np.float32)
        return arr if arr.size > 1 else None
    except Exception:
        return None


def build_finegrained_csv():
    """K-means で K クラスタに分割し、'fine_topic' 列を追加した CSV を保存"""
    if Path(CACHED_CSV).is_file():
        print(f"  ✅ キャッシュ存在: {CACHED_CSV}")
        return CACHED_CSV

    print(f"\n[K-means クラスタリング] K={K}, seed={KMEANS_SEED}")
    df = pd.read_csv(SRC_CSV)
    print(f"  行数: {len(df)}")

    # embedding パース
    print("  embedding パース中...")
    embeds = []
    valid_idx = []
    for i, s in enumerate(df["description_embedding"]):
        emb = parse_embedding(s)
        if emb is not None:
            embeds.append(emb)
            valid_idx.append(i)
        if (i+1) % 5000 == 0:
            print(f"    {i+1}/{len(df)}...")
    embeds = np.stack(embeds)
    print(f"  有効 embedding: {len(embeds)} 行 × {embeds.shape[1]} 次元")

    # MiniBatchKMeans (高速)
    from sklearn.cluster import MiniBatchKMeans
    print(f"  MiniBatchKMeans fit (K={K})...")
    km = MiniBatchKMeans(n_clusters=K, batch_size=2048, random_state=KMEANS_SEED, n_init=3, max_iter=200)
    labels = km.fit_predict(embeds)

    # cluster_id を fine_topic として df に書き込み
    fine_topic = np.full(len(df), -1, dtype=int)
    fine_topic[valid_idx] = labels
    df["fine_topic"] = [f"cluster_{c}" if c >= 0 else "_invalid" for c in fine_topic]

    # 無効行を除去
    df = df[df["fine_topic"] != "_invalid"].reset_index(drop=True)
    print(f"  有効行数 (保存): {len(df)}")
    print(f"  クラスタサイズ統計: min={pd.Series(labels).value_counts().min()}, "
          f"max={pd.Series(labels).value_counts().max()}, "
          f"mean={pd.Series(labels).value_counts().mean():.1f}")

    df.to_csv(CACHED_CSV, index=False)
    print(f"  ✅ Saved -> {CACHED_CSV}")
    return CACHED_CSV


def build_model(variant, bundle, device):
    use_topic_pe = USE_A and (variant == "pnode_pc")
    return BenchmarkTemporalVGAE(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        hidden_dim=128, latent_dim=2,
        initial_corp_vectors=bundle.init_vectors,
        link_score_mode="distance",
        variant=variant,
        pnode_potential_feature="mlp",
        year_min=YEAR_RANGE[0], year_max=YEAR_RANGE[1],
        topic_position_embedding=use_topic_pe,
    ).to(device)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = build_finegrained_csv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  K={K}  seed={SEED}  USE_A={USE_A}")

    bundle = load_author_topic_graph_bundle(
        csv_path, min_papers=MIN_PAPERS, year_range=YEAR_RANGE,
        topic_column="fine_topic",
    )
    n_topics = len(bundle.right_nodes)
    print(f"  Authors={bundle.num_corps}  Topics(K-means)={n_topics}  N={bundle.total_n}")

    all_years   = sorted(bundle.graphs.keys())
    train_years = [y for y in all_years if y < HOLDOUT]
    graphs_train = {y: bundle.graphs[y] for y in train_years}
    hist_edges_train = {
        y: bundle.hist_edges[y] for y in train_years if y in bundle.hist_edges
    }
    year_prev = max(train_years)
    year_next = HOLDOUT

    growth = bundle.topic_growth_by_year or {}
    all_gy = sorted(growth.keys())
    smoothed = {}
    for y in all_gy:
        nb = [yy for yy in all_gy if abs(yy - y) <= 1]
        smoothed[y] = torch.stack([growth[yy] for yy in nb]).mean(dim=0)

    print(f"  TrainYears={train_years}  Eval: {year_prev}→{year_next}")

    results = []
    for key, cfg in METHODS.items():
        set_seed(SEED)
        variant = cfg["variant"]
        trend_w = cfg["trend_weight"]
        print(f"\n[{key}] 学習中... (variant={variant}, trend_weight={trend_w})")

        model = build_model(variant, bundle, device)
        kw = dict(**TRAIN_KWARGS)
        if variant == "pnode_pc" and trend_w > 0:
            kw["trend_weight"]        = trend_w
            kw["topic_growth_by_year"] = smoothed
            if USE_A:
                kw["trend_loss_type"]    = "listnet"
                kw["trend_listnet_temp"] = 0.5
                kw["trend_warmup_epochs"] = max(EPOCHS // 3, 5)
        else:
            kw["trend_weight"]        = 0.0
            kw["topic_growth_by_year"] = None

        train_model_improved(
            model, graphs_train, bundle.num_corps, hist_edges_train, **kw,
        )
        model.eval()

        link_m = evaluate_val_future_link_metrics_for_years(
            model, bundle.graphs, bundle.num_corps, device, year_prev, year_next,
        )
        growth_for_eval = smoothed if variant == "pnode_pc" else None
        trend_m = evaluate_trend_metrics(
            model, bundle.graphs, bundle.num_corps, device,
            year_prev, year_next, topic_growth_by_year=growth_for_eval, ndcg_k=10,
        )
        sig = "*" if trend_m.get("spearman_p", 1.0) < 0.05 else ""
        sp_r = trend_m.get('spearman_r', float('nan'))
        sp_str = f"{sp_r:+.4f}{sig}" if sp_r == sp_r else " N/A "
        print(f"  {key:<12} Link={link_m.get('auc', float('nan')):.4f}  "
              f"Entry={trend_m.get('entry_auc', float('nan')):.4f}  "
              f"Spearman={sp_str}(p={trend_m.get('spearman_p', float('nan')):.4f})  "
              f"NDCG={trend_m.get('ndcg', float('nan')):.3f}  n_topics={n_topics}")

        results.append({
            "key": key, "K": K, "n_topics": n_topics,
            "link_auc":  link_m.get("auc", float("nan")),
            "entry_auc": trend_m.get("entry_auc", float("nan")),
            "spearman_r": trend_m.get("spearman_r", float("nan")),
            "spearman_p": trend_m.get("spearman_p", float("nan")),
            "ndcg_at_10": trend_m.get("ndcg", float("nan")),
            "phi_span":   trend_m.get("phi_span", float("nan")),
        })

    out = {"results": results, "settings": {
        "K": K, "year_range": list(YEAR_RANGE), "holdout": HOLDOUT,
        "min_papers": MIN_PAPERS, "epochs": EPOCHS, "seed": SEED, "use_a": USE_A,
    }}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
