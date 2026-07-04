"""
技術トレンド予測ベンチマーク。

問題設定:
  「将来リンク予測 AUC」ではなく「技術トレンド景観の推定」を評価する。

評価指標:
  Link-AUC    : 従来指標（補助）
  Entry-AUC   : 新規参入リンクのみ AUC（著者が初めて参入する技術）
  Spearman(Φ,g): Φランキング vs 実際の成長率（PC-PNODE 専用）
  NDCG@10     : 上位10成長トピックの予測精度（PC-PNODE 専用）

比較:
  Static   → Entry-AUC が最も低いはず（動的変化を捉えられない）
  NeuralODE → Entry-AUC 中程度、Spearman=N/A
  PNODE    → Entry-AUC 中程度、Spearman≈0（L_trend なし）
  PC-PNODE → Entry-AUC 最高、Spearman<0（谷=成長の対応）

使用例:
  python -m pnode_patent_runner.run_trend_benchmark
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.cope_experiment import load_author_topic_graph_bundle
from pnode_patent_runner.unified_training import (
    evaluate_val_future_link_metrics_for_years,
    train_model_improved,
)
from pnode_patent_runner.trend_evaluation import evaluate_trend_metrics

# ── 設定 ──────────────────────────────────────────────────────────────────────
DATA_CSV    = "data/processed/arxiv_cs_embedded_2020-2026_full.csv"
YEAR_RANGE  = (2022, 2025)
HOLDOUT     = 2025          # 学習: 2022-2024、テスト: 2024→2025
MIN_PAPERS  = 5
SEED        = int(os.environ.get("PNODE_SEED", 42))
EPOCHS      = 30
# B+E+F の改善を有効にする (環境変数 PNODE_USE_BEF=1 で切り替え)
USE_BEF     = bool(int(os.environ.get("PNODE_USE_BEF", "0")))
# Architecture A (TrendAdapter: z.detach() + linear + bias) を追加で有効化
USE_A       = bool(int(os.environ.get("PNODE_USE_A", "0")))
# Tier 1: X1 (Φ Anchor Loss) + X2 (ApproxNDCG)
USE_X12     = bool(int(os.environ.get("PNODE_USE_X12", "0")))
# Tier 2: X3 (PI-SDE 流 HJ 正則化) -- A+BEF に追加
USE_HJ      = bool(int(os.environ.get("PNODE_USE_HJ", "0")))
if USE_HJ:
    TAG = "_ahj"
elif USE_X12:
    TAG = "_ax12"
elif USE_A:
    TAG = "_abef"
elif USE_BEF:
    TAG = "_bef"
else:
    TAG = ""
OUTDIR      = Path("pnode_patent_runner/outputs/trend_benchmark")
OUT_JSON    = OUTDIR / f"trend_benchmark{TAG}_seed{SEED}.json"

METHODS = {
    "static":     dict(variant="static",     trend_weight=0.0),
    "neural_ode": dict(variant="neural_ode", trend_weight=0.0),
    "pnode":      dict(variant="pnode",      trend_weight=0.0),
    "pnode_pc":   dict(variant="pnode_pc",   trend_weight=1.0),
}

TRAIN_KWARGS = dict(
    num_epochs=EPOCHS,
    lr=1e-3,
    beta=0.01,
    pos_weight=5.0,
    latent_pred_weight=1.0,
    future_link_weight=10.0,
    potential_weight=0.1,
    trajectory_weight=0.1,
    num_neg_recon=800,
    num_neg_future=400,
    density_align_weight=0.05,
    loss_aux_warmup_epochs=5,
)


def build_model(variant, bundle, device):
    # Architecture A (TrendAdapter): pnode_pc のとき有効
    # USE_BEF=1 なら E (旧バイアス, encode() で適用) も A 経由で適用
    # USE_A=1 / USE_X12=1 なら A (detach() + linear + bias) を proper に適用
    use_topic_pe = (USE_BEF or USE_A or USE_X12 or USE_HJ) and (variant == "pnode_pc")
    return BenchmarkTemporalVGAE(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        hidden_dim=128,
        latent_dim=2,
        initial_corp_vectors=bundle.init_vectors,
        link_score_mode="distance",
        variant=variant,
        pnode_potential_feature="mlp",
        year_min=YEAR_RANGE[0],
        year_max=YEAR_RANGE[1],
        topic_position_embedding=use_topic_pe,
    ).to(device)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDA 非決定性を抑制 (atomicAdd, cudnn benchmark の影響を排除)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── データ読み込み ────────────────────────────────────────────────────────
    bundle = load_author_topic_graph_bundle(
        DATA_CSV, min_papers=MIN_PAPERS, year_range=YEAR_RANGE,
    )
    all_years  = sorted(bundle.graphs.keys())
    train_years = [y for y in all_years if y < HOLDOUT]
    graphs_train = {y: bundle.graphs[y] for y in train_years}
    hist_edges_train = {
        y: bundle.hist_edges[y] for y in train_years if y in bundle.hist_edges
    }
    graphs_full = bundle.graphs

    print(f"著者={bundle.num_corps}  トピック={len(bundle.right_nodes)}  年={all_years}")
    print(f"学習年={train_years}  ホールドアウト={HOLDOUT}")

    year_prev = max(train_years)
    year_next = HOLDOUT

    # ── 成長率の平滑化（±1年平均） ───────────────────────────────────────────
    growth_by_year = bundle.topic_growth_by_year or {}
    all_gy = sorted(growth_by_year.keys())
    smoothed: dict = {}
    for i, y in enumerate(all_gy):
        neighbors = [yy for yy in all_gy if abs(yy - y) <= 1]
        smoothed[y] = torch.stack([growth_by_year[yy] for yy in neighbors]).mean(dim=0)

    # ── 学習 + 評価ループ ─────────────────────────────────────────────────────
    results = []

    print("\n" + "=" * 65)
    print("  手法       Link-AUC  Entry-AUC  Spearman(Φ,g)  NDCG@10")
    print("=" * 65)

    for key, cfg in METHODS.items():
        set_seed(SEED)
        variant = cfg["variant"]
        trend_w = cfg["trend_weight"]

        print(f"\n[{key}] 学習中... (variant={variant}, trend_weight={trend_w})")

        model = build_model(variant, bundle, device)

        # 学習
        kw = dict(**TRAIN_KWARGS)
        if variant == "pnode_pc" and trend_w > 0:
            kw["trend_weight"]        = trend_w
            kw["topic_growth_by_year"] = smoothed
            if USE_HJ:
                # Tier 2 構成 (PI-SDE 流): A+BEF + HJ 正則化
                kw["trend_loss_type"]    = "listnet"
                kw["trend_listnet_temp"] = 0.5
                kw["trend_warmup_epochs"] = 10
                kw["hj_weight"]          = 0.05            # X3 (PI-SDE)
            elif USE_X12:
                # Tier 1 構成: ListNet (B) + X1 Anchor + curriculum (F) + adapter (A/E)
                # X2 (ApproxNDCG) は単独試験で不安定だったため X1 と組み合わせない。
                # X1 (Anchor Loss) は Φ span を target に固定することで
                # スケール不変な ListNet が Φ→0 に縮退するのを防ぐ。
                kw["trend_loss_type"]      = "listnet"      # B
                kw["trend_listnet_temp"]   = 0.5
                kw["trend_warmup_epochs"]  = 10             # F
                kw["trend_span_weight"]    = 0.3            # X1
                kw["trend_span_target"]    = 0.5            # X1
            elif False:  # 旧 X2 試験 (不採用)
                kw["trend_loss_type"]      = "approxndcg"
                kw["trend_listnet_temp"]   = 0.5
            elif USE_BEF or USE_A:
                kw["trend_loss_type"]    = "listnet"   # B
                kw["trend_listnet_temp"] = 0.5
                kw["trend_warmup_epochs"] = 10         # F
        else:
            kw["trend_weight"]        = 0.0
            kw["topic_growth_by_year"] = None

        train_model_improved(
            model, graphs_train, bundle.num_corps, hist_edges_train,
            **kw,
        )
        model = model.to(device)

        # チェックポイント保存
        ckpt_dir = OUTDIR / "ckpt"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"{key}_seed{SEED}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "variant": variant, "seed": SEED}, ckpt_path)

        model.eval()

        # ── 従来指標: Link-AUC ───────────────────────────────────────────────
        link_m = evaluate_val_future_link_metrics_for_years(
            model, graphs_full, bundle.num_corps, device, year_prev, year_next,
        )
        link_auc = link_m.get("auc", float("nan"))

        # ── 新規指標: Entry-AUC / Spearman / NDCG ───────────────────────────
        growth_for_eval = smoothed if variant == "pnode_pc" else None
        trend_m = evaluate_trend_metrics(
            model, graphs_full, bundle.num_corps, device,
            year_prev, year_next,
            topic_growth_by_year=growth_for_eval,
            ndcg_k=10,
        )

        entry_auc  = trend_m.get("entry_auc",  float("nan"))
        spearman_r = trend_m.get("spearman_r", float("nan"))
        spearman_p = trend_m.get("spearman_p", float("nan"))
        ndcg       = trend_m.get("ndcg",       float("nan"))
        phi_span   = trend_m.get("phi_span",   float("nan"))

        # 表示
        def fmt(v, prefix=""):
            if v != v:  # nan
                return "  N/A  "
            return f"{prefix}{v:+.4f}" if prefix else f"{v:.4f}"

        sig = ""
        if spearman_p == spearman_p and spearman_p < 0.05:
            sig = "*"
        elif spearman_p == spearman_p and spearman_p < 0.10:
            sig = "."

        print(
            f"  {key:<12} "
            f"Link={link_auc:.4f}  "
            f"Entry={fmt(entry_auc)}  "
            f"Spearman={fmt(spearman_r,''):>7}{sig}(p={spearman_p:.3f})  "
            f"NDCG={fmt(ndcg)}  "
            f"Φspan={phi_span:.4f}"
            if phi_span == phi_span else
            f"  {key:<12} "
            f"Link={link_auc:.4f}  "
            f"Entry={fmt(entry_auc)}  "
            f"Spearman=  N/A    "
            f"NDCG=  N/A  "
        )

        results.append({
            "key": key,
            "variant": variant,
            "link_auc": link_auc,
            "link_ap": link_m.get("ap", float("nan")),
            "entry_auc": entry_auc,
            "entry_ap": trend_m.get("entry_ap", float("nan")),
            "exit_auc": trend_m.get("exit_auc", float("nan")),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "ndcg_at_10": ndcg,
            "phi_span": phi_span,
        })

    # ── 最終サマリーテーブル ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  最終比較テーブル (ホールドアウト 2024→2025)")
    print("=" * 75)
    print(f"  {'手法':<12} {'Link-AUC':>9} {'Entry-AUC':>10} {'Spearman r':>12} {'NDCG@10':>9}")
    print("  " + "-" * 58)
    for r in results:
        def fv(v):
            return f"{v:.4f}" if v == v else "  N/A  "
        sp_str = f"{r['spearman_r']:+.4f}" if r['spearman_r'] == r['spearman_r'] else "  N/A  "
        star = " *" if (r['spearman_p'] == r['spearman_p'] and r['spearman_p'] < 0.05) else "  "
        print(f"  {r['key']:<12} {fv(r['link_auc']):>9} {fv(r['entry_auc']):>10} "
              f"{sp_str:>10}{star} {fv(r['ndcg_at_10']):>9}")
    print("=" * 75)
    print("  ※ Spearman(Φ,g): r<0 が期待動作（低Φ=谷が成長技術に対応）")
    print("  ※ Entry-AUC: 新規参入リンクのみ。Static はここで不利になる")
    print("  ※ Spearman/NDCG は PC-PNODE のみ（他手法は N/A）")

    # JSON 保存
    out = {"results": results, "settings": {
        "year_range": list(YEAR_RANGE), "holdout": HOLDOUT,
        "min_papers": MIN_PAPERS, "epochs": EPOCHS, "seed": SEED,
    }}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
