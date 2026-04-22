#!/usr/bin/env python3
"""
README_COPE.md のデータパイプライン・損失枠に沿い、ベースラインと CoPE-VGAE を比較する。

- データ: `load_cope_graph_bundle`（特許）/ `load_author_paper_graph_bundle`（著者–論文）/ `load_author_topic_graph_bundle`（著者–トピック）
- 学習: `unified_training.train_model_improved`（README 既定の β, pos_weight, λ 等）
- 評価: `evaluate_val_future_link_metrics`（最終 2 年の future-link ROC-AUC・AP・ECE）。任意: `--eval-horizon-gaps 1,2,3` で時系列キー列上の **k ステップ先**の AUC/AP/ECE に加え、同一ペアで **潜在 MSE / MAE / トレンド方向一致率**（`evaluate_latent_rollout_metrics_for_horizon_gaps`）も JSON に記録
- 任意: `--holdout-test-year Y` で **Y 年のエッジを学習に含めず**、`(year_prev→Y)` を最終テストとして報告（学習は `y<Y` のみ、`hist_edges` もその期間で再構成）

手法:
  Static, RNN+VGAE, NeuralODE, P-NODE（幾何デコーダのみ）,
  P-NODE+Res（勾配流＋学習可能残差場 h(z)）,
  P-NODE-Energy-TD（時間依存 Φ とボルツマン型リンク尤度・純勾配流 ODE）,
  CoPE-VGAE（Φ をデコーダと ODE で共有）

例（特許・リポジトリルート）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain patent \\
    --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 \\
    --epochs 10 --seed 42 --methods all

例（最終年 2020 をホールドアウトテスト。``--year-range`` に 2020 を含める）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain patent --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 --holdout-test-year 2020 \\
    --epochs 10 --seed 42 --methods all

例（Optuna の ``best_params_<study>.json`` を **CoPE（cope）のみ** に適用して他手法と比較）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain patent --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 --epochs 20 --methods all \\
    --optuna-best-json pnode_patent_runner/outputs/optuna/best_params_unified_vgae_cope.json

例（各手法ごとに Optuna を回した ``best_params`` を **手法キー別**に適用＝公平 HPO）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain patent --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 --epochs 20 --methods all \\
    --optuna-best-json-map pnode_patent_runner/outputs/optuna/optuna_paths_by_method.example.json

例（時間依存 Φ(z,year) / Neural ODE 年条件付け: CoPE→UnifiedVGAETD、P-NODE→勾配流 TD、Neural ODE→NeuralODEPredictorTime）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain patent --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 --epochs 10 --methods all \\
    --time-dependent-potential

例（学習曲線 PNG を出力: 損失・検証 AUC・potential/trajectory 等）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain patent --data data/processed/topic_info3.csv \\
    --year-range 2015 2019 --epochs 10 --methods pnode_energy \\
    --save-training-curves-dir pnode_patent_runner/outputs/cope_benchmark/curves

例（P-NODE-Energy-TD の .pt を保存 → HTML 可視化用）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain patent --data data/processed/topic_info3.csv \\
    --year-range 2015 2019 --epochs 10 --methods pnode_energy \\
    --save-checkpoint-dir pnode_patent_runner/outputs/cope_benchmark/ckpt
  python -m pnode_patent_runner.run_interactive_landscape_pnode_energy_td \\
    --data data/processed/topic_info3.csv --year-range 2015 2019 \\
    --load-checkpoint pnode_patent_runner/outputs/cope_benchmark/ckpt/pnode_energy_seed42.pt \\
    --output pnode_patent_runner/outputs/cope_landscape/map_pnode_energy_alt_dark_td.html

例（著者–論文・P-NODE_paper.ipynb 相当 CSV）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain arxiv \\
    --data notebooks/work/dataset/arxiv_cs_Data/arxiv_cs_embedded_2020-2026.csv \\
    --min-patents 5 \\
    --epochs 10 --seed 42 --methods all

例（著者–トピック・P-NODE_paper_topic.ipynb 相当。CSV に `topic` 列が必要）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain author_topic \\
    --min-patents 5 \\
    --epochs 10 --seed 42 --methods all
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_csv_for_domain(repo: Path, domain: str) -> Path:
    if domain == "patent":
        return repo / "notebooks/work/dataset/topic_info3.csv"
    # arxiv / author_topic は同一 ArXiv 埋め込み CSV（著者–論文は url、著者–トピックは topic 列を使用）
    # 全次元版 _full を優先（省略表示入りの CSV は preprocess で拒否される）
    for candidate in (
        repo / "data/processed/arxiv_cs_embedded_2020-2026_full.csv",
        repo / "notebooks/work/dataset/arxiv_cs_Data/arxiv_cs_embedded_2020-2026.csv",
        repo / "data/processed/arxiv_cs_embedded_2020-2026.csv",
    ):
        if candidate.is_file():
            return candidate
    return repo / "notebooks/work/dataset/arxiv_cs_Data/arxiv_cs_embedded_2020-2026.csv"


def _parse_horizon_gaps(s: str) -> List[int]:
    """カンマ区切り正整数。例 ``1,2,3`` → ``sorted(unique)``。"""
    raw = (s or "").strip()
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 1:
            raise SystemExit(f"--eval-horizon-gaps の各要素は 1 以上の整数: {v!r}")
        out.append(v)
    return sorted(set(out))


if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.cope_experiment import (
    BASELINE_KEYS,
    BASELINE_METHOD_SPECS,
    ModelBuildKw,
    build_baseline_model,
    load_author_paper_graph_bundle,
    load_author_topic_graph_bundle,
    load_cope_graph_bundle,
    parse_methods_csv,
    split_bundle_holdout_test_year,
)
from pnode_patent_runner.unified_training import (
    FUTURE_LINK_ECE_N_BINS,
    README_DEFAULT_BETA,
    README_DEFAULT_FUTURE_LINK_WEIGHT,
    README_DEFAULT_LATENT_PRED_WEIGHT,
    README_DEFAULT_NUM_NEG_FUTURE,
    README_DEFAULT_NUM_NEG_RECON,
    README_DEFAULT_POS_WEIGHT,
    README_DEFAULT_POTENTIAL_WEIGHT,
    README_DEFAULT_TRAJECTORY_WEIGHT,
    README_DEFAULT_ATTENTION_GROUND_WEIGHT,
    README_DEFAULT_POTENTIAL_REG_MODE,
    README_DEFAULT_TRAJECTORY_DELTA_SOURCE,
    README_DEFAULT_TRAJECTORY_GRAD_FLOOR,
    README_DEFAULT_TRAJECTORY_GRAD_FLOOR_WEIGHT,
    README_DEFAULT_TRAJECTORY_LOSS_TYPE,
    evaluate_future_link_metrics_for_horizon_gaps,
    evaluate_latent_rollout_metrics_for_horizon_gaps,
    evaluate_val_future_link_metrics,
    evaluate_val_future_link_metrics_for_years,
    train_model_improved,
)
from pnode_patent_runner.unified_training_td import (
    evaluate_future_link_metrics_for_horizon_gaps_td,
    evaluate_val_future_link_metrics_for_years_td,
    evaluate_val_future_link_metrics_td,
    train_model_td,
)
from pnode_patent_runner.unified_vgae_td import UnifiedVGAETD


def _load_optuna_best_json(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    ``run_optuna_unified_vgae`` が書き出す JSON から ``best_params`` とメタを返す。
    """
    if not path.is_file():
        raise SystemExit(f"Optuna JSON が見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    bp = raw.get("best_params")
    if not isinstance(bp, dict) or not bp:
        raise SystemExit(f"'best_params' が無いか空です: {path}")
    meta = {
        k: raw.get(k)
        for k in ("study_name", "storage", "best_value", "best_trial_number", "space", "cope_link_score")
        if k in raw
    }
    return meta, bp


def _model_kw_from_optuna_bp(args: Any, bp: Dict[str, Any]) -> ModelBuildKw:
    """run_optuna_unified_vgae の best_params から ModelBuildKw を組み立てる（全手法共通）。"""
    return ModelBuildKw(
        hidden_dim=int(bp.get("hidden_dim", args.hidden_dim)),
        latent_dim=int(bp.get("latent_dim", args.latent_dim)),
        link_score_mode=args.cope_link_score,
        cosine_logit_scale=float(bp.get("cosine_logit_scale", args.cosine_logit_scale)),
        w_pot_init=float(bp.get("w_pot_init", args.w_pot_init)),
        rnn_history_len=int(bp.get("rnn_history_len", args.rnn_history_len)),
        pnode_history_len=int(bp.get("pnode_history_len", args.pnode_history_len)),
        pnode_ode_method=str(args.pnode_ode_method),
        pnode_ode_n_steps=int(args.pnode_ode_n_steps),
        pnode_potential_feature=str(bp.get("pnode_potential_feature", args.pnode_potential_feature)),
        pnode_rff_frozen_basis=bool(bp.get("pnode_rff_frozen_basis", args.pnode_rff_frozen_basis)),
        pnode_density_calibrated=bool(bp.get("pnode_density_calibrated", args.pnode_density_calibrated)),
        pnode_density_log_weight=float(bp.get("pnode_density_log_weight", args.pnode_density_log_weight)),
        pnode_density_ema_momentum=float(bp.get("pnode_density_ema_momentum", args.pnode_density_ema_momentum)),
        pnode_hist_fuse_mode=str(bp.get("pnode_hist_fuse_mode", args.pnode_hist_fuse_mode)),
    )


def _train_kw_from_optuna_bp(args: argparse.Namespace, bp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_epochs": args.epochs,
        "potential_weight": float(bp.get("potential_weight", args.potential_weight)),
        "trajectory_weight": float(bp.get("trajectory_weight", args.trajectory_weight)),
        "lr": float(bp.get("lr", args.lr)),
        "latent_pred_weight": float(bp.get("latent_pred_weight", args.latent_pred_weight)),
        "future_link_weight": float(bp.get("future_link_weight", args.future_link_weight)),
        "num_neg_recon": args.num_neg_recon,
        "num_neg_future": args.num_neg_future,
        "beta": float(bp.get("beta", args.beta)),
        "pos_weight": float(bp.get("pos_weight", args.pos_weight)),
        "attention_ground_weight": float(
            bp.get("attention_ground_weight", args.attention_ground_weight)
        ),
        "potential_reg_mode": str(
            bp.get("potential_reg_mode", args.potential_reg_mode)
        ),
        "trajectory_delta_source": str(
            bp.get("trajectory_delta_source", args.trajectory_delta_source)
        ),
        "trajectory_loss_type": str(
            bp.get("trajectory_loss_type", args.trajectory_loss_type)
        ),
        "trajectory_grad_floor": float(
            bp.get("trajectory_grad_floor", args.trajectory_grad_floor)
        ),
        "trajectory_grad_floor_weight": float(
            bp.get("trajectory_grad_floor_weight", args.trajectory_grad_floor_weight)
        ),
    }


def _load_optuna_method_map(path: Path) -> Dict[str, Path]:
    """
    JSON: { "cope": "/abs/or/rel/best_params_unified_vgae_cope.json", "rnn": "...", ... }
    キーは static / rnn / neural_ode / pnode / pnode_residual / pnode_energy / cope。
    """
    if not path.is_file():
        raise SystemExit(f"Optuna マップ JSON が見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"Optuna マップは非空オブジェクトである必要があります: {path}")
    allowed = set(BASELINE_KEYS)
    out: Dict[str, Path] = {}
    for k, v in raw.items():
        key = str(k).strip().lower()
        if key not in allowed:
            raise SystemExit(
                f"Optuna マップの未知のキー: {k!r}。利用可能: {sorted(allowed)}（{path}）"
            )
        p = Path(str(v).strip())
        out[key] = p
    return out


def main() -> None:
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_out_dir = repo / "pnode_patent_runner/outputs/cope_benchmark"

    p = argparse.ArgumentParser(
        description="README 準拠パイプラインでの Temporal VGAE ベースライン比較",
    )
    p.add_argument(
        "--data-domain",
        type=str,
        choices=("patent", "arxiv", "author_topic"),
        default="patent",
        help="patent=企業–特許 / arxiv=著者–論文 / author_topic=著者–トピック（topic 列, P-NODE_paper_topic.ipynb）",
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="CSV パス。省略時はドメイン別の既定パス（arxiv/author_topic は ArXiv 埋め込み CSV を探索）",
    )
    p.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
    )
    p.add_argument("--years", type=str, default="")
    p.add_argument("--all-years", action="store_true")
    p.add_argument(
        "--min-patents",
        type=int,
        default=2,
        help="patent: 企業あたり最小特許数。arxiv/author_topic: 著者あたり最小論文（行）数（多く 5）",
    )
    p.add_argument(
        "--topic-column",
        type=str,
        default="topic",
        help="author_topic のトピック列名（既定 topic）",
    )
    p.add_argument(
        "--arxiv-year-min",
        type=int,
        default=2020,
        help="arxiv/author_topic 前処理の年フィルタ下限（--arxiv-no-year-filter で無効）",
    )
    p.add_argument(
        "--arxiv-year-max",
        type=int,
        default=2026,
        help="arxiv/author_topic 前処理の年フィルタ上限",
    )
    p.add_argument(
        "--arxiv-no-year-filter",
        action="store_true",
        help="arxiv/author_topic の年範囲フィルタを外す（CSV の全ての年）",
    )
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--methods",
        type=str,
        default="all",
        help="カンマ区切り: static,rnn,neural_ode,pnode,pnode_explicit,pnode_residual,pnode_energy,cope または all",
    )
    p.add_argument("--rnn-history-len", type=int, default=4)
    p.add_argument(
        "--pnode-history-len",
        type=int,
        default=1,
        metavar="K",
        help=(
            "P-NODE / Neural ODE: 直近 K 年の z を結合し融合層で 1 ステップ ODE 初値に射影（1=従来）。"
            "K>1 時は --pnode-hist-fuse-mode（GRU または Linear）。Neural ODE は K>1 で neural_ode_hist_fuse。"
        ),
    )
    p.add_argument(
        "--pnode-potential-feature",
        type=str,
        choices=("rff", "mlp"),
        default="mlp",
        help="P-NODE / P-NODE+Res の Φ ネット: rff=sin/cos 特徴、mlp=z から MLP（データ適応、既定）",
    )
    p.add_argument(
        "--pnode-rff-frozen-basis",
        action="store_true",
        help="pnode_potential_feature=rff のときのみ: 射影行列 B を学習しない（従来の固定 RFF 基底）",
    )
    p.add_argument(
        "--pnode-density-calibrated",
        action="store_true",
        help="P-NODE 系で CalibratedPotentialNet（Φ = φ_nn - w log p_hist）を使用（EMA 密度更新あり）",
    )
    p.add_argument(
        "--pnode-density-log-weight",
        type=float,
        default=1.0,
        help="--pnode-density-calibrated 時の log p_hist の係数の初期値",
    )
    p.add_argument(
        "--pnode-density-ema-momentum",
        type=float,
        default=0.05,
        help="--pnode-density-calibrated 時の μ の EMA モメンタム",
    )
    p.add_argument(
        "--pnode-hist-fuse-mode",
        type=str,
        choices=("linear", "gru"),
        default="gru",
        help="P-NODE 系で K>1 のときの履歴融合: gru（既定）または linear（従来）",
    )
    p.add_argument(
        "--pc-w-rho-init", type=float, default=0.1,
        help="PC-PNODE: w_ρ（密度引力強度）の初期値",
    )
    p.add_argument(
        "--pc-w-delta-init", type=float, default=0.1,
        help="PC-PNODE: w_Δ（トレンド引力強度）の初期値",
    )
    p.add_argument(
        "--pc-log-bandwidth-init", type=float, default=0.0,
        help="PC-PNODE: KDE帯域幅の log 初期値（exp(0)=1.0）",
    )
    p.add_argument(
        "--pc-density-align-weight", type=float, default=0.005,
        help="PC-PNODE: density-align 正則化損失の重み λ_da",
    )
    p.add_argument(
        "--pnode-ode-method",
        type=str,
        default="dopri5",
        choices=("dopri5", "rk4", "euler"),
        help="P-NODE 勾配流 ODE の積分法（rk4/euler は固定格子、--pnode-ode-n-steps で分割数）",
    )
    p.add_argument(
        "--pnode-ode-n-steps",
        type=int,
        default=4,
        help="rk4/euler 時の区間 [0,1] の分割数（大きいほど精密だが遅い）",
    )
    p.add_argument(
        "--loss-aux-warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help=(
            "N>1 のとき λ_pot・λ_traj を線形ランプ（epoch0 で 0、epoch N-1 で CLI 既定値）。"
            "N<=0 で無効（従来どおり一括）。N==1 は無効扱い（常に係数 1）。"
        ),
    )
    p.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="distance",
        help="README のインタラクティブ例に合わせ既定は distance",
    )
    p.add_argument("--cosine-logit-scale", type=float, default=5.0)
    p.add_argument("--w-pot-init", type=float, default=0.05)
    p.add_argument(
        "--beta",
        type=float,
        default=README_DEFAULT_BETA,
        help=f"KL 係数（README 既定 {README_DEFAULT_BETA}）",
    )
    p.add_argument(
        "--pos-weight",
        type=float,
        default=README_DEFAULT_POS_WEIGHT,
        help=f"再構成・未来リンク正例側（README 既定 {README_DEFAULT_POS_WEIGHT}）",
    )
    p.add_argument("--potential-weight", type=float, default=README_DEFAULT_POTENTIAL_WEIGHT)
    p.add_argument("--trajectory-weight", type=float, default=README_DEFAULT_TRAJECTORY_WEIGHT)
    p.add_argument(
        "--attention-ground-weight",
        type=float,
        default=README_DEFAULT_ATTENTION_GROUND_WEIGHT,
        help=(
            "pnode_explicit のみ: A(z)=exp(-Phi) を左パーティションの "
            "log(1+deg) 正規化目標に近づける補助損失の係数（0 で無効）"
        ),
    )
    p.add_argument(
        "--potential-reg-mode",
        type=str,
        default=README_DEFAULT_POTENTIAL_REG_MODE,
        choices=("l2", "log1p_sq", "centered_l2"),
        help="L_pot の形: l2(既定 0.01·Φ²) / log1p_sq / centered_l2（Φ 中心化）",
    )
    p.add_argument(
        "--trajectory-delta-source",
        type=str,
        default=README_DEFAULT_TRAJECTORY_DELTA_SOURCE,
        choices=("z", "mu"),
        help="L_traj の差分 (μ_{t+1}−·): z 再パラム(既定) または μ（_encoder 平均）",
    )
    p.add_argument(
        "--trajectory-loss-type",
        type=str,
        default=README_DEFAULT_TRAJECTORY_LOSS_TYPE,
        choices=("cosine", "smooth1mcos", "huber_vec"),
        help="L_traj: 余弦 / smooth_l1(1-cos) / 単位方向ベクトル差の smooth_l1",
    )
    p.add_argument(
        "--trajectory-grad-floor",
        type=float,
        default=README_DEFAULT_TRAJECTORY_GRAD_FLOOR,
        help="||∇Φ|| の希望下限; --trajectory-grad-floor-weight>0 のときペナ化",
    )
    p.add_argument(
        "--trajectory-grad-floor-weight",
        type=float,
        default=README_DEFAULT_TRAJECTORY_GRAD_FLOOR_WEIGHT,
        help="L_traj への (floor-||∇Φ||)_+^2 係数",
    )
    p.add_argument("--latent-pred-weight", type=float, default=README_DEFAULT_LATENT_PRED_WEIGHT)
    p.add_argument("--future-link-weight", type=float, default=README_DEFAULT_FUTURE_LINK_WEIGHT)
    p.add_argument("--num-neg-recon", type=int, default=README_DEFAULT_NUM_NEG_RECON)
    p.add_argument("--num-neg-future", type=int, default=README_DEFAULT_NUM_NEG_FUTURE)
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="省略時は {repo}/pnode_patent_runner/outputs/cope_benchmark/benchmark_<data_domain>_seed<seed>.json",
    )
    p.add_argument(
        "--optuna-best-json",
        type=str,
        default="",
        help=(
            "run_optuna_unified_vgae が出力した JSON（best_params 含む）。"
            "指定時、手法 key=cope の学習だけ best_params を上書き（他手法は CLI のまま）。"
            "--optuna-best-json-map に cope が含まれる場合はそちらを優先し、この引数は無視される。"
        ),
    )
    p.add_argument(
        "--optuna-best-json-map",
        type=str,
        default="",
        help=(
            "JSON ファイルパス。内容は { \"static\": \"path/to.json\", \"rnn\": \"...\", ... } 形式で、"
            "各手法キーに run_optuna_unified_vgae の出力を割り当てる（公平 HPO）。"
            "パスはリポジトリルートからの相対でも可。"
        ),
    )
    p.add_argument(
        "--holdout-test-year",
        type=int,
        default=None,
        metavar="YEAR",
        help=(
            "指定時、その年のエッジは学習損失に含めない。"
            "学習はその年より前のグラフのみ。hist_edges も学習期間で再構成。"
            "最終 AUC/AP は (直前年→Y) の future-link。--year-range に Y を含めること。"
        ),
    )
    p.add_argument(
        "--cope-density-calibrated",
        action="store_true",
        help="案1: CoPE のみ、潜在 μ の EMA 対角ガウス log p(z) で Φ=φ_nn - w log p を用いる（cope キー時のみ有効）",
    )
    p.add_argument(
        "--cope-density-log-weight",
        type=float,
        default=1.0,
        help="案1: log p 項の初期スケール（学習可能パラメータの初期値）",
    )
    p.add_argument(
        "--cope-density-ema-momentum",
        type=float,
        default=0.05,
        help="案1: μ の移動平均のモメンタム",
    )
    p.add_argument(
        "--time-dependent-potential",
        action="store_true",
        help=(
            "Φ(z,year): CoPE は UnifiedVGAETD、P-NODE は時間依存勾配流。"
            "Neural ODE は年埋め込み付きベクトル場（NeuralODEPredictorTime）。"
            "Static/RNN は従来どおり。"
        ),
    )
    p.add_argument(
        "--save-training-curves-dir",
        type=str,
        default="",
        metavar="DIR",
        help=(
            "各手法の学習曲線（損失・検証 AUC・potential/trajectory 等）を PNG で保存するディレクトリ。"
            "省略時は PNG を書き出さない。"
        ),
    )
    p.add_argument(
        "--save-checkpoint-dir",
        type=str,
        default="",
        metavar="DIR",
        help=(
            "各手法の学習後 state_dict を {key}_seed{seed}.pt で保存（メタ: year_min/max, hidden_dim 等）。"
            "P-NODE-Energy-TD の HTML 可視化: run_interactive_landscape_pnode_energy_td --load-checkpoint この .pt。"
        ),
    )
    p.add_argument(
        "--eval-horizon-gaps",
        type=str,
        default="",
        metavar="K_LIST",
        help=(
            "省略可。例: 1,2,3。sorted(years) 上で終端年から k ステップ遡った起点→終端の future-link を "
            "ロールアウト評価し、JSON に train_split_metrics_by_horizon_gap / final_metrics_by_horizon_gap を追加。"
        ),
    )
    args = p.parse_args()
    horizon_gaps = _parse_horizon_gaps(getattr(args, "eval_horizon_gaps", ""))

    data_str = (args.data or "").strip()
    data_path = Path(data_str) if data_str else _default_csv_for_domain(repo, args.data_domain)
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    try:
        method_keys = parse_methods_csv(args.methods)
    except ValueError as e:
        raise SystemExit(str(e))

    yr = tuple(args.year_range) if args.year_range is not None else None
    try:
        if args.data_domain == "patent":
            bundle = load_cope_graph_bundle(
                data_path,
                min_patents=args.min_patents,
                year_range=yr,
                years_csv=args.years,
                all_years=args.all_years,
            )
        else:
            ymin: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_min
            ymax: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_max
            if args.data_domain == "arxiv":
                bundle = load_author_paper_graph_bundle(
                    data_path,
                    min_papers=args.min_patents,
                    arxiv_year_min=ymin,
                    arxiv_year_max=ymax,
                    year_range=yr,
                    years_csv=args.years,
                    all_years=args.all_years,
                )
            else:
                bundle = load_author_topic_graph_bundle(
                    data_path,
                    min_papers=args.min_patents,
                    topic_column=args.topic_column,
                    arxiv_year_min=ymin,
                    arxiv_year_max=ymax,
                    year_range=yr,
                    years_csv=args.years,
                    all_years=args.all_years,
                )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))

    holdout = None
    if args.holdout_test_year is not None:
        try:
            holdout = split_bundle_holdout_test_year(bundle, int(args.holdout_test_year))
        except ValueError as e:
            raise SystemExit(str(e)) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"data_domain={args.data_domain}, csv={data_path}")
    print(
        f"グラフ: years={sorted(bundle.graphs.keys())}, "
        f"左パーティション数={bundle.num_corps}（企業または著者）, "
        f"N={bundle.total_n}, in_dim={bundle.in_dim}"
    )
    if holdout is not None:
        print(
            f"ホールドアウト: 学習年={list(holdout.train_years)}, "
            f"テスト遷移 {holdout.year_prev}→{holdout.holdout_test_year} "
            f"（{holdout.holdout_test_year} 年のエッジは学習に未使用）"
        )
    print(f"methods={method_keys}, epochs={args.epochs}, seed={args.seed}")
    if args.cope_density_calibrated:
        print(
            f"cope 案1: density_calibrated=True, log_w={args.cope_density_log_weight}, "
            f"ema_momentum={args.cope_density_ema_momentum}"
        )
    print(
        f"loss: beta={args.beta}, pos_weight={args.pos_weight}, "
        f"λ_lat={args.latent_pred_weight}, λ_fut={args.future_link_weight}, "
        f"λ_pot={args.potential_weight}, λ_traj={args.trajectory_weight}, "
        f"neg_recon={args.num_neg_recon}, neg_future={args.num_neg_future}"
    )
    if horizon_gaps:
        print(f"eval_horizon_gaps (index steps on sorted year keys): {horizon_gaps}")

    optuna_by_method: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], Path]] = {}
    optuna_map_path: Optional[Path] = None
    map_str = (args.optuna_best_json_map or "").strip()
    single_str = (args.optuna_best_json or "").strip()

    if map_str:
        optuna_map_path = Path(map_str)
        rel_base = repo
        for mkey, jpath in _load_optuna_method_map(optuna_map_path).items():
            p = jpath if jpath.is_absolute() else (rel_base / jpath)
            meta, bp = _load_optuna_best_json(p)
            optuna_by_method[mkey] = (meta, bp, p.resolve())
        print(f"Optuna マップを読込: {optuna_map_path.resolve()}")
        print(f"  → 次の手法に best_params を適用: {sorted(optuna_by_method.keys())}")

    if single_str:
        cope_path = Path(single_str)
        p = cope_path if cope_path.is_absolute() else (repo / cope_path)
        if "cope" in optuna_by_method:
            print(
                "警告: --optuna-best-json は無視（--optuna-best-json-map に cope が含まれる）"
            )
        else:
            meta, bp = _load_optuna_best_json(p)
            optuna_by_method["cope"] = (meta, bp, p.resolve())
            print(f"Optuna JSON を読込（cope）: {p.resolve()}")
            print(f"  → best_value={meta.get('best_value', 'N/A')}")

    if optuna_by_method:
        for mkey, (meta, _bp, jp) in sorted(optuna_by_method.items()):
            print(f"    [{mkey}] {jp}  (best_value={meta.get('best_value', 'N/A')})")
        loaded_methods = set(optuna_by_method.keys())
        if not loaded_methods.intersection(method_keys):
            print(
                "  警告: --methods に Optuna を適用した手法が含まれていません。"
                "読み込んだパラメータは使われません。"
            )
        elif "cope" in optuna_by_method and "cope" not in method_keys:
            print("  注意: cope 用の Optuna JSON はあるが --methods に cope が無い。")

    optuna_meta_cope: Optional[Dict[str, Any]] = None
    if "cope" in optuna_by_method:
        optuna_meta_cope = optuna_by_method["cope"][0]

    g_tr = holdout.graphs_train if holdout is not None else bundle.graphs
    h_tr = holdout.hist_edges_train if holdout is not None else bundle.hist_edges
    _train_years_sorted = sorted(g_tr.keys())
    _y_calendar_min = _train_years_sorted[0]
    _y_calendar_max = _train_years_sorted[-1]
    if args.time_dependent_potential and holdout is not None:
        # ホールドアウト年は学習グラフに無いが、TD の decode はテスト遷移の year_next（例: 2020）で
        # Φ(z,·) を参照するため、年埋め込みの暦年範囲はフルデータ（graphs_full）に合わせる。
        _full_years = sorted(holdout.graphs_full.keys())
        _y_calendar_min = _full_years[0]
        _y_calendar_max = _full_years[-1]
    if args.time_dependent_potential:
        _td_range_note = (
            "（ホールドアウトテスト年を含む暦年範囲）"
            if holdout is not None
            else "（学習グラフの暦年範囲）"
        )
        print(
            f"time_dependent_potential=True (Φ year range: {_y_calendar_min}..{_y_calendar_max} "
            f"{_td_range_note}"
        )

    if args.cope_density_calibrated and args.time_dependent_potential:
        print(
            "警告: --time-dependent-potential 時は CoPE の density_calibrated を未対応のため無視します。"
        )

    mb = ModelBuildKw(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        link_score_mode=args.cope_link_score,
        cosine_logit_scale=args.cosine_logit_scale,
        w_pot_init=args.w_pot_init,
        rnn_history_len=args.rnn_history_len,
        pnode_history_len=int(args.pnode_history_len),
        pnode_ode_method=str(args.pnode_ode_method),
        pnode_ode_n_steps=int(args.pnode_ode_n_steps),
        density_calibrated_potential=bool(args.cope_density_calibrated),
        density_log_weight=float(args.cope_density_log_weight),
        density_ema_momentum=float(args.cope_density_ema_momentum),
        time_dependent_potential=bool(args.time_dependent_potential),
        year_min=int(_y_calendar_min),
        year_max=int(_y_calendar_max),
        pnode_potential_feature=str(args.pnode_potential_feature),
        pnode_rff_frozen_basis=bool(args.pnode_rff_frozen_basis),
        pnode_density_calibrated=bool(args.pnode_density_calibrated),
        pnode_density_log_weight=float(args.pnode_density_log_weight),
        pnode_density_ema_momentum=float(args.pnode_density_ema_momentum),
        pnode_hist_fuse_mode=str(args.pnode_hist_fuse_mode),
        pc_w_rho_init=float(args.pc_w_rho_init),
        pc_w_delta_init=float(args.pc_w_delta_init),
        pc_log_bandwidth_init=float(args.pc_log_bandwidth_init),
        pc_density_align_weight=float(args.pc_density_align_weight),
    )

    name_by_key = {k: n for n, k in BASELINE_METHOD_SPECS}
    results: List[Dict[str, Any]] = []
    table_rows: List[Tuple[Any, ...]] = []

    for key in method_keys:
        label = name_by_key[key]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        entry = optuna_by_method.get(key)
        use_optuna = entry is not None
        optuna_bp: Optional[Dict[str, Any]] = None
        if use_optuna:
            _meta, optuna_bp, _src = entry
            assert optuna_bp is not None
            mb_run = _model_kw_from_optuna_bp(args, optuna_bp)
        else:
            mb_run = mb
        mb_run = dataclasses.replace(
            mb_run,
            time_dependent_potential=mb.time_dependent_potential,
            year_min=mb.year_min,
            year_max=mb.year_max,
            pnode_history_len=mb.pnode_history_len,
            pnode_ode_method=mb.pnode_ode_method,
            pnode_ode_n_steps=mb.pnode_ode_n_steps,
            pnode_potential_feature=mb.pnode_potential_feature,
            pnode_rff_frozen_basis=mb.pnode_rff_frozen_basis,
            pnode_density_calibrated=mb.pnode_density_calibrated,
            pnode_density_log_weight=mb.pnode_density_log_weight,
            pnode_density_ema_momentum=mb.pnode_density_ema_momentum,
            pnode_hist_fuse_mode=mb.pnode_hist_fuse_mode,
            pc_w_rho_init=mb.pc_w_rho_init,
            pc_w_delta_init=mb.pc_w_delta_init,
            pc_log_bandwidth_init=mb.pc_log_bandwidth_init,
            pc_density_align_weight=mb.pc_density_align_weight,
        )
        if key == "cope":
            mb_run = dataclasses.replace(
                mb_run,
                density_calibrated_potential=mb.density_calibrated_potential,
                density_log_weight=mb.density_log_weight,
                density_ema_momentum=mb.density_ema_momentum,
            )
        model = build_baseline_model(key, device, bundle, mb_run)
        if use_optuna:
            tw = _train_kw_from_optuna_bp(args, optuna_bp)
            if isinstance(model, UnifiedVGAETD):
                _, _, best_auc, hist = train_model_td(
                    model,
                    g_tr,
                    bundle.num_corps,
                    h_tr,
                    loss_aux_warmup_epochs=int(args.loss_aux_warmup_epochs),
                    **tw,
                )
            else:
                _, _, best_auc, hist = train_model_improved(
                    model,
                    g_tr,
                    bundle.num_corps,
                    h_tr,
                    loss_aux_warmup_epochs=int(args.loss_aux_warmup_epochs),
                    **tw,
                )
        else:
            _da_w = float(mb_run.pc_density_align_weight) if key == "pnode_pc" else 0.0
            train_kw = dict(
                num_epochs=args.epochs,
                potential_weight=args.potential_weight,
                trajectory_weight=args.trajectory_weight,
                lr=args.lr,
                latent_pred_weight=args.latent_pred_weight,
                future_link_weight=args.future_link_weight,
                num_neg_recon=args.num_neg_recon,
                num_neg_future=args.num_neg_future,
                beta=args.beta,
                pos_weight=args.pos_weight,
                density_align_weight=_da_w,
                attention_ground_weight=float(args.attention_ground_weight),
                potential_reg_mode=str(args.potential_reg_mode),
                trajectory_delta_source=str(args.trajectory_delta_source),
                trajectory_loss_type=str(args.trajectory_loss_type),
                trajectory_grad_floor=float(args.trajectory_grad_floor),
                trajectory_grad_floor_weight=float(args.trajectory_grad_floor_weight),
            )
            if isinstance(model, UnifiedVGAETD):
                _, _, best_auc, hist = train_model_td(
                    model,
                    g_tr,
                    bundle.num_corps,
                    h_tr,
                    loss_aux_warmup_epochs=int(args.loss_aux_warmup_epochs),
                    **train_kw,
                )
            else:
                _, _, best_auc, hist = train_model_improved(
                    model,
                    g_tr,
                    bundle.num_corps,
                    h_tr,
                    loss_aux_warmup_epochs=int(args.loss_aux_warmup_epochs),
                    **train_kw,
                )
        _save_ckpt = (getattr(args, "save_checkpoint_dir", None) or "").strip()
        if _save_ckpt:
            _cd = Path(_save_ckpt)
            _cd.mkdir(parents=True, exist_ok=True)
            _cp = _cd / f"{key}_seed{args.seed}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "year_min": int(mb.year_min),
                    "year_max": int(mb.year_max),
                    "hidden_dim": int(mb_run.hidden_dim),
                    "latent_dim": int(mb_run.latent_dim),
                    "cope_link_score": str(args.cope_link_score),
                    "cosine_logit_scale": float(args.cosine_logit_scale),
                    "method_key": key,
                    "data_domain": str(args.data_domain),
                    "seed": int(args.seed),
                },
                _cp,
            )
            print(f"  Saved checkpoint: {_cp.resolve()}")
        if isinstance(model, UnifiedVGAETD):
            val_train_m = evaluate_val_future_link_metrics_td(
                model, g_tr, bundle.num_corps, device
            )
        else:
            val_train_m = evaluate_val_future_link_metrics(
                model, g_tr, bundle.num_corps, device
            )
        train_split_auc = val_train_m["auc"]
        train_split_ap = val_train_m["ap"]
        train_split_ece = val_train_m.get("ece", float("nan"))
        if holdout is not None:
            if isinstance(model, UnifiedVGAETD):
                val_h = evaluate_val_future_link_metrics_for_years_td(
                    model,
                    holdout.graphs_full,
                    bundle.num_corps,
                    device,
                    holdout.year_prev,
                    holdout.holdout_test_year,
                )
            else:
                val_h = evaluate_val_future_link_metrics_for_years(
                    model,
                    holdout.graphs_full,
                    bundle.num_corps,
                    device,
                    holdout.year_prev,
                    holdout.holdout_test_year,
                )
            final_auc = val_h["auc"]
            final_ap = val_h["ap"]
            final_ece = val_h.get("ece", float("nan"))
        else:
            final_auc = train_split_auc
            final_ap = train_split_ap
            final_ece = train_split_ece
        table_rows.append(
            (
                label,
                key,
                best_auc,
                train_split_auc,
                train_split_ap,
                train_split_ece,
                final_auc,
                final_ap,
                final_ece,
            )
        )
        auc_curve = []
        for x in hist["val_auc"]:
            auc_curve.append(None if (isinstance(x, float) and np.isnan(x)) else float(x))
        row: Dict[str, Any] = {
            "label": label,
            "key": key,
            "best_val_auc": best_auc,
            "final_val_auc": final_auc,
            "final_val_ap": final_ap,
            "final_val_ece": final_ece,
            "train_split_val_auc": train_split_auc,
            "train_split_val_ap": train_split_ap,
            "train_split_val_ece": train_split_ece,
            "val_auc_per_epoch": auc_curve,
        }
        if hist.get("loss"):
            row["train_loss_per_epoch"] = [float(x) for x in hist["loss"]]
        tc_hist = hist.get("train_components")
        if isinstance(tc_hist, dict) and tc_hist:
            row["train_components_per_epoch"] = {
                k: [float(x) for x in v] for k, v in tc_hist.items() if isinstance(v, list)
            }
        leb = hist.get("last_epoch_breakdown")
        if isinstance(leb, dict) and leb:
            row["last_epoch_train_breakdown"] = {k: float(v) for k, v in leb.items()}
        if holdout is not None:
            row["holdout_test_year"] = holdout.holdout_test_year
            row["holdout_year_prev"] = holdout.year_prev
            row["train_years"] = list(holdout.train_years)
        if use_optuna:
            row["optuna_tuned"] = True
            row["optuna_tuned_method_key"] = key
            if key == "cope":
                row["optuna_tuned_cope"] = True
        if horizon_gaps:
            if isinstance(model, UnifiedVGAETD):
                row["train_split_metrics_by_horizon_gap"] = (
                    evaluate_future_link_metrics_for_horizon_gaps_td(
                        model, g_tr, bundle.num_corps, device, horizon_gaps, year_next=None
                    )
                )
                if holdout is not None:
                    row["final_metrics_by_horizon_gap"] = (
                        evaluate_future_link_metrics_for_horizon_gaps_td(
                            model,
                            holdout.graphs_full,
                            bundle.num_corps,
                            device,
                            horizon_gaps,
                            year_next=int(holdout.holdout_test_year),
                        )
                    )
                else:
                    row["final_metrics_by_horizon_gap"] = dict(
                        row["train_split_metrics_by_horizon_gap"]
                    )
            else:
                row["train_split_metrics_by_horizon_gap"] = (
                    evaluate_future_link_metrics_for_horizon_gaps(
                        model, g_tr, bundle.num_corps, device, horizon_gaps, year_next=None
                    )
                )
                if holdout is not None:
                    row["final_metrics_by_horizon_gap"] = (
                        evaluate_future_link_metrics_for_horizon_gaps(
                            model,
                            holdout.graphs_full,
                            bundle.num_corps,
                            device,
                            horizon_gaps,
                            year_next=int(holdout.holdout_test_year),
                        )
                    )
                else:
                    row["final_metrics_by_horizon_gap"] = dict(
                        row["train_split_metrics_by_horizon_gap"]
                    )
            row["train_split_latent_metrics_by_horizon_gap"] = (
                evaluate_latent_rollout_metrics_for_horizon_gaps(
                    model, g_tr, bundle.num_corps, device, horizon_gaps, year_next=None
                )
            )
            if holdout is not None:
                row["final_latent_metrics_by_horizon_gap"] = (
                    evaluate_latent_rollout_metrics_for_horizon_gaps(
                        model,
                        holdout.graphs_full,
                        bundle.num_corps,
                        device,
                        horizon_gaps,
                        year_next=int(holdout.holdout_test_year),
                    )
                )
            else:
                row["final_latent_metrics_by_horizon_gap"] = dict(
                    row["train_split_latent_metrics_by_horizon_gap"]
                )
        _curve_dir = (getattr(args, "save_training_curves_dir", None) or "").strip()
        if _curve_dir:
            from pnode_patent_runner.plot_training_curves import save_training_history_png

            _d = Path(_curve_dir)
            _d.mkdir(parents=True, exist_ok=True)
            _png = _d / f"training_curves_{key}.png"
            save_training_history_png(
                hist,
                _png,
                title=f"{label}  seed={args.seed}",
            )
            print(f"  Saved training curves: {_png.resolve()}")
        results.append(row)
        tag = f" [{key}←Optuna]" if use_optuna else ""
        if holdout is not None:
            print(
                f"\n[{label}]{tag} best_val_auc={best_auc:.4f}  "
                f"train_split_auc={train_split_auc:.4f}  train_split_ap={train_split_ap:.4f}  "
                f"train_split_ece={train_split_ece:.4f}  "
                f"HOLDOUT {holdout.year_prev}→{holdout.holdout_test_year}: "
                f"auc={final_auc:.4f} ap={final_ap:.4f} ece={final_ece:.4f}  "
                f"per_epoch={[round(x, 4) for x in hist['val_auc']]}"
            )
        else:
            print(
                f"\n[{label}]{tag} best_val_auc={best_auc:.4f}  final_val_auc={final_auc:.4f}  "
                f"final_val_ap={final_ap:.4f}  final_val_ece={final_ece:.4f}  "
                f"per_epoch={[round(x, 4) for x in hist['val_auc']]}"
            )
        if horizon_gaps:
            ho = row.get("final_metrics_by_horizon_gap", {})
            print(
                f"  horizon AUC (final): { {gk: float(hv['auc']) for gk, hv in sorted(ho.items(), key=lambda x: int(x[0]))} }"
            )
            lz = row.get("final_latent_metrics_by_horizon_gap", {})
            if lz:
                mse_h = {
                    gk: float(hv["mse"])
                    for gk, hv in sorted(lz.items(), key=lambda x: int(x[0]))
                    if isinstance(hv, dict) and "mse" in hv
                }
                dacc_h = {
                    gk: float(hv["direction_agreement"])
                    for gk, hv in sorted(lz.items(), key=lambda x: int(x[0]))
                    if isinstance(hv, dict) and "direction_agreement" in hv
                }
                print(f"  horizon latent MSE (final): {mse_h}")
                print(f"  horizon trend DirAcc (final): {dacc_h}")

    print("\n" + "=" * 72)
    if holdout is not None:
        h = holdout
        print(
            f"{'method':<14} {'key':<10} {'best_auc':>9} {'tr_auc':>9} {'tr_ap':>9} {'tr_ece':>9} "
            f"{'ho_auc':>9} {'ho_ap':>9} {'ho_ece':>9}  (H={h.year_prev}→{h.holdout_test_year})"
        )
        print("-" * 96)
        for label, key, best_auc, tsa, tsp, tse, f_auc, f_ap, fse in table_rows:
            print(
                f"{label:<14} {key:<10} {best_auc:9.4f} {tsa:9.4f} {tsp:9.4f} {tse:9.4f} "
                f"{f_auc:9.4f} {f_ap:9.4f} {fse:9.4f}"
            )
    else:
        print(
            f"{'method':<14} {'key':<12} {'best_auc':>10} {'final_auc':>10} "
            f"{'final_ap':>10} {'final_ece':>10}"
        )
        print("-" * 78)
        for label, key, best_auc, tsa, tsp, tse, f_auc, f_ap, fse in table_rows:
            print(
                f"{label:<14} {key:<12} {best_auc:10.4f} {f_auc:10.4f} {f_ap:10.4f} {fse:10.4f}"
            )
    print("=" * 96 if holdout is not None else "=" * 78)

    default_out_dir.mkdir(parents=True, exist_ok=True)
    _td_tag = "_td" if args.time_dependent_potential else ""
    out_json = (
        Path(args.output_json)
        if args.output_json.strip()
        else default_out_dir / f"benchmark_{args.data_domain}{_td_tag}_seed{args.seed}.json"
    )
    payload = {
        "readme_pipeline": True,
        "future_link_ece_n_bins": int(FUTURE_LINK_ECE_N_BINS),
        "eval_horizon_gaps": horizon_gaps,
        "latent_horizon_metrics_note": (
            "mse/mae: mean over graphs[year_next].active_mask nodes of (z_rollout - mu_encoder). "
            "direction_agreement: fraction where sign(||mu||_t-||mu||_{t-k}) equals sign(||z||_pred-||z||_start) "
            "among nodes with nonzero true increment; z uses eval encode (deterministic mu)."
        ),
        "data_domain": args.data_domain,
        "data": str(data_path),
        "time_dependent_potential": bool(args.time_dependent_potential),
        "phi_year_min": int(_y_calendar_min),
        "phi_year_max": int(_y_calendar_max),
        **(
            {"topic_column": args.topic_column}
            if args.data_domain == "author_topic"
            else {}
        ),
        "holdout_test_year": args.holdout_test_year,
        "years": sorted(bundle.graphs.keys()),
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dim": int(args.hidden_dim),
        "latent_dim": int(args.latent_dim),
        "link_score_mode": args.cope_link_score,
        "cope_density_calibrated": bool(args.cope_density_calibrated),
        "cope_density_log_weight": float(args.cope_density_log_weight),
        "cope_density_ema_momentum": float(args.cope_density_ema_momentum),
        "loss": {
            "beta": args.beta,
            "pos_weight": args.pos_weight,
            "latent_pred_weight": args.latent_pred_weight,
            "future_link_weight": args.future_link_weight,
            "potential_weight": args.potential_weight,
            "trajectory_weight": args.trajectory_weight,
            "num_neg_recon": args.num_neg_recon,
            "num_neg_future": args.num_neg_future,
            "loss_aux_warmup_epochs": int(args.loss_aux_warmup_epochs),
            "attention_ground_weight": float(args.attention_ground_weight),
            "potential_reg_mode": str(args.potential_reg_mode),
            "trajectory_delta_source": str(args.trajectory_delta_source),
            "trajectory_loss_type": str(args.trajectory_loss_type),
            "trajectory_grad_floor": float(args.trajectory_grad_floor),
            "trajectory_grad_floor_weight": float(args.trajectory_grad_floor_weight),
        },
        "pnode_history_len": int(args.pnode_history_len),
        "pnode_ode_method": str(args.pnode_ode_method),
        "pnode_ode_n_steps": int(args.pnode_ode_n_steps),
        "pnode_potential_feature": str(args.pnode_potential_feature),
        "pnode_rff_frozen_basis": bool(args.pnode_rff_frozen_basis),
        "pnode_density_calibrated": bool(args.pnode_density_calibrated),
        "pnode_density_log_weight": float(args.pnode_density_log_weight),
        "pnode_density_ema_momentum": float(args.pnode_density_ema_momentum),
        "pnode_hist_fuse_mode": str(args.pnode_hist_fuse_mode),
        "optuna_best_json": (
            str(optuna_by_method["cope"][2]) if "cope" in optuna_by_method else None
        ),
        "optuna_best_json_map": str(optuna_map_path.resolve())
        if optuna_map_path is not None
        else None,
        "optuna_meta": optuna_meta_cope,
        "optuna_meta_by_method": {
            k: v[0] for k, v in sorted(optuna_by_method.items())
        },
        "optuna_best_params_by_method": {
            k: v[1] for k, v in sorted(optuna_by_method.items())
        },
        "cope_optuna_best_params": (
            optuna_by_method["cope"][1] if "cope" in optuna_by_method else None
        ),
        "results": results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_json}")


if __name__ == "__main__":
    main()
