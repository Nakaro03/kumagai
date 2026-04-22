#!/usr/bin/env python3
"""
Temporal VGAE 系のハイパーパラメータを Optuna で調整する（**CoPE だけでなくベースラインも同一探索空間で公平に HPO 可能**）。

- 目的: ``train_model_improved`` 実行中の **最終 2 年 future-link ROC-AUC**（エポックごとの最大 = ``best_val_auc``）を **最大化**。
- データ: ``run_benchmark_comparison`` と同型の ``--data-domain`` / CSV / 年範囲。
- 任意: ``--holdout-test-year Y`` で学習を ``y<Y`` に限定（``hist_edges`` 再構成）。探索目的は **学習グラフの最後の2年** の AUC（テスト年 Y は trial に含めない）。
- ``--method cope|static|rnn|neural_ode|pnode`` で対象モデルを切替（既定 ``cope``）。各手法 **同じ trial 数** で別 study を回すと探索予算が揃う。

依存: ``pip install optuna``（``requirements.txt`` に記載）

例:
  python -m pnode_patent_runner.run_optuna_unified_vgae \\
    --data-domain arxiv \\
    --data data/processed/arxiv_cs_embedded_2020-2026_full.csv \\
    --year-range 2020 2026 --arxiv-year-min 2020 --arxiv-year-max 2026 \\
    --min-patents 5 --epochs 8 --n-trials 30 \\
    --cope-link-score cosine \\
    --storage sqlite:///pnode_patent_runner/outputs/optuna/unified_vgae_arxiv.db \\
    --study-name cope_arxiv_cosine

  python -m pnode_patent_runner.run_optuna_unified_vgae \\
    --data-domain patent \\
    --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 --min-patents 2 \\
    --epochs 10 --n-trials 20 \\
    --storage sqlite:///pnode_patent_runner/outputs/optuna/unified_vgae_patent.db
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import optuna
except ImportError as e:
    raise SystemExit(
        "Optuna が見つかりません。次を実行してください: pip install optuna\n" + str(e)
    ) from e

from pnode_patent_runner.cope_experiment import (
    ModelBuildKw,
    build_baseline_model,
    load_author_paper_graph_bundle,
    load_author_topic_graph_bundle,
    load_cope_graph_bundle,
    split_bundle_holdout_test_year,
)
from pnode_patent_runner.unified_training import train_model_improved
from pnode_patent_runner.unified_training_td import train_model_td
from pnode_patent_runner.unified_vgae_td import UnifiedVGAETD

OPTUNA_METHOD_CHOICES = ("cope", "static", "rnn", "neural_ode", "pnode")


def _default_csv_patent(repo: Path) -> Path:
    return repo / "notebooks/work/dataset/topic_info3.csv"


def _default_csv_arxiv(repo: Path) -> Path:
    for candidate in (
        repo / "data/processed/arxiv_cs_embedded_2020-2026_full.csv",
        repo / "notebooks/work/dataset/arxiv_cs_Data/arxiv_cs_embedded_2020-2026.csv",
        repo / "data/processed/arxiv_cs_embedded_2020-2026.csv",
    ):
        if candidate.is_file():
            return candidate
    return repo / "notebooks/work/dataset/arxiv_cs_Data/arxiv_cs_embedded_2020-2026.csv"


def _suggest_training_hparams(
    trial: "optuna.Trial",
    *,
    space: str,
    link_score_mode: str,
    tune_hidden_dim: bool,
    fixed_hidden_dim: int,
) -> Dict[str, Any]:
    """学習ループ・損失係数・（任意）hidden_dim を trial からサンプル。"""
    if space == "minimal":
        lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)
        beta = trial.suggest_float("beta", 0.005, 0.02, log=True)
        pos_weight = trial.suggest_float("pos_weight", 3.0, 8.0)
        latent_pred_weight = trial.suggest_float("latent_pred_weight", 0.5, 1.5)
        future_link_weight = trial.suggest_float("future_link_weight", 5.0, 15.0)
        potential_weight = trial.suggest_float("potential_weight", 0.005, 0.02, log=True)
        trajectory_weight = trial.suggest_float("trajectory_weight", 0.02, 0.08)
        w_pot_init = trial.suggest_float("w_pot_init", 0.02, 0.12)
        cosine_logit_scale = (
            trial.suggest_float("cosine_logit_scale", 4.0, 7.0)
            if link_score_mode == "cosine"
            else 5.0
        )
    elif space == "wide":
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        beta = trial.suggest_float("beta", 0.001, 0.1, log=True)
        pos_weight = trial.suggest_float("pos_weight", 1.0, 15.0)
        latent_pred_weight = trial.suggest_float("latent_pred_weight", 0.1, 5.0)
        future_link_weight = trial.suggest_float("future_link_weight", 1.0, 30.0)
        potential_weight = trial.suggest_float("potential_weight", 0.001, 0.1, log=True)
        trajectory_weight = trial.suggest_float("trajectory_weight", 0.01, 0.2)
        w_pot_init = trial.suggest_float("w_pot_init", 0.0, 0.2)
        cosine_logit_scale = (
            trial.suggest_float("cosine_logit_scale", 2.0, 12.0)
            if link_score_mode == "cosine"
            else 5.0
        )
    else:
        # default: README 付近を中心に探索
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        beta = trial.suggest_float("beta", 0.005, 0.03, log=True)
        pos_weight = trial.suggest_float("pos_weight", 2.0, 10.0)
        latent_pred_weight = trial.suggest_float("latent_pred_weight", 0.3, 2.0)
        future_link_weight = trial.suggest_float("future_link_weight", 4.0, 20.0)
        potential_weight = trial.suggest_float("potential_weight", 0.003, 0.03, log=True)
        trajectory_weight = trial.suggest_float("trajectory_weight", 0.02, 0.12)
        w_pot_init = trial.suggest_float("w_pot_init", 0.01, 0.15)
        cosine_logit_scale = (
            trial.suggest_float("cosine_logit_scale", 3.5, 8.0)
            if link_score_mode == "cosine"
            else 5.0
        )

    hidden_dim = (
        trial.suggest_categorical("hidden_dim", [64, 96, 128, 192, 256])
        if tune_hidden_dim
        else int(fixed_hidden_dim)
    )

    out: Dict[str, Any] = {
        "lr": lr,
        "beta": beta,
        "pos_weight": pos_weight,
        "latent_pred_weight": latent_pred_weight,
        "future_link_weight": future_link_weight,
        "potential_weight": potential_weight,
        "trajectory_weight": trajectory_weight,
        "w_pot_init": w_pot_init,
        "hidden_dim": int(hidden_dim),
    }
    if link_score_mode == "cosine":
        out["cosine_logit_scale"] = float(cosine_logit_scale)
    return out


def _load_bundle(args: argparse.Namespace, repo: Path):
    data_str = (args.data or "").strip()
    if args.data_domain == "patent":
        data_path = Path(data_str) if data_str else _default_csv_patent(repo)
    elif args.data_domain in ("arxiv", "author_topic"):
        data_path = Path(data_str) if data_str else _default_csv_arxiv(repo)
    else:
        raise SystemExit(f"不明な data-domain: {args.data_domain}")

    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    yr = tuple(args.year_range) if args.year_range is not None else None
    if args.data_domain == "patent":
        return data_path, load_cope_graph_bundle(
            data_path,
            min_patents=args.min_patents,
            year_range=yr,
            years_csv=args.years,
            all_years=args.all_years,
        )

    ymin = None if args.arxiv_no_year_filter else args.arxiv_year_min
    ymax = None if args.arxiv_no_year_filter else args.arxiv_year_max
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
    return data_path, bundle


def main() -> None:
    warnings.filterwarnings("ignore")
    repo = _REPO_ROOT
    default_db_dir = repo / "pnode_patent_runner/outputs/optuna"
    default_storage = f"sqlite:///{default_db_dir / 'unified_vgae_study.db'}"

    p = argparse.ArgumentParser(
        description="Optuna で CoPE / ベースラインを調整（val future-link AUC 最大化）。--method で対象を切替。",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=OPTUNA_METHOD_CHOICES,
        default="cope",
        help="cope=UnifiedVGAE / 他は BenchmarkTemporalVGAE の variant。公平な HPO では各手法同じ --n-trials で別 study を回す。",
    )
    p.add_argument(
        "--data-domain",
        type=str,
        choices=("patent", "arxiv", "author_topic"),
        default="arxiv",
    )
    p.add_argument("--data", type=str, default="", help="CSV。省略時はドメイン別の既定パス")
    p.add_argument("--year-range", nargs=2, type=int, default=None, metavar=("START", "END"))
    p.add_argument("--years", type=str, default="")
    p.add_argument("--all-years", action="store_true")
    p.add_argument("--min-patents", type=int, default=5)
    p.add_argument("--topic-column", type=str, default="topic")
    p.add_argument("--arxiv-year-min", type=int, default=2020)
    p.add_argument("--arxiv-year-max", type=int, default=2026)
    p.add_argument("--arxiv-no-year-filter", action="store_true")

    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128, help="--no-tune-hidden 時の固定値")
    p.add_argument(
        "--no-tune-hidden",
        action="store_true",
        help="hidden_dim を --hidden-dim 固定（探索から外す）",
    )
    p.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="cosine",
    )
    p.add_argument(
        "--space",
        type=str,
        choices=("default", "minimal", "wide"),
        default="default",
        help="探索空間の広さ（default=README 付近 / minimal=狭い / wide=広い）",
    )

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna RDB URL。省略時は outputs/optuna/unified_vgae_study.db（sqlite）",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default="",
        help="Optuna study 名。省略時は cope→unified_vgae_cope、それ以外→optuna_baseline_<method>",
    )
    p.add_argument(
        "--rnn-history-len",
        type=int,
        default=4,
        help="RNN ベースラインの履歴長（--method rnn 時、探索では 2〜6 も試す）",
    )
    p.add_argument(
        "--tune-latent-dim",
        action="store_true",
        help="各 trial で latent_dim を {2,4,8,16} から探索（指定しない場合は --latent-dim 固定）",
    )
    p.add_argument(
        "--pnode-history-len",
        type=int,
        default=1,
        metavar="K",
        help="--method pnode / pnode_energy 時: BenchmarkTemporalVGAE の履歴長（ベンチと同一意味）",
    )
    p.add_argument(
        "--pnode-ode-method",
        type=str,
        default="dopri5",
        choices=("dopri5", "rk4", "euler"),
        help="P-NODE 勾配流の積分法（run_benchmark_comparison と揃える）",
    )
    p.add_argument(
        "--pnode-ode-n-steps",
        type=int,
        default=4,
        help="rk4/euler 時の [0,1] 分割数",
    )
    p.add_argument(
        "--loss-aux-warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help="λ_pot・λ_traj の線形ウォームアップ（unified_training と同一。0 で無効）",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="最良 trial の要約を JSON で保存（省略時は storage 隣の best_params_<study>.json）",
    )
    p.add_argument(
        "--num-neg-recon",
        type=int,
        default=0,
        help="0（既定）で train_model_improved の README 既定を使用。>0 で上書き",
    )
    p.add_argument(
        "--num-neg-future",
        type=int,
        default=0,
        help="0（既定）で README 既定。>0 で上書き",
    )
    p.add_argument(
        "--holdout-test-year",
        type=int,
        default=None,
        metavar="YEAR",
        help=(
            "指定時、学習はその年より前のグラフのみ（run_benchmark_comparison のホールドアウトと同じ）。"
            "テスト年 Y は探索に含めない。"
        ),
    )
    p.add_argument(
        "--cope-density-calibrated",
        action="store_true",
        help="--method cope のみ: 案1の密度校準ポテンシャル（EMA 対角ガウス log p）",
    )
    p.add_argument("--cope-density-log-weight", type=float, default=1.0)
    p.add_argument("--cope-density-ema-momentum", type=float, default=0.05)
    p.add_argument(
        "--time-dependent-potential",
        action="store_true",
        help=(
            "Φ(z,year) を用いる学習（CoPE→UnifiedVGAETD、P-NODE→時間依存勾配流）。"
            "study 名の既定に _td が付く。"
        ),
    )
    args = p.parse_args()

    from pnode_patent_runner.unified_training import (
        README_DEFAULT_NUM_NEG_FUTURE,
        README_DEFAULT_NUM_NEG_RECON,
    )

    neg_r = args.num_neg_recon if args.num_neg_recon > 0 else README_DEFAULT_NUM_NEG_RECON
    neg_f = args.num_neg_future if args.num_neg_future > 0 else README_DEFAULT_NUM_NEG_FUTURE

    data_path, bundle = _load_bundle(args, repo)
    holdout = None
    if args.holdout_test_year is not None:
        try:
            holdout = split_bundle_holdout_test_year(bundle, int(args.holdout_test_year))
        except ValueError as e:
            raise SystemExit(str(e)) from e
    g_tr = holdout.graphs_train if holdout is not None else bundle.graphs
    h_tr = holdout.hist_edges_train if holdout is not None else bundle.hist_edges

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storage = (args.storage or "").strip() or default_storage
    if storage.startswith("sqlite:///") and not Path(
        storage.replace("sqlite:///", "", 1)
    ).parent.is_dir():
        Path(storage.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"data_domain={args.data_domain}, csv={data_path}")
    print(f"years={sorted(bundle.graphs.keys())}, N={bundle.total_n}, in_dim={bundle.in_dim}")
    if holdout is not None:
        print(
            f"ホールドアウト: 学習年={list(holdout.train_years)}, "
            f"テスト遷移 {holdout.year_prev}→{holdout.holdout_test_year}（探索は学習年のみ）"
        )
    study_name = (args.study_name or "").strip()
    if not study_name:
        study_name = (
            "unified_vgae_cope" if args.method == "cope" else f"optuna_baseline_{args.method}"
        )
        if args.time_dependent_potential:
            study_name = f"{study_name}_td"

    print(f"storage={storage}, study_name={study_name}, n_trials={args.n_trials}")
    print(
        f"method={args.method}, space={args.space}, link_score={args.cope_link_score}, "
        f"epochs/trial={args.epochs}"
    )
    if args.method == "cope":
        print(
            f"cope_density_calibrated={args.cope_density_calibrated}, "
            f"log_w={args.cope_density_log_weight}, ema={args.cope_density_ema_momentum}"
        )
    if args.cope_density_calibrated and args.time_dependent_potential:
        print(
            "警告: --time-dependent-potential では CoPE の density_calibrated は未対応のため無視されます。"
        )
    _ys_tr = sorted(g_tr.keys())
    _y_min_tr, _y_max_tr = int(_ys_tr[0]), int(_ys_tr[-1])
    if args.time_dependent_potential and holdout is not None:
        _ys_full = sorted(holdout.graphs_full.keys())
        _y_min_tr, _y_max_tr = int(_ys_full[0]), int(_ys_full[-1])
    print(
        f"time_dependent_potential={args.time_dependent_potential}, "
        f"phi_year_range=[{_y_min_tr}, {_y_max_tr}]"
        + (
            " (full calendar incl. holdout test year)"
            if (args.time_dependent_potential and holdout is not None)
            else ""
        )
    )

    def objective(trial: optuna.Trial) -> float:
        h = _suggest_training_hparams(
            trial,
            space=args.space,
            link_score_mode=args.cope_link_score,
            tune_hidden_dim=not args.no_tune_hidden,
            fixed_hidden_dim=args.hidden_dim,
        )
        torch.manual_seed(int(args.seed) + trial.number * 10007)
        np.random.seed(int(args.seed) + trial.number * 10007)

        rnn_len = int(args.rnn_history_len)
        if args.method == "rnn":
            rnn_len = int(trial.suggest_int("rnn_history_len", 2, 6))

        latent_dim = int(args.latent_dim)
        if args.tune_latent_dim:
            latent_dim = int(trial.suggest_categorical("latent_dim", [2, 4, 8, 16]))

        mb = ModelBuildKw(
            hidden_dim=int(h["hidden_dim"]),
            latent_dim=latent_dim,
            link_score_mode=args.cope_link_score,
            cosine_logit_scale=float(h.get("cosine_logit_scale", 5.0)),
            w_pot_init=float(h["w_pot_init"]),
            rnn_history_len=rnn_len,
            pnode_history_len=int(args.pnode_history_len),
            pnode_ode_method=str(args.pnode_ode_method),
            pnode_ode_n_steps=int(args.pnode_ode_n_steps),
            density_calibrated_potential=bool(args.cope_density_calibrated),
            density_log_weight=float(args.cope_density_log_weight),
            density_ema_momentum=float(args.cope_density_ema_momentum),
            time_dependent_potential=bool(args.time_dependent_potential),
            year_min=_y_min_tr,
            year_max=_y_max_tr,
        )

        model = build_baseline_model(args.method, device, bundle, mb)

        train_kw = dict(
            num_epochs=args.epochs,
            lr=float(h["lr"]),
            beta=float(h["beta"]),
            pos_weight=float(h["pos_weight"]),
            latent_pred_weight=float(h["latent_pred_weight"]),
            future_link_weight=float(h["future_link_weight"]),
            potential_weight=float(h["potential_weight"]),
            trajectory_weight=float(h["trajectory_weight"]),
            num_neg_recon=neg_r,
            num_neg_future=neg_f,
            loss_aux_warmup_epochs=int(args.loss_aux_warmup_epochs),
        )
        if isinstance(model, UnifiedVGAETD):
            _, _, best_auc, _hist = train_model_td(
                model,
                g_tr,
                bundle.num_corps,
                h_tr,
                **train_kw,
            )
        else:
            _, _, best_auc, _hist = train_model_improved(
                model,
                g_tr,
                bundle.num_corps,
                h_tr,
                **train_kw,
            )
        if np.isnan(float(best_auc)):
            return -1e9
        return float(best_auc)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=args.n_trials)

    bt = study.best_trial
    print("\n" + "=" * 60)
    print(f"best_value (max val AUC): {study.best_value:.6f}  trial={bt.number}")
    print("best_params:", json.dumps(bt.params, indent=2, ensure_ascii=False))
    print("=" * 60)

    out_json = Path(args.output_json.strip()) if args.output_json.strip() else None
    if out_json is None:
        default_db_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in study_name)
        out_json = default_db_dir / f"best_params_{safe_name}.json"

    payload = {
        "study_name": study_name,
        "optuna_method": args.method,
        "storage": storage,
        "data_domain": args.data_domain,
        "data": str(data_path),
        "holdout_test_year": args.holdout_test_year,
        "years": sorted(bundle.graphs.keys()),
        "n_trials": args.n_trials,
        "epochs_per_trial": args.epochs,
        "space": args.space,
        "cope_link_score": args.cope_link_score,
        "seed": args.seed,
        "latent_dim_cli": int(args.latent_dim),
        "tune_latent_dim": bool(args.tune_latent_dim),
        "pnode_history_len": int(args.pnode_history_len),
        "pnode_ode_method": str(args.pnode_ode_method),
        "pnode_ode_n_steps": int(args.pnode_ode_n_steps),
        "loss_aux_warmup_epochs": int(args.loss_aux_warmup_epochs),
        "time_dependent_potential": bool(args.time_dependent_potential),
        "phi_year_min": _y_min_tr,
        "phi_year_max": _y_max_tr,
        "best_value": study.best_value,
        "best_trial_number": bt.number,
        "best_params": bt.params,
        "user_attrs": dict(bt.user_attrs),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {out_json}")
    print(
        "比較検証（cope のみ）: python -m pnode_patent_runner.run_benchmark_comparison "
        f"--optuna-best-json {out_json}  ...（同一データ・--cope-link-score 整合）"
    )
    if args.method != "cope":
        print(
            "公平 HPO 比較: 各 --method ごとに同じ --n-trials で JSON を出し、"
            "マップ JSON を --optuna-best-json-map に渡して run_benchmark_comparison を実行。"
        )


if __name__ == "__main__":
    main()
