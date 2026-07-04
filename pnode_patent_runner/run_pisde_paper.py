"""
PI-SDE (btae400, Jiang & Wan 2024) を ArXiv 論文データに適用。

PI-SDE 元実装 /tmp/PI-SDE/ を利用し、当該リポジトリ内の以下を直接呼ぶ:
  - src.model.ForwardSDE  (Hamilton-Jacobi 内包 SDE)
  - src.train.run / run_leaveout
  - src.evaluation.evaluate_fit / evaluate_fit_leaveout

実行:
  PNODE_DOMAIN=paper python -m pnode_patent_runner.run_pisde_paper
  PNODE_DOMAIN=patent_energy python -m pnode_patent_runner.run_pisde_paper
"""
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

import torch
import numpy as np

# PI-SDE source を PYTHONPATH に追加
PISDE_ROOT = Path("/tmp/PI-SDE")
sys.path.insert(0, str(PISDE_ROOT))

# ── 設定 ─────────────────────────────────────────────────────────
DOMAIN = os.environ.get("PNODE_DOMAIN", "paper")
SEED   = int(os.environ.get("PNODE_SEED", 42))
EPOCHS = int(os.environ.get("PNODE_EPOCHS", 1500))   # PI-SDE 元は 3000、計算節約で 1500
LEAVEOUT = os.environ.get("PNODE_LEAVEOUT", "")      # 例 "3" でホールドアウト

if DOMAIN == "paper":
    DATA_NAME = "PNode_Paper"
    DATA_DIR  = "data/PNode_Paper"
    TRAIN_T   = [1, 2, 3]
elif DOMAIN == "patent_energy":
    DATA_NAME = "PNode_Patent_Energy"
    DATA_DIR  = "data/PNode_Patent_Energy"
    TRAIN_T   = list(range(1, 12))    # t=1..11 (years 2011..2021)
else:
    raise ValueError(f"unknown DOMAIN={DOMAIN}")

# PI-SDE が使う作業ディレクトリ (cwd 基準で動く)
WORKDIR = Path("/home/nakamuraroi/kumagai")
os.chdir(WORKDIR)

# 元論文の config_Veres を読み込み、引数を差し替える
import argparse


def make_args():
    """PI-SDE の args 互換オブジェクトを作る。"""
    ns = argparse.Namespace(
        seed=SEED,
        use_cuda=True,
        device=0,
        out_dir=f"RESULTS/{DATA_NAME}",
        data=DATA_NAME,
        data_path=f"{DATA_DIR}/alltime/fate_train.pt",
        data_dir=DATA_DIR,
        # MLP (元論文と同じ k_dims=[400,400])
        k_dims=[400, 400],
        activation="softplus",
        sigma_type="const",
        sigma_const=0.1,
        # 学習
        train_epochs=EPOCHS,
        train_lr=0.005,
        train_lambda=0.5,           # HJ 正則化の重み
        train_batch=0.1,
        train_clip=0.1,
        save=500,
        # 評価
        evaluate_n=10000,
        evaluate_data=None,
        evaluate_baseline=False,
        # 実行モード
        task="fate",
        train=False,
        evaluate=None,
        config=None,
        # Sinkhorn
        sinkhorn_scaling=0.7,
        sinkhorn_blur=0.1,
        ns=2000,
        # 時刻
        start_t=0,
        train_t=TRAIN_T,
    )
    ns.layers = len(ns.k_dims)
    return ns


def init_config(args):
    """PI-SDE の init_config 互換 (config_Veres.py から流用)。"""
    args.layers = len(args.k_dims)
    args.kDims = '_'.join(map(str, args.k_dims))
    name = (
        "{activation}-{kDims}-"
        "{train_lambda}-{sigma_type}-{sigma_const}-"
        "{train_clip}-{train_lr}"
    ).format(**args.__dict__)

    args.out_dir = os.path.join(args.out_dir, name, f"seed_{args.seed}")
    if args.task == "leaveout":
        args.out_dir = os.path.join(args.out_dir, args.leaveout_t)
    else:
        args.out_dir = os.path.join(args.out_dir, "alltime")

    os.makedirs(args.out_dir, exist_ok=True)
    args.train_pt  = os.path.join(args.out_dir, "train.{}.pt")
    args.done_log  = os.path.join(args.out_dir, "done.log")
    args.config_pt = os.path.join(args.out_dir, "config.pt")
    args.train_log = os.path.join(args.out_dir, "train.log")
    return args


def main():
    # PI-SDE モジュールを動的読み込み
    import src.train as train
    from src.evaluation import _evaluate_impute_model

    args = make_args()
    print("=" * 60)
    print(f"  PI-SDE run on {DATA_NAME}")
    print(f"  seed={SEED}  epochs={EPOCHS}  train_t={TRAIN_T}")
    if LEAVEOUT:
        print(f"  leaveout: t={LEAVEOUT}")
    print("=" * 60)

    # config_Veres.load_data がインポート時に走るので回避するため
    # 直接 train.run/run_leaveout を呼ぶ
    if LEAVEOUT:
        leaveouts = [int(LEAVEOUT)]
        config = train.run_leaveout(args, init_config, leaveouts=leaveouts)
    else:
        config = train.run(args, init_config)

    print(f"\nTraining complete. Results in {config.out_dir}")


if __name__ == "__main__":
    main()
