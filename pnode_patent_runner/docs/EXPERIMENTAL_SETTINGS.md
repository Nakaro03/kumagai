# 実験設定（論文・再現用）

本稿は [`run_benchmark_comparison.py`](../run_benchmark_comparison.py)・[`unified_training.py`](../unified_training.py) の実装に整合する。関連: [PNODE_PAPER_FRAMING.md](PNODE_PAPER_FRAMING.md) 評価段落、[README_COPE.md](../README_COPE.md) 損失・パイプライン。

---

## 1. データ分割（train / test）

### 1.1 既定（学習に最終年を含むプロトコル）

- 年キー集合 \(\mathcal{Y}\) は `--year-range` または `--years` / `--all-years` 等で決まる（ドメイン別: `cope_experiment.select_year_list`）。
- **学習**: 隣接年ペア \((y_t, y_{t+1})\) を **時系列順**にたどり、`train_model_improved`（または TD 系は `train_model_td`）で各ペアの損失を積算。
- **検証（主指標）**: 年を昇順に並べたとき **最終 2 年** \((y_{T-1}, y_T)\) の **future-link** について **ROC-AUC・AP・ECE**（`evaluate_val_future_link_metrics`）。
- 当該設定では、学習に**用いたノード集合**上で最終年を評価するため、**transductive** に近い。

### 1.2 ホールドアウト最終年（より厳しい汎化）

- CLI: `--holdout-test-year Y`。
- **学習**: \(y < Y\) のグラフのみ。\(Y\) 年のエッジは**損失に含めない**。`hist_edges` も学習期間の和集合のみで再構成（`split_bundle_holdout_test_year`）。
- **報告**: **遷移** \((Y\text{の直前年} \to Y)\) の **future-link** 指標を **final** として掲げる（JSON の `final_val_auc` / `final_val_ap` / `final_val_ece` 等）。`--year-range` には **Y を含める**。

### 1.3 長期（ホライズン）

- CLI: `--eval-horizon-gaps` 例: `1,2,3`。`sorted(graphs.keys())` 上の**インデックス差** \(k\) について、終端年 \(y_T\) まで **k ステップ** `rollout` した上で、同型の future-link 指標（および任意で潜在 MSE/MAE/方向一致）を JSON に格納。本文では **k を「長期」の操作定義**と明記する。

---

## 2. ハイパーパラメータ

### 2.1 損失・サンプリングの既定（`train_model_improved` / `compute_loss_standardized`）

| 記号/名前 | 既定（README 整合） | 上書き CLI 例 |
|-----------|-------------------|---------------|
| β（KL 係数） | `0.01` | `--beta` |
| 再構成・未来正例側重み | `pos_weight=5.0` | `--pos-weight` |
| λ_lat（潜在 MSE） | `1.0` | `--latent-pred-weight` |
| λ_fut（未来リンク BCE 外側係数） | `10.0` | `--future-link-weight` |
| λ_pot | `0.01` | `--potential-weight` |
| λ_traj | `0.05` | `--trajectory-weight` |
| 再構成負例本数 | `800` | `--num-neg-recon` |
| 未来負例本数 | `400` | `--num-neg-future` |
| pnode_explicit 接地 | `0.5` | `--attention-ground-weight`（0 で無効） |

- **L_pot の形**（`potential_reg_mode`）: `l2` / `log1p_sq` / `centered_l2`（`--potential-reg-mode`）。
- **L_traj**（`--trajectory-delta-source`, `--trajectory-loss-type`, 任意 `--trajectory-grad-floor` 等）: 主表に使うなら**事前登録**。
- **補助損失ウォームアップ**: `--loss-aux-warmup-epochs N`（N>1 で λ_pot・λ_traj の線形ランプ）。

### 2.2 モデル構造（ベンチ例）

- `--hidden-dim`, `--latent-dim`（可視化で 2D なら 2 固定を明示）
- P-NODE 系: `--pnode-history-len`, `--pnode-hist-fuse-mode`（`linear` | `gru`）, `--pnode-ode-method`（`dopri5` | `rk4` | `euler`）, `--pnode-ode-n-steps`
- リンクスコア: `--cope-link-score` `distance` | `cosine`, `--cosine-logit-scale`

### 2.3 比較の公平性

- **全手法で同一**データ・`epochs`・`seed`（および可能なら同一ハイパラ）。
- HPO する場合: **同トライアル予算**の Optuna を**手法キー別**に回し、`--optuna-best-json-map` で読み込む、**または** 1 本の表で「共通ハイパラ」と明言。

---

## 3. 評価プロトコル

### 3.1 主指標

- **ROC-AUC**、**Average Precision (AP)**、**ECE**（推論 logit 上、等幅 **15 ビン**; `FUTURE_LINK_ECE_N_BINS`）。

### 3.2 future-link サブサンプル（全ペア上ではない）

- 正例: 評価年の**観測**二部辺から **最大 1500** 本（`future_link_auc_scores` の `max_pos` 既定; 辺本数が少なければその全件）。
- 負例: 上記正例に現れる**アクティブ**左右ノード上で、**決定的**な手順＋**固定シード**の乱数で未観測辺を生成。`neg_ratio` 既定 1 なら正:負 ≒ 1:1 本数。
- **重要**: 指標は**全候補ペア**のランキングではない—本文に**必ず**記載する。

### 3.3 学習中診断（JSON / history）

- エポック損失内訳に **potential, trajectory, grad_phi_l2**（`grad_phi_l2` は軌道分岐有効時、ミニバッチ平均 ||∇Φ|| 相当）等が入る。

---

## 4. 可視化

| 内容 | 手段（リポ内） |
|------|----------------|
| 学習曲線 | [`plot_training_curves.py`](../plot_training_curves.py) + ベンチ JSON の `train_components_per_epoch` |
| 2D 潜在上の Φ・-∇Φ | `interactive_landscape_vector_field.py` 系、または BD 用 [`run_interactive_landscape_pnode_bd_vector_field.py`](../run_interactive_landscape_pnode_bd_vector_field.py) 等 |
| ODE 積分法の比較 | `grad_phi_l2` を併記し [`scripts/run_pnode_ode_diagnostics_sweep.sh`](../scripts/run_pnode_ode_diagnostics_sweep.sh) の例のように**同一 seed**で JSON 2 本 |

- **注意**: 潜在次元 \(D>2\) の場合、図は**射影/スライス**であり**全域の Φ ではない**—キャプションで限定する。

---

## 5. 再現性

1. **乱数**: `--seed` を本文と JSON に記録（`train_model_improved` 内で `torch.manual_seed` 等）。
2. **正例順序の再現**: `_future_link_pos_perm` が `(year_prev, year_next, max_pos, neg_ratio, 候補辺本数)` から導出する置換**のみ**に依存（同一数値で再現可）。
3. **CLI 固定**: `run_benchmark_comparison` の `output` JSON には `loss` ブロック・年・`seed`・主要フラグを保存。
4. **依存パッケージ**: Python, PyTorch, torch_geometric, torchdiffeq, sklearn 等の**バージョン**を Supplement か `requirements` に**固定**推奨。

---

## 6. 英文1段落（Method / Experiments 用ドラフト）

*Unless stated otherwise, we train on all consecutive year pairs in the selected range and report future-link ROC-AUC, average precision, and 15-bin expected calibration error on a subsample of at most 1500 positive edges in the target year, with paired negatives drawn under a fixed, reproducible subsampling and random pairing procedure (not a full-pairwise ranking). We optionally hold out a final test year *Y* from training and report metrics on the transition into *Y*; for long horizons we use multi-step rollouts with the same metric definition at each horizon gap* \(k\)*. We fix random seeds and log hyperparameters and CLI options in the benchmark JSON output.*

---

## 7. 相互リンク

- ポテンシャル2系（`pnode` / `pnode_explicit`）: [PNODE_POTENTIAL_BRANCHES.md](PNODE_POTENTIAL_BRANCHES.md)
- 消融と CLI: [PNODE_BOTTLENECK_AND_ABLATIONS.md](PNODE_BOTTLENECK_AND_ABLATIONS.md)
- 全体フロー: [../ARCHITECTURE.md](../ARCHITECTURE.md)
- ワークフロー詳細: [../PAPER_WORKFLOW.md](../PAPER_WORKFLOW.md)
