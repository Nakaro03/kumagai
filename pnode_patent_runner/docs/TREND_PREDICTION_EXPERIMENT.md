# 技術トレンド予測：性能比較実験設計

企業–特許／著者–論文／著者–トピックの **年次二部グラフ列**における **将来リンク（future-link）予測**を「技術トレンド予測」の操作化とする。実装の一次情報は [README_COPE.md](../README_COPE.md)、評価・ホールドアウトは [PAPER_WORKFLOW.md](../PAPER_WORKFLOW.md)、ベンチ CLI は [`run_benchmark_comparison.py`](../run_benchmark_comparison.py)。

---

## 比較手法（固定）

| 実装キー | 役割 |
|----------|------|
| `static` | 時間発展なし対照 |
| `rnn` | 系列潜在の RNN 予測 |
| `neural_ode` | 制約なし潜在 Neural ODE |
| `pnode` | **提案主軸**：ポテンシャル勾配流 ODE + 幾何デコーダ（デコーダに \(\Phi\) 項なし） |
| `cope` | 拡張：同一 \(\Phi\) をリンク logit にも使用（主表から外すか副次行にするか事前に固定） |

---

## 評価（固定）

- **主指標**: future-link の **ROC-AUC**、**AP**、**ECE**（`evaluate_val_future_link_metrics` 系；ECE は実装のビン数 `future_link_ece_n_bins` を論文に明記）。**Hits@K / MRR は主表スコープ外**（理由と執筆上の宣言は [RANKING_METRICS_SCOPE.md](RANKING_METRICS_SCOPE.md)）。
- **短期 vs 長期（推奨）**  
  - **短期**: 時系列キー昇順で **最終遷移 1 ステップ**（既定: 最後の 2 キー \((y_{T-1},y_T)\)）。  
  - **長期**: 同じキー列上で **\(k\) ステップ先**（例: \(k\in\{2,3\}\)）の future-link。  
  - **操作化**: 「ステップ」は **暦年の差ではなく** `sorted(graphs.keys())` の **インデックス差**（年が欠けていても一貫）。  
  - **実装**: [`unified_training.py`](../unified_training.py) の `rollout_z_pred_multistep` と `future_link_auc_scores`（インデックス差分だけロールアウト）。ベンチマークは **`--eval-horizon-gaps 1,2,3`** で各手法の JSON に `train_split_metrics_by_horizon_gap` / `final_metrics_by_horizon_gap`（キー `"1"`,`"2"`… に `auc`,`ap`,`ece`）を追加。図示は [`plot_horizon_benchmark.py`](../plot_horizon_benchmark.py)。査読用の **H_long 事前登録表・集約コマンド**は [LONG_HORIZON_PREREGISTRATION.md](LONG_HORIZON_PREREGISTRATION.md)。

---

## 1. データ分割（train / test）

| モード | 学習に使う年 | 主評価 | 備考 |
|--------|----------------|--------|------|
| **Transductive（CLI 既定）** | `sorted(years)` すべて | 最終 \(k\) ステップ遷移（例: `year_prev = years[-1-k]`, `year_next = years[-1]`） | ノード集合は学習期と重なる。**査読用の本文主表には用いず**、再現・感度として **Appendix（または補足）**に限定する。 |
| **ホールドアウト（本文主表に固定）** | `--holdout-test-year Y` により **\(y < Y\)** のみ | **\((Y\text{の直前キー} \to Y)\)** の future-link | `hist_edges` は学習期のみで再構成（`split_bundle_holdout_test_year`）。JSON の **`final_val_*` のみ**を本文の主表・主張に用いる（transductive の `final_val_*` と併記しない）。 |

**固定すべき記述**: CSV パス、`--data-domain`、`--year-range` / `--arxiv-year-min|max`、`--min-patents`、ホールドアウト有無と \(Y\)。

---

## 2. ハイパーパラメータ

| 区分 | 方針 |
|------|------|
| 既定 | [README_COPE.md](../README_COPE.md) の損失係数・`pos_weight`・負例本数に合わせる。 |
| 公平 HPO | 各手法に **同一 trial 数**の Optuna（[`run_optuna_unified_vgae.py`](../run_optuna_unified_vgae.py)）→ [`--optuna-best-json-map`](../run_benchmark_comparison.py) でベンチに流し込む（[PAPER_WORKFLOW.md](../PAPER_WORKFLOW.md) セクション 5）。 |
| デコーダ | `--cope-link-score distance|cosine` を **データごとに 1 つに固定**（両方勝つとは書かない）。 |
| 可視化用 2D | 主表用の `latent_dim` と **別実験**ならキャプションで必ず区別（[PAPER_WORKFLOW.md](../PAPER_WORKFLOW.md) セクション 6）。 |

---

## 3. 評価プロトコル

1. **正例**: `year_next` の観測二部エッジから **最大 1500** 本をサブサンプル（`max_pos`）。  
2. **負例**: その正例に現れる active 左右ノード上で、**固定シード**の乱数により生成（全ペアではない）。  
3. **指標**: ROC-AUC、AP、ECE を **同一ペア集合**で算出（正例サブサンプルは `(year_prev, year_next, max_pos, …)` から導く **決定的** `torch.Generator`／負例は対応する `numpy` シードで、学習中の他評価呼び出しの後でも同じ遷移なら同じ部分集合になる）。  
4. **多シード**: `torch.manual_seed` / `np.random.seed` を複数本取り、平均±標準誤差＋ **ペア差の Wilcoxon**（baseline ごと・\(k\) ごと；多重比較は Holm 等を事前登録）。  
5. **長期 \(k\ge2\)**: 上記「評価」の実装メモどおりロールアウト後に **同一プロトコル**でスコア算出。

---

## 4. 可視化（参考: `map_cope_alt_dark` 系）

- **テンプレート**: [`interactive_vector_field_alt_dark.html`](../interactive_vector_field_alt_dark.html)（成果物例のファイル名 `map_cope_alt_dark.html`。[README_COPE.md](../README_COPE.md) 参照）。  
- **手法だけ変更する手順**（解釈用・`latent_dim=2` 推奨）:  
  1. 手法ごとに [`run_benchmark_comparison.py`](../run_benchmark_comparison.py) で `--save-checkpoint-dir ... --methods <key>` 等により **同一データ条件**の `.pt` を保存。  
  2. [`run_interactive_landscape_cope_vector_field.py`](../run_interactive_landscape_cope_vector_field.py)（または P-NODE 向け TD 版ランドスケープ CLI）で **`--load-checkpoint`** に **その手法の ckpt** を指定し、`--html-template` を **`interactive_vector_field_alt_dark.html` のまま**、`--output` だけ `map_pnode_alt_dark.html` / `map_static_alt_dark.html` のように手法名を変えて出力。  
  3. 左パネルの見出し・キャプションに **手法名・シード・年範囲・`link_score_mode`** を明記し、主表の高次元実験と別であることを脚注する。

---

## 5. 再現性確保

| 項目 | 内容 |
|------|------|
| シード | `--seed` と負例 RNG の固定（実装の `np.random.default_rng(0)` 等）を論文・JSON に記載。 |
| 環境 | Python / PyTorch / CUDA、`torch_scatter` の有無、Docker（リポジトリ [Dockerfile](../../Dockerfile)）を Supplementary に列挙。 |
| CLI ログ | `run_benchmark_comparison` の JSON に `data`, `years`, `link_score_mode`, `loss` 係数、`holdout_test_year`, `future_link_ece_n_bins`（現行）を保存。多ホライズン実装後は `eval_horizon_gaps` を追加。 |
| 一括 | [`scripts/paper_benchmark_suite.example.sh`](../scripts/paper_benchmark_suite.example.sh) をベースにドメイン×シードを列挙。 |

---

## 再現コマンド例（現行ツール・単一ステップ主評価）

リポジトリルートで（パスは環境に合わせて変更）:

```bash
python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain patent \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2010 2020 \
  --holdout-test-year 2020 \
  --epochs 20 \
  --seed 42 \
  --methods static,rnn,neural_ode,pnode,cope \
  --cope-link-score distance
```

多ホライズン付きの例:

```bash
python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain patent \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2010 2020 \
  --holdout-test-year 2020 \
  --epochs 20 \
  --seed 42 \
  --methods all \
  --cope-link-score distance \
  --eval-horizon-gaps 1,2,3
```

曲線 PNG:

```bash
python -m pnode_patent_runner.plot_horizon_benchmark \
  pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed42.json \
  -o pnode_patent_runner/outputs/cope_benchmark/horizon_auc.png
```

---

## 関連ドキュメント

- [PNODE_PAPER_FRAMING.md](PNODE_PAPER_FRAMING.md) — 主張・消融・否定結果スケジュール  
- [PAPER_WORKFLOW.md](../PAPER_WORKFLOW.md) — 査読用チェックリスト
