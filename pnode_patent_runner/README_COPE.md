# CoPE-VGAE（`UnifiedVGAE`）

## 名称

| | |
|--|--|
| 略称 | **CoPE-VGAE** |
| 英語 | Consistent Potential-Energy Variational Graph Autoencoder |
| 日本語 | 一貫ポテンシャルエネルギー変分グラフオートエンコーダ |
| 実装クラス名 | `UnifiedVGAE`（`unified_vgae.py`） |

リンク尤度で使うスカラー場 **Φ** と、時間発展の **勾配流 ODE（−∇Φ）** が **同じ `PotentialNet`** から来るようにし、エネルギー解釈とデコーダを揃えるのが設計の要点です。

## アーキテクチャ概要

- **入力**: 年次の企業–特許二部グラフ（`torch_geometric.data.Data`）。特許ノードに結合埋め込み特徴、企業ノードは学習可能な `corp_embeddings` で上書き。
- **エンコーダ**: `SharedVGAEEncoder`（GAT）→ 各ノードの `μ`, `log σ²` → 再パラメータ化で `z`。
- **ポテンシャル・時間**: `GradientNeuralODEPredictor` 内の **`PotentialNet`** が Φ(z) を出力。ODE の速度場は **−∇Φ**（`GradientODEFunc`）。`predict_future` は `torchdiffeq` の `odeint` で 1 ステップ先の `z` を生成。
- **デコーダ（リンク logit）** — `decode_logits`（`link_score_mode` で選択）:
  - **`distance`**: `logit = r − ||z_i−z_j||² + w_pot · (Φ_i + Φ_j)`
  - **`cosine`**: L2 正規化後の内積にスケールをかけた項 `+ w_pot · (Φ_i + Φ_j)`（`w_pot = 0` なら従来の cosine デコーダと同形）
- 学習時は、再構成・未来リンクなどの尤度損失が **Φ** と **w_pot** に勾配を流します（下記「損失関数」、`unified_training.py`）。

## 損失関数（実装: `unified_training.compute_loss_standardized`）

年次ペア `(data_t, data_t1)` ごとに、次の **6 成分の和**を最小化します（`train_one_epoch` から呼び出し）。

**総損失**

`total_loss = recon_loss + β·kl_loss + λ_lat·L_lat + λ_fut·L_fut + λ_pot·L_pot + λ_traj·L_traj`

既定の係数例（`train_model_improved` 周辺）: `β=0.01`, `pos_weight=5.0`, `λ_lat=1.0`, `λ_fut=10.0`, `λ_pot=0.01`, `λ_traj=0.05`。負例本数は再構成側 `num_neg_recon=800`、未来リンク側 `num_neg_future=400`。

1. **再構成 `recon_loss`** — 時刻 `t` の **`z_t`** で `decode`。正例は `data_t` の全エッジ。負例は `sample_hard_negatives_v2` で二部 (企業, 特許) をランダム抽出し、**当年正例集合**および **`hist_edges`** に無いペアのみ。正例項は端点次数から `edge_rarity = 1/log(deg_u·deg_v)` を掛けた **`-mean log(pos_pred)` に `pos_weight`**。負例項は **`-mean log(1 - neg_pred)`**（`pos_weight` なし）。

2. **KL `kl_loss`** — `data_t.active_mask` 上のノードで標準的な ELBO 形: **`-0.5·mean(1 + logvar − μ² − exp(logvar))`**。総損失では **`β * kl_loss`**。

3. **ポテンシャル `potential_loss`** — `temporal_predictor` に **`potential_net` があるモデルのみ**（`UnifiedVGAE` は該当）。`φ = Φ(z_t)` に対し **`L_pot = 0.01·mean(φ²)`**（`potential_weight > 0` のとき）。`BenchmarkTemporalVGAE` の Static / RNN / NeuralODE では **0**。

4. **軌道整合 `trajectory_loss`** — 同様に `potential_net` ありのとき、`v = −∇_z Φ(z_t)` と教師変位 `d = μ_{t+1}.detach() − z_t`（`μ_{t+1}` は `data_{t+1}` を `no_grad` でエンコード）の **コサイン類似度**に対し **`mean(1 − cos(d, v))`**（active ノード上、`trajectory_weight > 0`）。

5. **潜在予測 `latent_pred_loss`** — **`z_{t+1}^{pred} = predict_future(z_history)`** と **`μ_{t+1}`** の **MSE**（`data_{t+1}.active_mask` 上）。重み `latent_pred_weight`。

6. **未来リンク `future_link_loss`** — **`z_{t+1}^{pred}`** で `decode`。正例は `data_{t+1}` のエッジ、負例は再び `sample_hard_negatives_v2`（`hist_edges` 除外）。**`-mean log(pos)·pos_weight − mean log(1−neg)`**。重み `future_link_weight`。

**補足**: `UnifiedVGAE.decode` は幾何項に **`w_pot·(Φ_i+Φ_j)`** を含むため、再構成・未来リンクの両方から **Φ** と **w_pot** に勾配が入る。評価用の future-link AUC は `evaluate_val_auc` / `future_link_auc_scores`（最終 2 年、`sklearn.roc_auc_score`）。

## データパイプライン（`data.py`）

特許 CSV（例: `notebooks/work/dataset/topic_info3.csv`）を想定。

1. **`preprocess_data`**: `description_embedding` と `metadata_embedding` をパースして連結 → `combined_vector`。`corporation` をリスト化、`year_month` を日付化して年範囲でフィルタ。
2. **`filter_active_corporations`**: 全期間で特許行数が `min_patents` 未満の企業を除外。
3. **`build_global_graphs`**: 企業 `0…C−1`、特許 `C…C+P−1` の固定インデックスで、**年ごとの** `Data` と **`hist_edges`**（過去に現れた企業–特許ペア）を返す。
4. **`calculate_initial_corp_vectors`**: 企業ごとに関与特許の `combined_vector` 平均を `corp_embeddings` の初期値に。

想定列の例: `description_embedding`, `metadata_embedding`, `corporation`, `year_month`, `patent_number`。

## 本ディレクトリ内の主なファイル

| ファイル | 内容 |
|----------|------|
| `unified_vgae.py` | `UnifiedVGAE`（CoPE-VGAE）本体 |
| `models.py` | `SharedVGAEEncoder`, `PotentialNet`, `GradientODEFunc`, `GradientNeuralODEPredictor` |
| `data.py` | 特許 CSV → 企業–特許グラフ |
| `data_arxiv.py` | 著者–論文・著者–トピック CSV（ArXiv 風）→ 二部グラフ（`P-NODE_paper.ipynb` / `P-NODE_paper_topic.ipynb` 相当） |
| `reexport_arxiv_embeddings.py` | `description_embedding` を省略なしの空白区切り数値列に直す（JSON 列 or `emb_*` ワイド列から） |
| `recompute_arxiv_embeddings_e5.py` | `intfloat/multilingual-e5-large` で `description` から埋め込みを再計算しパイプライン用 CSV に保存 |
| `checkpoint_utils.py` | チェックポイントの **shape が合うパラメータだけ**読み込み（企業数のズレ対策） |
| `interactive_landscape.py` | Plotly 用ペイロード `build_interactive_payload`（特許）/ `build_interactive_payload_author_paper`（論文） |
| `interactive_landscape_vector_field.py` | Φ のヒートマップ・等高線・|∇Φ| 色分け短矢印の計算と HTML 書き出し |
| `interactive_vector_field_template.html` | 上記 HTML の既定テンプレート（レイヤ表示の切替 UI） |
| `interactive_vector_field_alt_dark.html` | 同ペイロードの別レイアウト例（ダーク・左パネル・Viridis ヒート）。`--html-template` で指定 |
| `interactive_vector_field_author_paper.html` | 著者–論文ラベル版テンプレ（スライダー＝論文の年） |
| `run_interactive_landscape_cope_vector_field.py` | 企業–特許向けインタラクティブ地図 CLI |
| `run_interactive_landscape_arxiv_vector_field.py` | 著者–論文向けインタラクティブ地図 CLI（年は学習と揃える） |
| `run_train_unified_vgae_checkpoint.py` | `--data-domain patent|arxiv` で学習し `.pt` 保存 |
| `unified_training.py` | `compute_loss_standardized`, `train_one_epoch`, `evaluate_val_auc`, `train_model_improved` |
| `cope_experiment.py` | README 準拠の CSV→グラフ束・年次抽出・ベースラインモデル生成 |
| `run_benchmark_comparison.py` | 上記パイプライン + `train_model_improved` で 5 手法を比較（JSON 出力可） |
| `run_optuna_unified_vgae.py` | Optuna で **CoPE またはベースライン**（`--method`）を調整（最終 2 年 future-link AUC 最大化・sqlite 保存可） |
| `run_cope_effectiveness.py` | CoPE フル損失 vs アブレーションの AUC 比較（同パイプライン） |

## ベースライン比較（`run_benchmark_comparison.py`）

[`unified_training.py`](pnode_patent_runner/unified_training.py) の損失既定（README「損失関数」の β, pos_weight, λ, 負例本数）で各手法を同一データ・同一評価（最終 2 年の `evaluate_val_auc`）します。デコーダの **`--cope-link-score` 既定は `distance`**（下記 HTML 例と揃えるため）。結果は `pnode_patent_runner/outputs/cope_benchmark/benchmark_<data_domain>_seed<seed>.json` にも保存されます（**`data_domain` は `patent` / `arxiv` / `author_topic` いずれも同じ JSON 形で AUC・AP**）。

**ホールドアウト（任意・ドメイン共通）**: `--holdout-test-year Y` で **Y 年のエッジを学習損失に含めず**、`hist_edges` も学習期間のみで再構成。主な報告指標は **(直前年→Y)** の future-link AUC/AP（`train_split_*` は学習区間の最後2年）。`--year-range` に **Y を含める**こと。**`--data-domain patent` / `arxiv` / `author_topic` いずれも同じフラグ**。詳細は [PAPER_WORKFLOW.md](PAPER_WORKFLOW.md) の「ホールドアウト」。

**Optuna 連携:** [`run_optuna_unified_vgae.py`](pnode_patent_runner/run_optuna_unified_vgae.py) が出力する `best_params_*.json` を **`--optuna-best-json`** に渡すと、**CoPE（`cope`）だけ**がその最良ハイパラで学習され、他手法は CLI の損失・`lr` のまま比較できます。各手法を **同じ trial 数**で公平に HPO する場合は `--method` ごとに Optuna を回し、**`--optuna-best-json-map`** に手法キー別の JSON パスを渡す（`--cope-link-score` は Optuna 実行時と揃えること）。手順の表は [PAPER_WORKFLOW.md](PAPER_WORKFLOW.md) の「ハイパラ調整」。

```bash
cd /path/to/kumagai

python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain patent \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2010 2020 \
  --epochs 10 \
  --seed 42 \
  --methods all
```

**著者–論文（ArXiv 埋め込み CSV、`P-NODE_paper.ipynb` と同型）**では `--data-domain arxiv` と [`data_arxiv.py`](pnode_patent_runner/data_arxiv.py) を使います。想定列: `description_embedding`, `authors`, `year`, `url`（または `arxiv_id`）。**`description_embedding` は全次元の数値列が必須**（`...` 省略入りの CSV は無効）。ベンチマークでは `data/processed/arxiv_cs_embedded_2020-2026_full.csv` を推奨。`--data` 省略時は `_full.csv` → ノートブック配下の CSV → `data/processed` の短名を順に探します。著者フィルタは `--min-patents`（ノートブックでは多く 5）。

```bash
python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain arxiv \
  --min-patents 5 \
  --epochs 10 \
  --seed 42 \
  --methods all
```

**著者–トピック（`P-NODE_paper_topic.ipynb` と同型）**では `--data-domain author_topic` と [`cope_experiment.load_author_topic_graph_bundle`](pnode_patent_runner/cope_experiment.py) を使います。CSV には **`topic` 列**（既定名。変更は `--topic-column`）が必要で、トピック側の特徴はカテゴリ内 `paper_vector` の平均です。`--data` の探索パスは `arxiv` と同じ ArXiv 埋め込み CSV を想定します。

```bash
python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain author_topic \
  --min-patents 5 \
  --epochs 10 \
  --seed 42 \
  --methods all
```

## インタラクティブ潜在マップ（HTML）

**前提**: 可視化用は **`latent_dim=2`** のチェックポイント。リポジトリルートで実行する。

論文・スクリーンショット用の既定は **ダーク左パネル UI**（`interactive_vector_field_alt_dark.html`）で、出力ファイル名の例は **`map_cope_alt_dark.html`**（`run_interactive_landscape_cope_vector_field` の `--output` 既定と一致）。

```bash
cd /path/to/kumagai

python -m pnode_patent_runner.run_interactive_landscape_cope_vector_field \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2010 2020 \
  --load-checkpoint pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt \
  --cope-link-score distance
# 既定で pnode_patent_runner/outputs/cope_landscape/map_cope_alt_dark.html に書き出し
```

- **`--cope-link-score`**: 学習時と一致させる（`run_unified_landscape` 系は多くの場合 `distance`、`principled` 比較の既定は `cosine` のことがある）。
- **`--quiver-stride`**, **`--quiver-length`**, **`--n-mag-bins`**: 矢印の疎さ・長さ・|∇Φ| の色ビン数（再生成時）。
- **`--html-template`**: 埋め込み用 HTML。**省略時は `interactive_vector_field_alt_dark.html`**（上記ダーク UI）。従来の単ペイン風にする場合は `interactive_vector_field_template.html` を指定（`__PAYLOAD_B64__` / `__PAGE_TITLE__` / `__HEADING__` のプレースホルダは同一）。

HTML 内で **Φ ヒートマップ / 等高線 / 短矢印** の表示・透明度・等高線本数を切り替えられます。

### 著者–論文（ArXiv CSV）の可視化

[`run_interactive_landscape_arxiv_vector_field.py`](pnode_patent_runner/run_interactive_landscape_arxiv_vector_field.py) は [`load_author_paper_graph_bundle`](pnode_patent_runner/cope_experiment.py) を使い、スライダーの年は **CSV の論文 `year`**（年次グラフのキー）に対応します。学習（[`run_train_unified_vgae_checkpoint.py`](pnode_patent_runner/run_train_unified_vgae_checkpoint.py) の `--data-domain arxiv` や `run_benchmark_comparison --data-domain arxiv`）と **次を一致**させてください。

| 項目 | 意味 |
|------|------|
| `--arxiv-year-min` / `--arxiv-year-max`（または `--arxiv-no-year-filter`） | `preprocess_arxiv_data` に入る行の年フィルタ（学習・可視化で同じ） |
| `--year-range` / `--years` / `--all-years` | 束に含める **論文年**（グラフに残す年。学習と可視化で同じ） |
| `--min-patents` | 著者あたり最小 **論文行数**（学習と可視化で同じ） |
| `--cope-link-score` 他ハイパラ | チェックポイント学習時と同じ |

```bash
python -m pnode_patent_runner.run_train_unified_vgae_checkpoint \
  --data-domain arxiv \
  --year-range 2020 2024 \
  --arxiv-year-min 2020 --arxiv-year-max 2024 \
  --min-patents 5 \
  --epochs 20 \
  --cope-link-score cosine

python -m pnode_patent_runner.run_interactive_landscape_arxiv_vector_field \
  --data data/processed/arxiv_cs_embedded_2020-2026_full.csv \
  --year-range 2020 2024 \
  --arxiv-year-min 2020 --arxiv-year-max 2024 \
  --min-patents 5 \
  --load-checkpoint pnode_patent_runner/outputs/arxiv_landscape/unified_vgae_arxiv.pt \
  --cope-link-score cosine \
  --output pnode_patent_runner/outputs/arxiv_landscape/interactive_map_arxiv.html
```

## チェックポイントとデータの整合

学習時と **CSV・`min_patents`・前処理** がずれると、**企業数 `C` が 1 つでも変わり**、`corp_embeddings.weight` の shape がチェックポイントと一致しません。  
`run_interactive_landscape_cope_vector_field.py` は `load_state_dict_skip_shape_mismatch` により **一致するテンソルだけ**を読み込み、不一致の `corp_embeddings` は **現在のデータから計算した初期ベクトルのまま**残します。再現性を最優先する場合は、学習当時と同じデータ条件でグラフを構築してください。

**現在のデータ用に学習し直して .pt を作る**場合は [`run_train_unified_vgae_checkpoint.py`](pnode_patent_runner/run_train_unified_vgae_checkpoint.py) を使います（`--data-domain patent` なら `load_cope_graph_bundle`、`arxiv` なら `load_author_paper_graph_bundle` と同一条件）。可視化するときは **`--data`・`--min-patents`・年の切り方**を学習時と揃え、`--cope-link-score` も学習時と一致させてください。

```bash
python -m pnode_patent_runner.run_train_unified_vgae_checkpoint \
  --data-domain patent \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2010 2020 \
  --min-patents 2 \
  --epochs 30 \
  --cope-link-score distance \
  --save pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt
```

## 依存関係（目安）

- Python 3.8+
- PyTorch
- `torch_geometric`
- `torchdiffeq`（ODE 積分）
- `pandas`, `numpy`

（`torch-scatter` / `torch-sparse` は環境によって警告が出ることがありますが、フォールバックで動く場合があります。）

## 論文執筆・再現手順

査読対策・評価プロトコルの明記・ベンチマーク／消融コマンドの対応は、別紙 **[PAPER_WORKFLOW.md](PAPER_WORKFLOW.md)** にまとめています。本文の **Experimental setup** へ転記する対応表（評価・ホールドアウト・HPO）、**対称 HPO** の手順、**消融表**、**2D 可視化と主表の注意**は同ドキュメントのセクション2（写経表）・5・4・6を参照してください。多シード実行のシェル雛形は [pnode_patent_runner/scripts/paper_benchmark_suite.example.sh](scripts/paper_benchmark_suite.example.sh) を参照してください。

## 参考

- 数式・ベースラインとの対応の詳細は、別途 **案 A アーキテクチャ** のドキュメント（例: `plan_a_unified_energy_architecture.md`）があればそれを参照。
- 表記の定数: `unified_vgae.py` の `METHOD_SHORT_NAME` 等。
