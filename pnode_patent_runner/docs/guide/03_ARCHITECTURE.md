# 03. システムアーキテクチャ

[← 02_GETTING_STARTED](02_GETTING_STARTED.md) | [次: 04_USAGE →](04_USAGE.md)

---

## 1. 全体像（1 枚図）

```
[L1 生データ]                              [L2 前処理済]                      [L3 実行時]
─────────────                              ──────────────                     ──────────
PatentsView bulk:                          data/processed/:                    train/test:
  g_cpc_current.tsv      ─┐                bipartite_construction_firm.csv     Y_train=2012
  g_assignee_*.tsv       ─┼─►┌──────────┐  construction_firm_names.csv         Y_test=2015
  g_application.tsv      ─┘  │ build_*  │  cpc_content_construction.npz        H=3 years
  g_patent.tsv               │ scripts  │
                             └────┬─────┘
                                  ▼
                             ┌─────────────────────────────────────────┐
                             │           処理パイプライン (8 段階)        │
                             ├─────────────────────────────────────────┤
                             │ (1) 粒度化 (CPCサブグループ → グループ)    │
                             │ (2) 世界構築 (build_world @ Y)            │
                             │ (3) 特徴量抽出 (actor_scores: 4 特徴)     │
                             │ (4) 学習 (train_lr: hard-negative LR)    │
                             │ (5) 推論 (collect: 全候補スコア)          │
                             │ (6) 較正 (IsotonicRegression)             │
                             │ (7) 評価 (precision@k, AUC, ECE)         │
                             │ (8) 可視化 (UMAP + KDE + 矢印)            │
                             └────┬────────────────────────────────────┘
                                  ▼
                             ┌─────────────────────────────────────┐
                             │              出力                    │
                             ├─────────────────────────────────────┤
                             │ A. 推薦テーブル (CSV/JSON)           │
                             │ B. 集計メトリクス (AUC, P@k, ECE)    │
                             │ C. 可視化 (PNG, GIF)                 │
                             └─────────────────────────────────────┘
```

---

## 2. 主要モジュール (Python ファイル別)

### 2.1 構築モジュール（一度だけ実行）

| ファイル | 役割 | 入力 | 出力 |
|---|---|---|---|
| `build_firm_bipartite.py` | 生バルク → 二部グラフ CSV | g_cpc, g_assignee, g_application | `bipartite_<domain>_firm.csv` |
| `build_cpc_content_embeddings.py` | 特許タイトル → CPC 意味埋め込み | g_cpc, g_patent | `cpc_content_<domain>.npz` |

### 2.2 コアエンジン（毎回呼ばれる）

| ファイル | 役割 | 関数 |
|---|---|---|
| `recommender_firm.py` | **推薦エンジン本体** | `train_lr`, `build_world`, `actor_scores`, `collect` |
| `diagnose_entry_humanaware.py` | 二部 PPMI-SVD 埋め込み + Adamic-Adar | `bipartite_embed`, `cooc_graph` |

### 2.3 可視化スクリプト

| ファイル | 出力 | 内容 |
|---|---|---|
| `viz_final_integrated.py` | `viz_final_integrated.png` | **メイン**: hot/cold + 予測可能性 + 推薦 + Sweet Spot |
| `viz_predictability_map_v3.py` | `..._v3.png` | シンプル版 (1 ストーリー + 表) |
| `viz_predictability_map_v6.py` | `..._v6.png` | Nature 風 2 パネル (過去 vs 予測) |
| `viz_predictability_map_v7.py` | `..._v7.png` | 3 パネル (過去 + 予測 + ケース) |
| `viz_trends_in_latent.py` | `..._latent.png` | 4 時点スナップショット + 15 年トレンド要約 |
| `viz_trends_animation.py` | `..._animation.gif` | **アニメ GIF**（1 年刻み × 20 フレーム） |
| `viz_line_forecast_backtest.py` | `..._backtest.png` | 折れ線予測（バックテスト評価） |
| `viz_crossdomain_forecast.py` | `..._crossdomain.png` | 特許 vs arxiv CS の比較 |
| `compare_prediction_accuracy.py` | `compare_..._.png` | **棒グラフ**: 7 手法の精度比較 |

### 2.4 検証・診断スクリプト（研究知見の証拠）

| ファイル | 検証内容 |
|---|---|
| `diagnose_convergence_signal.py` | 収束予測のベース信号 |
| `diagnose_convergence_timing.py` | タイミング C-index |
| `diagnose_entry_humanaware.py` | 参入推薦 head-to-head |
| `diagnose_novelty_hazard.py` | brokerage + 前兆の検証 |
| `test_convergence_learnability.py` | 学習可能性 |
| `test_flow_vs_growthgrad.py` | 流れ vs 成長勾配 |
| `test_flow_vs_pointcloud.py` | 流れ vs 密度変化 |
| `test_humanaware_neuralode.py` | Neural ODE が静的を超えるか |
| `test_earlywarning_timing.py` | 月次早期警告信号 |
| `prototype_bipartite_sde.py` | Bipartite Latent SDE プロト |

---

## 3. データの流れ（時系列で追う）

### Step 1 — データ準備
```
PatentsView バルク → awk ストリーミング → 二部 CSV
(g_cpc, g_assignee, g_application)         (filing_date, assignee_id, cpc_group)
```

### Step 2 — テキスト埋め込み（補助）
```
特許タイトル → MiniLM (384d) → CPCごとに平均 → cpc_content.npz
```

### Step 3 — 実行時の処理（`recommender_firm.py`）

```python
# 疑似コード
df = load_bipartite_csv()
df["i"] = coarsen(df["i"])  # CPC group 粒度に集約

# build_world: 1 つの基準年 Y で 5 つの構造を作る
W       = sparse_cooc_matrix(df, Y=2015)           # CPC × CPC 共起
momentum= activity_per_cpc(df, Y=2015)             # 各 CPC の活動度
prior   = portfolio_per_firm(df, year <= 2015)     # 各企業の保有
nextf   = entries_per_firm(df, 2016..2018)         # 各企業の未来参入
Uemb, Cemb = ppmi_svd_bipartite(df, Y=2015)        # 二部埋め込み

# train_lr: 学習年 Y_train で hard-negative 学習
features = [relatedness, momentum, human, content]
sc, clf = LogisticRegression().fit(features, labels)

# collect: 評価年 Y_test で予測
for each test firm:
    candidates = all CPCs not in portfolio
    scores = clf.decision_function(features(firm, candidates))
    ranked = argsort(-scores)

# isotonic 較正
iso = IsotonicRegression().fit(raw_scores, hits)
calibrated_probs = iso.predict(raw_scores)
```

### Step 4 — 可視化

```python
# UMAP で 2D 化
xy = umap.UMAP().fit_transform(Cemb)

# 予測可能性 = 各 CPC の実ヒット率
hit_rate = n_hit / n_recommended per CPC

# KDE で 2D ヒートマップ化
heat = gaussian_kde(xy, weights=hit_rate)

# プロット
plt.imshow(heat) + scatter + arrows + firm overlay
```

---

## 4. 4 つの特徴量（中核）

| 特徴量 | 数式 | 直感的意味 |
|---|---|---|
| `relatedness` | Σ_{j ∈ 保有} AdamicAdar(g, j) | 自社既存技術への近さ |
| `momentum` | momentum[g] | 業界全体で今 g が活発か |
| `human-aware` | cos(Uemb[f], Cemb[g]) | 二部埋め込み上の距離 |
| `content` | cos(MiniLM[g], 自社平均) | テキストの意味的類似度 |

→ 4 特徴を `LogisticRegression` で融合。ただし**実証的に relatedness 単独が最強**（USAGE.md 参照）。

---

## 5. データ前提・想定

### 5.1 二部グラフ CSV のフォーマット

```csv
ts,u,i
2010-03-14,abc123-uuid,E04F21
2010-08-22,def456-uuid,E21B36
...
```

- `ts`: ISO 形式の出願日
- `u`: 企業/発明者の ID（UUID 文字列）
- `i`: CPC コード（サブグループ形式、例 `E04F21/04`）

### 5.2 CPC コンテンツ NPZ のフォーマット

```python
npz = np.load("cpc_content_construction.npz")
codes = npz["codes"]  # ['E02F5/106', 'E04F21/04', ...]
emb   = npz["emb"]    # shape [n_codes, 384]
```

### 5.3 企業名 CSV のフォーマット

```csv
u,org
abc123-uuid,"NABORS DRILLING TECHNOLOGIES USA, INC."
def456-uuid,"BASF SE"
```

---

## 6. 評価プロトコル

| 項目 | 設定 |
|---|---|
| 訓練年 Y_train | 2012 |
| 評価年 Y_test | 2015 |
| ホライズン H | 3 年 |
| 粒度 | CPC グループ（"/" 前、例 E04F21） |
| 候補集合 | 保有していない全 CPC（~200）|
| ハード負例 | 関連性 > 0 だが入らなかった CPC |
| 較正法 | IsotonicRegression（半分で fit、半分で評価） |

---

## 7. 拡張ポイント（他のドメインに適用）

「**energy**」や「**computing**」など別領域に切り替えるには:

1. `build_firm_bipartite.py --domain energy` でデータ構築
2. `build_cpc_content_embeddings.py --domain energy`
3. 各 viz スクリプトを `--domain energy` で実行

→ コード変更ほぼ不要（ドメインは引数で切り替え）。

---

## 8. 重要な設計判断（過去の試行錯誤の結論）

| 判断 | 理由 |
|---|---|
| LR (線形融合) を採用 | 複雑モデル (Neural SDE, GNN) はAdamic-Adarに勝てないと実証 |
| Hard-negative 学習 | ランダム負例だと "popularity 推薦" になる |
| Isotonic 較正 | Platt より柔軟、ECE 0.001 達成 |
| UMAP (not t-SNE/PCA) | 大域構造保持 + 局所詳細 |
| KDE 平滑化 | 個別 CPC のノイズを抑え "ゾーン" を可視化 |
| 固定 UMAP 座標 | アニメで各 CPC が同じ場所に居続ける |

詳細は [05_FINDINGS.md](05_FINDINGS.md) を参照。

---

## 次のステップ

→ [04_USAGE.md](04_USAGE.md) で各スクリプトの実行例と出力を学ぶ。
