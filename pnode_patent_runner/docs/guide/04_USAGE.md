# 04. 各スクリプトの使い方

[← 03_ARCHITECTURE](03_ARCHITECTURE.md) | [次: 05_FINDINGS →](05_FINDINGS.md)

---

## 索引

- §1 [準備系](§1-準備系-一度だけ実行) — データ・埋め込み構築
- §2 [推薦エンジン](§2-推薦エンジン) — メイン
- §3 [可視化群](§3-可視化群) — 全 PNG / GIF 生成
- §4 [比較・診断](§4-比較診断) — 精度比較、研究検証
- §5 [全スクリプト早見表](§5-全スクリプト早見表)

すべてのスクリプトは `--help` で引数一覧が出ます。

---

## §1 準備系（一度だけ実行）

### `build_firm_bipartite.py` — 二部グラフ構築

```bash
python pnode_patent_runner/build_firm_bipartite.py --domain construction
```

**入力**: `notebooks/work/dataset/PatentsViewBulkData/g_*.tsv`
**出力**:
- `data/processed/bipartite_construction_firm.csv`
- `data/processed/construction_firm_names.csv`

**所要時間**: 5-10 分（5 GB のバルクをストリーミング）

### `build_cpc_content_embeddings.py` — CPC コンテンツ埋め込み

```bash
python pnode_patent_runner/build_cpc_content_embeddings.py --domain construction
```

**入力**: 上記バルク + bipartite CSV
**出力**: `data/processed/cpc_content_construction.npz`
**所要時間**: 5-10 分（MiniLM 埋め込み）

---

## §2 推薦エンジン

### `recommender_firm.py` — メインの推薦エンジン

```bash
python pnode_patent_runner/recommender_firm.py --domain construction
```

**主要引数**:
- `--train-year` (default 2012)
- `--test-year` (default 2015)
- `--horizon` (default 3 years)
- `--seed` (default 42)

**出力例**:
```
P@5  full=0.157  relatedness-only=0.164  uplift=-0.007
P@10 full=0.117  relatedness-only=0.125  uplift=-0.008
ranking AUC (full) = 0.872  (base rate 0.012)
calibration ECE: uncalibrated 0.070 -> isotonic 0.001
```

**何を見るか**:
- **P@5**: 上位 5 件の的中率（高いほど良い）
- **AUC**: ランキング品質
- **ECE**: 較正の honest さ（低いほど良い）

---

## §3 可視化群

### A. メイン統合図 — `viz_final_integrated.py`

```bash
python pnode_patent_runner/viz_final_integrated.py --domain construction
```

**出力**: `viz_final_integrated.png`
**内容**: 2 パネル + 表
- (a) Hot vs Cold バブル（赤=加熱、青=冷却）
- (b) 予測可能性マップ + 企業推薦 + 金★ Sweet Spot
- 右側に企業情報と Top-5 推薦表

**使いどころ**: 教授・経営層への 5 分プレゼン用

### B. 予測可能性マップ シンプル版 — `viz_predictability_map_v3.py`

```bash
python pnode_patent_runner/viz_predictability_map_v3.py --domain construction
```

**出力**: `viz_predictability_map_v3.png`
**内容**: 1 ストーリーで明快
- 緑ゾーン（予測可能）+ 黒破線境界
- 1 社のポートフォリオ★ + 番号付き Top-5
- 右側に較正済み信頼度テーブル

### C. Nature 風 2 パネル — `viz_predictability_map_v6.py`

```bash
python pnode_patent_runner/viz_predictability_map_v6.py --domain construction
```

**出力**: `viz_predictability_map_v6.png`
**内容**: 過去フロー (青) vs 予測フロー (赤) を並列

### D. 3 パネル + ケース — `viz_predictability_map_v7.py`

```bash
python pnode_patent_runner/viz_predictability_map_v7.py --domain construction
```

**出力**: `viz_predictability_map_v7.png`
**内容**: 過去 + 予測 + ケーススタディの 3 連

### E. 時間スナップショット — `viz_trends_in_latent.py`

```bash
python pnode_patent_runner/viz_trends_in_latent.py --domain construction
```

**出力**: `viz_trends_in_latent.png`
**内容**: 2005, 2010, 2015, 2020 の 4 時点比較 + 15 年トレンド要約

### F. **アニメ GIF** — `viz_trends_animation.py`

```bash
python pnode_patent_runner/viz_trends_animation.py --domain construction
```

**主要引数**: `--fps 2`（再生速度）
**出力**: `viz_trends_animation.gif`（~700 KB）
**内容**: 2001-2020 の 20 フレーム動画

**使いどころ**: スライド埋め込み、業界変遷の 10 秒説明

### G. 折れ線予測 — `viz_line_forecast_backtest.py`

```bash
python pnode_patent_runner/viz_line_forecast_backtest.py --domain construction
```

**出力**: `viz_line_forecast_backtest.png`
**内容**: 上位 8 CPC の折れ線 + 3 手法のバックテスト MAPE

**重要**: バックテスト期間 (2016-2018) を使用。最新年（2021-2024）はデータラグで使えない。

### H. クロスドメイン比較 — `viz_crossdomain_forecast.py`

```bash
python pnode_patent_runner/viz_crossdomain_forecast.py
```

**出力**: `viz_crossdomain_forecast.png`
**内容**: 特許 (construction) vs arxiv CS の forecasting 精度比較

### I. ケーススタディ単独 — `case_study_firm.py`

```bash
python pnode_patent_runner/case_study_firm.py --domain construction
```

**出力**: 標準出力に企業ごとのテキスト出力（PNG なし）

---

## §4 比較・診断

### `compare_prediction_accuracy.py` — 7 手法の精度比較

```bash
python pnode_patent_runner/compare_prediction_accuracy.py --domain construction
```

**出力**:
- 標準出力に表
- `compare_prediction_accuracy.png` 棒グラフ

**比較する 7 手法**:
1. Random（下限）
2. Popularity (momentum)
3. **Adamic-Adar (relatedness)** ← 通常勝者
4. Content (MiniLM)
5. Human-aware (SVD)
6. Fusion LR (hard-neg)
7. Fusion + isotonic calibration

**何を見るか**: AA が ~14× base rate を達成し、複雑手法と同等 or 上回ることを確認。

### `prototype_bipartite_sde.py` — Neural SDE プロト（研究検証用）

```bash
python pnode_patent_runner/prototype_bipartite_sde.py --domain construction
```

**所要時間**: ~5 秒（GPU 必須でない、CPU でも動く）
**結論**: SDE は AA に負ける（−0.099）= proximity-bound 限界の確定的証拠

---

## §5 全スクリプト早見表

### 準備（一度だけ）

| スクリプト | 行頭で必要 |
|---|---|
| `build_firm_bipartite.py` | データ未構築なら |
| `build_cpc_content_embeddings.py` | データ未構築なら |

### 推薦・評価

| スクリプト | 目的 |
|---|---|
| `recommender_firm.py` | メインの推薦 + 評価 |
| `recommender_inventor.py` | 発明者レベル版（参考）|
| `compare_prediction_accuracy.py` | 7 手法比較 |

### 可視化（PNG）

| スクリプト | 用途 |
|---|---|
| `viz_final_integrated.py` | プレゼン用メイン |
| `viz_predictability_map_v3.py` | シンプル説明用 |
| `viz_predictability_map_v6.py` | Nature 風 2 パネル |
| `viz_predictability_map_v7.py` | 3 パネル + ケース |
| `viz_trends_in_latent.py` | 時間軸スナップショット |
| `viz_line_forecast_backtest.py` | 時系列予測 |
| `viz_crossdomain_forecast.py` | 特許 vs 論文 |

### 可視化（GIF）

| スクリプト | 用途 |
|---|---|
| `viz_trends_animation.py` | 業界変遷アニメ（必見） |

### 検証（研究の証拠）

| スクリプト | 検証内容 |
|---|---|
| `diagnose_*.py` 群 | 各仮説の検証 |
| `test_*.py` 群 | 各アーキ・信号の検証 |
| `prototype_bipartite_sde.py` | Neural SDE の限界実証 |

---

## 別ドメインで実行

「**energy**」「**computing**」も同じ方法論で実行可能:

```bash
# 1. ドメイン用データ構築
python pnode_patent_runner/build_firm_bipartite.py --domain energy
python pnode_patent_runner/build_cpc_content_embeddings.py --domain energy

# 2. 推薦エンジン
python pnode_patent_runner/recommender_firm.py --domain energy

# 3. 可視化
python pnode_patent_runner/viz_final_integrated.py --domain energy
```

⚠️ ただし **energy** は CPC が 317 しかないため、`viz_predictability_map_v3.py` 等で表示が変化することに注意。

---

## 典型的なワークフロー（30 分で全部回す）

```bash
cd /home/nakamuraroi/kumagai

# 1. 推薦エンジン動作確認（30 秒）
python pnode_patent_runner/recommender_firm.py --domain construction

# 2. 比較棒グラフ（30 秒）
python pnode_patent_runner/compare_prediction_accuracy.py --domain construction

# 3. メイン統合図（1-2 分）
python pnode_patent_runner/viz_final_integrated.py --domain construction

# 4. アニメ GIF（2-3 分）
python pnode_patent_runner/viz_trends_animation.py --domain construction

# 5. 折れ線バックテスト（30 秒）
python pnode_patent_runner/viz_line_forecast_backtest.py --domain construction
```

→ 5 つのファイル（PNG×4 + GIF×1）が生成され、プロジェクトの主要成果物が揃う。

---

## 次のステップ

→ [05_FINDINGS.md](05_FINDINGS.md) で研究で分かったこと、限界、論文化戦略を学ぶ。
