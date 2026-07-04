# 02. 環境構築とクイックスタート

[← 01_OVERVIEW](01_OVERVIEW.md) | [次: 03_ARCHITECTURE →](03_ARCHITECTURE.md)

---

## 1. 環境構築

### 1.1 Python と必須ライブラリ

Python 3.8 以上が必要。本プロジェクトは Python 3.8 で開発・検証されています。

```bash
# 必須ライブラリ
pip install numpy pandas scipy scikit-learn matplotlib umap-learn statsmodels

# 埋め込み用（テキスト処理）
pip install sentence-transformers

# Neural SDE プロト（任意、検証目的のみ）
pip install torch torchsde torchdiffeq
```

### 1.2 動作環境

| 項目 | 推奨 |
|---|---|
| OS | Linux / macOS（Windows は WSL 推奨）|
| メモリ | 16 GB 以上 |
| ディスク | 10 GB 以上の空き |
| GPU | 任意（Neural SDE プロト時のみ有用） |

### 1.3 動作確認

```bash
python -c "
import pandas, scipy, sklearn, matplotlib, umap, statsmodels
print('all core libs OK')
"
```

→ エラーなく "all core libs OK" が出れば準備完了。

---

## 2. データの準備

### 2.1 ディレクトリ構造（重要）

このプロジェクトは以下のディレクトリ前提:

```
/home/nakamuraroi/kumagai/
├── pnode_patent_runner/           # スクリプト群
│   ├── docs/guide/                # 本ドキュメント
│   ├── recommender_firm.py        # 推薦エンジン
│   ├── viz_*.py                   # 可視化群
│   └── ...
├── data/
│   └── processed/                 # 前処理済みデータ
│       ├── bipartite_construction_firm.csv     # 必須
│       ├── construction_firm_names.csv          # 必須
│       └── cpc_content_construction.npz         # 必須
└── notebooks/work/dataset/PatentsViewBulkData/  # 生バルク（オプション）
    ├── g_cpc_current.tsv
    ├── g_assignee_disambiguated.tsv
    ├── g_application.tsv
    └── g_patent.tsv
```

### 2.2 必要なデータファイル

**最小構成（既に存在する場合はスキップ可）**:

| ファイル | 必須 | サイズ目安 | 入手方法 |
|---|---|---|---|
| `data/processed/bipartite_construction_firm.csv` | ◎ | ~40 MB | §2.3 で構築 |
| `data/processed/construction_firm_names.csv` | ◎ | ~3 MB | §2.3 で構築（同時） |
| `data/processed/cpc_content_construction.npz` | ◎ | ~8 MB | §2.4 で構築 |

これらが既にある場合は §3 へ。なければ次節へ。

### 2.3 二部グラフ（firm × CPC）の構築

PatentsView の公開バルクデータから構築します。

```bash
# Step 1: 生バルクファイル取得（PatentsView S3）
mkdir -p notebooks/work/dataset/PatentsViewBulkData
cd notebooks/work/dataset/PatentsViewBulkData

# CPC（特許 → 技術分類、3.1 GB）
curl -O https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip
unzip g_cpc_current.tsv.zip

# Assignee（特許 → 企業、342 MB zip → 1.1 GB tsv）
curl -O https://s3.amazonaws.com/data.patentsview.org/download/g_assignee_disambiguated.tsv.zip
unzip g_assignee_disambiguated.tsv.zip

# Application（特許 → 出願日、~150 MB zip）
curl -O https://s3.amazonaws.com/data.patentsview.org/download/g_application.tsv.zip
unzip g_application.tsv.zip

# Patent（特許 → タイトル、~300 MB zip）
curl -O https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip
unzip g_patent.tsv.zip

cd /home/nakamuraroi/kumagai

# Step 2: 二部グラフ構築（5-10 分）
python pnode_patent_runner/build_firm_bipartite.py --domain construction
```

→ `data/processed/bipartite_construction_firm.csv` と
  `data/processed/construction_firm_names.csv` が生成される。

### 2.4 CPC コンテンツ埋め込みの構築

```bash
python pnode_patent_runner/build_cpc_content_embeddings.py --domain construction
```

→ `data/processed/cpc_content_construction.npz` が生成される（5-10 分）。

⚠️ 初回は `sentence-transformers/all-MiniLM-L6-v2` モデル（~80 MB）をダウンロード。

---

## 3. 最初の実行（30 秒で動く）

```bash
cd /home/nakamuraroi/kumagai

# 推薦エンジンを動かす（firm-level、3 年予測）
python pnode_patent_runner/recommender_firm.py --domain construction
```

出力（例）:

```
domain=construction CPC=231 train=2012 test=2015 horizon=3y  [v2: hard-neg training + isotonic calibration]

Fusion weights (standardized): {'relatedness': 0.34, 'momentum': 0.46, 'human-aware': 0.45, 'content': 0.45}

(2)+(4) precision@k on held-out test (full fusion vs relatedness-only):
  P@5   full=0.060  relatedness-only=0.098  uplift=-0.037
  P@10  full=0.058  relatedness-only=0.078  uplift=-0.020
  P@20  full=0.049  relatedness-only=0.061  uplift=-0.012
  ranking AUC (full) = 0.951  (base rate 0.0008)

(3) Calibration ECE: uncalibrated 0.033 -> isotonic 0.000
  isotonic reliability buckets (predicted vs realized hit-rate):
     pred  actual        n
   0.0000  0.0000     8117
   ...
```

これが**推薦エンジンの基本動作確認**です。30 秒〜数分で完了。

---

## 4. 主要可視化の生成（順次）

```bash
# 1. 統合図（hot/cold + 予測可能性 + 推薦、約 1-2 分）
python pnode_patent_runner/viz_final_integrated.py --domain construction

# 2. 予測可能性マップ v3（シンプル版、約 1-2 分）
python pnode_patent_runner/viz_predictability_map_v3.py --domain construction

# 3. 時間発展アニメ GIF（約 2-3 分、20 フレーム）
python pnode_patent_runner/viz_trends_animation.py --domain construction

# 4. 予測精度比較棒グラフ（約 30 秒）
python pnode_patent_runner/compare_prediction_accuracy.py --domain construction
```

各スクリプトは `pnode_patent_runner/viz_*.png` / `.gif` を生成します。

---

## 5. 出力先

すべての可視化と中間ファイルは:

```
pnode_patent_runner/
├── viz_final_integrated.png
├── viz_trends_in_latent.png
├── viz_trends_animation.gif
├── viz_predictability_map_v3.png
├── viz_predictability_map_v7.png
├── viz_crossdomain_forecast.png
├── compare_prediction_accuracy.png
└── ... 等
```

直接画像ビューアで開くか、Markdown 内に貼り付けて使用してください。

---

## 6. うまく動かない時のチェックリスト

| 症状 | 対応 |
|---|---|
| `ModuleNotFoundError: pandas` 等 | `pip install ...` し直す |
| `FileNotFoundError: bipartite_construction_firm.csv` | §2.3 を実行してデータ構築 |
| `FileNotFoundError: cpc_content_construction.npz` | §2.4 を実行 |
| `KeyError: None` 等 | データの year 範囲が想定外。`--y0` `--y1` 引数を確認 |
| UMAP が遅い・固まる | データを subset（`--n-eval 500` 等）で試す |
| Memory error | スワップを増やすか、データを構築済 CSV を別の機械から転送 |

---

## 次のステップ

→ [03_ARCHITECTURE.md](03_ARCHITECTURE.md) でシステム内部の動きを理解。
→ または [04_USAGE.md](04_USAGE.md) で各スクリプトの使い方を学ぶ。
