# 技術トレンド予測プロジェクト — ドキュメント

初めての人向けに、このプロジェクトの全体像と各機能を理解できるように分割した資料です。

---

## このプロジェクトは何か（1 行で）

> **特許・論文データから「企業が次に取り組む技術」を、較正された honest な信頼度付きで予測し、潜在空間上で可視化するシステム。**

---

## ドキュメント構成

下の順に読むのが推奨です。

| 番号 | ドキュメント | 内容 | 読了目安 |
|---|---|---|---|
| **01** | [OVERVIEW.md](01_OVERVIEW.md) | 何を解こうとしているか、主要な概念、研究の核心 | 10 分 |
| **02** | [GETTING_STARTED.md](02_GETTING_STARTED.md) | 環境構築、データ準備、最初の実行 | 20 分 |
| **03** | [ARCHITECTURE.md](03_ARCHITECTURE.md) | システム設計、データフロー、モジュール構成 | 15 分 |
| **04** | [USAGE.md](04_USAGE.md) | 各スクリプトの使い方と実行例 | 30 分 |
| **05** | [FINDINGS.md](05_FINDINGS.md) | 研究で分かったこと、proximity-bound 限界 | 15 分 |
| **06** | [VISUALIZATIONS.md](06_VISUALIZATIONS.md) | **可視化の読み方ガイド**（色・記号・図の使い分け） | 15 分 |

---

## 全体所要時間

- **読むだけ**: 1.5 時間で全体を把握
- **手を動かして再現**: 半日（データ取得を含む）

---

## クイック索引（やりたいことから探す）

| やりたいこと | 行き先 |
|---|---|
| まず何ができるか知りたい | [01_OVERVIEW.md](01_OVERVIEW.md) |
| 環境構築したい | [02_GETTING_STARTED.md](02_GETTING_STARTED.md) §1 |
| 自分のデータで実行したい | [02_GETTING_STARTED.md](02_GETTING_STARTED.md) §3 |
| システム全体の仕組みを知りたい | [03_ARCHITECTURE.md](03_ARCHITECTURE.md) §1 |
| 特定のスクリプトの使い方 | [04_USAGE.md](04_USAGE.md) スクリプト索引 |
| 可視化の見方 | [06_VISUALIZATIONS.md](06_VISUALIZATIONS.md) |
| **初めて見る人向けのシンプルな可視化** | [viz_clear.png](../viz_clear.png) — 1 枚で完結、説明吹き出し付き |
| 研究の結論を知りたい | [05_FINDINGS.md](05_FINDINGS.md) §総括 |
| トップ会議への投稿戦略 | [05_FINDINGS.md](05_FINDINGS.md) §論文化戦略 |

---

## 関連する旧ドキュメント

このプロジェクトには探索期に書かれた多数の docs があります。**主要なものだけ**:

| ファイル | 内容 |
|---|---|
| [PROGRESS_REPORT_TREND_PREDICTION.md](../PROGRESS_REPORT_TREND_PREDICTION.md) | 教授向け進捗報告書（Q&A 形式）|
| [X3_DESCRIPTIVE_FRAMING.md](../X3_DESCRIPTIVE_FRAMING.md) | X3-clean の記述ツール framing |
| [PISDE_ROLE_DECISION.md](../PISDE_ROLE_DECISION.md) | PI-SDE の役割決定（descriptive vs predictive）|

その他は探索期の試行（X1–X5 PI-SDE 系、DRIFT 等）のメモであり、**現在の deliverable には直接関係しません**。

---

## 主要な成果物（5 つ）

| 種類 | ファイル | 何のため |
|---|---|---|
| 推薦エンジン | `recommender_firm.py` | 企業 → CPC 推薦 |
| 可視化（統合）| `viz_final_integrated.png` | hot/cold + 予測可能性 + 推薦 |
| 可視化（トレンド）| `viz_trends_animation.gif` | 15 年の業界動態 |
| 予測比較 | `compare_prediction_accuracy.png` | 7 手法の精度比較 |
| クロスドメイン | `viz_crossdomain_forecast.png` | 特許 vs 論文の予測比較 |

---

## サポート

質問・不明点は以下のドキュメントを順に確認してください:
1. まず該当する番号のドキュメントを開く
2. 索引から関連トピックを探す
3. それでも分からなければ、コードを直接読む

各スクリプトはセルフドキュメントされており、`--help` で使い方が表示されます。

```bash
python pnode_patent_runner/recommender_firm.py --help
```

それでは [01_OVERVIEW.md](01_OVERVIEW.md) からどうぞ。
