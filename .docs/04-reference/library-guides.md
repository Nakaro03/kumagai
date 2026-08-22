# ライブラリガイド（Library Guides）

## Context7 MCP Server

実装前にライブラリの最新ドキュメントを取得:

1. `resolve-library-id`: ライブラリIDを取得
2. `get-library-docs`: 最新ドキュメントを取得

詳細は[MCP使用ガイド](../02-workflow/mcp-usage.md)を参照。

## 主要ライブラリ

| ライブラリ | 用途 |
| --- | --- |
| `torch` (2.4.1+cu121) | ニューラルモデル全般 |
| `torch_geometric` | GAT/VGAE等のグラフニューラルネットワーク |
| `torchdiffeq` | Neural ODE（`odeint`, `odeint_adjoint`） |
| `statsmodels` | OLS（HC1/クラスタ頑健SE）、Gate 0の回帰分析 |
| `scikit-learn` | ロジスティック回帰（GEM）、AUC/AP評価 |
| `pandas` / `numpy` | データ前処理・特徴量構築 |

新しいライブラリを導入する場合、このセクションと`04-reference/build-test.md`に追記すること。
