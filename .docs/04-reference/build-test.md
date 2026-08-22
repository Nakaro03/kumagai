# ビルド＆テスト（Build & Test）

## セットアップ

システムPython（`python3`、torch 2.4.1+cu121, torch_geometric, torchdiffeq, statsmodels, pandas, scikit-learnがインストール済み）を使用する。専用venvは現状必須ではない——`python3`の解決先が正しいことを`which python3`で確認する。

```bash
# 依存関係の追加インストールが必要な場合
python3 -m pip install <package>
```

## 実行

すべてのスクリプトは**リポジトリルート（`/home/*/kumagai`）から`-m`付きで実行する**（`pnode_patent_runner`はパッケージとして扱われ、内部で`from pnode_patent_runner.xxx import ...`という絶対importを使うため）:

```bash
# 正しい実行方法
python3 -m pnode_patent_runner.diagnose_exit_hazard --domain construction

# 誤り（pnode_patent_runner/ディレクトリの中からの直接実行はModuleNotFoundErrorになる）
cd pnode_patent_runner && python3 diagnose_exit_hazard.py  # NG
```

例外: `run_*.py`系の一部スクリプトは`sys.path.insert(0, str(_REPO))`で自己解決するため、`pnode_patent_runner/`配下から直接実行できる場合もある（ファイル冒頭の`_REPO = Path(__file__).resolve().parents[1]`パターンの有無を確認する）。

## テスト

```bash
# 回帰テスト（共有インフラのみ、実データ不要）
python3 pnode_patent_runner/test_eval_negative_sampling.py

# 構文チェック
python3 -m py_compile pnode_patent_runner/<file>.py
```

自動テストスイート（pytest）は共有インフラ（`techtrend_common.py`等）に対しては整備を推奨するが、実験・診断スクリプトは合成データでの前後比較で妥当性を確認する運用（[TDD・DDDワークフロー](../02-workflow/tdd-ddd-workflow.md)参照）。

## 前提条件

- **GPU**: RTX 3090（24GB）が利用可能。学習を伴うスクリプト（`run_dual_force_patent_domain.py`, `run_tap_node_patent_domain.py`, `run_benchmark_comparison.py`等）で必要。訓練不要の診断スクリプト（`diagnose_*.py`）はGPU不要
- **メモリ**: construction等の大規模ドメイン（CPC maingroup数千種類）は素朴な3次元テンソル実装だとGPUメモリを容易に使い切る。行列積ベースの実装（分配法則の活用）が必須な場面がある（`dual_force_models.py`のforward()実装を参照）

## 環境変数

現状このリポジトリ自体は必須の環境変数を持たない。MCPサーバーの認証は`claude mcp`側で管理される。
