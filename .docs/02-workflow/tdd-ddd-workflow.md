# TDD・DDDワークフロー（TDD・DDD Workflow）

## 適用範囲（本プロジェクトでの現実的な運用）

本プロジェクトは実験・診断スクリプトが中心の研究コードであり、汎用アプリ開発と同じ厳密さでTDDを全コードに適用することは非現実的。以下の使い分けを原則とする：

- **共有インフラ（`techtrend_common.py`、評価関数群、`unified_training.py`等の複数スクリプトから再利用されるコード）**: TDD推奨。回帰テストの実例: `test_eval_negative_sampling.py`（評価関数の負例サンプリングバグの修正を、合成データでの修正前後比較として固定化）。
- **単発の実験・診断スクリプト（`diagnose_*.py`, `techtrend_arch_*.py`等）**: 先にテストを書くのではなく、**訓練不要ベースラインと事前登録した停止規則を先に書く**ことがこのプロジェクトにおける「レッド」に相当する（[研究の規律](../03-domain-specific/research-discipline.md)参照）。結果の再現性は、合成データでの前後比較（コミット前後でスクリプトの出力が変わらない/意図通り変わることを確認）で担保する。

## TDDの基本サイクル（共有インフラに適用する場合）

### 1. レッド（Red）

仕様に対して失敗する（エラーになる）テストコードを書く

- テストケース = 実行可能な仕様
- テストが失敗することを確認する

### 2. グリーン（Green）

レッドのテストをもとに成功する（パスする）コードを書く

- テストが通る最小限の実装を行う

### 3. リファクタリング（Refactoring）

- コードの重複を削除する
- 命名を改善する
- テストは常にグリーンのまま維持する

## 実践例（本プロジェクトでの回帰テストの実例）

```python
# 1. レッド相当: バグを再現する合成データケースを書く
def test_no_true_positive_leaks_into_negatives() -> None:
    # 真の正例数がmax_posを超える合成グラフを作り、
    # 修正前のコードでは負例に真の正例が混入することを確認する

# 2. グリーン: 修正後のコードで混入がゼロであることを確認
    assert not leaked, f"true positive(s) mislabeled as negative: {leaked[:5]}"

# 3. リファクタリング: 決定性・重複負例の排除も同じテストで担保
    assert torch.equal(pos_ei, pos_ei2), "positive sampling is not deterministic"
```

## 注意点

- **共有インフラの変更は先にテストを書く**: 複数の実験スクリプトが依存するコードほどTDDの価値が高い
- **単発スクリプトは訓練不要ベースライン+停止規則を先に書く**: ニューラルモデルを書く前に、必ず何を上回れば「意味がある」かを事前登録する（[研究の規律](../03-domain-specific/research-discipline.md)）
- **設計の記録**: 設計判断はdocstringや、対応するGitHub Issueの固定コメントに記録する（ローカルmdでの新規作成は禁止。[設計記録の方針](./documentation-policy.md)参照）

## 関連ドキュメント

- [PR作成ワークフロー](./pr-workflow.md)
- [設計記録の方針](./documentation-policy.md)
- [研究の規律](../03-domain-specific/research-discipline.md): 訓練不要天井・事前登録ゲートという、このプロジェクト独自の「レッド」に相当する規律
- [コード構造](../01-coding-standards/code-structure.md)
