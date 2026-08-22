# PR作成ワークフロー（PR Workflow）

## 規模に応じた2つのパス

- **小規模変更**（バグ修正、単一の診断スクリプト、単一の設計文書追加）: Epic Issueを介さず、`main`への直接PRでよい。[Gitワークフロー](./git-workflow.md)の「小規模変更」フローを参照。
- **大規模な検証プログラム**（複数ドメイン・複数手法を横断する新しい研究方向の立ち上げ等）: 以下のEpic Issue + PRで管理する。

## Epic Issue + PRによる管理

```bash
# 1. Epic Issue作成（機能全体の管理）
gh issue create \
  --title "Epic: 機能全体の名前" \
  --body "
## 実装Plan
- [ ] PR: サブタスク1
- [ ] PR: サブタスク2
- [ ] PR: サブタスク3

## 完了条件
- [ ] Gate 0/S/Lの判定が出ている
- [ ] 結果がoutputs/配下に保存され、docs/の設計文書に反映されている
"

# 2. ブランチ作成
git checkout -b feature/xxx

# 3. Draft PR作成（Epic Issueを参照、充実した本文）
gh pr create --draft \
  --title "feat: サブタスク1"
# - Part of #<Epic Issue番号>
# - 概要、背景・目的、変更内容、影響範囲、テスト等を記載

# 4. 実装

# 5. PR description更新（実装に応じて更新）

# 6. Ready for review
gh pr ready

# 7. マージ
```

## Epic Issueの役割

- 機能全体・検証プログラム全体の進捗管理
- 依存関係の明確化
- 完了条件の定義（本プロジェクトでは「Gate判定が出ている」ことが多い）

## PRの役割

- 個別タスクの実装
- レビュー単位

## サブタスク管理の方針

**基本方針**: 基本はPRのみで管理、必要に応じてSub Issueも作成可。

## PR bodyテンプレート（ブランチ別）

| ケース | ベース ← ヘッド | 想定スコープ | PR body冒頭 | Issueクローズ方針 |
| ------ | --------------------------------------- | -------------- | ---------------- | ----------------- |
| A | `epic/feature-name ← feature/subtask-*` | サブタスク | `Part of #<Epic>` | - （Issueなし） |
| B | `main ← epic/feature-name` | Epic統合 | `Closes #<Epic>` | EpicをClose |
| C | `main ← feature/*` | 小規模／Hotfix | `Closes #<Issue>`（Issueがある場合） | 単体でClose |

> **重要**:
>
> - **基本的にSub Issueは作成しない**: Epic配下のサブタスクはPRのみで管理する。
> - **GitHubのクローズ動作**: `Closes #123`は、PRがマージされた時点でIssueをクローズする。

**コマンド例（ケースC、本プロジェクトで最も頻度が高い）**:

```bash
gh pr create --base main \
  --head fix/eval-negative-sampling \
  --title "fix: 評価関数の負例サンプリングバグ修正" \
  --body "$(cat <<'EOF'
Closes #1

## Summary
future_link_auc_scores() 等3箇所が、サブサンプル後の正例集合から負例棄却集合を
構築しており、真の正例を負例として誤サンプリングしうるバグを修正。

## Testing
- [x] 合成データでの修正前後比較テスト追加（test_eval_negative_sampling.py）
- [x] 実データ(construction 2021)で契約前後の再現性を確認
EOF
)"
```
