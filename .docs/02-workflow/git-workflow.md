# Gitワークフロー（Git Workflow）

## ブランチ戦略

**原則**: mainは常に安定（動作確認済み。本リポジトリは研究コードのためpytest等の自動テストは限定的——[TDD・DDDワークフロー](./tdd-ddd-workflow.md)のドメイン適用範囲を参照）

**⚠️ 重要**: mainブランチへの直接コミットは**禁止**されています。すべての変更は作業ブランチを作成し、PRを通じてマージしてください。

> **注記**: 本ファイル導入（2026-08-21）以前のコミットは直接main、または単一のfeatureブランチなしでの作業が実績として存在する（`git log`参照）。過去分の移行対応は不要。以降の作業からこのルールを適用する。

**階層構造**:

```
main (安定版)
  ↑
epic/feature-name (統合・テスト用、大きな検証プログラム単位)
  ↑
  ├─ feature/subtask-1 (個別タスク、例: 1つのGate 0診断スクリプト)
  ├─ feature/subtask-2 (個別タスク、例: 評価バグの修正)
  └─ feature/subtask-3 (個別タスク)
```

小規模な変更（バグ修正、単一の診断スクリプト追加等）は、epicを介さず`main`への直接PRでよい。

**運用フロー（小規模変更、本プロジェクトで最も頻度が高い）**:

```bash
git checkout -b fix/eval-negative-sampling
# → 実装 → コミット
gh pr create --base main \
  --title "fix: xxx" \
  --body "$(cat <<'EOF'
## Summary
...

## Testing
- [x] 該当する検証スクリプトを実行し結果を確認
EOF
)"
# → レビュー → マージ
```

**運用フロー（大規模な検証プログラム、Epic単位）**:

```bash
# 1. Epicブランチ作成
git checkout -b epic/predictability-map-writeup

# 2. 個別タスクブランチ作成（epicから分岐）
git checkout -b feature/finish-verification-b

# 3. 実装 → epicにマージ
gh pr create --base epic/predictability-map-writeup \
  --title "feat: 検証Bの全ドメイン化"
# PR本文: Part of #<Epic Issue番号>、概要、背景・目的、変更内容、影響範囲、テスト

# → レビュー → マージ（epicへ）→ 次のタスクを繰り返す

# 5. Epic完了後、mainにマージ
gh pr create --base main \
  --head epic/predictability-map-writeup \
  --title "Epic: predictability-map-writeup" \
  --body "Closes #<Epic Issue番号>"
```

**メリット**:

- mainが常に安定
- epicブランチで統合検証
- 個別タスクごとにレビュー

**注意点**:

- 動作確認できていないコード・検証スクリプトが失敗するコードはepicにもマージしない
- epicが長生きしすぎないよう、適度な粒度に分割

## 並行作業戦略（git worktree）

複数のEpicやタスクを並行開発する場合、git worktreeを使用して独立した作業ディレクトリを作成する。

```bash
cd /path/to/kumagai
git checkout -b branch-a

git worktree add ../kumagai-branch-b branch-b

git worktree list
git worktree remove path
git worktree prune
```

**注意点**:

- 各worktreeは独立した作業ツリーだが、gitリポジトリは共有
- 同じブランチを複数のworktreeでチェックアウトできない
- 大容量データ（`data/processed/`, `outputs/`）はworktree間で重複しない（`.gitignore`対象、シンボリックリンクではなく実体を1つに保つ）
