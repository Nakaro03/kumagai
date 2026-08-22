# 設計記録の方針（Documentation Policy）

## 最終ルール（2026-08-22改訂）: ローカルの新規設計・計画mdファイルは禁止

**実装計画・設計文書をローカルの新しいMarkdownファイルとして作成することは禁止する。** 一般的なAGENTS.mdテンプレートが定める「設計専用ディレクトリ・ファイルの作成禁止」の原則をそのまま踏襲する。

理由: 本プロジェクトは数ヶ月にわたる研究プログラムであり、以前は`pnode_patent_runner/docs/*.md`に生きた設計文書を置く運用をしていたが、これは以下の問題を生んだ:

- 同じ研究方向について、どのファイルが「最新の正しい結論」かが分かりにくくなる（複数ファイルに分散し、互いの関係が曖昧）
- ローカルファイルはレビュー・検索・通知の対象になりにくく、チームや後続セッションとの共有が弱い

## 新しい記録先: GitHub Issueの固定コメント1件を更新し続ける

**最終設計を一元化するため、研究方向ごとに1つのGitHub Issueを作り、その中の固定コメント1件だけをコメントIDで指定して更新し続ける。** 新しい実験結果が出るたびに新しいコメントやファイルを追加するのではなく、同じコメントを`gh api`でPATCH更新し、常に「現時点の最終設計」を1箇所に保つ。

**運用手順**:

```bash
# 1. 研究方向ごとにIssueを作成（初回のみ）
gh issue create --title "Exit Hazard: 撤退リスク予測" --body "調査中"

# 2. 固定コメントを1件だけ投稿し、コメントIDを控える
gh issue comment <issue番号> --body-file initial_design.md
# → 返ってきたURLの末尾（#issuecomment-XXXXXXXXX）がコメントID

# 3. 以降の更新は、同じコメントIDをPATCHで上書きする（新しいコメントを追加しない）
gh api -X PATCH repos/<owner>/<repo>/issues/comments/<comment_id> -f body="$(cat updated_design.md)"
```

- コメントの編集履歴はGitHub側で保持される（"edited"表示から遡って確認可能）ため、履歴を失うことにはならない
- 1つの研究方向（Exit Hazard、Collaboration Tie等）につき、Issueは1つ・固定コメントも1つに保つ。関連するPRからは`Part of #<Issue番号>`で参照する
- Issue本体（description）とこの固定コメントの役割を混同しない: Issue本体は「何を調べているか」の短い要約、固定コメントが「現時点の最終設計・結論」の本体

## 既存のローカル設計文書（`pnode_patent_runner/docs/*.md`）の扱い

2026-08-22以前に作成された`DUAL_FORCE_REDESIGN.md`, `EXIT_HAZARD_DESIGN.md`, `COLLABORATION_TIE_DESIGN.md`, `RESEARCH_PLAN_PREDICTABILITY.md`等は**削除しない**。歴史的record・過去の実験ログとしてそのまま残す。ただし、**これらのファイルへの新規追記は行わない**——該当する研究方向に新しい進展があった場合は、対応するGitHub Issueの固定コメントを更新する（Issueがまだ無い研究方向は、この機会に1つ作成する）。

## テストと実装が仕様書（共有インフラに適用）

**原則**: ドキュメントではなく、コードが真実

1. **テストが仕様書**: テストケース = 実行可能な仕様
2. **実装が設計書**: docstring = 設計判断、責務、制約
3. **PRが実装の文脈**: PR description = なぜこの実装にしたか

## 記録先（本プロジェクトでは5つ + 出力データ）

- **テスト**: 共有インフラの振る舞い（実行可能な仕様）
- **コードのdocstring**: 設計判断、責務、制約
- **PR description**: 実装の背景、他の案、トレードオフ
- **コミットメッセージ**: なぜこの実装にしたか
- **GitHub Issueの固定コメント（コメントID指定でPATCH更新）**: 研究プログラムの生きた設計文書（Gate判定、実験履歴、否定的結果を含む）——ローカルmdの代替
- **`outputs/**/*.json`**（小規模なもののみ、force-addで.gitignoreを越えて追跡）: 検証結果の生データ

## 原則

**上記以外の場所に設計情報を記録しない。** 特に以下を禁止:

- ローカルの新しい設計専用ディレクトリ・ファイルの追加（`docs/`, `_docs/`, `design/`, `*_DESIGN.md`, `*_PLAN.md`等）
- 既存の`pnode_patent_runner/docs/*.md`への新規追記
- 同じ研究方向について複数のIssue・複数の固定コメントを作ること（分散を防ぐという目的に反する）

## 検索方法

```bash
# 研究方向のIssueを探す
gh issue list --state all --search "exit hazard"

# 固定コメントの現在の内容を確認する
gh issue view <issue番号> --comments

# PRで設計判断を探す
gh pr list --search "exit hazard"

# コードで探す
rg "def diagnose_exit_hazard" --type py -A 20

# 過去の（凍結された）ローカル設計文書を探す
ls pnode_patent_runner/docs/*.md
```
