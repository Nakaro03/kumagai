# MCP使用ガイド（MCP Usage Guide）

## 作業方針

作業の際に、Codex MCPやContext7 MCP、Serena MCP、kiri MCPを使って、調査させて、まとめて報告する。

エージェントはMCPの調査結果をレビューして、あらゆる観点から評価して点数と理由をつけてMCPに返答する。点数が80点以上になったら、PRやIssueを作成or修正する。80点未満の場合は、スコアと理由を報告して止める（勝手にPR/Issueを作らない）。

Codex MCPが動かないか1分以上応答がない場合はCodexコマンド（`codex mcp-server`が起動しているCLI本体、または`codex exec`）を使用する。それも1分以上応答がない場合は他のMCPで暫定的に採点して、後からリファクタする。

## このプロジェクトでの設定状況

`claude mcp list` をこのリポジトリ（`/home/*/kumagai`）から実行して確認する。

- **Codex MCP**: `codex mcp-server`（プロジェクトローカルスコープで登録済み）。`cwd`パラメータで他リポジトリ（`../kumagai-patent-analysis`等）を対象に読み取り専用調査もできる（`sandbox: read-only`, `approval-policy: never`推奨）。
- **Context7 MCP**: `npx -y @upstash/context7-mcp`（プロジェクトローカルスコープ）。ライブラリドキュメント検索専用——**学術文献・一般的なWeb記事の調査には向かない**。その場合は`WebSearch`ツールで代替する（本プロジェクトの実績: 論文・アーキテクチャの先行研究調査はWebSearchが実質的なContext7代替）。
- **Serena MCP**: `serena start-mcp-server --context claude-code --project-from-cwd`。**活性化されたプロジェクトのディレクトリにスコープが固定される**（`cwd`引数では切り替わらない）。姉妹リポジトリ（`kumagai-patent-analysis`）を調査する場合は、`Explore` subagentで代替する。
- **kiri MCP**: `kiri --repo . --db .kiri/index.duckdb --watch`。Serenaと同じ理由でこのリポジトリにスコープが固定される。姉妹リポジトリの調査には`general-purpose` subagentで代替する。

**姉妹リポジトリ（`kumagai-patent-analysis`）を調査する4系統レビューの実施例**: Codex CLI（`cwd`指定で直接読み取り）／Context7→WebSearch／Serena→Explore subagent／kiri→general-purpose subagent、という代替パターンが実績として確立している。詳細は`kumagai-patent-analysis`側のissue #2, #5, #10, #11, #12を参照（同じ手法の先例）。

## スコアリングの重み（4系統フルレビュー時）

姉妹リポジトリで確立した慣例: Serena 30% / Codex 30% / kiri 25% / Context7 15%。単一MCP（Codexのみ等）で完結させる場合は、その旨と理由を記録する（例: 実データクエリの反復検証が中心で、他系統の付加価値が薄いと判断した場合）。

## Context7 MCP Server

実装前にライブラリの最新ドキュメントを取得:

1. `resolve-library-id`: ライブラリIDを取得
2. `get-library-docs`: 最新ドキュメントを取得

**例**: `/pytorch/pytorch`, `/rtqichen/torchdiffeq`, `/pyg-team/pytorch_geometric`

## Serena MCP Server

コードベース操作に使用:

**検索**:
- `find_symbol`: シンボル検索（クラス、関数等）
- `find_referencing_symbols`: 参照箇所を検索
- `get_symbols_overview`: ファイルの高レベル概要（新しいファイルを読む最初の一手）

**編集**:
- `replace_symbol_body`: シンボルの本体を置換
- `insert_after_symbol` / `insert_before_symbol`: シンボルの前後に挿入

**活用例**:
- 実験スクリプト間で共有されるヘルパー（`techtrend_common.py`等）のリファクタリング時のシンボル検索
- 大規模な変更時のパターン検索

## Codex MCP Server

AI支援によるコーディングタスク・独立監査に使用:

- `codex`: 新規スレッドを開始。`prompt`, `sandbox`（`read-only`推奨）, `approval-policy`（`never`推奨）, `cwd`を指定
- `codex-reply`: 同じスレッドで会話を継続（`threadId`必須）——スコアのやり取りや追加検証依頼はこちらを使う

**活用例（本プロジェクトの実績）**:
- 実データ（pickle, CSV, raw PatentsViewテーブル）への直接クエリを伴う実現可能性調査
- 既存コードのバグ監査（例: 評価関数の負例サンプリングバグ発見・修正レビュー）
- 新しい研究アイデアの先行研究チェック（プロジェクト内ですでに試された内容との重複確認）

## kiri MCP Server

コードベースの理解と分析に特化したツール

**主要機能**:
- `context_bundle`: コードコンテキストの抽出（🎯 PRIMARY TOOL）。目標を具体的なキーワードで指定
- `files_search`: 関数名・クラス名・エラーメッセージ等の正確なキーワード検索。新しい研究アイデアの**先行実装が既にあるかの確認**に有用（本プロジェクトでは複数回、「新しいと思ったアイデアが実は既存だった」ことをこの検索で発見している）
- `snippets_get`: 特定ファイルパスからコードスニペットを取得
- `semantic_rerank`: セマンティック類似度による再ランク付け
- `deps_closure`: 依存関係グラフのトラバース

**ベストプラクティス**:
1. 新しい実験アイデアを思いついたら、実装前に必ず`files_search`/`context_bundle`で類似の既存実装がないか確認する
2. `context_bundle`を最初のステップとして使用（具体的なキーワードを含む明確な目標を指定）
3. 抽象的な動詞（"understand", "explore"）ではなく、具体的なキーワードを使用
