# リポジトリガイドライン

## プロジェクト概要（Overview）

`pnode_patent_runner/` を中心とした、特許・論文の二部グラフ上での技術トレンド予測を研究するリポジトリ。Neural ODE / ポテンシャル流 / 温度発展GNN・時間点過程などのアーキテクチャを、firm×CPC・inventor×CPC・inventor×topic 等の複数ドメインで系統的にベンチマークし、訓練不要ベースライン（人気度・関連性・momentum・構造的近接度）との比較で「学習モデルが本当に増分を持つか」を検証する。

**技術スタック**: PyTorch, torch_geometric, torchdiffeq, statsmodels, scikit-learn, pandas。詳細は [ビルド＆テスト](.docs/04-reference/build-test.md) を参照。

**姉妹リポジトリ**: `../kumagai-patent-analysis`（KG-ATLAS、熊谷組向け実運用ツール）。本リポジトリの研究知見がKG-ATLASの予測系機能（Prophet予測・white-space検出等）の妥当性検証に接続する。詳細は [プロジェクトマップ](.docs/03-domain-specific/project-map.md) を参照。

**主要機能**: プロジェクトに応じて設定

**設計書**: **ローカルに実装計画・設計のmdファイルを新規作成することは禁止**。最終設計は研究方向ごとに1つのGitHub Issueへ一元化し、その中の固定コメント1件をIDで指定して更新し続ける（新しいコメント・新しいファイルを都度追加しない）。運用ルールは [設計記録の方針](.docs/02-workflow/documentation-policy.md) を参照。`pnode_patent_runner/docs/*.md` の既存文書（2026-08-22以前に作成）はそのまま残すが、今後の更新はGitHub Issue側で行う。

## 作業方針

**⚠️ 重要**: main ブランチへの直接コミットは**禁止**されています。すべての変更は作業ブランチを作成し、PR を通じてマージしてください。

> **移行に関する注記（2026-08-21）**: 本ファイル導入以前は、AIエージェントによる作業は main への直接コミットで行われていました（`git log`参照）。本ファイル導入以降の作業はこのブランチ+PRルールに従います。過去のコミットを遡って修正する必要はありません。

作業をなるべく小さな単位に分けてレビューしやすくすること、必要ならば全体の epic issue による管理や、小さい単位でブランチ&PR を管理するようにする（sub issue は必ずしも作る必要はない）。

作業の際に、Codex MCP や Context7 MCP, Serena MCP, kiri MCP を使って、調査させて、まとめて報告して。あなたは MCP の調査結果をレビューして、あらゆる観点から評価して点数と理由をつけて MCP に返答して。点数が 80 点以上になったら、PR や Issue を作成 or 修正して。

Codex MCP が動かないか 1 分以上応答がない場合は Codex コマンドを使用する。それも 1 分以上応答がない場合は他の MCP で暫定的に採点して、後からリファクタする。

詳細は [MCP 使用ガイド](.docs/02-workflow/mcp-usage.md) を参照してください。

## 知識ベース（Knowledge Base）

このプロジェクトの詳細な知識は、`.docs/` ディレクトリにモジュール化して格納されています。エージェントと人間の両方が必要に応じて該当するファイルを参照してください。

### クイックナビゲーション

| カテゴリ | 用途 | 主要ファイル |
| --- | --- | --- |
| **[エンジニアリング原則](.docs/00-core-principles.md)** | 常に参照 | 思考と成熟、シンプルな実装、安定した基盤、プロセスとコミュニケーション |
| **[コーディング規約](.docs/01-coding-standards/)** | コーディング時 | [命名規則](.docs/01-coding-standards/naming-conventions.md)、[コード構造](.docs/01-coding-standards/code-structure.md)、[型アノテーション](.docs/01-coding-standards/type-annotations.md)、[Docstring ガイド](.docs/01-coding-standards/docstring-guide.md) |
| **[ワークフロー・プロセス](.docs/02-workflow/)** | 開発プロセス時 | [MCP 使用](.docs/02-workflow/mcp-usage.md)、[PR 作成](.docs/02-workflow/pr-workflow.md)、[Git ワークフロー](.docs/02-workflow/git-workflow.md)、[TDD・DDD ワークフロー](.docs/02-workflow/tdd-ddd-workflow.md)、[設計記録](.docs/02-workflow/documentation-policy.md) |
| **[ドメイン固有](.docs/03-domain-specific/)** | 研究実験時 | [研究の規律](.docs/03-domain-specific/research-discipline.md)、[プロジェクトマップ](.docs/03-domain-specific/project-map.md)、[データとドメイン](.docs/03-domain-specific/data-and-domains.md) |
| **[リファレンス](.docs/04-reference/)** | 必要時参照 | [セキュリティ](.docs/04-reference/security.md)、[ビルド＆テスト](.docs/04-reference/build-test.md)、[ライブラリガイド](.docs/04-reference/library-guides.md)、[チェックリスト](.docs/04-reference/checklists.md) |

### 想定ユースケース

1. **新規実験・診断スクリプト実装時**: `00-core-principles.md` → `03-domain-specific/research-discipline.md`（訓練不要天井を先に測る規律）→ `01-coding-standards/`
2. **PR 作成時**: `02-workflow/pr-workflow.md` → `02-workflow/git-workflow.md` → `04-reference/checklists.md`
3. **MCP 使用時**: `02-workflow/mcp-usage.md` → `04-reference/checklists.md`
4. **セットアップ時**: `04-reference/build-test.md`
5. **研究の現在地を把握したい時**: `03-domain-specific/project-map.md`

詳細は [知識ベース README](.docs/README.md) を参照してください。

## メンテナンスポリシー（Maintenance policy）

- 繰り返し指示された内容は`.docs/`ディレクトリの該当ファイルに反映を検討
- 冗長性を排除し、簡潔で密度の濃い文書を維持
- 実験の最終設計・生きた履歴は研究方向ごとのGitHub Issueの固定コメントに一元化する（[設計記録の方針](.docs/02-workflow/documentation-policy.md)参照）、AGENTS.md はエントリーポイントとして最小限に保つ
- 知識ベースの構造は`.docs/README.md`を参照
