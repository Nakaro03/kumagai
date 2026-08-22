# チェックリスト・コマンド例（Checklists & Commands）

## 新しい実験・診断スクリプトを始める前のチェックリスト

- [ ] [プロジェクトマップ](../03-domain-specific/project-map.md)で、同じ・近縁のアイデアが既に試されていないか確認した（kiri MCPの`files_search`、またはgrepで既存コードを検索した）
- [ ] 訓練不要ベースラインを、ニューラルモデルより先に測定する計画になっている
- [ ] 停止規則（何を上回れば「意味がある」か）を結果を見る前に決めた
- [ ] holdout分割・複数seed（最低3〜5、できれば10）を使う設計になっている
- [ ] 将来情報の漏洩がないか（`year <= Y`のデータのみから特徴量を作っているか）確認した

## PR作成チェックリスト

- [ ] 関連する検証スクリプトが実行できる（該当する場合）
- [ ] 関連するIssueまたはEpic Issueを参照している（`Closes #123`または`Part of #123`）
- [ ] PR本文に変更内容、背景、影響範囲を記載している
- [ ] `outputs/`配下に新しい結果JSONを追加した場合、小規模なもの（数MB程度）のみforce-addで追跡している。学習済みチェックポイント（`.pt`）や大容量データは含めない

## コードレビューチェックリスト

- [ ] 命名規則に従っている（[命名規則](../01-coding-standards/naming-conventions.md)参照）。ただし年・horizon・在籍年数等の研究変数（`Y`, `H`, `K`）は数理的記法として許容する
- [ ] マジックナンバーが定数化されている（[コード構造](../01-coding-standards/code-structure.md)参照）
- [ ] Docstringが適切に記載されている（[Docstringガイド](../01-coding-standards/docstring-guide.md)参照）
- [ ] 統計的主張（AUC等）に信頼区間・検定・サンプル数の記載がある

## MCP使用時のチェックリスト

- [ ] 適切なMCPツールを選択している（[MCP使用ガイド](../02-workflow/mcp-usage.md)参照）
- [ ] 姉妹リポジトリを調査する場合、Serena/kiriの代替（Explore/general-purpose subagent）を使っている
- [ ] 調査結果をレビューし、80点以上を確認している。80点未満の場合はスコアと理由を報告して止めている
- [ ] Codex MCPが応答しない場合のフォールバック手順を理解している

## コマンド例

```bash
# 依存関係インストール
python3 -m pip install <package>

# 診断スクリプト実行（リポジトリルートから）
python3 -m pnode_patent_runner.diagnose_exit_hazard --domain construction

# 回帰テスト実行
python3 pnode_patent_runner/test_eval_negative_sampling.py

# PR作成（小規模変更）
gh pr create --base main --title "fix: xxx"

# PR作成（Epic配下のサブタスク）
gh pr create --base epic/feature-name --head feature/subtask-1 --title "feat: サブタスク1"
```

詳細は各ワークフローファイルを参照してください。
