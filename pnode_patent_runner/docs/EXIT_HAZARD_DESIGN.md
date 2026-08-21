# 技術関係の撤退リスク(Exit Hazard)— Honest-Tool 設計仕様

**Status**: Gate 0 通過(2026-08-21)。**ニューラルモデルは提案しない** — 訓練不要のスコアそのものが成果物という結論。

- 検証コード: `diagnose_exit_hazard.py`
- 結果: `outputs/predictability_map/exit_hazard_construction.json`(firm×CPC)、
  `outputs/predictability_map/exit_hazard_agrifood.json`(inventor×CPC、交差ドメイン検証)
- 4系統代替レビュー(Codex MCP中心、本タスクはCodexのみで完結——理由は§5): 84/100

---

## 1. 背景

本プロジェクトのWHERE(参入)/HOW-MUCH(成長率)/WHEN(タイミング)/WHAT-NEW(新規結合)という
4タスク族は、static/RNN/NeuralODE/PNODE/Dual-Force/TAP-NODE/Jump-ODE-TPPの全アーキテクチャで
共通の壁にぶつかっている——構造的・慣性的ベースライン(人気度・関連性・momentum)が学習モデルを
上回るか、学習モデルが天井に届かない(`WHY_NEURALODE_FAILS_ja.md`)。

本書は、この4タスク族の**鏡像**——「企業(または発明者)が既存の技術分野への出願を**やめる**」
というEXITタスクを検証する。粗い代理指標(`trend_evaluation.py`のExit-AUC、author_topicドメイン、
1年欠測=撤退という単純な定義)は既にほぼチャンスレベル(static .489/NeuralODE .495/PNODE .516)
だったが、これは(1) 発明者レベルで企業レベルではない、(2) 在籍実績(established tenure)を
要求しない、(3) H年ラグの定義がない、という3点で本来の問いを検証していなかった。

## 2. Gate 0 設計(2026-08-21、Codex MCP設計)

- **在籍(established)ペア** (u,c): u が maingroup c に [Y-K+1, Y] の**全年**で出願している(K=3)。
- **EXIT=1** ⟺ u が (Y, Y+H] の**どの年にも** c で出願しない(H∈{2,3})。
- **domain_silent フラグ**: u がこのドメインのCPCスコープ内で (Y, Y+H] に一切出願がない
  (競合リスクとして分離報告。**真の企業消滅ではない**——元データが既にドメインのCPCで
  フィルタ済みのため、他分野では活動している可能性がある)。
- **訓練不要ベースライン**(すべて Y 以前のデータのみで計算):
  - `streak_length` / `recent_activity`(直近の在籍年数)——**独立ではない**
    (Spearman ρ≈0.92〜0.93、実質1つの「関係の持続性(persistence)」シグナル)
  - `firm_trend`(企業全体のポートフォリオ規模の推移)
  - `global_momentum`(その技術分野自体の全体的な縮小・拡大)
- **事前登録した停止規則**: どのベースラインもAUC≥0.60またはPR-lift≥1.25倍を
  cutoff横断で一貫してクリアしなければ、投資せず「予測不可能」として記録して終了。
- **評価**: 企業(エンティティ)クラスタ頑健ブートストラップ信頼区間(1エンティティが
  複数の相関したペアを持つため、ペア単位ブートストラップは楽観的すぎる)。

## 3. 結果

### construction(firm×CPC、K=3)

**persistence_family(streak/recent_activity)が全12セル(H∈{2,3}×cutoff∈{2015,16,17}×
domain_silent込み/除外)でAUC≥0.60をクリア。** 本プロジェクトの新規ターゲット変数の中で
初めて、事前登録した停止規則を一貫して通過した。

- AUC 0.62〜0.72(企業クラスタ頑健95%信頼区間は一貫して0.5を含まない)
- PR-lift 1.4〜1.7倍
- 実測撤退率による経験的リスク層別(H3/Y2017の例、streakは連続在籍年数):
  streak=3 → 24.7%、streak=4 → 17.9%、streak=5 → 6.8%、streak=6 → 6.7%
  (単調ではあるが完全な線形ではない。特にdomain_silent除外・希薄なビンでは小さな逆転あり——
  「較正済み確率」ではなく「経験的リスク層別」と呼ぶべき、Codex MCP指摘)
- `global_momentum_inv`(分野縮小)も複数セルでAUC 0.59〜0.60、PR-lift最大2.26倍。
- `firm_trend_inv`(企業全体の縮小)は弱く不安定(0.48〜0.62)。

### agrifood(inventor×CPC、K=3、交差ドメイン検証)

同じ質的パターンが**別ドメイン・別ノード種別(発明者)**で再現。persistence_familyは
大半のセルでゲートをクリア(AUC 0.56〜0.64、constructionよりやや弱い)。
`global_momentum_inv`はagrifoodの方がむしろ強く一貫(AUC最大0.66、PR-lift最大2.2倍)。
経験的リスク層別も同じ単調な形状(streak=3→30〜35%、streak=6→8〜15%)。

**解釈上の注意**(Codex MCP指摘): agrifoodは「発明者×技術」の持続性という一般的主張を
補強するものであり、firm-levelの2つ目の再現ではない。企業ポートフォリオ管理ツールとしての
主張の直接的根拠はconstruction(firm×CPC)に限定される。

## 4. 結論と位置づけ

> 確立された行為主体×技術の関係は、単純な在籍期間依存性と分野レベルの縮小によって
> 予測可能な撤退リスクを示す。ニューラルな時間発展は不要である。

これは本プロジェクトの他の"honest tool"路線(X3-clean記述的勾配流、TAP-NODEの可視化としての
価値、conformal較正の転移性)と同型の着地——**訓練不要のスコアそのものが成果物**であり、
Dual-Force/TAP-NODE型のニューラルモデル投資は正当化されない。CPC-CPC収束予測(AUC 0.83〜0.89、
`project_convergence_signal`)ほど科学的に強い信号ではないが、予測単位(「この企業とこの技術の
関係が撤退リスクにある」)が直接解釈可能なため、実務的には同等以上に有用な可能性がある。

## 5. 次のタスク(未着手、優先順)

1. **経験的リスク層別を正式な較正評価に格上げ**: 現状は同一cutoffサンプルでのビン集計。
   out-of-time(訓練cutoffと評価cutoffを分離)での確率較正・Brier score・信頼性図を追加する。
2. **もう1つのfirm-level交差検証**: agrifoodはinventor-levelのため、construction以外の
   firm×CPCデータ(現状`_firm.csv`はconstructionのみ)を用意できれば真の2つ目のfirm-level再現になる。
3. **KG-ATLAS(`kumagai-patent-analysis`)への接続検討**: 企業ポートフォリオ管理向けの
   「撤退リスクスコア」機能として、既存のexecutive dashboard(`plot_executive_dashboard.py`の
   「撤退検討 BOTTOM-10」)に接続できないか検討する——ただしこれは静的なランキング表示であり、
   本書のリスクスコアとは別物。接続する場合は較正済み確率(タスク1)が前提。
4. `firm_trend`ベースラインの弱さの原因(企業全体の活動指標がCPCレベルの撤退と結びつきにくい)
   を掘り下げるか、優先度を下げて次段に進むかを判断する。

## 6. レビューについて

本タスクは前段のGate-0設計・結果検証・手法修正(企業クラスタブートストラップ、
persistence_familyの重複排除、domain_silentの命名修正、較正表の追加)を**Codex MCPとの
反復的なやり取り**のみで完結させた(スコア 38→74→84)。Context7/Serena/kiriによる並行検証は
実施していない——理由: 本タスクは新規の外部文献接続ではなく既存インフラ(`techtrend_common.py`)
の直接的な再利用・拡張であり、事前の文献調査(`RELATED_WORK_SURVEY.md`)は既に候補A/B/C全体を
カバー済みだったため、Codexとの実データ検証の反復サイクルの方が投資対効果が高いと判断した。
将来この方向に本格投資する場合は、4系統フルレビューを別途実施することを推奨する。
