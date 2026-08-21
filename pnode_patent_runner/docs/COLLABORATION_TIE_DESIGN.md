# 新規共同出願関係(共同発明者ネットワーク形成)予測 — 設計仕様

**Status**: 実現可能性調査(Codex MCP, 2026-08-21)完了・**Gate 0は未実行**。データ抽出パイプライン新設が必要なため、本書は着手前の設計文書(`DUAL_FORCE_REDESIGN.md`と同型)。実データ検証済みだがまだ「結果」ではない——`EXIT_HAZARD_DESIGN.md`との違いに注意。

---

## 1. 背景

本セッションでfirm×CPC(参入・成長率・タイミング)・CPC×CPC(新規結合)・firm/inventor×CPCの関係
持続性(EXIT、`EXIT_HAZARD_DESIGN.md`)まで検証した。いずれも既存の二部グラフ(行為主体×技術)の
上での予測課題だった。本候補は**グラフ対象そのものを変える**——firm×firmまたはinventor×inventor
の**共同出願関係の新規形成**(これまで一度も共著関係がなかった2者が初めて同じ特許に名を連ねる)
を予測する。CPC-CPC収束予測(AUC 0.83〜0.89、`project_convergence_signal`)と表面上似ているが、
Codex MCPの指摘通り**同一シグナルの言い換えではない**——同じCPCに複数の発明者がいても
共同出願するとは限らないため、収束予測の高いAUCがそのまま転移する保証はない。

## 2. 実現可能性調査で確認された事実(Codex MCP、実データ検証)

**現状の処理済みCSV(`data/processed/bipartite_*.csv`)には共著関係を検出する情報がない**:
出願単位の識別子(patent_id)が破棄され `ts,u,i` の3列だけが残る
(`extract_domain_bipartite.py:91`、`build_firm_bipartite.py:74`)。

**ただしローカルにあるPatentsView生テーブルには残っている**(`g_inventor_disambiguated.tsv`、
`g_assignee_disambiguated.tsv`、`g_application.tsv`、`g_patent.tsv`、`g_cpc_current.tsv`)。
これらをpatent_idで結合し、construction関連CPC(E02/E03/E04/E21)・2000〜2021年で集計した実測:

| 粒度 | 対象特許数 | 複数主体の特許 | 割合 | ユニークtie数 | 新規tie数/年(2017〜2021) |
|---|---:|---:|---:|---:|---|
| **発明者(inventor)** | 118,895 | 66,880 | **56.25%** | 197,244 | 14,535 / 14,936 / 19,730 / 20,419 / 19,261 |
| **企業(firm/organization assignee)** | 105,650 | 2,148 | 2.03% | 1,650 | 121 / 147 / 134 / 145 / 114 |

**結論**: 発明者レベルは主実験に十分な密度がある。企業レベルは希薄すぎて主実験には不向き
(補助的な確認には使える可能性はある)。

## 3. 先行研究の有無

このリポジトリ内(`WHY_NEURALODE_FAILS_ja.md`、`PAPER_WORKFLOW.md`、`RESEARCH_PLAN_PREDICTABILITY.md`
の4タスク族マップ、`RELATED_WORK_SURVEY.md`)を含む全体検索で、共同発明者・共同出願人ネットワークの
**形成予測**に関する先行実装は見つからなかった。近い既存コードは2つのみ、いずれも別物:
`diagnose_convergence_signal.py`の「同一発明者によるCPCペアの共起」(発明者は固定でCPCペアを見る、
本候補の逆)と、`interactive_landscape.py`の共同出願人メタデータ表示(予測ではなく表示のみ)。

## 4. Gate 0 設計(実行後に本書を更新する、2026-08-21 Codex MCP事前登録)

### 4.1 データ抽出(新規実装が必要)

生テーブルからinventor×patentの多重度を保持したまま、年次のinventor-inventor tie形成イベントを
再構築する新しいスクリプトが必要(既存の`techtrend_common.py`は転用不可、patent_idを扱う抽出層
そのものを新設する)。

### 4.2 ラベルと評価対象

- 予測対象年 Y 以前にアクティブな発明者ペアのうち、**過去に一度も共著関係がないペア**を評価対象とする。
- ラベル=1: Y+1年(または[Y+1,Y+3])に初めて共著。

### 4.3 訓練不要ベースライン(ニューラルモデルの前に必須)

- 次数・人気度の積(preferential attachment)
- 共有CPCのTF-IDFコサイン類似度 / Jaccard
- 共著者ネットワーク上のAdamic-Adar(共通の共著者)
- 共有assignee(同じ企業に所属したことがあるか)の履歴
- 直近の活動量・recency

### 4.4 事前登録した「興味深い」の基準(Codex MCP, 2026-08-21)

CPC-CPC収束予測(0.83〜0.89)の焼き直しで終わらせないため、以下を事前登録する:

1. **easy set**(過去未リンクの全アクティブペア)は診断用のみ、主張の根拠にしない。
2. **hard set**: 事前活動量・次数・CPC類似度(・可能ならassignee露出)でマッチさせた負例。
3. **構造天井**: preferential attachment・shared-CPC類似度・共著Adamic-Adar、およびその組合せ。
4. **非自明サブセット**: 共著Adamic-Adar=0のペア(真に新規な結合)+ 異なるassignee間のペアに限定した評価。
5. **判定基準**: hard setでPR-lift≥1.5倍**かつ**(ΔAUC≥0.03または有意なrecall@k改善)を、
   複数年・可能なら第2ドメインで再現して初めて「興味深い」と認める。
   AA/shared-CPCだけで高性能に達し、AA=0のサブセットで性能が消失する場合は、
   CPC-CPC収束予測と同じ「predictable-but-trivial」の別バリエーションとして記録する。

## 5. 次のタスク(未着手、優先順)

1. **ユーザーの着手判断待ち**: 生PatentsViewテーブルからのデータ抽出パイプラインは新規開発
   (既存インフラの転用ではない)であり、着手前に規模感を確認する。
2. 着手する場合: 4.1のデータ抽出スクリプトを実装し、4.3の訓練不要天井を先に測定する
   (この時点でニューラルモデルは一切実装しない、本プロジェクトの規律)。
3. 天井が4.4の基準を満たさない場合は「9件目のsimple-beats-complex」相当として記録して終了。
   満たす場合のみ、学習モデルへの投資を検討する。

## 6. レビューについて

`EXIT_HAZARD_DESIGN.md`と同様、本タスクもCodex MCPとの反復(実現可能性調査80/100 →
Gate 0基準の事前登録の明確化)のみで完結させた。Context7/Serena/kiriの並行実行は未実施——
実データ検証(生PatentsViewテーブルへの実際のクエリ)が必要な段階だったため、Codexの
read-onlyサンドボックスでの直接検証を優先した。データ抽出パイプラインの実装に進む場合は、
実装レビューとして4系統フルレビューを行うことを推奨する。
