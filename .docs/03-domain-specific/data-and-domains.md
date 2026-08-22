# データとドメイン（Data & Domains）

## 利用可能なドメイン

`data/processed/bipartite_{domain}.csv`（列: `ts, u, i` — 出願日, 主体, CPC/IPCコード）:

- `construction`, `agrifood`, `energy`, `semiconductor`, `pharma`, `computing`（特許、`u`=inventor）
- `bipartite_construction_firm.csv`のみfirm-level（`u`=applicant）。**他ドメインのfirm-levelデータは現状存在しない**（新しいfirm-level交差検証をしたい場合はデータ抽出が必要）
- `author_topic`ドメイン（論文、著者×トピック）

## CPC/IPC粒度

- `maingroup`（例: `A01M31`）が主に使われる粒度。`subclass`は飽和しやすく比較にほぼ使えない、`subgroup`は逆に粒度が細かすぎて評価ペアが希薄になる（`predictability_ceilings.py`のgranularityミスマッチが過去に発覚した実績あり——真の人気度天井は0.61で0.93ではなかった）
- IPC/FI/Fタームデータは現行のPatentsViewスナップショットには含まれない（`kumagai-patent-analysis`側の生データには存在するため、必要ならそちらを参照）

## 既知のデータ品質上の注意点

- **公開ラグによる右打ち切り**: 直近1〜2年のデータは系統的に過小（出願から公開まで約18ヶ月）。評価ウィンドウはデータ境界から2年以上離すか、既知の打ち切り率で補正する。年範囲は`--y1`を2021年以前に設定するのが安全（`diagnose_exit_hazard.py`のデフォルト参照）
- **将来情報の漏洩**: `build_sets(df, Y, H)`のようなヘルパーは常に`year <= Y`のデータのみからprior/特徴量を構築する規約。新しい特徴量を追加する際は必ずこの規約を守る
- **評価の負例サンプリング**: `future_link_auc_scores()`系の評価関数は、真の正例数がサブサンプル上限（`max_pos=1500`）を超えるドメイン・年で誤った負例を混入させるバグが過去にあった（修正済み、`test_eval_negative_sampling.py`参照）。新しい評価関数を書く際は同じ落とし穴（負例棄却集合をサブサンプル前の全正例から構築すること）に注意する

## 生データへのアクセス

処理済みCSVは`patent_id`等の識別子を保持していないため、特許間の引用・共同出願関係を再構築する必要がある場合は、ローカルのraw PatentsViewテーブル（`g_inventor_disambiguated.tsv`, `g_assignee_disambiguated.tsv`, `g_application.tsv`, `g_patent.tsv`, `g_cpc_current.tsv`）に直接アクセスする必要がある（`docs/COLLABORATION_TIE_DESIGN.md`の実現可能性調査を参照）。
