# プロジェクトマップ（What's Been Tried）

新しい予測ターゲット・アーキテクチャを提案する前に、このファイルで既存の到達点を確認すること。詳細な数値・実験設計は各リンク先の`pnode_patent_runner/docs/*.md`を参照。

## タスク族ごとの到達点

| タスク族 | 状態 | 主要な証拠 | 主要文書 |
| --- | --- | --- | --- |
| **WHERE**（firm-CPC参入予測） | 閉じている（simple-beats-complex） | Task B "+126%"はtransductiveアーティファクトと判明・撤回。inductive評価では関連性ベースラインに-45%〜-66% | `WHY_NEURALODE_FAILS_ja.md` |
| **HOW-MUCH**（成長率予測） | 閉じている | X5/DRIFT全滅。momentum/persistenceが圧勝 | `docs/RESEARCH_PLAN_PREDICTABILITY.md` |
| **WHEN**（参入タイミング、Neural Jump-ODE点過程含む） | 閉じている（8th simple-beats-complex） | Arch A/B/C（`techtrend_arch_{a,b,c}.py`）全滅。点過程という構造的に正しい枠組みでも失敗 | `WHY_NEURALODE_FAILS_ja.md` |
| **WHAT-NEW**（CPC-CPC新規結合/convergence） | predictable-but-trivial | AUC 0.83〜0.89と予測可能だが、Adamic-Adarを学習モデルが一切上回れない。真に新規な"jump"はchanceレベル | `diagnose_convergence_signal.py`, `diagnose_novelty_hazard.py` |
| **EXIT**（技術関係の撤退・持続性） | **唯一Gate 0を通過した正の発見** | streak/recent_activityというtraining-freeシグナルがAUC 0.60〜0.72、2ドメインで再現 | `docs/EXIT_HAZARD_DESIGN.md` |
| **Collaboration tie**（共同出願関係の新規形成） | 実現可能性確認済み・Gate 0未実行 | inventor-levelは十分密（年間15〜20K件の新規tie）。firm-levelは希薄すぎる | `docs/COLLABORATION_TIE_DESIGN.md` |
| **Dual-Force / TAP-NODE**（アテンション機構） | rank_renormの優位性はshared encoder統制で消失 | GEMには有意に勝つが、static/RNN/NeuralODE/PNODEに対しては同じエンコーダ下で有意差なし | `docs/DUAL_FORCE_REDESIGN.md` |
| **重複出願検出の研究化**（測定・データ品質） | 姉妹リポジトリで提案済み・未実行 | 6,864/44,564件(15.5%)がフラグ対象。人手ラベリング未着手 | `kumagai-patent-analysis` issue #11 |
| **クロスソース埋め込み統合の評価** | 姉妹リポジトリで提案済み・未実行 | P0修正は本番反映済み。Recall@k評価が欠けている | `kumagai-patent-analysis` issue #12 |

## 既に却下された近縁アイデア（再提案する前に読むこと）

- **発明者・出願人の「人流ベクトル」可視化での技術トレンド予測**: 4系統レビューで22〜38点。継続的な移動が予測シグナルを持つという、既に反証済みの枠組みの焼き直し（発明者データの疎さも追加の障害）
- **CPC×CPCの新規結合創発予測**: 「WHAT-NEW」として既に検証済み（上表参照）。文献的には新しく見えても中身は既存

## 姉妹リポジトリとの接続

`../kumagai-patent-analysis`（KG-ATLAS）は実運用ツール。研究の知見がツールの予測系機能の妥当性検証に直結する。姉妹リポジトリ側の4系統レビュー実績（issue #2, #5, #10, #11, #12, #13）も参照。特にissue #13（`/lead-lag`の統計的未検証な出力）は、研究とは独立に早期対応が推奨される運用上の懸念として記録されている。
