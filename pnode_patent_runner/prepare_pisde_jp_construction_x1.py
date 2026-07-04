"""
日本語建築・土木論文 (J-STAGE) を PI-SDE + X1 用に準備。

入力:  data/raw_jstage_construction.jsonl
出力:  data/PNode_JP_Construction_X1/alltime/fate_train.pt
       data/PNode_JP_Construction_X1/leaveout{1..10}/fate_train.pt

処理:
  1. JP/EN タイトル + ジャーナル名で BERT embedding (sonoisa or e5)
  2. PCA → 50D
  3. k-means で K=40 topics に分割
  4. 年ごとの centroid + growth 計算
"""
from __future__ import annotations

import json, sys, os
from pathlib import Path
from collections import Counter

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

RAW   = Path("data/raw_jstage_construction.jsonl")
OUT_DIR = Path("data/PNode_JP_Construction_X1/alltime")
OUT_PT  = OUT_DIR / "fate_train.pt"

YEARS = list(range(2015, 2026))
PCA_DIM = 50
N_TOPICS = 40
EMB_MODEL = "sonoisa/sentence-bert-base-ja-mean-tokens"  # JP-specific
BATCH = 64


def load_records():
    recs = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("year"):
                continue
            if r["year"] < YEARS[0] or r["year"] > YEARS[-1]:
                continue
            # 必要: JP タイトル (なければ EN)
            t = (r.get("title_ja") or "").strip() or (r.get("title_en") or "").strip()
            if not t:
                continue
            recs.append({
                "title": t,
                "journal": r.get("journal_ja") or r.get("journal_en") or "",
                "year": r["year"],
                "doi": r.get("doi", ""),
                "cdjournal": r.get("cdjournal", ""),
            })
    return recs


def embed_texts(texts, model_name=EMB_MODEL, batch=BATCH):
    """Sentence-BERT で texts を embed → (N, 768)"""
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    print(f"Embedding {len(texts)} texts (batch={batch})...")
    emb = model.encode(texts, batch_size=batch, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def label_clusters(texts, cluster_ids, k):
    """各クラスタの代表ラベルを (頻出文字列ベース) 抽出"""
    import re
    labels = []
    for c in range(k):
        idx = np.where(cluster_ids == c)[0]
        if len(idx) == 0:
            labels.append(f"topic_{c:02d}")
            continue
        # 各タイトルから 2-3 文字の語を全部数える (CJK + 英字)
        all_tokens = []
        for i in idx[: min(200, len(idx))]:
            t = texts[i]
            t = re.sub(r"[「」『』、。・,\.!\?\(\)\[\]【】]", " ", t)
            # 2-4 文字単位の漢字シーケンス + 英単語
            tokens = re.findall(r"[一-龥]{2,4}|[A-Za-z]{3,}", t)
            all_tokens.extend(tokens)
        ctr = Counter(all_tokens)
        # 共通すぎる助詞/接尾辞を除外
        stops = {"研究", "について", "に関する", "に関して", "場合", "結果",
                 "考察", "Study", "Research", "Analysis", "study", "research",
                 "based", "Based"}
        top = [(w, n) for w, n in ctr.most_common(20) if w not in stops][:3]
        if not top:
            labels.append(f"topic_{c:02d}")
        else:
            label = "+".join(w for w, n in top)
            labels.append(f"{c:02d}:{label}")
    return labels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW}...")
    recs = load_records()
    print(f"  records: {len(recs)}")
    if len(recs) == 0:
        raise RuntimeError("no records found")

    # year distribution
    yc = Counter(r["year"] for r in recs)
    print("  year:", {y: yc.get(y, 0) for y in YEARS})

    # embedding 入力: "title  (journal_name)"  - ジャーナル名を含めて文脈強化
    embed_texts_in = [
        f"{r['title']}  ({r['journal']})" if r["journal"] else r["title"]
        for r in recs
    ]

    cache_p = OUT_DIR / "embeds.npy"
    if cache_p.exists():
        print(f"Loading cached embeds <- {cache_p}")
        embeds = np.load(cache_p)
    else:
        embeds = embed_texts(embed_texts_in)
        np.save(cache_p, embeds)
        print(f"  cached -> {cache_p}")
    print(f"  emb shape: {embeds.shape}")

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X = pca.fit_transform(embeds).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    print(f"  PCA → {X.shape}, explained var = {pca.explained_variance_ratio_.sum():.3f}")

    # k-means clustering on full embedding (not PCA) for better semantics
    from sklearn.cluster import KMeans
    print(f"  KMeans K={N_TOPICS}...")
    km = KMeans(n_clusters=N_TOPICS, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(embeds)
    n_topics = N_TOPICS
    topic_names = label_clusters([r["title"] for r in recs], cluster_ids, n_topics)
    print(f"  topics: {n_topics}")

    # 年ごとに分割
    xp, topics_per_year, centroids_per_year, growth_per_year = [], [], [], []
    counts_per_year = {}
    y_list = []
    for k, yr in enumerate(YEARS):
        mask = np.array([r["year"] == yr for r in recs])
        X_yr = X[mask]
        topics_yr = cluster_ids[mask]

        xp.append(torch.tensor(X_yr, dtype=torch.float32))
        topics_per_year.append(torch.tensor(topics_yr, dtype=torch.long))
        y_list.append(float(k))

        cents = torch.zeros(n_topics, PCA_DIM, dtype=torch.float32)
        for j in range(n_topics):
            sub = X_yr[topics_yr == j]
            if len(sub) > 0:
                cents[j] = torch.tensor(sub.mean(axis=0), dtype=torch.float32)
        centroids_per_year.append(cents)

        for j in range(n_topics):
            counts_per_year[(yr, j)] = int((topics_yr == j).sum())

        active = int((cents.abs().sum(-1) > 0).sum())
        print(f"  t={k:>2}  year={yr}  N={int(mask.sum()):>5}  active topics={active}/{n_topics}")

    # growth rate
    for k, yr in enumerate(YEARS):
        g_yr = torch.zeros(n_topics, dtype=torch.float32)
        if k == 0:
            growth_per_year.append(g_yr); continue
        prev_yr = YEARS[k - 1]
        for j in range(n_topics):
            c_now = float(counts_per_year[(yr, j)])
            c_prev = float(counts_per_year[(prev_yr, j)])
            g_yr[j] = (c_now - c_prev) / (c_prev + 1.0)
        growth_per_year.append(g_yr)

    # normalize
    all_g = torch.stack([g for g in growth_per_year if g.abs().sum() > 0])
    g_mean = all_g.mean().item()
    g_std  = all_g.std().item() + 1e-8
    growth_per_year_norm = [(g - g_mean) / g_std for g in growth_per_year]

    data_pt = {
        "xp":          xp,
        "y":           y_list,
        "topics":      topics_per_year,
        "topic_names": topic_names,
        "centroids":   centroids_per_year,
        "growth":      growth_per_year,
        "growth_norm": growth_per_year_norm,
        "n_topics":    n_topics,
    }
    torch.save(data_pt, OUT_PT)
    print(f"\n✅ Saved -> {OUT_PT}")

    # leaveout 1..last (=10) for last year prediction
    last_t = len(YEARS) - 1
    for lo in [last_t]:
        out_lo = Path(f"data/PNode_JP_Construction_X1/leaveout{lo}/fate_train.pt")
        out_lo.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_pt, out_lo)
        print(f"  Saved -> {out_lo}")

    # 簡易レポート
    print(f"\n[Top growing topics at t={last_t} ({YEARS[last_t]})]")
    g_last = growth_per_year[last_t].numpy()
    order = np.argsort(-g_last)
    for i in order[:8]:
        print(f"  {topic_names[i]:<40}  g={g_last[i]:+.3f}  count={counts_per_year[(YEARS[last_t], i)]:>4}")
    print(f"\n[Top declining topics at t={last_t}]")
    for i in order[-5:][::-1]:
        print(f"  {topic_names[i]:<40}  g={g_last[i]:+.3f}  count={counts_per_year[(YEARS[last_t], i)]:>4}")


if __name__ == "__main__":
    main()
