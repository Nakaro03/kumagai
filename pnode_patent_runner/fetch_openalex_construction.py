"""
OpenAlex API から建築/土木論文を取得し、X1 形式で保存。

検索条件:
  - Concept: Civil engineering (C147176958), Structural engineering (C66938386), 等
  - 年: 2020-2025
  - 言語: 英語のみ
  - abstract あり

出力構造 (X1 形式):
  data/PNode_Construction_X1/alltime/fate_train.pt
    {xp, y, topics, topic_names, centroids, growth, growth_norm, n_topics}

トピック分類: 各論文の primary subfield (sub-concept) を topic として使用
"""
from __future__ import annotations

import os
import sys
import time
import json
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import requests

_REPO = Path(__file__).resolve().parents[1]
os.chdir(_REPO)

# 建築・土木関連の concept
CONCEPT_IDS = [
    "C147176958",  # Civil engineering
    "C66938386",   # Structural engineering
    "C187320778",  # Geotechnical engineering
    "C144024400",  # Architecture
    "C66938386",   # Structural engineering (重複可)
    "C2778572881", # Construction
    "C160487506",  # Highway engineering
    "C16160715",   # Traffic engineering
    "C93907247",   # Geotechnics
]

# 建築関連キーワード (より広く)
CONSTRUCTION_KEYWORDS = [
    # 一般構造
    "construction", "structural", "concrete", "reinforced", "bridge",
    "building", "geotechnical", "civil engineering", "infrastructure",
    "earthquake", "seismic", "foundation", "steel", "masonry",
    "highway", "pavement", "tunnel", "soil", "bim",
    # 材料
    "rebar", "composite material", "fiber reinforced", "polymer concrete",
    "geosynthetic", "geopolymer", "ultra-high performance",
    # 構造解析
    "finite element", "fem", "vibration", "modal analysis",
    "shear strength", "compressive strength", "bearing capacity",
    # 建設手法
    "prefabricat", "modular construction", "3d printing", "additive manufacturing",
    "sustainable construction", "green building",
    # その他
    "retrofit", "rehabilitat", "monitoring", "smart building",
    "earthquake resistant", "wind loading",
]

# 検索クエリ (search パラメータ用)
SEARCH_QUERIES = [
    "civil engineering",
    "structural engineering",
    "construction technology",
    "geotechnical engineering",
    "earthquake engineering",
]

OUT_DIR = Path("data/PNode_Construction_X1_v3/alltime")
YEARS = list(range(2015, 2026))          # 2015-2025 (11 年)
MAX_PER_YEAR = 4000                       # 各年の最大論文数 (rate limit 余裕あり)
MAX_PAGES_PER_QUERY = 30                  # 多めにページング
PCA_DIM = 50
TOP_K_TOPICS = 40
PER_PAGE = 100                            # API 1 リクエストあたり (rate limit 配慮で小さく)


def fetch_works_concept(concept_ids: List[str], year: int, per_page=200, max_pages=15):
    """concept_ids フィルタで論文取得"""
    concept_filter = "|".join(concept_ids)
    return _fetch_paged(
        filter_str=f"concepts.id:{concept_filter},publication_year:{year}",
        per_page=per_page, max_pages=max_pages, search=None,
    )


def fetch_works_search(query: str, year: int, per_page=200, max_pages=8):
    """search パラメータで title/abstract 検索"""
    return _fetch_paged(
        filter_str=f"publication_year:{year}",
        per_page=per_page, max_pages=max_pages, search=query,
    )


def _fetch_paged(filter_str: str, per_page=200, max_pages=10, search=None):
    base_url = "https://api.openalex.org/works"
    results = []
    cursor = "*"
    page = 0
    while page < max_pages:
        params = {
            "filter": filter_str,
            "per-page": per_page, "cursor": cursor,
            "select": "id,title,abstract_inverted_index,publication_year,concepts",
        }
        if search:
            params["search"] = search
        url = base_url + "?" + urllib.parse.urlencode(params)
        try:
            r = requests.get(url, timeout=30)
            d = r.json()
        except Exception as e:
            print(f"    request failed: {e}")
            break
        works = d.get("results", [])
        if not works: break
        results.extend(works)
        cursor = d.get("meta", {}).get("next_cursor")
        if not cursor: break
        page += 1
        time.sleep(0.15)
    return results


def reconstruct_abstract(inv_idx: Optional[Dict]) -> str:
    """OpenAlex の abstract_inverted_index を文字列に復元"""
    if not inv_idx:
        return ""
    word_positions = []
    for word, positions in inv_idx.items():
        for p in positions:
            word_positions.append((p, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def primary_subfield(concepts: List[Dict]) -> Optional[str]:
    """論文の primary concept (level=2 or 3 の最高 score) を返す"""
    if not concepts:
        return None
    # level=2 or 3 で建築関連の subfield を優先
    candidates = [c for c in concepts if c.get("level") in (2, 3)]
    if not candidates:
        candidates = concepts
    top = max(candidates, key=lambda c: c.get("score", 0))
    return top["display_name"]


def is_construction_paper(title: str, abstract: str, concepts: List[Dict]) -> bool:
    """v1 と同じ strict 判定: キーワード AND concept score > 0.3"""
    text = (title + " " + abstract).lower()
    has_keyword = any(kw in text for kw in CONSTRUCTION_KEYWORDS)
    has_concept = False
    for c in concepts:
        cid = c.get("id", "").split("/")[-1]
        if cid in CONCEPT_IDS and c.get("score", 0) > 0.3:    # v1 と同じ閾値
            has_concept = True
            break
    return has_keyword and has_concept


def main():
    print("=" * 60)
    print("  Construction Papers via OpenAlex API")
    print("=" * 60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_papers = []
    seen_ids = set()
    for yr in YEARS:
        print(f"\n[year {yr}] fetching...")
        # 戦略 1: concept フィルタ
        works_c = fetch_works_concept(CONCEPT_IDS, yr, max_pages=MAX_PAGES_PER_QUERY)
        # 戦略 2: search query (複数クエリの結果を統合)
        works_s = []
        for q in SEARCH_QUERIES:
            works_s.extend(fetch_works_search(q, yr, max_pages=5))
        # 重複排除
        works_all = works_c + works_s
        unique = []
        for w in works_all:
            if w["id"] not in seen_ids:
                seen_ids.add(w["id"])
                unique.append(w)
        print(f"  concept={len(works_c)}, search={len(works_s)}, unique={len(unique)}")
        if len(unique) > MAX_PER_YEAR:
            unique = unique[:MAX_PER_YEAR]
        works = unique
        for w in works:
            abs_text = reconstruct_abstract(w.get("abstract_inverted_index"))
            if not abs_text or len(abs_text) < 50:
                continue
            title = w.get("title", "") or ""
            concepts = w.get("concepts", [])
            # 厳格フィルタ: 建築関連キーワード + concept スコア
            if not is_construction_paper(title, abs_text, concepts):
                continue
            sub = primary_subfield(concepts)
            all_papers.append({
                "id": w["id"], "title": title,
                "abstract": abs_text, "year": yr, "topic": sub,
            })
    print(f"\nTotal papers with abstracts: {len(all_papers)}")
    if len(all_papers) < 100:
        print("⚠️  Not enough papers. Check API limits.")
        return

    df = pd.DataFrame(all_papers)
    print(f"  unique topics: {df['topic'].nunique()}")
    print(f"  top 20 topics:\n{df['topic'].value_counts().head(20)}")

    # v3: strict AND filter で誤検出は最小化済み → 白リストは緩く
    # top-K でフィルタするだけにする
    top_topics = df["topic"].value_counts().head(TOP_K_TOPICS).index.tolist()
    df = df[df["topic"].isin(top_topics)].reset_index(drop=True)
    print(f"  After top-{TOP_K_TOPICS} filter: {len(df)} papers")
    topic_names = sorted(df["topic"].dropna().unique())
    topic_to_id = {t: i for i, t in enumerate(topic_names)}
    n_topics = len(topic_names)
    print(f"  After top-{TOP_K_TOPICS} filter: {len(df)} papers, {n_topics} topics")

    # 埋め込み (sentence-transformers MiniLM 384D)
    print("\nGenerating embeddings (sentence-transformers MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (df["title"].fillna("") + ". " + df["abstract"]).tolist()
    embeds = model.encode(texts, batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True)
    print(f"  embeddings shape: {embeds.shape}")

    # PCA 50D
    print(f"\nPCA → {PCA_DIM}D...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X = pca.fit_transform(embeds).astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    print(f"  explained var: {pca.explained_variance_ratio_.sum():.4f}")

    # X1 形式に整形
    xp, topics_per_year, y_list, centroids_per_year = [], [], [], []
    counts_per_year = {}
    for k, yr in enumerate(YEARS):
        mask = df["year"].values == yr
        if mask.sum() == 0: continue
        X_yr = X[mask]
        topics_yr = df.loc[mask, "topic"].map(topic_to_id).values

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
        print(f"  year {yr} (t={k}): {mask.sum()} papers, {int((cents.abs().sum(-1) > 0).sum())} active topics")

    # 成長率
    growth_per_year = []
    for k, yr in enumerate(YEARS):
        g = torch.zeros(n_topics, dtype=torch.float32)
        if k > 0:
            prev_yr = YEARS[k - 1]
            for j in range(n_topics):
                c_now = float(counts_per_year[(yr, j)])
                c_prev = float(counts_per_year[(prev_yr, j)])
                g[j] = (c_now - c_prev) / (c_prev + 1.0)
        growth_per_year.append(g)

    all_g = torch.stack([g for g in growth_per_year if g.abs().sum() > 0])
    g_mean, g_std = all_g.mean().item(), all_g.std().item() + 1e-8
    growth_norm = [(g - g_mean) / g_std for g in growth_per_year]

    data_pt = {
        "xp": xp, "y": y_list, "topics": topics_per_year,
        "topic_names": topic_names,
        "centroids": centroids_per_year,
        "growth": growth_per_year, "growth_norm": growth_norm,
        "n_topics": n_topics,
    }
    out_pt = OUT_DIR / "fate_train.pt"
    torch.save(data_pt, out_pt)
    print(f"\n✅ Saved -> {out_pt}")

    # 統計表示
    print(f"\n[統計]")
    print(f"  topics: {n_topics}")
    g_last = growth_per_year[-1].numpy()
    order = np.argsort(-g_last)
    print(f"  最新年 (year {YEARS[-1]}) 成長率上位:")
    for i in order[:7]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {topic_names[i][:50]:<50} g={g_last[i]:+.3f}")
    print(f"  下位:")
    for i in order[-5:][::-1]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {topic_names[i][:50]:<50} g={g_last[i]:+.3f}")


if __name__ == "__main__":
    main()
