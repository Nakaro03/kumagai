"""
arXiv API から建築・土木隣接論文を取得し、X1 形式で保存。

arXiv API:
  - エンドポイント: http://export.arxiv.org/api/query
  - 無料、レート制限緩い (~3 秒間隔推奨)
  - max_results 上限 2000/query

検索戦略:
  cs.* / eess.* / physics.* / cond-mat.* の建築隣接カテゴリで複数キーワード検索

出力: data/PNode_ArXiv_Construction_X1/alltime/fate_train.pt
"""
from __future__ import annotations

import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import requests

_REPO = Path(__file__).resolve().parents[1]
os.chdir(_REPO)

# 建築隣接の arXiv カテゴリ
RELEVANT_CATEGORIES = [
    # 工学・物理
    "eess.SY",        # Systems and Control
    "eess.SP",        # Signal Processing (構造ヘルスモニタリング)
    "cs.RO",          # Robotics (建設ロボット)
    "cs.CY",          # Computers and Society (スマートシティ)
    "cs.CV",          # Computer Vision (画像解析 → 損傷検知)
    "cs.LG",          # ML (機械学習応用)
    "physics.app-ph", # Applied Physics
    "physics.geo-ph", # Geophysics (地盤)
    "cond-mat.mtrl-sci", # Materials Science
    "cond-mat.soft",  # Soft Materials (concrete)
    "math.OC",        # Optimization and Control
    "stat.AP",        # Applied Statistics
]

# 建築・土木関連キーワード (検索クエリ)
SEARCH_QUERIES = [
    "construction",
    "structural engineering",
    "concrete",
    "reinforced concrete",
    "earthquake engineering",
    "geotechnical",
    "civil engineering",
    "building information modeling",
    "structural health monitoring",
    "construction robotics",
    "smart construction",
    "construction automation",
    "infrastructure",
    "bridge engineering",
    "seismic design",
    "foundation engineering",
    "soil mechanics",
    "pavement engineering",
    "masonry",
    "prefabricated construction",
]

CONSTRUCTION_KEYWORDS = [
    "construction", "structural", "concrete", "reinforced",
    "bridge", "building", "geotechnical", "civil engineering",
    "infrastructure", "earthquake", "seismic", "foundation",
    "steel", "masonry", "highway", "pavement", "tunnel",
    "soil", "bim", "rebar", "composite material", "geopolymer",
    "finite element", "vibration", "shear strength",
    "compressive strength", "prefabricat", "retrofit",
]

OUT_DIR = Path("data/PNode_ArXiv_Construction_X1/alltime")
YEARS = list(range(2015, 2026))
PCA_DIM = 50
TOP_K_TOPICS = 40
MAX_PER_QUERY = 500     # 1 クエリあたり最大件数
SLEEP_BETWEEN = 3       # arXiv API は 3 秒間隔推奨


def fetch_arxiv_query(query: str, start: int = 0, max_results: int = 100,
                      year: Optional[int] = None) -> List[Dict]:
    """arXiv API で検索"""
    base = "http://export.arxiv.org/api/query"
    # search_query syntax: ti:keyword AND abs:keyword
    q = f'all:"{query}"'
    if year:
        q = f'({q}) AND submittedDate:[{year}01010000 TO {year}12312359]'
    params = {
        "search_query": q,
        "start": start,
        "max_results": min(max_results, 2000),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"    error: {e}")
        return []

    # ATOM feed をパース
    from xml.etree import ElementTree as ET
    NS = {"atom": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"    parse error: {e}")
        return []

    results = []
    for entry in root.findall("atom:entry", NS):
        title = entry.findtext("atom:title", "", NS).strip().replace("\n", " ")
        summary = entry.findtext("atom:summary", "", NS).strip().replace("\n", " ")
        published = entry.findtext("atom:published", "", NS)
        pub_year = int(published[:4]) if published else None
        # primary category
        primary = entry.find("arxiv:primary_category", NS)
        primary_cat = primary.get("term") if primary is not None else None
        # all categories
        cats = [c.get("term") for c in entry.findall("atom:category", NS)]
        results.append({
            "title": title,
            "abstract": summary,
            "year": pub_year,
            "primary_category": primary_cat,
            "categories": cats,
        })
    return results


def is_construction_relevant(paper: Dict) -> bool:
    """論文が建築関連かをキーワードで判定"""
    text = (paper["title"] + " " + paper["abstract"]).lower()
    has_keyword = any(kw in text for kw in CONSTRUCTION_KEYWORDS)
    # カテゴリ的にも建築隣接かチェック
    has_relevant_cat = any(c in RELEVANT_CATEGORIES for c in paper.get("categories", []))
    return has_keyword and has_relevant_cat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=" * 70)
    print(f"  arXiv API: 建築隣接論文取得")
    print(f"  Categories: {len(RELEVANT_CATEGORIES)} 個")
    print(f"  Queries:    {len(SEARCH_QUERIES)} 個")
    print(f"  Years:      {YEARS[0]}-{YEARS[-1]}")
    print(f"=" * 70)

    all_papers = []
    seen_titles = set()
    for q_idx, q in enumerate(SEARCH_QUERIES):
        print(f"\n[Query {q_idx+1}/{len(SEARCH_QUERIES)}] '{q}'")
        results = fetch_arxiv_query(q, start=0, max_results=MAX_PER_QUERY)
        n_new = 0
        for p in results:
            if not p["year"] or p["year"] not in YEARS:
                continue
            if not p["abstract"] or len(p["abstract"]) < 50:
                continue
            if p["title"] in seen_titles:
                continue
            if not is_construction_relevant(p):
                continue
            seen_titles.add(p["title"])
            all_papers.append(p)
            n_new += 1
        print(f"  retrieved: {len(results)}, new construction-relevant: {n_new}")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nTotal unique construction papers: {len(all_papers)}")
    if len(all_papers) < 50:
        print("⚠️  Not enough papers")
        return

    df = pd.DataFrame(all_papers)
    print(f"  unique primary categories: {df['primary_category'].nunique()}")
    print(f"  top 10 categories:\n{df['primary_category'].value_counts().head(10)}")

    # トピックとして primary_category を使用
    df["topic"] = df["primary_category"]
    top_topics = df["topic"].value_counts().head(TOP_K_TOPICS).index.tolist()
    df = df[df["topic"].isin(top_topics)].reset_index(drop=True)
    topic_names = sorted(df["topic"].dropna().unique())
    topic_to_id = {t: i for i, t in enumerate(topic_names)}
    n_topics = len(topic_names)
    print(f"  After top-{TOP_K_TOPICS} filter: {len(df)} papers, {n_topics} topics")

    # 埋め込み (sentence-transformers)
    print("\nGenerating embeddings...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (df["title"].fillna("") + ". " + df["abstract"]).tolist()
    embeds = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True)
    print(f"  embeddings shape: {embeds.shape}")

    # PCA
    print(f"\nPCA → {PCA_DIM}D...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(PCA_DIM, embeds.shape[0] - 1), random_state=42)
    X = pca.fit_transform(embeds).astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    actual_dim = pca.n_components_
    print(f"  explained var: {pca.explained_variance_ratio_.sum():.4f}, dim: {actual_dim}")

    # X1 形式
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
        cents = torch.zeros(n_topics, actual_dim, dtype=torch.float32)
        for j in range(n_topics):
            sub = X_yr[topics_yr == j]
            if len(sub) > 0:
                cents[j] = torch.tensor(sub.mean(axis=0), dtype=torch.float32)
        centroids_per_year.append(cents)
        for j in range(n_topics):
            counts_per_year[(yr, j)] = int((topics_yr == j).sum())
        print(f"  year {yr} (t={k}): {mask.sum()} papers, {int((cents.abs().sum(-1) > 0).sum())} active topics")

    growth_per_year = []
    for k, yr in enumerate(YEARS):
        g = torch.zeros(n_topics, dtype=torch.float32)
        if k > 0:
            prev_yr = YEARS[k - 1]
            for j in range(n_topics):
                c_now = float(counts_per_year.get((yr, j), 0))
                c_prev = float(counts_per_year.get((prev_yr, j), 0))
                g[j] = (c_now - c_prev) / (c_prev + 1.0)
        growth_per_year.append(g)
    all_g = torch.stack([g for g in growth_per_year if g.abs().sum() > 0])
    g_mean, g_std = all_g.mean().item(), all_g.std().item() + 1e-8
    growth_norm = [(g - g_mean) / g_std for g in growth_per_year]

    data_pt = {
        "xp": xp, "y": y_list, "topics": topics_per_year,
        "topic_names": topic_names, "centroids": centroids_per_year,
        "growth": growth_per_year, "growth_norm": growth_norm,
        "n_topics": n_topics,
    }
    out_pt = OUT_DIR / "fate_train.pt"
    torch.save(data_pt, out_pt)
    print(f"\n✅ Saved -> {out_pt}")

    # 統計
    print(f"\n[統計]")
    print(f"  topics (primary_category): {topic_names[:20]}")
    g_last = growth_per_year[-1].numpy()
    order = np.argsort(-g_last)
    print(f"\n  {YEARS[-1]} 年 成長率上位:")
    for i in order[:7]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {topic_names[i]:<20} g={g_last[i]:+.3f}")
    print(f"  下位:")
    for i in order[-5:][::-1]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {topic_names[i]:<20} g={g_last[i]:+.3f}")


if __name__ == "__main__":
    main()
