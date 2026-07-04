"""
arXiv API から建築・土木隣接論文を**大規模**に取得 (目標 ~15k 件)。

戦略:
  1. 検索クエリ数を 20 → 60+ に拡張 (専門用語多数)
  2. クエリごとに 2000 件 (arXiv 上限) まで取得 + pagination
  3. キーワード/カテゴリフィルタを維持して建築関連だけ採用

出力: data/PNode_ArXiv_Construction_X1_v2/alltime/fate_train.pt
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

RELEVANT_CATEGORIES = [
    "eess.SY", "eess.SP", "eess.IV", "eess.AS",
    "cs.RO", "cs.CY", "cs.CV", "cs.LG", "cs.AI",
    "cs.HC", "cs.CE", "cs.SE", "cs.MA", "cs.DB", "cs.NE",
    "physics.app-ph", "physics.geo-ph", "physics.flu-dyn",
    "physics.comp-ph", "physics.ao-ph",
    "cond-mat.mtrl-sci", "cond-mat.soft", "cond-mat.dis-nn",
    "cond-mat.stat-mech",
    "math.OC", "math.NA", "math.AP",
    "stat.AP", "stat.ML",
    "astro-ph.EP",
    "q-fin.RM",
]

# 大幅拡張クエリ (約 60 個)
SEARCH_QUERIES = [
    # 一般建築・土木
    "construction", "structural engineering", "civil engineering",
    "building construction", "civil infrastructure",
    "building information modeling", "BIM",
    # コンクリート・材料
    "concrete", "reinforced concrete", "fiber reinforced concrete",
    "high performance concrete", "ultra high performance concrete",
    "cement composite", "geopolymer concrete",
    "fly ash", "silica fume", "rebar",
    # 構造解析
    "finite element analysis", "modal analysis structural",
    "structural dynamics", "structural vibration",
    "structural reliability", "structural optimization",
    "structural identification", "model updating",
    # 構造ヘルスモニタリング
    "structural health monitoring", "damage detection structural",
    "vibration based monitoring", "acoustic emission monitoring",
    "fiber optic sensor structural",
    # 地震工学
    "earthquake engineering", "seismic design", "seismic response",
    "seismic isolation", "base isolation",
    "seismic vulnerability", "fragility curve",
    # 地盤
    "geotechnical engineering", "soil mechanics",
    "foundation engineering", "pile foundation",
    "soil structure interaction", "liquefaction soil",
    "retaining wall", "slope stability",
    # 橋・トンネル
    "bridge engineering", "bridge monitoring",
    "tunnel construction", "tunneling engineering",
    # 道路・舗装
    "pavement engineering", "asphalt pavement",
    "highway engineering", "road construction",
    # ロボット・自動化
    "construction robotics", "construction automation",
    "additive manufacturing construction", "3D printing construction",
    # スマート
    "smart construction", "smart city infrastructure",
    "smart building", "intelligent building",
    "digital twin construction", "construction digital twin",
    # 検査・損傷
    "crack detection concrete", "damage assessment building",
    "non-destructive testing concrete",
    # サステナビリティ
    "sustainable construction", "green building", "energy efficient building",
    # 設計
    "structural design optimization", "topology optimization structural",
    # 既存
    "masonry", "prefabricated construction", "modular construction",
]

CONSTRUCTION_KEYWORDS = [
    "construction", "structural", "concrete", "reinforced",
    "bridge", "building", "geotechnical", "civil engineering",
    "infrastructure", "earthquake", "seismic", "foundation",
    "steel", "masonry", "highway", "pavement", "tunnel",
    "soil", "bim", "rebar", "geopolymer",
    "finite element", "vibration", "shear strength",
    "compressive strength", "prefabricat", "retrofit",
    "structural health", "damage detection", "modal analysis",
    "topology optimization", "fiber reinforced",
    "cement", "asphalt", "rebar", "pile", "slope",
    "ductility", "stiffness", "deflection",
]

OUT_DIR = Path("data/PNode_ArXiv_Construction_X1_v2/alltime")
YEARS = list(range(2015, 2026))
PCA_DIM = 50
TOP_K_TOPICS = 40
MAX_PER_QUERY = 2000   # arXiv 上限
PAGE_SIZE = 200
SLEEP_BETWEEN = 3


def fetch_arxiv_query(query: str, max_total: int = 2000) -> List[Dict]:
    """1 クエリ × pagination"""
    base = "http://export.arxiv.org/api/query"
    results = []
    start = 0
    while start < max_total:
        params = {
            "search_query": f'all:"{query}"',
            "start": start,
            "max_results": PAGE_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = base + "?" + urllib.parse.urlencode(params)
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"    page {start}: error {e}")
            break

        from xml.etree import ElementTree as ET
        NS = {"atom": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            break
        entries = root.findall("atom:entry", NS)
        if not entries:
            break
        for entry in entries:
            title = entry.findtext("atom:title", "", NS).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", "", NS).strip().replace("\n", " ")
            published = entry.findtext("atom:published", "", NS)
            pub_year = int(published[:4]) if published else None
            primary = entry.find("arxiv:primary_category", NS)
            primary_cat = primary.get("term") if primary is not None else None
            cats = [c.get("term") for c in entry.findall("atom:category", NS)]
            results.append({
                "title": title, "abstract": summary, "year": pub_year,
                "primary_category": primary_cat, "categories": cats,
            })
        if len(entries) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN)
    return results


def is_construction_relevant(paper: Dict) -> bool:
    text = (paper["title"] + " " + paper["abstract"]).lower()
    has_keyword = any(kw in text for kw in CONSTRUCTION_KEYWORDS)
    has_relevant_cat = any(c in RELEVANT_CATEGORIES for c in paper.get("categories", []))
    return has_keyword and has_relevant_cat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"  arXiv 大規模建築論文取得  (queries={len(SEARCH_QUERIES)}, max/query={MAX_PER_QUERY})")
    print("=" * 70)

    all_papers = []
    seen_titles = set()
    for q_idx, q in enumerate(SEARCH_QUERIES):
        print(f"\n[{q_idx+1}/{len(SEARCH_QUERIES)}] '{q}'")
        results = fetch_arxiv_query(q, max_total=MAX_PER_QUERY)
        n_new = 0
        for p in results:
            if not p["year"] or p["year"] not in YEARS: continue
            if not p["abstract"] or len(p["abstract"]) < 50: continue
            if p["title"] in seen_titles: continue
            if not is_construction_relevant(p): continue
            seen_titles.add(p["title"])
            all_papers.append(p)
            n_new += 1
        print(f"  retrieved: {len(results)}, new: {n_new}, total cumul: {len(all_papers)}")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nTotal unique construction papers: {len(all_papers)}")
    if len(all_papers) < 100:
        print("⚠️  Not enough papers")
        return

    df = pd.DataFrame(all_papers)
    df["topic"] = df["primary_category"]
    top_topics = df["topic"].value_counts().head(TOP_K_TOPICS).index.tolist()
    df = df[df["topic"].isin(top_topics)].reset_index(drop=True)
    topic_names = sorted(df["topic"].dropna().unique())
    topic_to_id = {t: i for i, t in enumerate(topic_names)}
    n_topics = len(topic_names)
    print(f"  After top-{TOP_K_TOPICS} filter: {len(df)} papers, {n_topics} topics")

    print("\nGenerating embeddings...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (df["title"].fillna("") + ". " + df["abstract"]).tolist()
    embeds = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True)

    print(f"\nPCA → {PCA_DIM}D...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(PCA_DIM, embeds.shape[0] - 1), random_state=42)
    X = pca.fit_transform(embeds).astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    actual_dim = pca.n_components_
    print(f"  explained var: {pca.explained_variance_ratio_.sum():.4f}")

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

    torch.save({
        "xp": xp, "y": y_list, "topics": topics_per_year,
        "topic_names": topic_names, "centroids": centroids_per_year,
        "growth": growth_per_year, "growth_norm": growth_norm,
        "n_topics": n_topics,
    }, OUT_DIR / "fate_train.pt")
    print(f"\n✅ Saved -> {OUT_DIR / 'fate_train.pt'}")

    # 統計
    print(f"\n[統計]")
    g_last = growth_per_year[-1].numpy()
    order = np.argsort(-g_last)
    print(f"  {YEARS[-1]} 年 成長率上位:")
    for i in order[:8]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {topic_names[i]:<22} g={g_last[i]:+.3f}")
    print(f"  下位:")
    for i in order[-5:][::-1]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {topic_names[i]:<22} g={g_last[i]:+.3f}")


if __name__ == "__main__":
    main()
