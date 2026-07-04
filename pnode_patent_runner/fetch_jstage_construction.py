"""
J-STAGE API から日本建築・土木論文を収集 (建築 + 土木 = 建設全体)。

使用 API: https://api.jstage.jst.go.jp/searchapi/do?service=3
出力: data/raw_jstage_construction.jsonl  (title, journal, year, authors, doi)
"""
from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path
import re
from xml.etree import ElementTree as ET

import requests

OUT = Path("/home/nakamuraroi/kumagai/data/raw_jstage_construction.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 建設 (建築 + 土木) を広めに covering する keyword
QUERIES = [
    # 建築 (Architecture)
    "建築", "建築計画", "建築構造", "建築意匠", "建築環境", "建築設備",
    "都市計画", "住宅", "建築史", "建築材料", "意匠設計",
    # 土木 (Civil Engineering)
    "土木", "土木工学", "コンクリート", "鉄筋コンクリート", "鋼構造",
    "橋梁", "トンネル", "舗装", "地盤工学", "地盤改良",
    "耐震設計", "耐震補強", "免震", "制震",
    # 建設材料・施工
    "建設", "施工管理", "建設材料", "鉄骨造", "木造建築",
    # 環境・防災
    "都市環境", "ヒートアイランド", "BIM", "建設DX",
    "防災計画", "減災", "津波防災",
]

YEAR_FROM = 2015
YEAR_UNTIL = 2025

# Construction-related journal codes for filter (whitelist)
JOURNAL_WHITELIST = {
    "jaabe",        # AIJ Journal of Asian Arch & Building Eng
    "jaabe_jp",
    "aijt",         # AIJ technical journal
    "aija",         # AIJ Architecture Annual Articles
    "aijs",         # AIJ Structural Journal
    "aijp",         # AIJ Planning Annual Articles
    "aijax",        # AIJ Architecture Annual Examples
    "aijaxs",
    "aijaxe",
    "aijaxep",
    "aijaxeu",
    "aijaxh",
    "aijaxss",
    "aijaxsi",
    "aijl",         # AIJ Letters
    "aijlap",
    "aijls",
    "aijaej",
    "journalcpij",  # 都市計画論文集
    "cpij",
    "jsce",         # 土木学会
    "jscejam",
    "jscejb1",
    "jscejb2",
    "jscejb3",
    "jscejba",
    "jscejc",
    "jscejd1",
    "jscejd2",
    "jscejd3",
    "jscejea",
    "jscejeb",
    "jscejer",
    "jscejf1",
    "jscejf3",
    "jscejf4",
    "jscejf5",
    "jscejf6",
    "jscejg",
    "jscejh",
    "jscejhe",
    "jscejhsce",
    "jscejam23",
    "jscejhe23",
    "jscejb2_23",
}


def fetch_query(text: str, year_from: int, year_until: int, max_records: int = 3000):
    """Page through J-STAGE for a single keyword."""
    out = []
    start = 1
    count = 1000  # max per request
    while True:
        url = (
            "https://api.jstage.jst.go.jp/searchapi/do?"
            f"service=3&text={urllib.parse.quote(text)}"
            f"&pubyearfrom={year_from}&pubyearuntil={year_until}"
            f"&start={start}&count={count}"
        )
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} for query={text!r} start={start}")
                break
        except Exception as e:
            print(f"  request error for {text!r}: {e}")
            break

        # Parse XML — strip namespaces for ease of querying
        try:
            it = ET.iterparse(__import__("io").BytesIO(r.content))
            for _, el in it:
                if "}" in el.tag:
                    el.tag = el.tag.split("}", 1)[1]
            root = it.root
        except ET.ParseError as e:
            print(f"  xml parse error for {text!r}: {e}")
            break

        total = root.findtext("totalResults", default="0")
        try:
            total = int(total)
        except Exception:
            total = 0

        entries = root.findall("entry")
        if not entries:
            break

        for e in entries:
            title_ja = e.findtext("article_title/ja", default="") or ""
            title_en = e.findtext("article_title/en", default="") or ""
            j_ja  = e.findtext("material_title/ja", default="") or ""
            j_en  = e.findtext("material_title/en", default="") or ""
            cdjournal = e.findtext("cdjournal", default="") or ""
            year = e.findtext("pubyear", default="") or ""
            doi  = e.findtext("doi", default="") or ""

            authors_ja = [a.text for a in e.findall("author/ja/name") if a.text]
            authors_en = [a.text for a in e.findall("author/en/name") if a.text]

            try:
                year_int = int(year)
            except Exception:
                continue
            if year_int < year_from or year_int > year_until:
                continue

            out.append({
                "doi": doi.strip(),
                "title_ja": title_ja.strip(),
                "title_en": title_en.strip(),
                "journal_ja": j_ja.strip(),
                "journal_en": j_en.strip(),
                "cdjournal": cdjournal.strip().lower(),
                "year": year_int,
                "authors_ja": authors_ja,
                "authors_en": authors_en,
                "query": text,
            })

        print(f"    [{text!r}] start={start}, got={len(entries)}, total≈{total}, cumulative={len(out)}")

        start += count
        if start > total or start > max_records:
            break

        time.sleep(1.5)  # be gentle to J-STAGE
    return out


def main():
    all_hits = []
    seen_doi = set()
    seen_key = set()

    for q in QUERIES:
        print(f"\n>>> query={q!r}")
        hits = fetch_query(q, YEAR_FROM, YEAR_UNTIL, max_records=5000)
        # dedupe across queries
        new_count = 0
        for h in hits:
            doi = h["doi"]
            key = doi if doi else f"{h['title_ja']}_{h['year']}_{h['cdjournal']}"
            if doi and doi in seen_doi: continue
            if key in seen_key: continue
            seen_doi.add(doi) if doi else None
            seen_key.add(key)
            all_hits.append(h)
            new_count += 1
        print(f"    {new_count} new (total: {len(all_hits)})")

    # Write
    with open(OUT, "w", encoding="utf-8") as f:
        for h in all_hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_hits)} unique records -> {OUT}")

    # Year distribution
    from collections import Counter
    yc = Counter(h["year"] for h in all_hits)
    print("\nYear distribution:")
    for y in sorted(yc.keys()):
        bar = "█" * min(60, yc[y] // 50)
        print(f"  {y}: {yc[y]:>6}  {bar}")

    # Journal distribution (top 30)
    jc = Counter(h["cdjournal"] for h in all_hits if h["cdjournal"])
    print(f"\nTop 30 journals (out of {len(jc)}):")
    for cd, n in jc.most_common(30):
        print(f"  {cd:<20} {n:>5}")


if __name__ == "__main__":
    main()
