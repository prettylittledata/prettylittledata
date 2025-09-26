import json, re, time, math, hashlib
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict

import feedparser
import requests
import pandas as pd
import numpy as np
from dateutil import parser as dtp
from unidecode import unidecode

# -------- Settings --------
OUT_PATH = "data/signals.json"

RSS_SOURCES = {
    "fashion": [
        "https://www.vogue.com/rss",
        "https://www.businessoffashion.com/feed",
        "https://www.theguardian.com/fashion/rss",
        "https://www.dazed.com/fashion.rss",
        "https://fashionista.com/.rss/full/",
    ],
    "design": [
        "https://www.dezeen.com/feed/",
        "https://www.itsnicethat.com/feed",
        "https://eyeondesign.aiga.org/feed/",
        "https://www.creativeboom.com/feed/",
    ],
    "culture/beauty/wellness": [
        "https://www.thecut.com/feed",
        "https://www.allure.com/feed/all.xml",
        "https://www.refinery29.com/en-us/rss.xml",
        "https://www.wellandgood.com/feed/",
    ],
}

# YouTube (RSS feeds require only channel_id)
YOUTUBE_CHANNELS = {
    "fashion": [
        "UCi6b0l6g2WfZ3jz6U0n3tSg",  # BoF (example; replace if needed)
    ],
    "design": [
        "UC0QHWhjbe5fGJEP1s3xVK8g",  # Dezeen (example)
    ],
}

# Wikipedia Pageviews API (to ground common entities)
WIKI_TITLES = [
    "Longevity", "Gorpcore", "Quiet luxury", "Mob wife", "Cottagecore",
    "Coquette", "Y2K", "Mermaidcore", "Art deco", "Brutalism",
]

STOP_PHRASES = set("""
the a an and or but for with from by of in on to at into onto over under as is are was were be being been
vs versus near me guide review reviews best top cheap luxury budget brand brands what when where why how
""".split())

# map noisy synonyms to a canonical label
SYNONYMS = {
    "quiet luxury": ["stealth wealth"],
    "art deco": ["art-deco"],
    "y2k": ["y2k aesthetic"],
}

MIN_DATE = datetime.now(timezone.utc) - timedelta(days=28)

# POS-lite patterns with simple heuristics (no heavy NLP libs in Actions)
# We'll use regex & token rules to keep it robust.
def tokenize(text):
    text = unidecode(text.lower())
    text = re.sub(r"[#@/\\|_]", " ", text)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def candidate_phrases(text):
    toks = tokenize(text)
    # build 2-4grams
    for n in (2,3,4):
        for i in range(len(toks)-n+1):
            grams = toks[i:i+n]
            if any(t in STOP_PHRASES for t in grams): 
                continue
            # filter patterns: allow hyphen in adjectives; no all-numeric
            if all(re.search(r"[a-z]", g) for g in grams):
                yield " ".join(grams)

def canonicalize(phrase):
    p = phrase.strip()
    for canon, variants in SYNONYMS.items():
        if p == canon or p in variants:
            return canon
    return p

def fetch_rss_items():
    rows = []
    for cat, feeds in RSS_SOURCES.items():
        for url in feeds:
            try:
                d = feedparser.parse(url)
                for e in d.entries:
                    dt = None
                    for k in ("published", "updated", "created"):
                        if getattr(e, k, None):
                            try:
                                dt = dtp.parse(getattr(e, k)).astimezone(timezone.utc)
                                break
                            except: pass
                    if not dt: dt = datetime.now(timezone.utc)
                    if dt < MIN_DATE: 
                        continue
                    title = e.title if hasattr(e,"title") else ""
                    summ = e.summary if hasattr(e,"summary") else ""
                    link = e.link if hasattr(e,"link") else ""
                    rows.append(dict(source="rss", category=cat, date=dt.isoformat(), title=title, summary=summ, url=link))
            except Exception as ex:
                print("RSS error", url, ex)
    return rows

def fetch_youtube_items():
    rows = []
    for cat, chans in YOUTUBE_CHANNELS.items():
        for ch in chans:
            url = f"https://www.youtube.com/feeds/videos.xml?channel_id={ch}"
            try:
                d = feedparser.parse(url)
                for e in d.entries:
                    dt = dtp.parse(e.published).astimezone(timezone.utc) if hasattr(e,"published") else datetime.now(timezone.utc)
                    if dt < MIN_DATE: 
                        continue
                    rows.append(dict(source="youtube", category=cat, date=dt.isoformat(),
                                     title=e.title, summary="", url=e.link))
            except Exception as ex:
                print("YT error", ch, ex)
    return rows

def wiki_pageviews_last_30d(title):
    t = title.replace(" ", "_")
    end = (datetime.utcnow()-timedelta(days=1)).strftime("%Y%m%d")
    start = (datetime.utcnow()-timedelta(days=30)).strftime("%Y%m%d")
    api = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{t}/daily/{start}/{end}"
    try:
        r = requests.get(api, timeout=20)
        if r.status_code == 200:
            js = r.json()
            vals = [v["views"] for v in js.get("items",[])]
            if len(vals)>=7:
                return dict(last_7=sum(vals[-7:]), last_28=sum(vals[-28:]))
    except Exception as ex:
        print("wiki error", title, ex)
    return dict(last_7=0, last_28=0)

def zscore(s):
    if len(s)==0: return []
    arr = np.array(s, dtype=float)
    return list((arr - arr.mean()) / (arr.std(ddof=1) if arr.std(ddof=1)>0 else 1.0))

# ---------- Collect ----------
rows = []
rows += fetch_rss_items()
rows += fetch_youtube_items()

df = pd.DataFrame(rows)
if df.empty:
    # Write empty scaffold to avoid breaking the site
    with open(OUT_PATH, "w") as f:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "signals":[]}, f, ensure_ascii=False, indent=2)
    raise SystemExit("No rows collected")

df["date"] = pd.to_datetime(df["date"], utc=True)

# ---------- Phrase extraction ----------
records = []
for _, r in df.iterrows():
    text = f"{r.title} {r.summary}"
    for ph in candidate_phrases(text):
        c = canonicalize(ph)
        records.append({
            "phrase": c,
            "category": r.category,
            "source": r.source,
            "date": r.date,
            "url": r.url
        })
phr = pd.DataFrame(records)
if phr.empty:
    with open(OUT_PATH, "w") as f:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "signals":[]}, f, ensure_ascii=False, indent=2)
    raise SystemExit("No phrases extracted")

# drop ultra-generic phrases (single token slips)
phr = phr[phr["phrase"].str.count(" ") >= 1]

# ---------- Aggregate windows ----------
now = pd.Timestamp.utcnow()
w28 = now - pd.Timedelta(days=28)
w7  = now - pd.Timedelta(days=7)

def window_count(dfw):
    return dfw.groupby(["phrase"]).size().rename("mentions").reset_index()

vol28 = window_count(phr[phr["date"] >= w28])
vol7  = window_count(phr[phr["date"] >= w7])

# merge
agg = vol28.merge(vol7, on="phrase", how="left", suffixes=("_28d","_7d")).fillna(0)

# source breadth (unique sources in 28d)
breadth = phr[phr["date"]>=w28].groupby("phrase")["source"].nunique().rename("source_breadth")
agg = agg.merge(breadth, on="phrase", how="left").fillna(0)

# recency = fraction of mentions in last 7d
agg["recency"] = (agg["mentions_7d"] / agg["mentions_28d"].replace(0, np.nan)).fillna(0)

# growth = mentions_7d vs average per week in 28d
agg["growth"] = (agg["mentions_7d"] / (agg["mentions_28d"]/4).replace(0,np.nan)).fillna(0)

# attach categories and top links
topcat = (phr[phr["date"]>=w28]
          .groupby(["phrase","category"]).size().reset_index(name="c")
          .sort_values(["phrase","c"], ascending=[True,False])
          .drop_duplicates("phrase"))
agg = agg.merge(topcat[["phrase","category"]], on="phrase", how="left")

# collect latest links
def latest_links(p):
    sub = phr[(phr["phrase"]==p) & (phr["date"]>=w28)].sort_values("date", ascending=False).head(6)
    return [{"url":u, "date":str(d)} for u,d in zip(sub["url"], sub["date"]) if isinstance(u,str) and u]

link_map = {p: latest_links(p) for p in agg["phrase"]}
agg["links"] = agg["phrase"].map(link_map)

# ---------- Wikipedia grounding ----------
wiki_data = {}
for t in WIKI_TITLES:
    wiki_data[t.lower()] = wiki_pageviews_last_30d(t)
agg["wiki_last7"]  = agg["phrase"].map(lambda p: wiki_data.get(p.lower(),{}).get("last_7",0))
agg["wiki_last28"] = agg["phrase"].map(lambda p: wiki_data.get(p.lower(),{}).get("last_28",0))

# ---------- Scoring ----------
for col in ("mentions_7d","mentions_28d","growth","source_breadth","recency","wiki_last7"):
    if col not in agg: agg[col]=0

zs = {
    "z_volume": zscore(list(agg["mentions_7d"])),
    "z_growth": zscore(list(agg["growth"])),
    "z_breadth": zscore(list(agg["source_breadth"])),
    "z_recency": zscore(list(agg["recency"])),
}
for k,v in zs.items(): agg[k]=v

agg["interest_score"] = (
      agg["z_volume"]
    + 1.5*agg["z_growth"]
    + 0.5*agg["z_breadth"]
    + 0.5*agg["z_recency"]
)

# keep clean phrases
def ok_phrase(p):
    # avoid starting/ending with hyphen, digits-only, and brandy junk
    if re.search(r"^\d+$", p): return False
    if p.startswith("-") or p.endswith("-"): return False
    if any(tok in STOP_PHRASES for tok in p.split()): return False
    return True

agg = agg[agg["phrase"].apply(ok_phrase)]

# rank
agg = agg.sort_values("interest_score", ascending=False).head(200)

# build final objects
signals = []
for _,r in agg.iterrows():
    signals.append({
        "signal": r["phrase"],
        "category": r.get("category","mixed"),
        "volume_7d": int(r["mentions_7d"]),
        "volume_28d": int(r["mentions_28d"]),
        "growth_ratio": float(round(r["growth"],2)),
        "source_breadth": int(r["source_breadth"]),
        "recency_share": float(round(r["recency"],3)),
        "wiki_views_7d": int(r["wiki_last7"]),
        "interest_score": float(round(r["interest_score"],3)),
        "latest_links": r["links"],
    })

out = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "horizon_days": 28,
    "signals": signals
}

# ensure folder exists
import os
os.makedirs("data", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Wrote {OUT_PATH} with {len(signals)} signals.")
