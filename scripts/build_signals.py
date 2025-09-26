import json, re, os, math
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import feedparser, requests, pandas as pd, numpy as np
from dateutil import parser as dtp
from unidecode import unidecode

OUT_PATH = "data/signals.json"
SOURCES_PATH = "data/sources.json"
FILTERS_PATH = "data/filters.json"

HORIZON_DAYS = 28
MIN_DATE = datetime.now(timezone.utc) - timedelta(days=HORIZON_DAYS)

# -------- helpers --------
def read_json(path, default):
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception: return default

cfg = read_json(SOURCES_PATH, {"categories":{}, "wikipedia_titles":[], "social":{"enable":False}})
flt = read_json(FILTERS_PATH, {"synonyms":{}, "banlist":[], "ban_regex":[], "ban_entities":[], "allowlist":{}})

SOCIAL_ENABLE = (os.getenv("SOCIAL_ENABLE","false").lower()=="true")

BAN = set(w.strip().lower() for w in flt.get("banlist",[]))
BAN_ENT = set(w.strip().lower() for w in flt.get("ban_entities",[]))
BAN_RE = [re.compile(p, re.I) for p in flt.get("ban_regex",[])]
ALLOW_RE = {k:[re.compile(p, re.I) for p in v] for k,v in flt.get("allowlist",{}).items()}

SYN = {k.lower():[w.lower() for w in v] for k,v in flt.get("synonyms",{}).items()}

def canon(phrase):
    p = phrase.strip().lower()
    for k, vs in SYN.items():
        if p==k or p in vs: return k
    return p

def tokenize(text):
    text = unidecode(text.lower())
    text = re.sub(r"[#@/\\|_]", " ", text)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def candidate_phrases(text):
    toks = tokenize(text)
    for n in (2,3,4):
        for i in range(len(toks)-n+1):
            grams = toks[i:i+n]
            if any(g in BAN for g in grams): continue
            if not all(re.search(r"[a-z]", g) for g in grams): continue
            yield " ".join(grams)

def banned(phrase):
    p = phrase.lower()
    if any(r.search(p) for r in BAN_RE): return True
    if p in BAN or p in BAN_ENT: return True
    if p.startswith("-") or p.endswith("-"): return True
    if re.fullmatch(r"\d+", p): return True
    return False

def passes_allowlist(phrase, category):
    regs = ALLOW_RE.get(category, [])
    if not regs: return True  # no allowlist → pass
    return any(r.search(phrase) for r in regs)

def zscore(arr):
    a = np.array(arr, dtype=float)
    sd = a.std(ddof=1) if len(a)>1 else 0
    if sd==0: return np.zeros_like(a)
    return (a - a.mean())/sd

# -------- collectors --------
def fetch_rss_items():
    rows = []
    for cat, spec in cfg.get("categories",{}).items():
        for url in spec.get("rss",[]):
            try:
                d = feedparser.parse(url)
                for e in d.entries:
                    dt = None
                    for k in ("published","updated","created"):
                        if getattr(e,k,None):
                            try:
                                dt = dtp.parse(getattr(e,k)).astimezone(timezone.utc); break
                            except: pass
                    if not dt: dt = datetime.now(timezone.utc)
                    if dt < MIN_DATE: continue
                    rows.append(dict(source="rss",category=cat,date=dt.isoformat(),
                                     title=getattr(e,"title",""),summary=getattr(e,"summary",""),url=getattr(e,"link","")))
            except Exception as ex:
                print("RSS error", url, ex)
    return rows

def fetch_youtube_items():
    rows = []
    for cat, spec in cfg.get("categories",{}).items():
        for ch in spec.get("youtube_channels",[]):
            try:
                url = f"https://www.youtube.com/feeds/videos.xml?channel_id={ch}"
                d = feedparser.parse(url)
                for e in d.entries:
                    dt = dtp.parse(e.published).astimezone(timezone.utc) if hasattr(e,"published") else datetime.now(timezone.utc)
                    if dt < MIN_DATE: continue
                    rows.append(dict(source="youtube",category=cat,date=dt.isoformat(),
                                     title=e.title,summary="",url=e.link))
            except Exception as ex:
                print("YT error", ch, ex)
    return rows

# stubs for social — return [] until you wire scrapers
def fetch_social_items():
    if not SOCIAL_ENABLE or not cfg.get("social",{}).get("enable",False):
        return []
    # Placeholders to keep pipeline stable.
    # Implement your TikTok/IG fetch here and return the same row schema.
    return []

def wiki_views(title):
    t = title.replace(" ", "_")
    end = (datetime.utcnow()-timedelta(days=1)).strftime("%Y%m%d")
    start = (datetime.utcnow()-timedelta(days=30)).strftime("%Y%m%d")
    api = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{t}/daily/{start}/{end}"
    try:
        r = requests.get(api, timeout=20)
        if r.status_code == 200:
            items = r.json().get("items",[])
            vals = [v["views"] for v in items]
            return dict(last_7=sum(vals[-7:]) if len(vals)>=7 else 0,
                        last_28=sum(vals[-28:]) if len(vals)>=28 else sum(vals))
    except Exception as ex:
        print("wiki error", title, ex)
    return dict(last_7=0,last_28=0)

# -------- run --------
rows = []
rows += fetch_rss_items()
rows += fetch_youtube_items()
rows += fetch_social_items()

df = pd.DataFrame(rows)
if df.empty:
    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH,"w") as f:
        json.dump({"generated_at": datetime.utcnow().isoformat()+"Z","signals":[]}, f, ensure_ascii=False, indent=2)
    raise SystemExit("No rows collected")

df["date"] = pd.to_datetime(df["date"], utc=True)

# phrase extraction with filters
rec = []
for _, r in df.iterrows():
    text = f"{r.title} {r.summary}"
    for ph in candidate_phrases(text):
        ph = canon(ph)
        if banned(ph): continue
        # try allowlist; if fails, keep but mark low-priority
        allowed = passes_allowlist(ph, r.category)
        rec.append({"phrase":ph,"category":r.category,"source":r.source,"date":r.date,"url":r.url,"allowed":allowed})

phr = pd.DataFrame(rec)
# drop single-token just in case
phr = phr[phr["phrase"].str.count(" ")>=1]
if phr.empty:
    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH,"w") as f:
        json.dump({"generated_at": datetime.utcnow().isoformat()+"Z","signals":[]}, f, ensure_ascii=False, indent=2)
    raise SystemExit("No phrases")

now = pd.Timestamp.utcnow()
w28 = now - pd.Timedelta(days=HORIZON_DAYS)
w7  = now - pd.Timedelta(days=7)

def wc(dfw): return dfw.groupby(["phrase"]).size().rename("mentions").reset_index()

vol28 = wc(phr[phr["date"]>=w28])
vol7  = wc(phr[phr["date"]>=w7])

agg = vol28.merge(vol7, on="phrase", how="left", suffixes=("_28d","_7d")).fillna(0)
breadth = phr[phr["date"]>=w28].groupby("phrase")["source"].nunique().rename("source_breadth")
agg = agg.merge(breadth, on="phrase", how="left").fillna(0)

agg["recency"] = (agg["mentions_7d"] / agg["mentions_28d"].replace(0,np.nan)).fillna(0)
agg["growth"]  = (agg["mentions_7d"] / (agg["mentions_28d"]/4).replace(0,np.nan)).fillna(0)

topcat = (phr[phr["date"]>=w28]
          .groupby(["phrase","category"]).size().reset_index(name="c")
          .sort_values(["phrase","c"], ascending=[True,False])
          .drop_duplicates("phrase"))
agg = agg.merge(topcat[["phrase","category"]], on="phrase", how="left")

def latest_links(p):
    sub = phr[(phr["phrase"]==p) & (phr["date"]>=w28)].sort_values("date", ascending=False).head(6)
    return [{"url":u, "date":str(d)} for u,d in zip(sub["url"], sub["date"]) if isinstance(u,str) and u]

agg["links"] = agg["phrase"].map(lambda p: latest_links(p))

# Wikipedia grounding
wiki_titles = [t for t in cfg.get("wikipedia_titles",[])]
wmap = {t.lower(): wiki_views(t) for t in wiki_titles}
agg["wiki_last7"] = agg["phrase"].map(lambda p: wmap.get(p.lower(),{}).get("last_7",0))

# z-scores
agg["z_volume"]  = zscore(agg["mentions_7d"])
agg["z_growth"]  = zscore(agg["growth"])
agg["z_breadth"] = zscore(agg["source_breadth"])
agg["z_recency"] = zscore(agg["recency"])

# allowed boost (soft)
allowed_map = phr.groupby("phrase")["allowed"].max().to_dict()
agg["allowed_boost"] = agg["phrase"].map(lambda p: 0.4 if allowed_map.get(p,False) else 0)

agg["interest_score"] = (
    agg["z_volume"] + 1.5*agg["z_growth"] + 0.5*agg["z_breadth"] + 0.5*agg["z_recency"] + agg["allowed_boost"]
)

# final clean & rank
agg = agg.sort_values("interest_score", ascending=False).head(200)

signals = []
for _, r in agg.iterrows():
    signals.append({
        "signal": r["phrase"],
        "category": r.get("category","mixed"),
        "volume_7d": int(r["mentions_7d"]),
        "volume_28d": int(r["mentions_28d"]),
        "growth_ratio": float(round(r["growth"],2)),
        "source_breadth": int(r["source_breadth"]),
        "recency_share": float(round(r["recency"],3)),
        "wiki_views_7d": int(r.get("wiki_last7",0)),
        "interest_score": float(round(r["interest_score"],3)),
        "latest_links": r["links"]
    })

os.makedirs("data", exist_ok=True)
with open(OUT_PATH,"w") as f:
    json.dump({"generated_at": datetime.utcnow().isoformat()+"Z",
               "horizon_days": HORIZON_DAYS,
               "signals": signals}, f, ensure_ascii=False, indent=2)

print(f"Wrote {OUT_PATH} with {len(signals)} signals.")
