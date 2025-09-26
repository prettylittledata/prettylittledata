#!/usr/bin/env python3
import argparse, time, feedparser, pandas as pd, trafilatura
from urllib.parse import quote_plus, urlparse

def fetch(url):
    try:
        dl = trafilatura.fetch_url(url, no_ssl=True, timeout=30)
        if not dl: return ""
        return trafilatura.extract(dl, include_comments=False) or ""
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="TXT: one Google News query per line")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--hl", default="en-US"); ap.add_argument("--gl", default="US"); ap.add_argument("--ceid", default="US:en")
    args = ap.parse_args()

    cutoff = time.time() - args.days*86400
    queries = [l.strip() for l in open(args.queries) if l.strip() and not l.startswith("#")]
    rows = []
    for q in queries:
        url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={args.hl}&gl={args.gl}&ceid={args.ceid}"
        parsed = feedparser.parse(url)
        for e in parsed.entries:
            pub = int(time.mktime(getattr(e, "published_parsed", time.gmtime())))
            if pub < cutoff: continue
            link = getattr(e,"link","") or ""
            title = getattr(e,"title","") or ""
            text = fetch(link)
            rows.append({
                "source":"gnews","query":q,"domain":urlparse(link).netloc,
                "url":link,"created_utc":pub,"title":title,"text":(title+"\n\n"+text).strip()
            })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
