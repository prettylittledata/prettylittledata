#!/usr/bin/env python3
import argparse, time, feedparser, trafilatura, pandas as pd
from urllib.parse import urlparse

def epoch(dt_struct):
    try: return int(time.mktime(dt_struct))
    except: return None

def fetch_article(url):
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True, timeout=30)
        if not downloaded: return ""
        return trafilatura.extract(downloaded, include_comments=False) or ""
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", required=True, help="path to txt with 1 RSS URL per line")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    cutoff = time.time() - args.days*86400
    urls = [l.strip() for l in open(args.feeds) if l.strip() and not l.strip().startswith("#")]
    rows = []
    for feed in urls:
        parsed = feedparser.parse(feed)
        for e in parsed.entries:
            pub = epoch(getattr(e, "published_parsed", None)) or epoch(getattr(e, "updated_parsed", None)) or int(time.time())
            if pub < cutoff: continue
            link = getattr(e, "link", "") or ""
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            text = fetch_article(link)
            fulltext = (title + "\n\n" + (text if text else summary)).strip()
            rows.append({
                "source":"rss",
                "domain": urlparse(link).netloc,
                "url": link,
                "created_utc": pub,
                "title": title,
                "text": fulltext
            })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
