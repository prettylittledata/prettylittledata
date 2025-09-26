#!/usr/bin/env python3
"""
Search YouTube and collect titles+descriptions for recent videos.
Env: YOUTUBE_API_KEY
"""
import os, argparse, time, requests, pandas as pd
API = "https://www.googleapis.com/youtube/v3"

def yt_get(path, **params):
    r = requests.get(API+path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="TXT: one search query per line")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--max_per_query", type=int, default=25)
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()

    key = os.environ.get("YOUTUBE_API_KEY")
    if not key: raise SystemExit("Missing YOUTUBE_API_KEY")

    cutoff_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - args.days*86400))
    queries = [l.strip() for l in open(args.queries) if l.strip() and not l.startswith("#")]

    rows=[]
    for q in queries:
        pageToken=None; pulled=0
        while pulled < args.max_per_query:
            data = yt_get("/search",
                key=key, q=q, part="snippet", type="video",
                order="date", publishedAfter=cutoff_iso, maxResults=min(50, args.max_per_query-pulled),
                pageToken=pageToken)
            for item in data.get("items",[]):
                sn = item["snippet"]; vid = item["id"]["videoId"]
                title = sn.get("title",""); desc = sn.get("description","")
                pub = int(pd.Timestamp(sn.get("publishedAt")).timestamp())
                url = f"https://www.youtube.com/watch?v={vid}"
                rows.append({"source":"youtube","query":q,"url":url,"created_utc":pub,"title":title,"text":desc})
            pulled += len(data.get("items",[]))
            pageToken = data.get("nextPageToken")
            if not pageToken: break

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} videos -> {args.out}")

if __name__ == "__main__":
    main()
