#!/usr/bin/env python3
import argparse, time, requests, pandas as pd
from datetime import datetime, timedelta, timezone

API = "https://www.googleapis.com/youtube/v3"

def iso_after(days):
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat().replace("+00:00","Z")

def to_epoch(iso_str):
    try:
        return int(datetime.fromisoformat(iso_str.replace("Z","+00:00")).timestamp())
    except Exception:
        return int(time.time())

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    def get_transcript(video_id):
        try:
            parts = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return " ".join(p["text"] for p in parts)[:15000]
        except Exception:
            return ""
except Exception:
    def get_transcript(video_id): return ""

def search(api_key, q, published_after, max_items=100):
    items, token = [], None
    fetched = 0
    while True:
        params = {
            "part":"snippet","type":"video","order":"date",
            "maxResults":50,"q":q,"key":api_key,"publishedAfter":published_after
        }
        if token: params["pageToken"] = token
        r = requests.get(API + "/search", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("items", []))
        fetched += len(data.get("items", []))
        token = data.get("nextPageToken")
        if not token or fetched >= max_items: break
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="txt file with 1 search query per line")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--max_per_query", type=int, default=100)
    args = ap.parse_args()

    api_key = None
    import os
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("Missing YOUTUBE_API_KEY env var")

    published_after = iso_after(args.days)
    queries = [l.strip() for l in open(args.queries) if l.strip() and not l.startswith("#")]

    rows = []
    for q in queries:
        for item in search(api_key, q, published_after, max_items=args.max_per_query):
            vid = item["id"]["videoId"]
            sn = item["snippet"]
            title = sn.get("title","")
            desc  = sn.get("description","") or ""
            when  = sn.get("publishedAt")
            epoch = to_epoch(when)
            url = f"https://www.youtube.com/watch?v={vid}"
            transcript = get_transcript(vid)
            text = (title + "\n\n" + desc + ("\n\n" + transcript if transcript else "")).strip()
            rows.append({
                "source":"youtube",
                "query": q,
                "url": url,
                "video_id": vid,
                "channel": sn.get("channelTitle",""),
                "created_utc": epoch,
                "title": title,
                "text": text
            })
            time.sleep(0.2)  # gentle pacing
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
