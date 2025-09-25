#!/usr/bin/env python3
"""
Polite Reddit scraper for PrettyData.
- Reads subs, days, limit; outputs CSV with created_utc + text (title+selftext).
- Handles 429 TooManyRequests with exponential/backoff using Retry-After.
- Sleeps between items to be nice.
Env: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
"""

import os, time, argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
import praw
import prawcore

def utc_now(): return datetime.now(timezone.utc)

def make_reddit():
    cid = os.environ.get("REDDIT_CLIENT_ID")
    csec = os.environ.get("REDDIT_CLIENT_SECRET")
    ua = os.environ.get("REDDIT_USER_AGENT", "prettydata-scraper/0.1 (by u/anonymous)")
    if not (cid and csec and ua):
        raise SystemExit("Missing reddit creds. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.")
    # requestor timeout helps if Reddit is slow
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua,
                       requestor_kwargs={"timeout": 30})

def backoff_sleep(exc, attempt, base=30, cap=600):
    wait = None
    # try to respect Retry-After if present
    try:
        wait = int(exc.response.headers.get("Retry-After", ""))
    except Exception:
        pass
    if not wait:
        wait = min(cap, base * (2 ** (attempt - 1)))
    # add a little jitter
    wait = int(wait + 3)
    print(f"[rate-limit] sleeping {wait}s (attempt {attempt})…")
    time.sleep(wait)

def scrape_sub(reddit, sub, since_epoch, per_sub_limit, item_sleep):
    out, seen = [], 0
    sr = reddit.subreddit(sub)
    # ask for extra to survive filtering by date
    fetch_limit = per_sub_limit * 3
    attempt = 0
    while True:
        try:
            for s in sr.new(limit=fetch_limit):
                if s.created_utc < since_epoch:
                    # older than our window; we can stop early
                    break
                title = s.title or ""
                body = (s.selftext or "").strip()
                text = (title + "\n\n" + body).strip()
                out.append({
                    "source": "reddit",
                    "subreddit": sub,
                    "url": f"https://www.reddit.com{s.permalink}",
                    "created_utc": int(s.created_utc),
                    "title": title,
                    "text": text
                })
                seen += 1
                if seen >= per_sub_limit:
                    return out
                if item_sleep > 0:
                    time.sleep(item_sleep)
            # finished listing without errors
            return out
        except prawcore.exceptions.TooManyRequests as e:
            attempt += 1
            backoff_sleep(e, attempt)
            continue
        except prawcore.exceptions.RequestException as e:
            # transient network issue
            attempt += 1
            backoff_sleep(e, attempt, base=10, cap=180)
            continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--limit", type=int, default=1000, help="max items per subreddit")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds to sleep between items")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    reddit = make_reddit()
    since_epoch = int((utc_now() - timedelta(days=args.days)).timestamp())

    rows = []
    for sub in args.subs:
        print(f"[scrape] r/{sub} last {args.days}d, limit {args.limit}")
        rows.extend(scrape_sub(reddit, sub, since_epoch, args.limit, args.sleep))
        # pause between subreddits
        time.sleep(2)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[done] wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
