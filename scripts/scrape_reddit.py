#!/usr/bin/env python3
import os, argparse
from datetime import datetime, timedelta
import praw, pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subs", nargs="+", default=["HomeDecorating","InteriorDesign","CozyPlaces"])
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--limit", type=int, default=1500)  # per subreddit approx
    p.add_argument("-o", "--out", default="data/decor_reddit_raw.csv")
    return p.parse_args()

def main():
    args = parse_args()
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "PrettyLittleData/0.1")
    )
    reddit.read_only = True

    end = datetime.utcnow()
    start = end - timedelta(days=args.days)

    rows = []
    for sub in args.subs:
        sr = reddit.subreddit(sub)
        for post in sr.new(limit=args.limit):
            created = datetime.utcfromtimestamp(post.created_utc)
            if not (start <= created <= end):
                continue
            try:
                post.comments.replace_more(limit=0)
                comments = [c.body for c in post.comments.list() if hasattr(c, "body")]
            except Exception:
                comments = []
            rows.append({
                "source": "reddit",
                "subreddit": sub,
                "post_id": post.id,
                "created_utc": int(post.created_utc),
                "title": post.title or "",
                "selftext": post.selftext or "",
                "text": f"{post.title}\n\n{post.selftext}\n\n" + "\n".join(comments)
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")

if __name__ == "__main__":
    main()
