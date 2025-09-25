#!/usr/bin/env python3
import argparse, json, os, time, pandas as pd
from pytrends.request import TrendReq

def load_terms(path):
    if path.lower().endswith(".json"):
        return [t.strip() for t in json.load(open(path)).get("terms", []) if t.strip()]
    return [l.strip() for l in open(path) if l.strip()]

def chunks(lst, n=5):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms-file", required=True, help="JSON {'terms':[...]} or .txt")
    ap.add_argument("--geo", default="", help="country code like US, GB, ES or empty for worldwide")
    ap.add_argument("--timeframe", default="today 12-m", help="e.g. 'today 12-m','today 5-y','now 7-d'")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    terms = load_terms(args.terms_file)
    py = TrendReq(hl='en-US', tz=0)
    frames = []
    for batch in chunks(terms, 5):
        try:
            py.build_payload(batch, timeframe=args.timeframe, geo=args.geo)
            df = py.interest_over_time()
            if df.empty: 
                time.sleep(2); 
                continue
            isP = df["isPartial"].rename("isPartial").reset_index()
            df = df.drop(columns=["isPartial"]).reset_index()
            long = df.melt(id_vars=["date"], var_name="Term", value_name="Value")
            long = long.merge(isP, on="date", how="left")
            frames.append(long)
            time.sleep(2)  # be polite
        except Exception:
            time.sleep(5)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date","Term","Value","isPartial"])
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows -> {args.out}")

if __name__ == "__main__":
    main()
