#!/usr/bin/env python3
"""
Unsupervised phrase discovery + trend ranking.
Input: CSV with columns [created_utc, title, text, source, url]
Outputs:
  - all_top_phrases.csv (ranked by slope)
  - all_phrase_counts_over_time.csv (weekly)
  - all_increasing_terms.csv / all_decreasing_terms.csv
  - all_new_terms.csv (first seen in last N weeks + positive growth)
"""
import argparse, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import linregress

def weekfloor(ts):
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    # monday week start:
    return (dt - pd.to_timedelta(dt.weekday(), unit='D')).date()

def build_vocab(texts, min_df=5, max_df=0.6, ngram=(1,3)):
    vec = CountVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df,
                          stop_words='english', lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    vec.fit(texts)
    return vec

def counts_by_week(df, vec):
    df = df.copy()
    df["period"] = df["created_utc"].apply(weekfloor)
    weeks = sorted(df["period"].unique())
    # transform all docs once to keep vocabulary
    X_all = vec.transform(df["doc"])
    vocab = np.array(vec.get_feature_names_out())
    # aggregate by week
    records = []
    for w in weeks:
        idx = (df["period"]==w).values.nonzero()[0]
        if len(idx)==0: continue
        Xw = X_all[idx,:].sum(axis=0).A1
        nz = np.nonzero(Xw)[0]
        for j in nz:
            records.append((w, vocab[j], int(Xw[j])))
    out = pd.DataFrame(records, columns=["period","Term","Count"])
    return out

def slope_table(ct):
    # pivot to numeric x for regression
    weeks = sorted(ct["period"].unique())
    week_ix = {w:i for i,w in enumerate(weeks)}
    rows = []
    for term, grp in ct.groupby("Term"):
        y = np.zeros(len(weeks))
        for _,r in grp.iterrows():
            y[week_ix[r["period"]]] = r["Count"]
        x = np.arange(len(weeks))
        if y.sum()==0: continue
        try:
            lr = linregress(x, y)
            rows.append([term, y.sum(), lr.slope, lr.pvalue, int((y>0).sum())])
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["Term","Total","slope","pvalue","weeks_present"])

def first_seen(ct):
    fs = ct.groupby("Term")["period"].min().rename("first_seen").reset_index()
    ls = ct.groupby("Term")["period"].max().rename("last_seen").reset_index()
    return fs.merge(ls, on="Term")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("-o", "--out-prefix", default="data/all")
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--max_df", type=float, default=0.6)
    ap.add_argument("--new_weeks", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    if df.empty:
        # write empty outputs
        empty = pd.DataFrame(columns=["Term","Total","slope","pvalue","weeks_present","first_seen","last_seen"])
        empty.to_csv(f"{args.out_prefix}_top_phrases.csv", index=False)
        pd.DataFrame(columns=["period","Term","Count"]).to_csv(f"{args.out_prefix}_phrase_counts_over_time.csv", index=False)
        empty.to_csv(f"{args.out_prefix}_increasing_terms.csv", index=False)
        empty.to_csv(f"{args.out_prefix}_decreasing_terms.csv", index=False)
        empty.to_csv(f"{args.out_prefix}_new_terms.csv", index=False)
        print("No data; wrote empty outputs"); return

    df["created_utc"] = df["created_utc"].fillna(0).astype(int)
    df["doc"] = (df["title"].fillna("") + "\n\n" + df["text"].fillna("")).astype(str)

    vec = build_vocab(df["doc"], min_df=args.min_df, max_df=args.max_df, ngram=(1,3))
    ct = counts_by_week(df, vec)

    # store time series
    ct.to_csv(f"{args.out_prefix}_phrase_counts_over_time.csv", index=False)

    # slope table + first/last seen
    st = slope_table(ct)
    fl = first_seen(ct)
    joined = st.merge(fl, on="Term", how="left").sort_values("slope", ascending=False)

    joined.to_csv(f"{args.out_prefix}_top_phrases.csv", index=False)

    inc = joined[(joined["slope"]>0)].sort_values(["slope","Total"], ascending=False).head(200)
    dec = joined[(joined["slope"]<0)].sort_values(["slope","Total"], ascending=True).head(200)
    inc.to_csv(f"{args.out_prefix}_increasing_terms.csv", index=False)
    dec.to_csv(f"{args.out_prefix}_decreasing_terms.csv", index=False)

    # "new": first_seen within last N weeks and positive slope
    weeks = sorted(ct["period"].unique())
    cutoff = weeks[-args.new_weeks] if len(weeks)>args.new_weeks else weeks[0]
    newest = joined[(joined["first_seen"]>=cutoff) & (joined["slope"]>0)].sort_values(["slope","Total"], ascending=False).head(200)
    newest.to_csv(f"{args.out_prefix}_new_terms.csv", index=False)

    print("Wrote trend files under", args.out_prefix)

if __name__ == "__main__":
    main()
