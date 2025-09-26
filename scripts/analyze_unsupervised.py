#!/usr/bin/env python3
"""
Unsupervised phrase discovery (1–3-gram) + weekly trend slopes.
Writes:
  all_top_phrases.csv
  all_phrase_counts_over_time.csv
  all_increasing_terms.csv
  all_decreasing_terms.csv
  all_new_terms.csv
"""
import argparse, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import linregress

def weekfloor(ts):
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return (dt - pd.to_timedelta(dt.weekday(), unit='D')).date()

def build_vec(texts, min_df=4, max_df=0.6, ngram=(1,3)):
    return CountVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df,
                           stop_words='english', lowercase=True,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b").fit(texts)

def counts_by_week(df, vec):
    df = df.copy()
    df["period"] = df["created_utc"].apply(weekfloor)
    weeks = sorted(df["period"].unique())
    X = vec.transform(df["doc"])
    vocab = np.array(vec.get_feature_names_out())
    rec=[]
    for w in weeks:
        idx = np.where(df["period"].values==w)[0]
        if len(idx)==0: continue
        row = X[idx,:].sum(axis=0).A1
        nz = np.nonzero(row)[0]
        for j in nz:
            rec.append((w, vocab[j], int(row[j])))
    return pd.DataFrame(rec, columns=["period","Term","Count"])

def slope_table(ct):
    weeks = sorted(ct["period"].unique())
    idx = {w:i for i,w in enumerate(weeks)}
    rec=[]
    for term, g in ct.groupby("Term"):
        y = np.zeros(len(weeks))
        for _,r in g.iterrows(): y[idx[r["period"]]] = r["Count"]
        x = np.arange(len(weeks))
        if y.sum()==0: continue
        lr = linregress(x,y)
        rec.append([term, int(y.sum()), lr.slope, lr.pvalue, int((y>0).sum())])
    return pd.DataFrame(rec, columns=["Term","Total","slope","pvalue","weeks_present"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("-o","--out-prefix", default="data/all")
    ap.add_argument("--min_df", type=int, default=4)
    ap.add_argument("--max_df", type=float, default=0.6)
    ap.add_argument("--new_weeks", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    if df.empty:
        for name in ["top_phrases","phrase_counts_over_time","increasing_terms","decreasing_terms","new_terms"]:
            pd.DataFrame().to_csv(f"{args.out_prefix}_{name}.csv", index=False)
        print("No data; wrote empty outputs"); return

    df["created_utc"] = df["created_utc"].fillna(0).astype(int)
    df["doc"] = (df["title"].fillna("") + "\n\n" + df["text"].fillna("")).astype(str)

    vec = build_vec(df["doc"], args.min_df, args.max_df)
    ct = counts_by_week(df, vec)
    ct.to_csv(f"{args.out_prefix}_phrase_counts_over_time.csv", index=False)

    st = slope_table(ct)
    fs = ct.groupby("Term")["period"].min().rename("first_seen").reset_index()
    ls = ct.groupby("Term")["period"].max().rename("last_seen").reset_index()
    join = st.merge(fs, on="Term", how="left").merge(ls, on="Term", how="left")
    join.sort_values("slope", ascending=False).to_csv(f"{args.out_prefix}_top_phrases.csv", index=False)

    inc = join[join["slope"]>0].sort_values(["slope","Total"], ascending=False).head(200)
    dec = join[join["slope"]<0].sort_values(["slope","Total"], ascending=True).head(200)
    inc.to_csv(f"{args.out_prefix}_increasing_terms.csv", index=False)
    dec.to_csv(f"{args.out_prefix}_decreasing_terms.csv", index=False)

    weeks = sorted(ct["period"].unique())
    cutoff = weeks[-args.new_weeks] if len(weeks)>args.new_weeks else weeks[0]
    new = join[(join["first_seen"]>=cutoff) & (join["slope"]>0)].sort_values(["slope","Total"], ascending=False).head(200)
    new.to_csv(f"{args.out_prefix}_new_terms.csv", index=False)

    print("Unsupervised analysis done under", args.out_prefix)

if __name__ == "__main__":
    main()
