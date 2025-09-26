#!/usr/bin/env python3
"""
Unsupervised phrase discovery (1–3-grams) + weekly trend slopes + smarter ranking.
Writes:
  all_top_phrases.csv
  all_phrase_counts_over_time.csv
  all_increasing_terms.csv
  all_decreasing_terms.csv
  all_new_terms.csv
"""
import os, re, argparse, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from scipy.stats import linregress

def weekfloor(ts: int):
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return (dt - pd.to_timedelta(dt.weekday(), unit='D')).date()

URL_RE   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
HTML_RE  = re.compile(r"&[a-z]+;")
UTM_RE   = re.compile(r"\butm_[a-z]+\b")
SPACE_RE = re.compile(r"\s+")

NOISE_STOPWORDS = {
    "http","https","www","com","net","org","href","amp","cdn","utm","xml","rss",
    "html","htm","php","gif","jpg","jpeg","png","svg"
}

# domain-generic single words we want to hide from the TOP lists (unigrams only)
GENERIC_UNIGRAMS = {
    "design","designer","designers","trend","trends",
    "fashion","style","styles","decor","beauty","ideas","home"
}

def clean_text(s: str) -> str:
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = UTM_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s)
    return s.strip()

def load_extra_stopwords(path="data/stopwords_extra.txt"):
    if os.path.exists(path):
        return [w.strip().lower() for w in open(path) if w.strip() and not w.startswith("#")]
    return []

def build_vec(texts, min_df=4, max_df=0.6, ngram=(1,3), extra_stop=None):
    stop = set(ENGLISH_STOP_WORDS) | set(NOISE_STOPWORDS)  # <-- adds the/the/and/for/etc.
    if extra_stop:
        stop |= set(extra_stop)
    # token_pattern excludes tokens <3 chars
    return CountVectorizer(
        ngram_range=ngram,
        min_df=min_df,
        max_df=max_df,
        stop_words=list(stop),
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    ).fit(texts)

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
        y = np.zeros(len(weeks), dtype=float)
        for _,r in g.iterrows():
            y[idx[r["period"]]] = r["Count"]
        x = np.arange(len(weeks), dtype=float)
        if y.sum()==0: continue
        lr = linregress(x,y)
        rec.append([term, int(y.sum()), float(lr.slope), float(lr.pvalue), int((y>0).sum())])
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
    doc = (df["title"].fillna("") + "\n\n" + df["text"].fillna("")).astype(str)
    df["doc"] = doc.map(clean_text)

    extra_stop = load_extra_stopwords()
    vec = build_vec(df["doc"], args.min_df, args.max_df, ngram=(1,3), extra_stop=extra_stop)

    ct = counts_by_week(df, vec)
    ct.to_csv(f"{args.out_prefix}_phrase_counts_over_time.csv", index=False)

    st = slope_table(ct)
    fs = ct.groupby("Term")["period"].min().rename("first_seen").reset_index()
    ls = ct.groupby("Term")["period"].max().rename("last_seen").reset_index()
    join = st.merge(fs, on="Term", how="left").merge(ls, on="Term", how="left")

    # ranking scores
    join["score_up"]   = join["slope"] / np.sqrt(join["Total"] + 1)
    join["score_down"] = (-join["slope"]) / np.sqrt(join["Total"] + 1)

    join.sort_values("slope", ascending=False).to_csv(f"{args.out_prefix}_top_phrases.csv", index=False)

    def drop_generic_unigrams(df_in: pd.DataFrame):
        is_unigram = df_in["Term"].str.count(" ") == 0
        generic = df_in["Term"].str.lower().isin(GENERIC_UNIGRAMS)
        return df_in[~(is_unigram & generic)]

    inc = join[join["slope"]>0].copy()
    inc = drop_generic_unigrams(inc)
    inc["score"] = inc["score_up"]
    inc = inc.sort_values(["score","slope","Total"], ascending=[False, False, False]).head(200)
    inc.to_csv(f"{args.out_prefix}_increasing_terms.csv", index=False)

    dec = join[join["slope"]<0].copy()
    dec = drop_generic_unigrams(dec)
    dec["score"] = dec["score_down"]
    dec = dec.sort_values(["score","slope","Total"], ascending=[False, True, True]).head(200)
    dec.to_csv(f"{args.out_prefix}_decreasing_terms.csv", index=False)

    weeks = sorted(ct["period"].unique())
    cutoff = weeks[-args.new_weeks] if len(weeks) > args.new_weeks else weeks[0]
    new = join[(join["first_seen"]>=cutoff) & (join["slope"]>0)].copy()
    new = drop_generic_unigrams(new)
    new["score"] = new["score_up"]
    new = new.sort_values(["score","slope","Total"], ascending=[False, False, False]).head(200)
    new.to_csv(f"{args.out_prefix}_new_terms.csv", index=False)

    print("Unsupervised analysis updated under", args.out_prefix)

if __name__ == "__main__":
    main()
