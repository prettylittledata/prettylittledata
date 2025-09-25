#!/usr/bin/env python3
"""
PrettyData analyzer
- Inputs: CSV with columns created_utc (epoch seconds), text (string)
- Outputs: *_top_words.csv, *_top_bigrams.csv, *_term_counts_over_time.csv,
           *_trend_results.csv, *_increasing_terms.csv, *_decreasing_terms.csv
- Fallbacks: if no significant trends, show top by slope; if still empty, top by total mentions.
"""

import os, re, json, argparse
import pandas as pd
import statsmodels.api as sm

STOP = set((
    "a an the and or of in to for with on at from by about into over after before between out up "
    "down off above under again further then once here there all any both each few more most other "
    "some such no nor not only own same so than too very can will just don dont should shouldve now "
    "is are was were be been being have has had having do does did doing i me my myself we our ours "
    "ourselves you your yours yourself yourselves he him his himself she her hers herself it its "
    "itself they them their theirs themselves what which who whom this that these those am isnt "
    "arent wasnt werent havent hasnt hadnt doesnt didnt wont wouldnt shant shouldnt mustnt cant "
    "cannot could couldnt might mightnt neednt look also thinking any tips feel feels like really "
    "help please get got one two small big question thanks thank"
).split())

def tok(text: str):
    return [w.strip(".,!?;:()[]{}'\"").lower() for w in str(text).split() if w]

def contains_term(text: str, term: str) -> bool:
    t_esc = re.escape(term)
    t_esc = re.sub(r"\s+", r"\\s+", t_esc)
    return re.search(rf"\b{t_esc}\b", str(text), flags=re.I) is not None

def load_terms(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing terms file: {path}")
    if path.lower().endswith(".json"):
        data = json.load(open(path))
        return list(dict.fromkeys([t.strip() for t in data.get("terms", []) if t.strip()]))
    return list(dict.fromkeys([line.strip() for line in open(path) if line.strip()]))

def analyze(raw_csv: str, terms_file: str, prefix: str):
    if not os.path.exists(raw_csv):
        print(f"[warn] Missing {raw_csv}. Run scraper first.")
        return

    terms = load_terms(terms_file)
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(raw_csv)
    df["date"] = pd.to_datetime(df.get("created_utc"), unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["period"] = df["date"].dt.to_period("W").dt.start_time  # weekly start

    # ---------- tokens ----------
    words, bigrams = [], []
    for txt in df["text"].astype(str):
        tokens = tok(txt)
        words.extend([w for w in tokens if w and w not in STOP])
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in STOP and w2 not in STOP:
                bigrams.append(f"{w1} {w2}")

    if words:
        (pd.Series(words).value_counts()
         .rename_axis("word").reset_index(name="count")
         .to_csv(f"data/{prefix}_top_words.csv", index=False))
    else:
        pd.DataFrame(columns=["word", "count"]).to_csv(f"data/{prefix}_top_words.csv", index=False)

    if bigrams:
        (pd.Series(bigrams).value_counts()
         .rename_axis("bigram").reset_index(name="count")
         .to_csv(f"data/{prefix}_top_bigrams.csv", index=False))
    else:
        pd.DataFrame(columns=["bigram", "count"]).to_csv(f"data/{prefix}_top_bigrams.csv", index=False)

    # ---------- term counts over time ----------
    recs = []
    for term in terms:
        grp = df.groupby("period")["text"].apply(lambda col: sum(contains_term(t, term) for t in col)) \
                .reset_index(name="Count")
        grp["Term"] = term
        recs.append(grp)
    ts = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["period","Count","Term"])
    ts.to_csv(f"data/{prefix}_term_counts_over_time.csv", index=False)

    # also keep totals per term for fallback
    totals = ts.groupby("Term", as_index=False)["Count"].sum().rename(columns={"Count":"Total"})
    weeks = ts.groupby("Term", as_index=False)["period"].nunique().rename(columns={"period":"Weeks"})
    totals = totals.merge(weeks, on="Term", how="left")

    # ---------- trend regression ----------
    out = []
    for term, sub in ts.groupby("Term"):
        if sub.empty or len(sub) < 3 or sub["Count"].sum() == 0:
            out.append({"Term": term, "estimate": 0.0, "p_value": 1.0})
            continue
        x = pd.to_datetime(sub["period"])
        xnum = (x - x.min()) / pd.Timedelta(days=1)
        X = sm.add_constant(xnum.values)
        y = sub["Count"].values
        try:
            model = sm.OLS(y, X).fit()
            est = float(model.params[1]); p = float(model.pvalues[1])
        except Exception:
            est, p = 0.0, 1.0
        out.append({"Term": term, "estimate": est, "p_value": p})

    trend = (pd.DataFrame(out).sort_values("p_value")
             if out else pd.DataFrame(columns=["Term","estimate","p_value"]))
    trend = trend.merge(totals, on="Term", how="left")
    trend.to_csv(f"data/{prefix}_trend_results.csv", index=False)

    # ---------- pick rows: significant → slope → count ----------
    def pick(df: pd.DataFrame, positive=True, k=12):
        if df.empty:
            return df
        df = df.copy()
        df["mode"] = ""
        # 1) significant
        sig = df[(df["p_value"] < 0.05) & ((df["estimate"] > 0) if positive else (df["estimate"] < 0))]
        sig = sig.sort_values("estimate", ascending=not positive).head(k)
        if not sig.empty:
            sig["mode"] = "significant"
            return sig[["Term","estimate","p_value","Total","Weeks","mode"]]
        # 2) top by slope
        slope = df[df["estimate"] > 0] if positive else df[df["estimate"] < 0]
        slope = slope.sort_values("estimate", ascending=not positive).head(k)
        if not slope.empty:
            slope["mode"] = "slope"
            return slope[["Term","estimate","p_value","Total","Weeks","mode"]]
        # 3) top by count
        cnt = df.sort_values("Total", ascending=False).head(k) if positive else df.sort_values("Total").head(k)
        cnt["mode"] = "count"
        return cnt[["Term","estimate","p_value","Total","Weeks","mode"]]

    inc = pick(trend, positive=True, k=12)
    dec = pick(trend, positive=False, k=12)
    inc.to_csv(f"data/{prefix}_increasing_terms.csv", index=False)
    dec.to_csv(f"data/{prefix}_decreasing_terms.csv", index=False)

    print(f"[ok] wrote CSVs for '{prefix}' to ./data")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/decor_reddit_raw.csv")
    ap.add_argument("--terms-file", default="data/terms_decor.json")
    ap.add_argument("--prefix", default="decor")
    args = ap.parse_args()
    analyze(args.raw, args.terms_file, args.prefix)
