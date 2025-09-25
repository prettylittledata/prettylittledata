#!/usr/bin/env python3
import os, re, json, argparse
import pandas as pd
import statsmodels.api as sm

STOP = set(("a an the and or of in to for with on at from by about into over after before between "
            "out up down off above under again further then once here there all any both each few "
            "more most other some such no nor not only own same so than too very can will just don "
            "dont should shouldve now is are was were be been being have has had having do does did "
            "doing i me my myself we our ours ourselves you your yours yourself yourselves he him "
            "his himself she her hers herself it its itself they them their theirs themselves what "
            "which who whom this that these those am isnt arent wasnt werent havent hasnt hadnt "
            "doesnt didnt wont wouldnt shant shouldnt mustnt cant cannot could couldnt might "
            "mightnt neednt look also thinking any tips feel feels like really help please get got "
            "one two small big question").split())

def tok(t):
    return [w.strip(".,!?;:()[]{}'\"").lower() for w in str(t).split() if w]

def contains_term(text, term):
    t_esc = re.escape(term)
    t_esc = re.sub(r"\s+", r"\\s+", t_esc)
    return re.search(rf"\b{t_esc}\b", str(text), flags=re.I) is not None

def load_terms(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing terms file: {path}")
    if path.endswith(".json"):
        return json.load(open(path))["terms"]
    return [line.strip() for line in open(path) if line.strip()]

def analyze(raw_csv, terms_file, prefix):
    if not os.path.exists(raw_csv):
        print(f"Missing {raw_csv}. Run scraper first.")
        return

    terms = load_terms(terms_file)
    df = pd.read_csv(raw_csv)
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["period"] = df["date"].dt.to_period("W").dt.start_time  # weekly bucket starts

    # -------- Unigrams / Bigrams --------
    words, bigrams = [], []
    for txt in df["text"].astype(str):
        toks = tok(txt)
        words += [w for w in toks if w not in STOP]
        for i in range(len(toks) - 1):
            w1, w2 = toks[i], toks[i + 1]
            if w1 not in STOP and w2 not in STOP:
                bigrams.append(f"{w1} {w2}")

    os.makedirs("data", exist_ok=True)

    if words:
        pd.Series(words).value_counts() \
            .rename_axis("word").reset_index(name="count") \
            .to_csv(f"data/{prefix}_top_words.csv", index=False)
    else:
        pd.DataFrame(columns=["word", "count"]).to_csv(f"data/{prefix}_top_words.csv", index=False)

    if bigrams:
        pd.Series(bigrams).value_counts() \
            .rename_axis("bigram").reset_index(name="count") \
            .to_csv(f"data/{prefix}_top_bigrams.csv", index=False)
    else:
        pd.DataFrame(columns=["bigram", "count"]).to_csv(f"data/{prefix}_top_bigrams.csv", index=False)

    # -------- Term counts over time --------
    recs = []
    for term in terms:
        grp = df.groupby("period")["text"].apply(lambda col: sum(contains_term(t, term) for t in col)) \
                .reset_index(name="Count")
        grp["Term"] = term
        recs.append(grp)

    ts = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["period", "Count", "Term"])
    ts.to_csv(f"data/{prefix}_term_counts_over_time.csv", index=False)

    # -------- Trend regression --------
    out = []
    for term, sub in ts.groupby("Term"):
        x = pd.to_datetime(sub["period"])
        if len(sub) < 3 or sub["Count"].sum() == 0:
            out.append({"Term": term, "estimate": 0.0, "p_value": 1.0})
            continue
        xnum = (x - x.min()) / pd.Timedelta(days=1)
        X = sm.add_constant(xnum.values)
        y = sub["Count"].values
        try:
            model = sm.OLS(y, X).fit()
            out.append({"Term": term, "estimate": float(model.params[1]), "p_value": float(model.pvalues[1])})
        except Exception:
            out.append({"Term": term, "estimate": 0.0, "p_value": 1.0})

    trend = pd.DataFrame(out).sort_values("p_value") if out else pd.DataFrame(columns=["Term", "estimate", "p_value"])
    trend.to_csv(f"data/{prefix}_trend_results.csv", index=False)

    inc = trend[(trend.estimate > 0) & (trend.p_value < 0.05)].sort_values("estimate", ascending=False)
    dec = trend[(trend.estimate < 0) & (trend.p_value < 0.05)].sort_values("estimate")
    inc.to_csv(f"data/{prefix}_increasing_terms.csv", index=False)
    dec.to_csv(f"data/{prefix}_decreasing_terms.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/decor_reddit_raw.csv")
    ap.add_argument("--terms-file", default="data/terms_decor.json")
    ap.add_argument("--prefix", default="decor")
    args = ap.parse_args()
    analyze(args.raw, args.terms_file, args.prefix)
