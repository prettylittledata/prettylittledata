#!/usr/bin/env python3
"""
PrettyData analyzer
- Works for any CSV with at least: created_utc (epoch seconds), text (string)
- Tracks unigrams/bigrams + weekly term mentions + trend regression
- Falls back to "top by slope" if nothing is statistically significant

Outputs (into data/ using --prefix):
  <prefix>_top_words.csv
  <prefix>_top_bigrams.csv
  <prefix>_term_counts_over_time.csv
  <prefix>_trend_results.csv
  <prefix>_increasing_terms.csv
  <prefix>_decreasing_terms.csv
"""

import os, re, json, argparse
import pandas as pd
import statsmodels.api as sm

# ---- stopwords (quick + pragmatic) ----
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

# ---- helpers ----
def tok(text: str):
    """Very simple tokenizer that lowercases and strips punctuation."""
    return [w.strip(".,!?;:()[]{}'\"").lower() for w in str(text).split() if w]

def contains_term(text: str, term: str) -> bool:
    """Whole-word-ish match for multiword terms, whitespace-insensitive."""
    t_esc = re.escape(term)
    t_esc = re.sub(r"\s+", r"\\s+", t_esc)
    return re.search(rf"\b{t_esc}\b", str(text), flags=re.I) is not None

def load_terms(path: str):
    """Load terms from JSON {'terms': [...]} or from a plain text file (one per line)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing terms file: {path}")
    if path.lower().endswith(".json"):
        data = json.load(open(path))
        return list(dict.fromkeys([t.strip() for t in data.get("terms", []) if t.strip()]))
    # fallback: newline list
    return list(dict.fromkeys([line.strip() for line in open(path) if line.strip()]))

# ---- core analysis ----
def analyze(raw_csv: str, terms_file: str, prefix: str):
    if not os.path.exists(raw_csv):
        print(f"[warn] Missing {raw_csv}. Run scraper first.")
        return

    terms = load_terms(terms_file)
    if not terms:
        print(f"[warn] No terms loaded from {terms_file}.")
    os.makedirs("data", exist_ok=True)

    # Load
    df = pd.read_csv(raw_csv)
    # created_utc -> datetime (UTC), drop bad rows
    df["date"] = pd.to_datetime(df.get("created_utc"), unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    # weekly bucket (start of week)
    df["period"] = df["date"].dt.to_period("W").dt.start_time

    # --- unigrams / bigrams ---
    words, bigrams = [], []
    for txt in df["text"].astype(str):
        tokens = tok(txt)
        words.extend([w for w in tokens if w and w not in STOP])
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in STOP and w2 not in STOP:
                bigrams.append(f"{w1} {w2}")

    # pandas-version-proof value_counts -> reset_index
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

    # --- term counts over time ---
    recs = []
    for term in terms:
        grp = (
            df.groupby("period")["text"]
              .apply(lambda col: sum(contains_term(t, term) for t in col))
              .reset_index(name="Count")
        )
        grp["Term"] = term
        recs.append(grp)

    ts = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["period", "Count", "Term"])
    ts.to_csv(f"data/{prefix}_term_counts_over_time.csv", index=False)

    # --- trend regression (OLS on week index) ---
    out = []
    for term, sub in ts.groupby("Term"):
        if sub.empty:
            out.append({"Term": term, "estimate": 0.0, "p_value": 1.0})
            continue
        x = pd.to_datetime(sub["period"])
        # need at least 3 time points to estimate a slope usefully
        if len(sub) < 3 or sub["Count"].sum() == 0:
            out.append({"Term": term, "estimate": 0.0, "p_value": 1.0})
            continue
        xnum = (x - x.min()) / pd.Timedelta(days=1)  # days since first bucket
        X = sm.add_constant(xnum.values)
        y = sub["Count"].values
        try:
            model = sm.OLS(y, X).fit()
            est = float(model.params[1])
            p = float(model.pvalues[1])
        except Exception:
            est, p = 0.0, 1.0
        out.append({"Term": term, "estimate": est, "p_value": p})

    trend = (pd.DataFrame(out).sort_values("p_value")
             if out else pd.DataFrame(columns=["Term", "estimate", "p_value"]))
    trend.to_csv(f"data/{prefix}_trend_results.csv", index=False)

    # --- choose significant results, or fallback to top slopes if none ---
    def top_or_sig(tr: pd.DataFrame, direction="+", k=12) -> pd.DataFrame:
        if tr.empty:
            return tr
        df = tr.copy()
        df["sig"] = df["p_value"] < 0.05
        if direction == "+":
            sig = df[(df.estimate > 0) & df.sig].sort_values("estimate", ascending=False)
            top = df[df.estimate > 0].nlargest(k, "estimate")
        else:
            sig = df[(df.estimate < 0) & df.sig].sort_values("estimate")  # most negative first
            top = df[df.estimate < 0].nsmallest(k, "estimate")
        outdf = sig if not sig.empty else top
        return outdf[["Term", "estimate", "p_value"]]

    inc = top_or_sig(trend, "+", 12)
    dec = top_or_sig(trend, "-", 12)
    inc.to_csv(f"data/{prefix}_increasing_terms.csv", index=False)
    dec.to_csv(f"data/{prefix}_decreasing_terms.csv", index=False)

    print(f"[ok] Wrote CSVs with prefix '{prefix}' into ./data")

# ---- CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/decor_reddit_raw.csv",
                    help="Path to raw CSV with created_utc,text")
    ap.add_argument("--terms-file", default="data/terms_decor.json",
                    help="JSON {'terms':[...]} or .txt (one term per line)")
    ap.add_argument("--prefix", default="decor", help="Output prefix")
    args = ap.parse_args()
    analyze(args.raw, args.terms_file, args.prefix)
