#!/usr/bin/env python3
import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="list of CSV files to combine")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        if os.path.exists(p):
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
        print(f"Combined {len(frames)} files -> {args.out}")
    else:
        pd.DataFrame().to_csv(args.out, index=False)
        print("No inputs found; wrote empty CSV")

if __name__ == "__main__":
    main()
