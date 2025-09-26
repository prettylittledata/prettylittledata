#!/usr/bin/env python3
import argparse, os, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("-o","--out", required=True)
args = ap.parse_args()
frames = []
for p in args.inputs:
    if os.path.exists(p) and os.path.getsize(p)>0:
        try: frames.append(pd.read_csv(p))
        except: pass
if frames:
    pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
else:
    pd.DataFrame(columns=["source","url","created_utc","title","text"]).to_csv(args.out, index=False)
print(f"Combined {len(frames)} files -> {args.out}")
