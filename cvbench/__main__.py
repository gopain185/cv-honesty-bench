"""python -m cvbench [replicates]: bias of every validation scheme in every scenario."""

import sys
import time

import pandas as pd

from .harness import pivot, run, summarise

pd.set_option("display.width", 200)
t = time.time()
results = run(replicates=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
summary = summarise(results)
print(pivot(summary, "bias").round(4).to_string(), end="\n\n")
print(pivot(summary, "rmse").round(4).to_string(), end="\n\n")
print(summary.round(4).to_string(index=False))
print(f"{len(results)} estimates in {time.time() - t:.0f}s")
