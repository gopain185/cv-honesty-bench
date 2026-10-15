"""Run every applicable scheme on many replicates of every scenario and compare with the truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scenarios import SCENARIOS
from .schemes import SCHEMES, true_score

SEED = 20261016
TOLERANCE = 0.01  # |bias| in AUC below which a scheme counts as honest for a scenario


def run(scenarios: list[str] | None = None, replicates: int = 8, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for name in scenarios or list(SCENARIOS):
        sc = SCENARIOS[name]
        for r in range(replicates):
            data = sc.generate(np.random.default_rng([seed, r, list(SCENARIOS).index(name)]))
            truth = true_score(data)
            for scheme, (fn, applies) in SCHEMES.items():
                if applies(data):
                    est = fn(data, seed + r)
                    rows.append({"scenario": name, "replicate": r, "scheme": scheme,
                                 "estimate": est, "truth": truth, "error": est - truth})
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Bias, spread and RMSE of each scheme's error, with an honest/optimistic/pessimistic verdict."""
    g = results.groupby(["scenario", "scheme"], sort=False).error
    out = pd.DataFrame({"bias": g.mean(), "sd": g.std(ddof=1), "rmse": g.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                        "n": g.size()}).reset_index()
    out["verdict"] = np.select([out.bias > TOLERANCE, out.bias < -TOLERANCE], ["optimistic", "pessimistic"], "honest")
    return out


def pivot(summary: pd.DataFrame, value: str = "bias") -> pd.DataFrame:
    order = [s for s in SCHEMES if s in set(summary.scheme)]
    return summary.pivot(index="scenario", columns="scheme", values=value).reindex(
        index=[s for s in SCENARIOS if s in set(summary.scenario)], columns=order)
