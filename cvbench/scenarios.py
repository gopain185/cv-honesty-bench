"""Synthetic data-generating processes with a known deployment distribution.

Each scenario returns a training set and a large test set drawn from the distribution the model will
actually face (new groups, the future, unseen rows), so the true score of a model is known and every
cross-validation scheme can be judged by how far its estimate lands from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

N_FEATURES = 8
TEST_SIZE = 20_000


@dataclass
class Data:
    X: pd.DataFrame
    y: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    groups: np.ndarray | None = None  # group id per training row, if the data has groups
    time: np.ndarray | None = None  # timestamp per training row, if the data is ordered in time
    cat_col: str | None = None  # a high-cardinality categorical column that needs target encoding


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    generate: Callable[[np.random.Generator], Data]
    safe: tuple[str, ...]  # schemes expected not to be optimistic (bias < TOLERANCE); checked by the tests
    traps: tuple[str, ...] = ()  # schemes expected to be clearly optimistic (bias > 2 * TOLERANCE)
    tags: tuple[str, ...] = field(default=())


def _signal(X: np.ndarray) -> np.ndarray:
    """A fixed non-linear logit: main effects, one interaction and a threshold."""
    return 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3] + 0.8 * (X[:, 4] > 0.5) - 0.4


def _bernoulli(rng, logit):
    return (rng.random(len(logit)) < 1 / (1 + np.exp(-logit))).astype(int)


def _frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def iid(rng, n: int = 3000) -> Data:
    X, Xt = rng.normal(size=(n, N_FEATURES)), rng.normal(size=(TEST_SIZE, N_FEATURES))
    return Data(_frame(X), _bernoulli(rng, _signal(X)), _frame(Xt), _bernoulli(rng, _signal(Xt)))


def small_iid(rng) -> Data:
    return iid(rng, n=150)


def grouped(rng, n_groups: int = 150, per_group: int = 20) -> Data:
    """Rows come in groups (customers, patients, sites). Each group has its own outcome offset and its
    own feature profile, so a flexible model can recognise a group from its features and memorise the
    offset. At deployment every group is new."""

    def draw(k):
        centre = rng.normal(scale=1.0, size=(k, N_FEATURES))
        offset = rng.normal(scale=1.5, size=k)
        g = np.repeat(np.arange(k), per_group)
        X = centre[g] + rng.normal(scale=0.35, size=(len(g), N_FEATURES))
        return X, _bernoulli(rng, _signal(X) + offset[g]), g

    X, y, g = draw(n_groups)
    Xt, yt, _ = draw(TEST_SIZE // per_group)
    return Data(_frame(X), y, _frame(Xt), yt, groups=g)


def drift(rng, n: int = 3000) -> Data:
    """The relation between features and outcome changes smoothly over time (covariate shift on x0, the effect
    of x5 flips sign); the model is deployed on the period after the training window."""

    def draw(t):
        X = rng.normal(size=(len(t), N_FEATURES))
        X[:, 0] += 1.5 * t
        return X, _bernoulli(rng, _signal(X) + (1.5 - 3.0 * t) * X[:, 5])

    t = np.sort(rng.random(n))
    X, y = draw(t)
    Xt, yt = draw(rng.uniform(1.0, 1.25, TEST_SIZE))
    return Data(_frame(X), y, _frame(Xt), yt, time=t)


def regimes(rng, n: int = 3000, steps: int = 60, horizon: float = 0.1) -> Data:
    """A slowly wandering offset on the outcome (market regime, season, a policy that keeps changing), and a
    feature that tracks time (account age, a date-derived column). Rows close in time share the offset, so a
    model can learn it from its neighbours in time; at deployment the offset has moved on."""
    walk = np.cumsum(rng.normal(size=steps + int(steps * horizon) + 1))
    walk = 1.5 * (walk - walk[:steps].mean()) / walk[:steps].std()

    def draw(t):
        X = rng.normal(size=(len(t), N_FEATURES))
        X[:, 7] = t + rng.normal(scale=0.01, size=len(t))
        offset = walk[np.minimum((t * steps).astype(int), len(walk) - 1)]
        return X, _bernoulli(rng, _signal(X) + offset)

    t = np.sort(rng.random(n))
    X, y = draw(t)
    Xt, yt = draw(rng.uniform(1.0, 1.0 + horizon, TEST_SIZE))
    return Data(_frame(X), y, _frame(Xt), yt, time=t)


def duplicated(rng, n_unique: int = 1200) -> Data:
    """Each record appears one to four times with a small jitter (re-submitted forms, augmentation done
    before splitting, a join that fanned out). Deployment rows are new records."""
    base = rng.normal(size=(n_unique, N_FEATURES))
    yb = _bernoulli(rng, _signal(base))
    reps = rng.integers(1, 5, n_unique)
    src = np.repeat(np.arange(n_unique), reps)
    X = base[src] + rng.normal(scale=0.02, size=(len(src), N_FEATURES))
    Xt = rng.normal(size=(TEST_SIZE, N_FEATURES))
    return Data(_frame(X), yb[src], _frame(Xt), _bernoulli(rng, _signal(Xt)), groups=src)


def high_cardinality(rng, n: int = 3000, n_levels: int = 600) -> Data:
    """A categorical column with many rare levels and only a weak real effect, which has to be target
    encoded. Encoding it on the whole training set before splitting leaks each row's own label."""

    def draw(k, level_effect):
        X = rng.normal(size=(k, N_FEATURES))
        lev = rng.integers(0, n_levels, k)
        return X, lev, _bernoulli(rng, _signal(X) + level_effect[lev])

    effect = rng.normal(scale=0.3, size=n_levels)
    X, lev, y = draw(n, effect)
    Xt, levt, yt = draw(TEST_SIZE, effect)
    F, Ft = _frame(X), _frame(Xt)
    F["merchant"], Ft["merchant"] = lev, levt
    return Data(F, y, Ft, yt, cat_col="merchant")


RANDOM_SPLITS = ("holdout", "kfold", "stratified_kfold", "repeated_kfold")

SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("iid", "independent rows, deployment from the same distribution", iid, safe=RANDOM_SPLITS),
    Scenario("small_n", "150 independent rows: unbiased schemes differ in variance", small_iid,
             safe=("kfold", "repeated_kfold"), tags=("variance",)),
    Scenario("grouped", "150 groups of 20 rows with group-level offsets; deployment on new groups", grouped,
             safe=("group_kfold",), traps=RANDOM_SPLITS),
    Scenario("duplicated", "records repeated 1-4 times with jitter; deployment on new records", duplicated,
             safe=("group_kfold",), traps=RANDOM_SPLITS),
    Scenario("regimes", "outcome offset wandering over time, a time-proxy feature; deployment after training",
             regimes, safe=("forward_holdout", "time_series_split"), traps=RANDOM_SPLITS),
    Scenario("drift", "smooth drift over time; deployment on the period after training", drift,
             safe=("time_series_split", "forward_holdout")),
    Scenario("leaky_encoding", "high-cardinality categorical needing target encoding", high_cardinality,
             safe=("kfold", "stratified_kfold", "repeated_kfold"), traps=("kfold_leaky_encoding",)),
]}
