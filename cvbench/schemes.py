"""Validation schemes and the model they validate.

Every scheme receives the training data only and returns its estimate of the deployment ROC AUC. The model
pipeline is the same for all schemes; only the way rows are split (and, for one scheme, where the target
encoding is fitted) changes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import (GroupKFold, KFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     TimeSeriesSplit, train_test_split)

from .scenarios import Data

SMOOTHING = 10.0


def target_encode(train_cat: pd.Series, y_train: np.ndarray, apply_to: pd.Series) -> np.ndarray:
    """Plain smoothed mean of y per level, learned from (train_cat, y_train), applied to apply_to.

    Used only by the leaky scheme: applied to the rows it was learned from, it contains their labels.
    """
    prior = y_train.mean()
    stats = pd.DataFrame({"c": train_cat.to_numpy(), "y": y_train}).groupby("c").y.agg(["sum", "count"])
    enc = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
    return apply_to.map(enc).fillna(prior).to_numpy()


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
                                          early_stopping=False, random_state=seed)


def fit_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_ev: pd.DataFrame, cat_col: str | None) -> np.ndarray:
    """The pipeline under validation: target-encode cat_col on the training part only, then boost.

    Training rows get cross-fitted encodings (each row encoded by a model that did not see it), so the
    booster does not learn to trust an encoding that contains the row's own label.
    """
    if cat_col is not None:
        X_tr, X_ev = X_tr.copy(), X_ev.copy()
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=0)
        X_tr[cat_col] = enc.fit_transform(X_tr[[cat_col]], y_tr).ravel()
        X_ev[cat_col] = enc.transform(X_ev[[cat_col]]).ravel()
    return make_model().fit(X_tr, y_tr).predict_proba(X_ev)[:, 1]


def true_score(d: Data) -> float:
    """AUC on the deployment sample of the model refitted on the whole training set."""
    return float(roc_auc_score(d.y_test, fit_predict(d.X, d.y, d.X_test, d.cat_col)))


def _cv(d: Data, splits, cat_col=None, X=None) -> float:
    X = d.X if X is None else X
    scores = []
    for tr, va in splits:
        if len(np.unique(d.y[va])) < 2:
            continue
        p = fit_predict(X.iloc[tr], d.y[tr], X.iloc[va], cat_col)
        scores.append(roc_auc_score(d.y[va], p))
    return float(np.mean(scores))


def holdout(d: Data, seed: int) -> float:
    idx = np.arange(len(d.y))
    tr, va = train_test_split(idx, test_size=0.2, random_state=seed, stratify=d.y)
    return _cv(d, [(tr, va)], d.cat_col)


def kfold(d: Data, seed: int) -> float:
    return _cv(d, KFold(5, shuffle=True, random_state=seed).split(d.X), d.cat_col)


def stratified_kfold(d: Data, seed: int) -> float:
    return _cv(d, StratifiedKFold(5, shuffle=True, random_state=seed).split(d.X, d.y), d.cat_col)


def repeated_kfold(d: Data, seed: int) -> float:
    return _cv(d, RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed).split(d.X, d.y), d.cat_col)


def group_kfold(d: Data, seed: int) -> float:
    return _cv(d, GroupKFold(5).split(d.X, d.y, d.groups), d.cat_col)


def time_series_split(d: Data, seed: int) -> float:
    return _cv(d, TimeSeriesSplit(5).split(d.X), d.cat_col)  # rows are already in time order


def forward_holdout(d: Data, seed: int) -> float:
    cut = int(0.8 * len(d.y))
    return _cv(d, [(np.arange(cut), np.arange(cut, len(d.y)))], d.cat_col)


def kfold_leaky_encoding(d: Data, seed: int) -> float:
    """The common mistake: target-encode on the full training set, then cross-validate."""
    X = d.X.copy()
    X[d.cat_col] = target_encode(d.X[d.cat_col], d.y, d.X[d.cat_col])
    return _cv(d, KFold(5, shuffle=True, random_state=seed).split(X), None, X)


Scheme = Callable[[Data, int], float]

SCHEMES: dict[str, tuple[Scheme, Callable[[Data], bool]]] = {
    "holdout": (holdout, lambda d: True),
    "kfold": (kfold, lambda d: True),
    "stratified_kfold": (stratified_kfold, lambda d: True),
    "repeated_kfold": (repeated_kfold, lambda d: True),
    "group_kfold": (group_kfold, lambda d: d.groups is not None),
    "time_series_split": (time_series_split, lambda d: d.time is not None),
    "forward_holdout": (forward_holdout, lambda d: d.time is not None),
    "kfold_leaky_encoding": (kfold_leaky_encoding, lambda d: d.cat_col is not None),
}
