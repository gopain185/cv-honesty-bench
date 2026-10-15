import numpy as np
import pandas as pd
import pytest

from cvbench.harness import TOLERANCE, pivot, run, summarise
from cvbench.scenarios import SCENARIOS
from cvbench.schemes import SCHEMES, kfold_leaky_encoding, target_encode, true_score

REPLICATES = 6


@pytest.fixture(scope="module")
def summary():
    return summarise(run(replicates=REPLICATES))


def _bias(summary, scenario, scheme):
    row = summary[(summary.scenario == scenario) & (summary.scheme == scheme)]
    assert len(row) == 1, (scenario, scheme)
    return float(row.bias.iloc[0])


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_generators_are_deterministic_and_well_formed(name):
    a = SCENARIOS[name].generate(np.random.default_rng(1))
    b = SCENARIOS[name].generate(np.random.default_rng(1))
    assert a.X.equals(b.X) and np.array_equal(a.y, b.y)
    assert list(a.X.columns) == list(a.X_test.columns)
    assert set(np.unique(a.y)) == {0, 1} and set(np.unique(a.y_test)) == {0, 1}
    if a.time is not None:
        assert np.all(np.diff(a.time) >= 0), "time-ordered scenarios must be sorted by time"
    if a.groups is not None:
        assert len(a.groups) == len(a.y)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_safe_schemes_are_not_optimistic(summary, name):
    for scheme in SCENARIOS[name].safe:
        assert _bias(summary, name, scheme) < TOLERANCE, (name, scheme)


@pytest.mark.parametrize("name", [n for n, s in SCENARIOS.items() if s.traps])
def test_trap_schemes_are_clearly_optimistic(summary, name):
    for scheme in SCENARIOS[name].traps:
        assert _bias(summary, name, scheme) > 2 * TOLERANCE, (name, scheme)


def test_on_iid_data_random_splits_agree_with_the_truth(summary):
    for scheme in SCENARIOS["iid"].safe:
        assert abs(_bias(summary, "iid", scheme)) < TOLERANCE


def test_small_data_repeated_cv_has_less_spread_than_holdout(summary):
    s = summary[summary.scenario == "small_n"].set_index("scheme")
    assert s.loc["repeated_kfold", "sd"] < s.loc["holdout", "sd"]
    assert s.loc["kfold", "rmse"] < s.loc["holdout", "rmse"]


def test_schemes_only_run_where_they_apply(summary):
    table = pivot(summary)
    assert table.loc["iid", ["group_kfold", "time_series_split", "kfold_leaky_encoding"]].isna().all()
    assert not np.isnan(table.loc["grouped", "group_kfold"])
    assert not np.isnan(table.loc["leaky_encoding", "kfold_leaky_encoding"])


def test_plain_target_encoding_contains_the_rows_own_label():
    cat = pd.Series(np.arange(100))  # every level appears once
    y = np.r_[np.zeros(50), np.ones(50)]
    enc = target_encode(cat, y, cat)
    assert np.all(enc[y == 1] > enc[y == 0])  # perfectly separates the training labels


def test_leaky_scheme_beats_truth_on_a_pure_noise_categorical():
    d = SCENARIOS["leaky_encoding"].generate(np.random.default_rng(3))
    d.X["merchant"] = np.random.default_rng(4).integers(0, 1500, len(d.y))  # no signal at all
    d.X_test["merchant"] = np.random.default_rng(5).integers(0, 1500, len(d.y_test))
    assert kfold_leaky_encoding(d, 0) > true_score(d) + 0.02


def test_every_scheme_is_registered_with_a_guard():
    assert set(SCHEMES) >= {s for sc in SCENARIOS.values() for s in sc.safe + sc.traps}
