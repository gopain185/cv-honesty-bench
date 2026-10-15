# cv-honesty-bench

A benchmark for **validation schemes**. Seven synthetic tabular tasks each come with a deployment sample whose
labels are known: new groups, the period after training, or new records. Every cross-validation estimate can
therefore be checked against the score the model actually gets once deployed.

A CV score is a promise. This benchmark measures how often, and by how much, each scheme breaks it.

- **Kaggle notebook:** [Does Your CV Tell the Truth? Six Traps](https://www.kaggle.com/code/locdunhat/does-your-cv-tell-the-truth-six-traps)
- **Kaggle dataset:** [CV Honesty Benchmark](https://www.kaggle.com/datasets/locdunhat/cv-honesty-benchmark) (CC0)

## Results

Bias = mean over 8 replicates of (CV estimate − deployment AUC). Model: `HistGradientBoostingClassifier`,
metric ROC AUC, scikit-learn 1.9.1. Positive values mean the scheme was optimistic.

| scenario | holdout | K-fold | strat. K-fold | repeated K-fold | GroupKFold | TimeSeriesSplit | forward holdout | K-fold + leaky TE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iid (control) | +0.002 | −0.002 | −0.004 | −0.001 | | | | |
| small_n (150 rows) | −0.060 | −0.011 | −0.014 | −0.019 | | | | |
| grouped | **+0.112** | **+0.101** | **+0.101** | **+0.101** | −0.004 | | | |
| duplicated | **+0.193** | **+0.198** | **+0.196** | **+0.197** | −0.005 | | | |
| regimes | **+0.095** | **+0.094** | **+0.093** | **+0.094** | | −0.023 | −0.021 | |
| drift | +0.008 | +0.005 | +0.003 | +0.005 | | −0.037 | −0.066 | |
| leaky_encoding | +0.009 | +0.000 | +0.004 | +0.002 | | | | **+0.078** |

On `small_n`, bias isn't the issue. Spread is: the standard deviation of the estimate is 0.155 for one holdout
and 0.067 for repeated 5-fold.

What it shows:

- **Grouped rows and duplicated records** are where random splits fail hardest, at +0.10 and +0.20 AUC of
  optimism. GroupKFold on the group, or on the source record, is within 0.005 of the truth.
- **Time is two problems.**
  - With *regimes* (rows close in time share something, and a feature tracks time), random K-fold is about
    +0.09 optimistic and forward validation is close to honest.
  - With *smooth drift*, random K-fold happens to land near the truth, and forward validation is pessimistic,
    because each forward fold trains on less, older data than the final model.

  Time-ordered validation never over-promised here, but its pessimism is real.
- **Target encoding fitted on the full training set** before splitting adds +0.08 of optimism. Fitting the
  encoder inside each training fold, with cross-fitting for the training rows, removes it.
- **On iid data every random scheme is honest.** The traps come from the data, not from the schemes.

## Scenarios

| name | structure | deployment |
| --- | --- | --- |
| `iid` | none | same distribution |
| `small_n` | 150 rows | same distribution |
| `grouped` | 150 groups × 20 rows, group offsets, group-specific feature profiles | new groups |
| `duplicated` | records repeated 1–4 times with small jitter | new records |
| `regimes` | an outcome offset that wanders over time; `x7` tracks time | the next 10% of time |
| `drift` | covariate shift on `x0`; the effect of `x5` flips sign over time | the next 25% of time |
| `leaky_encoding` | 600-level categorical with a weak real effect | same distribution |

Each scenario declares the schemes expected to be safe (bias below +0.01) and the traps expected to be clearly
optimistic (bias above +0.02). The test suite checks both on 6 fresh replicates, so the claims above are
tested rather than only reported.

## Usage

```bash
pip install -r requirements.txt
python -m cvbench 8       # bias and RMSE tables, about 10 minutes on a laptop
python -m pytest -q       # about 8-10 minutes (reruns the benchmark with 6 replicates)
python kaggle/build_kaggle.py   # Kaggle dataset + notebook
```

To test another scheme, add a function `(data, seed) -> estimate` to `cvbench/schemes.py`, register it in
`SCHEMES` with a guard saying where it applies, and rerun.

## Layout

```
cvbench/scenarios.py   data-generating processes, each with a deployment sample
cvbench/schemes.py     the model pipeline, the validation schemes, the true score
cvbench/harness.py     replicates, bias / sd / RMSE, verdicts
tests/                 checks that the stated safe and trap schemes behave as claimed
kaggle/                dataset and notebook builders, datasheet
```

All data is synthetic. License: MIT.
