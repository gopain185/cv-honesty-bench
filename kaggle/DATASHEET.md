# CV Honesty Benchmark

Seven synthetic binary-classification tasks for testing **validation schemes**. Each task comes with its own
deployment sample (new groups, the future, new records), so the true score of a model is known and any
cross-validation estimate can be checked against it.

Companion notebook: [Does Your CV Tell the Truth? Six Traps](https://www.kaggle.com/code/locdunhat/does-your-cv-tell-the-truth-six-traps)

## Files

For each scenario there are two files: `<scenario>_train.csv` (what you would validate on) and
`<scenario>_deploy.csv` (20,000 rows the model will actually face, with labels).

| scenario | train rows | structure in train | what deployment looks like |
| --- | --- | --- | --- |
| `iid` | 3,000 | none | same distribution |
| `small_n` | 150 | none | same distribution |
| `grouped` | 3,000 | `group`: 150 groups × 20 rows, group-level outcome offsets | new groups only |
| `duplicated` | ≈3,000 | `group`: source record of rows repeated 1–4 times with jitter | new records |
| `regimes` | 3,000 | `time`: outcome offset wandering over time, `x7` tracks time | the next 10% of time |
| `drift` | 3,000 | `time`: covariate shift on `x0`, effect of `x5` flips sign | the next 25% of time |
| `leaky_encoding` | 3,000 | `merchant`: 600-level categorical with a weak real effect | same distribution |

Columns: `x0`–`x7` numeric features, optional `time` / `group` / `merchant`, and `y` (0/1). The rows of the
time-ordered scenarios are sorted by `time`.

`results.csv` has every estimate from the reference run: 8 replicates per scenario, one row per scheme, with
columns `scenario, replicate, scheme, estimate, truth, error`. `summary.csv` has bias, sd and RMSE per
scenario and scheme. The model is scikit-learn `HistGradientBoostingClassifier` (120 iterations, 15 leaves,
learning rate 0.08); the metric is ROC AUC.

## How to use

Pick a validation scheme and run it on `<scenario>_train.csv`. Then fit the model on the whole training file
and score it on `<scenario>_deploy.csv`. The gap between the two numbers is the error of your scheme. To get
bias and variance rather than one draw, rerun the generators in the `cv-honesty-bench` code with new seeds.

## Provenance

All data is simulated. The generators in `cvbench/scenarios.py` (repository `cv-honesty-bench` by Loc Nguyen,
MIT) produced these files with seed `[20261016, 0, scenario index]`. The outcome is a fixed non-linear logistic
function of `x0`–`x4`, plus the structure listed above. No real people or records are involved.

## Limitations

- One model family (gradient boosting) and one metric (ROC AUC). Other models memorise groups or duplicates
  more or less readily, so the size of each bias will change, but its sign generally won't.
- The tolerance for calling a scheme honest (|bias| < 0.01 AUC) is a choice, not a standard.
- The scenarios are deliberately clean, with one problem each. Real data often has several at once.

## License

CC0 1.0.
