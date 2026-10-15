# Does your CV tell the truth? Six traps for validation schemes

A cross-validation score is a promise about how the model will do once deployed. This notebook checks that
promise on synthetic data where the answer is known. Each scenario comes with a large deployment sample
(new groups, the future, new records), so for every replicate we can compare what the validation scheme
*said* with what the model then *did*.

- **Model:** scikit-learn `HistGradientBoostingClassifier`, the same for every scheme.
- **Metric:** ROC AUC.
- **Schemes:** holdout, K-fold, stratified K-fold, repeated stratified K-fold, GroupKFold, TimeSeriesSplit, a
  forward holdout, and K-fold after target-encoding on the full training set. A scheme only runs where it
  makes sense; GroupKFold, for example, needs groups.
- **Scenarios:** one control (`iid`) and six traps: small data, grouped rows, duplicated records, time
  regimes, smooth drift, and a high-cardinality categorical that needs target encoding.
- **Replicates:** each scenario is regenerated 8 times with new seeds, and every scheme is rerun each time.
  Bias is the mean of `estimate - truth`.

Everything is synthetic and runs on CPU. The package source is embedded below (MIT); the data is also
published as the [CV Honesty Benchmark](https://www.kaggle.com/datasets/locdunhat/cv-honesty-benchmark)
dataset (CC0).
