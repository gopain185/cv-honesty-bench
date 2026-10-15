## What the numbers say

The numbers in this section come from the reference run (scikit-learn 1.9.1). A rerun above may differ in
the third decimal.

1. **Grouped rows and duplicated records are where random splits fail hardest:** about +0.10 and +0.20 AUC
   of optimism. The same customer, or the same record, lands on both sides of the split, and the model is
   rewarded for recognising it. GroupKFold on the group (or on the source record) brings the bias to within
   0.005.
2. **Time is two different problems.**
   - With **regimes**, rows close in time share something the features don't explain, and a feature tracks
     time. Random K-fold gives the model neighbours from the same period and is about +0.09 optimistic.
     Forward validation is close to honest; it is slightly pessimistic because each fold trains on less,
     older data.
   - With **smooth drift**, random K-fold happens to land near the truth (+0.005), and forward validation is
     pessimistic (-0.04 to -0.07), for the same reason.

   Time-ordered validation is still the safe default, since it never promised more than the model
   delivered here. But its pessimism is real, and it grows with how much the world changes within a fold.
3. **Target encoding fitted outside the folds** adds +0.08 AUC of optimism on a categorical with only a weak
   real effect. It is the classic leak, and it is invisible in the CV score itself. Fitting the encoder inside
   each training fold, with cross-fitting for the training rows, removes it.
4. **With 150 rows, all schemes are roughly unbiased, but one holdout is a coin toss.** Its estimate has a
   standard deviation of about 0.16 AUC, against about 0.07 for 5-fold and repeated 5-fold. On small data,
   the number of folds and repeats matters more than which random split you use.
5. **On the iid control, every random scheme is honest** (|bias| < 0.005). The traps above are caused by the
   data, not by the schemes themselves.

**Rule of thumb:** split the way the model will meet new data. If it will meet new customers, split by
customer. If it will meet the future, split by time. Fit every preprocessing step that looks at the target
inside the training folds.
