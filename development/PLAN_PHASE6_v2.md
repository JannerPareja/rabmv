# Analysis plan of phase 6, version 2 (1 November 2026)

Phase 6 is part of the development of RabMV (method), not of the study. Version 1 of its plan
(`PROTOCOL_PHASE6.md`) is kept unchanged, and so are its outputs (`RESULTS_PHASE6.md`, `decisions.json`).
This version changes how the comparison with EBM is reported and adds two analyses. All of them were decided
after seeing the results of version 1, and the method article states this in its methods section.

## Changes and their reasons

1. **Estimation instead of a non-inferiority decision for AUC-PR.** With 181 events the paired AUC-PR
   differences between folds have a standard deviation of about 0.03. Under the corrected resampled t-test the
   correction term n_test/n_train does not shrink with more repetitions, so even with infinitely many repetitions
   the smallest difference a one-sided 95 % bound can separate is about 0.026. A margin of 0.01 in AUC-PR cannot
   be resolved by these data under any repetition scheme (table `resolution.csv`). The comparison of RabMV with
   EBM is therefore reported as estimation, as recommended for comparing classifiers (Benavoli et al., 2017):
   the mean paired difference with its corrected 95 % confidence interval and the posterior probabilities of the
   Bayesian correlated t-test with ROPE ±0.01, for AUC-PR, AUC-ROC and the Brier score, together with
   calibration. The numbers are those already computed by version 1 (no result changes).
2. **Exploratory shortcut ablations.** The pre-specified ablation removed 32 % of the gap between EBM_registry
   and RabMV_registry. To explain the rest, on the saved models (no training):
   * EBM_registry with each missing-value score replaced by the mean score of the term over the measured
     training cases (RabMV's own rule). Setting a score to 0 or re-centring a term adds a constant to every case
     and cannot change AUC-PR, so the pre-specified ablation is not an artefact of centring;
   * categories that practically only one form produces (fewer than 5 cases in the other class, at least 5 in
     their own, whole window) set to not measured at scoring time, in both EBM_registry and RabMV_registry.
3. **Resolution of the data**: for each metric and comparison, the standard deviation of the paired
   differences, the corrected standard error with 20 folds and its floor with infinitely many repetitions, and
   the smallest separable difference.

## Unchanged

Data, folds, models, search, thresholds, metrics, the secondary comparisons, the pre-specified ablation and every
complementary evaluation of version 1.

## Reference

Benavoli A, Corani G, Demšar J, Zaffalon M (2017). Time for a change: a tutorial for comparing multiple
classifiers through Bayesian analysis. JMLR 18(77).
