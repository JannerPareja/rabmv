# Pre-specified protocol of phase 6: confirmation of RabMV with the Valledupar data

Version 1 · written 31 October 2026, before running any task of phase 6.
Author of the study: Janer Alfonso Pareja Gutiérrez. Method: RabMV (Registry-Aware Boosting Machine of Valledupar).

The SHA-256 fingerprint of this file is stored in `development/config.py` (`PROTOCOL_SHA256`). The analysis
command refuses to run if the file was changed. Any change after the first task of phase 6 is a new version
of the protocol, declared with its reason, and the results that depend on it are reported as exploratory.

## 0. What was known when this protocol was written (transparency)

Phase 5 (repetition 1 of the 5×5, folds 0–4) was run and analysed. Its results were used to choose the
design of this phase and are therefore not confirmatory:

* With no balancing and the Bayesian optimizer, mean AUC-PR over folds 0–4: EBM 0.4888, RabMV 0.4765,
  RabMV_registry 0.4392, RabMV_common_education 0.4469, EBM_registry 0.5011. Over the four optimizers,
  RabMV − EBM ranged from −0.012 to +0.006 (a tie); RabMV calibration slope 0.97–1.00, EBM 0.91–0.94.
* SMOTE and ADASYN lowered AUC-PR and calibration of both models and create values that do not exist in the
  data (binary variables at fractional values; bins of RabMV occupied only by synthetic cases). They are not
  used in this phase.
* Class weights (with prior correction) did not improve AUC-PR. They are not used in this phase.
* EBM_registry learned a score of −1.69 logit for missing education in fold 0 (none · bayes); in Valledupar
  education is missing almost only among attempts (form 356 of 2016 had no such field; «no information»).
  This motivated the shortcut ablation of section 6, specified here before seeing it.
* Between-fold variability is large (AUC-PR from about 0.37 to 0.66 depending on the fold).

## 1. Data, folds and models

* Data: `datos_modelado.csv` of `etl.py` version 2, SHA-256 `37AFEDCB577CC27F…` (checked by the code).
  Window 2016–2022: 2,351 events, 181 completed suicides (forms 105 · Cesar) and 2,170 suicide attempts
  (form 356). In Valledupar the form coincides with the class.
* Outer folds: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)`. Folds 0–4 are repetition 1
  (phase 5). **Folds 5–24 (repetitions 2–5) were never used to choose anything and decide this phase.**
* Within each outer fold, exactly as in phase 5: X_thresh = 20 % of the outer training set (stratified,
  seed 42); hyperparameter search on the rest with inner stratified 5-fold CV and 24 evaluations of the
  Bayesian optimizer (Optuna TPE, seed 42); refit with the best configuration; threshold maximizing F1 on
  X_thresh; Platt and isotonic recalibration fitted on X_thresh; every metric on the test fold.
* No balancing. Variants (inputs as in phase 5):

| Variant | Input | Role |
|---|---|---|
| EBM | Competencia 1 matrix of the 8-concept core (InterpretML 0.7.8) | Comparator |
| RabMV | Same matrix | The method on the comparator's ground |
| RabMV_registry | Registry-aware input (catalog and mappings, common resolution, marginalization, categorical) | The method as designed |
| EBM_registry | Exactly the registry-aware input; EBM learns its own value for missing | Isolates marginalization against a learned missing value |
| RabMV_common_education | Competencia 1 matrix with education from the registry (marginalized) | Diagnostic: splits the registry difference |

## 2. Primary question: non-inferiority of RabMV against EBM (H1)

* Quantity: d = AUC-PR(RabMV) − AUC-PR(EBM) on each of the 20 test folds 5–24 (paired by fold).
* Test: corrected resampled t-test for repeated k-fold CV (Nadeau and Bengio, 2003; Bouckaert and Frank,
  2004): t = d̄ / √((1/n + n_test/n_train) · s²), n = 20, df = 19, with n_test and n_train the mean sizes of the
  test fold and the outer training set.
* **H1 holds if both:** (a) the one-sided 95 % lower confidence bound of d̄ is greater than −0.01
  (non-inferiority margin 0.01 of AUC-PR); and (b) RabMV is weakly calibrated on folds 5–24: mean calibration
  intercept within [−0.2, 0.2] and mean calibration slope within [0.8, 1.2] (original probabilities).
* Reported alongside (does not change the decision): Bayesian correlated t-test (Corani and Benavoli, 2015;
  Benavoli et al., 2017) with ρ = n_test / (n_test + n_train) and ROPE ±0.01: P(d < −0.01), P(|d| ≤ 0.01),
  P(d > 0.01).

## 3. Superiority (H2), only if H1 holds (fixed sequence)

* **H2 holds if both:** the one-sided corrected t-test rejects d̄ ≤ 0 at α = 0.05; and the Bayesian posterior
  probability P(d > 0.01) exceeds 0.95.
* If H1 does not hold, H2 is not tested and is reported as «not tested».

## 4. Secondary comparisons (one family, Holm correction, two-sided α = 0.05)

Same 20 folds, same corrected t-test; Bayesian probabilities, Cliff's δ and the Wilcoxon signed-rank test
are reported as complements.

| # | Comparison | What it measures | Direction seen in phase 5 |
|---|---|---|---|
| S1 | RabMV_registry − EBM_registry | Marginalization against a learned missing value, same input | Negative |
| S2 | EBM_registry − EBM | What EBM gains from the registry-aware input | Positive |
| S3 | RabMV_registry − RabMV | Cost of the registry-aware input for RabMV | Negative |
| S4 | RabMV_common_education − RabMV | Part of S3 due to education | Negative |
| S5 | RabMV_registry − RabMV_common_education | Rest of S3 | Negative, small |

## 5. Sensitivity analyses of the comparisons

* The same tests on the 25 folds 0–24 (includes the folds used to choose the design).
* The same tests on AUC-ROC, Brier score and F1.

## 6. Shortcut ablation of EBM_registry (pre-specified)

* For each saved EBM_registry model (folds 0–24 and the temporal fold), every learned missing-value score is
  set to 0 (the «missing» bin of each main effect and the «missing» row and column of each pair), without
  refitting and without changing any other score or the intercept. The test fold is scored again.
* Quantities per fold: AUC-PR of the ablated model; Δ_ablation = AUC-PR(ablated) − AUC-PR(EBM_registry); and
  A = AUC-PR(ablated) − AUC-PR(RabMV_registry).
* Pre-specified reading: if Δ_ablation is negative and the share of the S1 gap removed by the ablation,
  (AUC-PR(EBM_registry) − AUC-PR(ablated)) / (AUC-PR(EBM_registry) − AUC-PR(RabMV_registry)), is at least 0.5 on
  folds 5–24, the advantage of EBM_registry is attributed mainly to its missing-value score (recognition of the
  form). A is tested with the corrected t-test and the Bayesian correlated t-test (ROPE ±0.01), reported, not
  part of the Holm family.
* Declared limitation: setting a score to 0 does not re-centre the shape function; the ablated model is the
  learned model with no contribution from missingness, which is what RabMV's marginalization prescribes.

## 7. Supporting counts (aggregates)

From the data, per class: cases with missing education (total and in 2016 versus 2017–2022), and the
frequency of each category of the insurance scheme in the registry-aware input. Counts from 1 to 4 are shown
as «<5».

## 8. Temporal validation (descriptive)

The five variants are fitted with the cases notified in 2016–2020 (same search, X_thresh and threshold rule)
and tested on 2021–2022. Reported: AUC-PR, AUC-ROC, Brier, calibration, F1, and the paired differences of
section 4. One split: no test is applied.

## 9. Final models (for publication and documentation)

Each variant is searched (inner 5-fold CV, 24 Bayesian evaluations) and refitted on every case of 2016–2022.
The published equation, shape functions, pair tables and term tables come from these models. The operating
threshold is the F1 threshold on out-of-fold probabilities (5 stratified folds, seed 42, the chosen
configuration refitted in each); the same out-of-fold probabilities give an internal validation, slightly
optimistic because the configuration was chosen with every case (declared).

## 10. Complementary evaluations (descriptive, all reported)

* Calibration: Brier, ECE, calibration intercept and slope, binned calibration curves; original, Platt and
  isotonic probabilities.
* Equity by sex: sensitivity (TPR) in women and in men and its gap with the threshold of each fold (mean over
  folds), and pooled by repetition.
* Clinical utility: decision curve (net benefit) for thresholds 0.05–0.50 (step 0.01) against «alert
  everyone» and «alert no one», computed per repetition (each case once) and averaged over repetitions 2–5.
* Stability of the interpretation: Spearman correlation of term importances between folds and Kendall's W per
  variant; frequency of each selected interaction; hyperparameters chosen per fold.
* Computational cost: search, refit and prediction time.
* Pooled precision-recall and calibration curves; AUC-PR per fold and repetition.

## 11. Rules

1. RabMV, EBM, the inputs and the search are not modified according to the results of folds 5–24.
2. Seed 42, the folds, the tests, the margins and the thresholds above do not change after seeing results.
3. Every result of this protocol is reported, favourable or not, with the discussion of its probable cause.
4. Reporting follows TRIPOD+AI.
5. Privacy: only aggregates leave the private work folder; no individual row; counts from 1 to 4 are
   suppressed.
6. Not included, and why: SMOTE / ADASYN (invalid on binary and encoded variables, section 0); class weights
   (no gain, section 0); ablation of components of the boosting engine (the engine is not claimed as new);
   external datasets and robustness to seeds or label noise (left for the peer review, if requested).

## References

* Benavoli A, Corani G, Demšar J, Zaffalon M (2017). Time for a change: a tutorial for comparing multiple
  classifiers through Bayesian analysis. JMLR 18(77).
* Bouckaert RR, Frank E (2004). Evaluating the replicability of significance tests for comparing learning
  algorithms. PAKDD.
* Corani G, Benavoli A (2015). A Bayesian approach for comparing cross-validated algorithms on multiple data
  sets. Machine Learning 100.
* Nadeau C, Bengio Y (2003). Inference for the generalization error. Machine Learning 52.
* Van Calster B et al. (2016). A calibration hierarchy for risk models was defined: from utopia to empirical
  data. J Clin Epidemiol 74.
* Vickers AJ, Elkin EB (2006). Decision curve analysis: a novel method for evaluating prediction models. Med
  Decis Making 26.
* Collins GS et al. (2024). TRIPOD+AI statement. BMJ 385.
