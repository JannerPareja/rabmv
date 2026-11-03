# Simulation v2 of RabMV: pre-specified plan (ADEMP)

Written on 1 Nov 2026, before any evaluation seed (101–200) was run. It is frozen together with the code
(`simulation_v2/FROZEN_PROTOCOL.json`); the evaluation refuses to run if any frozen file changes.
Simulation v1 (`simulation/`, frozen on 26–27 Oct 2026) is not modified: v2 imports its population.

## Why a second simulation

Simulation v1 validated the checks and the marginalization with the form identical to the class. Phase 6
(real data) showed three things v1 does not cover:

- the real mechanism is a missingness shortcut: education is missing in 476 of 2,170 attempts against
  fewer than 5 of 181 completed suicides;
- a difference of 0.01 in AUC-PR cannot be resolved with 181 events;
- the novelty search found direct competitors:
  - Shukla, Mohanty and McCaslin 2026 (audit of label–environment confounding);
  - the Missingness Demographic Leakage Audit (Patel and Beedala 2026);
  - pattern submodels (Mercaldo and Blume 2020);
  - EBM's native missing-value bin.

The truth is known only in simulation, and a target registry can be as large as needed: with 10,000
events per target registry, the test noise is negligible.

## A. Aims

| Aim | Question |
|---|---|
| O1 | How should what was not measured be treated when the measurement pattern depends on the form and the form is associated with the class to a varying degree? |
| O2 | Which treatment keeps its performance in target registries with another form structure (transportability)? |
| O3 | What does the rule cost when the missingness carries no class information? |
| O4 | Do RabMV's checks, the battery of Shukla et al. and a simple missingness–class test detect the problem and name the right concept? |
| O5 | Does each method recover the true shape of the gap concept, and how much of its logit comes from the missingness itself? |
| O6 | Where must the rule lose? (Missingness that is real information and is stable across registries.) |

## B. Data-generating mechanism

**Population.** The population is v1's (`simulation.generator.population`: declared assumptions, same
draws). A new recorder (`simulation_v2/generator.py`) adds the following parameters.

**Form–class coupling π.**
- P(form B | completed) = π and P(form B | attempt) = 1 − π.
- π = 1 is Valledupar; π = 0.5 makes the form independent of the class.

**Gap concept.** Two arms: education (weak effect, as in the real data) and occupation (OR ≈ 2.2).

**Form B.** It keeps the other differences of v1: questions not asked, single choice, blanks recorded as
"No", and education in 5 levels.

**Mechanisms of the gap.**

| Code | Mechanism | π |
|---|---|---|
| E | Structural: form A does not measure the concept in 2016 (as form 356) | 1, 0.85, 0.5 |
| MAR | Blanks made by people in every form, about 10 %, depending on age | 0.5 |
| MNAR | Blanks that depend on the true value (low education, not employed), stable in every registry | 0.5 |

**No-difference scenarios.**
- `no_difference`: both classes with the complete form, nothing missing.
- `no_difference_year`: the same, with a completed-suicide share that grows with the year (logit 1.0 + 0.6 (year − 2019) of keeping each completed suicide). This is a label–era association without any differential measurement.

**Conditions.** 12 conditions with 180 events, plus `E_pi100_*` with 60 events (descriptive).

**Target registries.** 10,000 events each (about 790 completed suicides), with natural prevalence. Domain facts (the user, 1 Nov 2026):
- completed suicides come from vital statistics (RUAF-ND, Legal Medicine) and each municipality codes them with its own instrument;
- the Valledupar forms are becoming more similar to SIVIGILA;
- a field written once in the paper form becomes 1 for the option written and 2 for the others in the SIVIGILA structure (the mechanism of check B).

| Target | Structure |
|---|---|
| T0 | The training registry |
| T1 | An instrument for completed suicides that measures what form 356 measures (no structural gaps; people's blanks remain) |
| T2 | Form independent of the class (π = 0.5): generalization to other registries |
| T3 | Another municipality: its completed-suicide instrument does not measure the gap concept in any year |
| T4 | Convergence: from 2020 on, completed suicides are recorded with the complete instrument |

## C. Estimands and performance measures

**Primary.** Excess log-loss over the oracle (the true logit) in target T3. It is a proper scoring rule
(discrimination and calibration together).

T3 was chosen as the primary target instead of T1 in the desk test, before freezing. A model that learned
the shortcut "missing → attempt" is harmless where nothing is missing (T1). It fails where the missingness
reappears with another meaning (T3), which is the realistic case for another municipality.

**Secondary.** In T0–T4: AUC-PR, AUC-ROC, Brier, calibration intercept (calibration in the large) and
calibration slope.

**Shortcut** (T0). Each NaN is replaced by draws from the measured training values of its concept (5
draws), and two quantities are computed:
- the mean absolute change of the logit in the cases with NaN;
- the AUC-PR the model gains from the missingness itself.

**Shape recovery** (T1). RMSE of the partial dependence of the gap concept against its true effect. Both
curves are centred and weighted by the frequencies of the codes.

**Audits** (training registry), each reported as a proportion of replicates:
- detection of the gap concept;
- alerts on honest concepts (measured the same way by both forms);
- the overall verdict of Shukla et al.;
- attribution AUROCs.

**Monte Carlo.**
- 100 replicates per condition (seeds 101–200); n_sim = (SD / target MCSE)² (Morris, White and Crowther 2019).
  - In the desk test, the SDs of the paired differences were at most 0.042 for C1 (differences near 0.25) and at most 0.003 for C2 (margin 0.01).
  - Even with twice those SDs, 100 replicates give an MCSE ≤ 0.0084 for C1 and ≤ 0.0006 for C2.
  - For detection, 200 pooled replicates per criterion: all detected gives a Wilson lower limit of 0.981.
- Every mean is reported with its MCSE. Confidence intervals use t with 249 degrees of freedom; proportions use Wilson intervals.

## D. Methods

All methods except `naive` receive the same harmonized matrix: RabMV's eligible concepts in their common
resolution, with NaN where nothing was measured. Each preprocessing strategy runs on both engines (RabMV
and InterpretML's EBM 0.7.8, main effects only), so the strategy is not confounded with the engine.

**Hyperparameters.** They are fixed per engine by `simulation_v2/calibrate.py`, a rule written before
running it:
- seeds 1–20, no-difference scenario;
- the lowest mean excess log-loss among 12 combinations per engine (bins × minimum leaf × smoothing).

So each engine is tuned for the population, never for the missingness.

| Method | Description |
|---|---|
| `rabmv` | RabMV, `missing='marginalize'` (the proposal) |
| `rabmv_impute` / `ebm_impute` | Median (numeric, ordinal) or mode (nominal, binary) |
| `rabmv_indicator` / `ebm_indicator` | Imputation plus a missing indicator per column with NaN in training |
| `rabmv_mice` / `ebm_mice` | m = 5 chained-equation imputations without the outcome (deployable; Sperrin 2020, Hoogland 2020), probabilities averaged |
| `rabmv_pattern` / `ebm_pattern` | Pattern submodels (Mercaldo and Blume 2020) |
| `drop` | RabMV refitted without the gap concept |
| `ebm_native` | EBM with its native missing bin |
| `ebm_marginalize` | The same fitted EBM with RabMV's rule applied after the fit: each missing-bin score is set to the mean score of the measured cases of its term. This separates the rule from fitting with it |
| `trees_native` | HistGradientBoosting with native NaN (learned split side, as XGBoost) |
| `naive` | RabMV on the matrix of an ETL that declares everything exact (v1 baseline) |
| `rabmv_sensitivity` | RabMV, marginalized, with the `sensitivity` eligibility mode (admits partially comparable concepts); descriptive |
| `oracle` | True logit |

**Audits.**

*RabMV's checks* (A, B, C, E; α = 0.01, calibrated and frozen in v1). They run with the undeclared schema
(everything exact, one version) and with the correct one.

*The battery of Shukla et al. (2026)*, as written in their sections 4 and 5. The environment is the form and
the year.
- Test A: χ² and Cramér's V with 500 permutations.
- Test B: logistic gain with the environment shuffled within folds, 200 permutations.
- Test C: years without completed suicides, Poisson tail.
- Overall rule: FAIL if ≥ 2 FAIL; CAUTION if ≥ 1 FAIL or ≥ 2 CAUTION.
- Attribution: iid, year-blocked, form-blocked and residualized AUROC.

Declared deviations from their protocol:
- NaN in the features is imputed by the median;
- the attribution uses the logistic model of test B;
- blocked AUROCs pool the out-of-fold predictions.

*Missingness–class test.* Fisher's exact test per concept, α = 0.01.

## E. Pre-specified criteria (180 events)

| Criterion | Rule |
|---|---|
| C1 Transport | In `E_pi100_education` and `E_pi100_occupation`, target T3, the upper 95 % limit of the mean Δ excess log-loss (rabmv − comparator) is < 0 against `ebm_native`, `rabmv_indicator`, `ebm_indicator` and `trees_native` |
| C2 No cost of the rule | In `MAR_education` and `MAR_occupation`, T0, the 90 % CI of the mean Δ AUC-PR is within ±0.01 for `rabmv` − `rabmv_mice` and for `ebm_marginalize` − `ebm_native` |
| C3 Detection | Wilson lower limit of the sensitivity for the gap concept (both `E_pi100_*`, undeclared schema) ≥ 0.90, and of the per-concept specificity in `no_difference` (16 concepts) ≥ 0.99 |
| C4 Localization | Wilson lower limit of P(gap concept alerted and no honest concept alerted) ≥ 0.90 (both `E_pi100_*`, undeclared schema) |

**Descriptive:**
- every other condition, target and comparison;
- 60 events;
- the engine difference in `no_difference`;
- the Shukla battery;
- O5 and O6.

**Declared predictions** (reported whatever happens):
1. With π = 1, Shukla's test A fails in every replicate (V = 1) whether or not the schema declares the gaps. Their form-blocked attribution is undefined, because each held-out form has one class (their corollary 4).
2. In `no_difference_year`, test A can flag a label–era association that is not a measurement difference, while RabMV's checks do not alert.
3. In MNAR, `ebm_native` and the indicators beat the rule in T0–T2, because the blank is real and stable information. That is the limit of the method.

## Changes made in the desk test, before freezing

The desk test (seeds 101–103, here and on the user's PC) exists to find errors before freezing; none of
these changes uses the evaluation seeds.

1. **Primary target T3 instead of T1** (section C). The reason is mechanistic, not numerical: a shortcut on
   the missingness cannot act where nothing is missing.
2. **Both engines for every strategy and an engine calibration** (section D). With the same fixed
   hyperparameters, the two engines differed by more than the strategies did, which would have confounded
   the comparison. After calibration, the engine difference in `no_difference` remains: about 0.001 of
   excess log-loss and 0.01 of AUC-PR in favour of EBM. It is reported, not hidden.
3. **`ebm_marginalize`** (RabMV's rule applied to a fitted EBM) and **`rabmv_sensitivity`** (descriptive).
   With π < 1, the strict eligibility rule excludes the partially comparable concepts (f3–f6) even when the
   form is unrelated to the class, so it gives up real information. `rabmv_sensitivity` and `naive` measure
   that cost.
4. **Conditions renamed** from `null` to `no_difference`, because pandas reads the text "null" as a missing
   value.
5. **Faster code without changing any estimand**:
   - the imputations are computed once and shared by both engines;
   - the logistic models of Shukla's test B use the Newton–Cholesky solver;
   - Shukla's χ² is computed with numpy;
   - the text normalization of `rabmv` is memoized inside the simulation (a test checks identical
     harmonized data).
6. **Size, decided from the timing of the test level on the user's PC** (about 4.5 min per task with 4
   processes, so roughly 90 h for 3,500 tasks):
   - 100 replicates instead of 250, justified by the SDs observed above;
   - target registries of 10,000 events instead of 20,000;
   - the engine calibration was rerun with 10,000-event targets and the same rule.
     - EBM: the same combination (1024 bins, minimum leaf 2, 75 smoothing rounds).
     - RabMV: smoothing κ = 2 instead of 10 (128 bins, minimum leaf 60). The two tied within 1e-4 of excess
       log-loss (0.0076), and the rule takes the first combination of the grid.

7. **Robust calibration intercept** (found in the first 2.7 h of the evaluation on the user's PC, before any
   result was read). The unguarded Newton step for the calibration intercept overflowed in a few
   very badly calibrated models. It was replaced by Brent's method on a monotone equation, which always
   converges and gives the same value as Newton where Newton converges. The protocol was refrozen and the
   evaluation was restarted from the first task; the 103 tasks already run were discarded.

## F. Computation, outputs and rules

**Commands.**
- `python -m simulation_v2.calibrate` (about 8 min; results shipped in `results/calibration/`).
- `python -m simulation_v2.run test --processes N`: 3 seeds, for errors and timing.
- `python -m simulation_v2.run freeze`
- `python -m simulation_v2.run evaluation --processes N`
- `python -m simulation_v2.run summary --level evaluation`

**Outputs.**
- One JSON line per task.
- The partial-dependence curves of the first 10 seeds.
- Tables with MCSE, `criteria.json`, figures in English and `RESULTS.md`.

Every task is checkpointed as it finishes, and progress with the estimated time left is printed after
every task. Only synthetic data are used, so the outputs and the generated data can be published.
