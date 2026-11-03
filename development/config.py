"""Configuration of the development of RabMV with real data (phase 5). The other modules only read it.

It reproduces Competencia 1 (c1/config.py) in everything that defines the comparison —seed, balancing,
optimizers, budget, inner CV, X_thresh and genetic-algorithm parameters— and changes only what the
development of RabMV requires:

* Features: the comparable core of 8 concepts (phase 4), not the 30 variables of Competencia 1.
  ``escolaridad_sin_info`` is not used: it is not a concept, and its frequency differs between forms
  (a shortcut).
* Window 2016–2022 (2,351 events): the database contains no attempts from 2013–2015.
* Only repetition 1 of the 5×5 (folds 0–4). Repetitions 2–5 are reserved for confirmation.

Design version 2 (27 Oct 2026, after the test level), decided before running the development level:

* RabMV searches interactions in the same range as EBM (0–10 pairs; grid 0, 5, 10), so both models have the
  same model class. The registry-aware variants can use them too (purified pairs, ``rabmv.core.purification``).
* Prior correction of the class weights for both models (``models.BalancedWeights``).
* Diagnostic variant ``RabMV_common_education``: the Competencia 1 matrix with education taken from the
  registry (common five-level resolution, missing = NaN marginalized), to measure how much of the difference
  between the matrix and the registry-aware input comes from education.

Phase 6 (31 Oct 2026, ``PROTOCOL_PHASE6.md``): levels ``confirmation`` (folds 5–24), ``temporal`` and ``final``,
each with no balancing and the Bayesian search for the five variants, and the constants of the pre-specified
analysis (margin, ROPE, α, calibration criterion, decision-curve thresholds). Same design version: the levels only
add tasks; phase 5 checkpoints are reused for folds 0–4.
"""
SEED = 42
DESIGN_VERSION = 3                # stored in each checkpoint; a work folder of another version is rejected
# Version 3 (27 Oct 2026): same design as 2; each checkpoint also stores the fitted model and everything needed to
# analyse it (shape functions, probabilities of every partition, recalibrated probabilities, oversampling
# diagnostics), so that figures, equations and analyses never require running the training again.

# ----------------------------------------------------------------------------- data
# Fingerprint of datos_modelado.csv produced by etl.py version 2 (27 Oct 2026, with mechanism).
DATA_SHA256 = '37afedcb577cc27f6d53adf45ff4844d5af1b5eb86ea2db1e3bfd892540dcdbe'
TARGET = 'target'
WINDOW = (2016, 2022)
EXPECTED_EVENTS = {'cases': 2351, 'completed': 181}
# Column names of the Valledupar modelling table (etl.py), encoded as in Competencia 1
SCALE = ['edad', 'estrato', 'sexo', 'ocupacion']       # StandardScaler
ORDINAL_WITH_MISSING = ['escolaridad']                 # fold median + StandardScaler
ONE_HOT = ['area', 'tip_ss', 'estado_civil']           # OneHotEncoder + StandardScaler
CORE = SCALE + ORDINAL_WITH_MISSING + ONE_HOT
SENSITIVE_COLUMN = 'sexo'                              # 0 = woman, 1 = man
# Pre-specified eligible concepts of the Valledupar catalog (phase 4): the registry-aware input must be these.
CONCEPTS = ['edad', 'sexo', 'area', 'regimen', 'ocupacion_activa', 'estrato', 'estado_civil', 'escolaridad']
FORM_BY_CLASS = {0: '356', 1: '105_cesar'}             # in Valledupar the form coincides with the class

# ----------------------------------------------------------------------------- validation
# Repetition 1 of the 5×5 of Competencia 1: RepeatedStratifiedKFold(5, 5, seed) and folds 0–4.
OUTER, REPETITIONS = 5, 5
LEVELS = {
    'test':        dict(inner=5, budget=3, folds=[0]),
    'development': dict(inner=5, budget=24, folds=[0, 1, 2, 3, 4]),
    # Phase 6 (confirmation, protocol PROTOCOL_PHASE6.md, fixed before running):
    # repetitions 2–5 of the same 5×5 (folds 5–24), never used to choose anything;
    'confirmation': dict(inner=5, budget=24, folds=list(range(5, 25))),
    # temporal validation: fit on 2016–2020, test on 2021–2022 (fold number TEMPORAL_FOLD);
    'temporal':    dict(inner=5, budget=24, folds=[100]),
    # final models: search and refit on every case of 2016–2022; out-of-fold probabilities from FINAL_OOF folds
    # give the operating threshold and an internal validation of the published model (fold number 200).
    'final':       dict(inner=5, budget=24, folds=[200]),
}
THRESHOLD_FRACTION = 0.20         # X_thresh: 20 % of the outer training set
# Phase 6: the design chosen in phase 5 (no balancing, Bayesian search) for the five variants.
PHASE6_LEVELS = ('confirmation', 'temporal', 'final')
TEMPORAL_FOLD, FINAL_FOLD = 100, 200
TEMPORAL_SPLIT_YEAR = 2020        # fit with year ≤ 2020, test with year ≥ 2021
FINAL_OOF = 5                     # folds of the out-of-fold probabilities of the final models

# ----------------------------------------------------------------------------- design
BALANCING = ['none', 'smote', 'adasyn', 'weights']
OPTIMIZERS = ['grid', 'random', 'bayes', 'genetic']
# Variants: InterpretML's EBM and RabMV with the same matrix (Competencia 1 pipeline); registry-aware RabMV
# (catalog and mappings, marginalization and categorical features), which does not admit oversampling by
# interpolation: only 'none' and 'weights'.
# RabMV_common_education (diagnostic): the C1 matrix with education in the registry's common resolution and
# its missing values marginalized (NaN does not admit oversampling by interpolation either).
# EBM_registry (added 29 Oct 2026, same design version: it only adds tasks): InterpretML's EBM with exactly the
# registry-aware input of RabMV_registry (same concepts, common resolution, nominal features as categorical, NaN
# where there is no measurement). EBM learns a value of its own for a missing value; RabMV marginalizes it. The
# comparison RabMV_registry − EBM_registry isolates that difference.
VARIANTS = ['EBM', 'RabMV', 'RabMV_registry', 'RabMV_common_education', 'EBM_registry']
BALANCING_BY_VARIANT = {'EBM': BALANCING, 'RabMV': BALANCING, 'RabMV_registry': ['none', 'weights'],
                        'RabMV_common_education': ['none', 'weights'], 'EBM_registry': ['none', 'weights']}
REGISTRY_INPUT = ('RabMV_registry', 'EBM_registry')       # variants that receive the registry-aware matrix
OVERSAMPLING_NEIGHBOURS = 5       # SMOTE / ADASYN, 1:1 ratio

# Genetic algorithm (DEAP), as in Competencia 1
GA_GENERATIONS = 3
GA_TOURNAMENT = 3
GA_P_CROSSOVER = 0.7
GA_P_MUTATION = 0.2
GA_INDPB = 0.3
GA_SIGMA = 0.2
GA_ELITE = 1


PHASE6_BALANCING, PHASE6_OPTIMIZER = 'none', 'bayes'
# SHA-256 of development/PROTOCOL_PHASE6.md (version 1, 31 Oct 2026): the analysis refuses another protocol file.
PROTOCOL_SHA256 = 'b4773364e4a3de6580fc2c8a9866361772920330a7301c9ecf99f55dbfa38859'
# Pre-specified analysis of phase 6 (PROTOCOL_PHASE6.md; changing any value requires a new protocol version)
NI_MARGIN = 0.01                  # non-inferiority margin in AUC-PR (RabMV − EBM > −0.01)
ROPE = 0.01                       # region of practical equivalence of the Bayesian correlated t-test
ALPHA = 0.05                      # one-sided for non-inferiority and superiority; two-sided (Holm) for the rest
BAYES_PROBABILITY = 0.95          # posterior probability required by the Bayesian criteria
CALIBRATION_INTERCEPT = (-0.2, 0.2)    # weak calibration of RabMV: mean calibration intercept within this range
CALIBRATION_SLOPE = (0.8, 1.2)         # and mean calibration slope within this range
DCA_THRESHOLDS = (0.05, 0.50, 0.01)    # decision curve: from, to, step
BOOTSTRAP = 2000                  # resamples for the confidence intervals of pooled quantities


def combinations(level=None):
    """List of (variant, balancing, optimizer) of the design (of a level), in a fixed order."""
    if level in PHASE6_LEVELS:
        return [(v, PHASE6_BALANCING, PHASE6_OPTIMIZER) for v in VARIANTS]
    return [(v, b, o) for v in VARIANTS for b in BALANCING_BY_VARIANT[v] for o in OPTIMIZERS]
