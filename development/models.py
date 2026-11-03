"""Variants, pipelines, balancing methods and search spaces of the development.

Reproduces c1/modelos.py, c1/balanceo.py and c1/preprocesamiento.py of Competencia 1 for EBM and RabMV:

* Pipeline: ColumnTransformer → balancing → model. Oversampling comes after the ColumnTransformer and only
  sees the set the model is fitted on (never X_thresh nor the test fold).
* Weights: w_i = n / (2 · n_class(i)), identical to class_weight='balanced', passed as sample_weight (EBM and
  RabMV have no class_weight).
* InterpretML's EBM: outer_bags=8, early_stopping_rounds=50, n_jobs=1, study seed; Competencia 1 space
  (interactions, learning_rate, max_leaves; grid of 24).
* RabMV: n_bags=8, early_stopping_rounds=50; space of EBMP v2 in Competencia 1 (learning_rate, max_leaves,
  min_samples_leaf, smoothing) plus the interactions of EBM (0–10). Grid of 24 = interactions {0, 5, 10} ×
  learning_rate {0.03, 0.06, 0.1, 0.2} × smoothing {0, 3} (min_samples_leaf 10, max_leaves 3): the grid fixes
  the secondary parameters at their defaults, as the EBM grid does, and contrasts no smoothing with strong
  smoothing; random, Bayesian and genetic search the whole continuous space.
* Class weights with prior correction (King and Zeng, 2001): fitting with weights w₁ (events) and w₀ shifts
  the logit by log(w₁/w₀); ``BalancedWeights`` subtracts it, which returns the probabilities to the scale of
  the data without changing the ranking (AUC) or the decisions (the X_thresh threshold moves with them).

The ``RabMV_registry`` variant receives the harmonized concepts (NaN = not measured or missing) without a
ColumnTransformer, with ``missing='marginalize'`` and nominal features as categorical. Oversampling by
interpolation does not apply to that input (it would create non-existent categories and does not accept NaN).
``RabMV_common_education`` (diagnostic) receives the Competencia 1 matrix with education in the registry's
common resolution: education is only scaled (no imputation) and its NaN is marginalized.
"""
import inspect
from dataclasses import dataclass, field

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rabmv import RabMVClassifier

from . import config as C


# ----------------------------------------------------------------------------- search spaces
@dataclass
class Param:
    """Hyperparameter of a search space (as in Competencia 1).

    Attributes
    ----------
    name : str
    kind : {'float', 'int', 'cat'}
    low, high : float, optional
    log : bool
    options : list
    grid : list
        Grid Search levels (the full grid has 24 points).
    """
    name: str
    kind: str
    low: float = None
    high: float = None
    log: bool = False
    options: list = field(default_factory=list)
    grid: list = field(default_factory=list)


SPACE_EBM = [
    Param('interactions', 'int', 0, 10, grid=[0, 5, 10]),
    Param('learning_rate', 'float', 0.03, 0.2, log=True, grid=[0.03, 0.06, 0.1, 0.2]),
    Param('max_leaves', 'int', 2, 4, grid=[2, 3]),
]
SPACE_RABMV = [
    Param('interactions', 'int', 0, 10, grid=[0, 5, 10]),
    Param('learning_rate', 'float', 0.03, 0.2, log=True, grid=[0.03, 0.06, 0.1, 0.2]),
    Param('max_leaves', 'int', 2, 4, grid=[3]),
    Param('min_samples_leaf', 'int', 2, 50, log=True, grid=[10]),
    Param('smoothing', 'cat', options=[0.0, 0.5, 3.0], grid=[0.0, 3.0]),
]
SPACES = {'EBM': SPACE_EBM, 'RabMV': SPACE_RABMV, 'RabMV_registry': SPACE_RABMV,
          'RabMV_common_education': SPACE_RABMV, 'EBM_registry': SPACE_EBM}
MARGINALIZING = ('RabMV_registry', 'RabMV_common_education')


def grid_size(variant):
    """Points of the Grid Search grid (= budget of the development level)."""
    return int(np.prod([len(p.grid) for p in SPACES[variant]]))


# ----------------------------------------------------------------------------- balancing
def balanced_weights(y):
    """Weights w_i = n / (k · n_class(i)), identical to class_weight='balanced'."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y], dtype=float)


def sampler(balancing):
    """Oversampling step of the pipeline (or 'passthrough'), as in Competencia 1."""
    if balancing == 'smote':
        return SMOTE(sampling_strategy='auto', k_neighbors=C.OVERSAMPLING_NEIGHBOURS, random_state=C.SEED)
    if balancing == 'adasyn':
        return ADASYN(sampling_strategy='auto', n_neighbors=C.OVERSAMPLING_NEIGHBOURS, random_state=C.SEED)
    return 'passthrough'


class BalancedWeights(ClassifierMixin, BaseEstimator):
    """Fit ``base_estimator`` with balanced sample_weight (equivalent to class_weight='balanced').

    With ``prior_correction=True`` the predicted logit is corrected by the offset that the weights introduce
    (King and Zeng, 2001, "Logistic regression in rare events data"): maximizing the weighted likelihood with
    weights w₁ for the events and w₀ for the non-events estimates logit p + log(w₁/w₀), so

        p = σ(logit p_w − log(w₁/w₀)),   with balanced weights log(w₁/w₀) = log(n₀/n₁).

    It is exact for the intercept of a correctly specified logistic model and, being a constant shift of the
    logit, it does not change the ranking of the cases (AUC-PR, AUC-ROC) nor the decisions taken with a
    threshold chosen on X_thresh. It only returns the probabilities to the scale of the data (Brier, ECE,
    calibration intercept).

    Parameters
    ----------
    base_estimator : estimator
        Must accept ``sample_weight`` in ``fit``.
    prior_correction : bool, default=True
    """

    def __init__(self, base_estimator, prior_correction=True):
        """Store the base estimator without fitting it (scikit-learn convention)."""
        self.base_estimator = base_estimator
        self.prior_correction = prior_correction

    def __sklearn_tags__(self):
        """Classifier that requires ``y``; accepts NaN if the base estimator does."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = 'classifier'
        tags.target_tags.required = True
        tags.input_tags.allow_nan = self.base_estimator.__sklearn_tags__().input_tags.allow_nan
        return tags

    def fit(self, X, y):
        """Fit a copy of the base estimator with class-balanced weights."""
        if 'sample_weight' not in inspect.signature(self.base_estimator.fit).parameters:
            raise TypeError(f'{type(self.base_estimator).__name__}.fit() does not accept sample_weight')
        self.estimator_ = clone(self.base_estimator)
        w = balanced_weights(y)
        self.estimator_.fit(X, y, sample_weight=w)
        self.classes_ = self.estimator_.classes_
        positive = np.asarray(y) == self.classes_[1]
        self.offset_ = float(np.log(w[positive][0] / w[~positive][0])) if self.prior_correction else 0.0
        return self

    def predict_proba(self, X):
        """Probabilities of the fitted estimator, with the prior correction of the weights."""
        p = self.estimator_.predict_proba(X)
        if not self.offset_:
            return p
        q = np.clip(p[:, 1], 1e-15, 1 - 1e-15)
        p1 = 1 / (1 + np.exp(-(np.log(q / (1 - q)) - self.offset_)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """Predicted class with threshold 0.5 on the corrected probability (the study uses X_thresh)."""
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


# ----------------------------------------------------------------------------- pipelines
def preprocessing(impute=True):
    """ColumnTransformer of Competencia 1 restricted to the core (not fitted).

    age, stratum, sex and occupation: StandardScaler; education: fold median + StandardScaler (with
    ``impute=False``, only StandardScaler: its NaN reach the model, which marginalizes them); area,
    insurance scheme and marital status: OneHotEncoder + StandardScaler.
    """
    education = (SkPipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]) if impute
                 else StandardScaler())
    ct = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), C.SCALE),
            ('ord', education, C.ORDINAL_WITH_MISSING),
            ('cat', SkPipeline([('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                                ('sc', StandardScaler())]), C.ONE_HOT),
        ],
        remainder='drop', verbose_feature_names_out=True)
    ct.set_output(transform='pandas')
    return ct


def estimator(variant, categorical=None):
    """Base estimator with the fixed values of the study."""
    if variant == 'EBM':
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(outer_bags=8, early_stopping_rounds=50, random_state=C.SEED, n_jobs=1)
    if variant == 'EBM_registry':                            # same input as RabMV_registry; EBM's own NaN handling
        from interpret.glassbox import ExplainableBoostingClassifier
        categorical = set(categorical or [])
        types = ['nominal' if j in categorical else 'auto' for j in range(len(C.CONCEPTS))]
        return ExplainableBoostingClassifier(outer_bags=8, early_stopping_rounds=50, random_state=C.SEED, n_jobs=1,
                                             feature_types=types)
    if variant == 'RabMV':
        return RabMVClassifier(n_bags=8, early_stopping_rounds=50, interactions=0, random_state=C.SEED)
    if variant == 'RabMV_registry':
        return RabMVClassifier(n_bags=8, early_stopping_rounds=50, interactions=0, random_state=C.SEED,
                               missing='marginalize', categorical_features=list(categorical or []))
    if variant == 'RabMV_common_education':
        return RabMVClassifier(n_bags=8, early_stopping_rounds=50, interactions=0, random_state=C.SEED,
                               missing='marginalize')
    raise ValueError(variant)


def pipeline(variant, balancing, categorical=None):
    """Complete pipeline of a combination (not fitted).

    Parameters
    ----------
    variant : {'EBM', 'RabMV', 'RabMV_registry', 'RabMV_common_education', 'EBM_registry'}
    balancing : {'none', 'smote', 'adasyn', 'weights'}
    categorical : list of int, optional
        Nominal columns of the registry-aware input.

    Returns
    -------
    imblearn.pipeline.Pipeline
    """
    if balancing not in C.BALANCING_BY_VARIANT[variant]:
        raise ValueError(f'{variant} × {balancing} does not apply')
    est = estimator(variant, categorical)
    if balancing == 'weights':
        est = BalancedWeights(est)
    if variant in C.REGISTRY_INPUT:
        return Pipeline([('bal', 'passthrough'), ('clf', est)])
    return Pipeline([('prep', preprocessing(impute=variant not in MARGINALIZING)), ('bal', sampler(balancing)),
                     ('clf', est)])


def prediction_model(fitted):
    """The fitted pipeline without its oversampling step (which only acts in fit).

    A fitted SMOTE / ADASYN keeps its neighbour search, which stores training rows; the other steps keep only
    cuts, shape functions, medians, scales and categories. Predictions are identical to those of ``fitted``.
    """
    return SkPipeline([(name, step) for name, step in fitted.steps if name != 'bal'])


def _plain(v):
    """Arrays and lists as plain numpy/python objects (readable without the model's library)."""
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def model_card(model):
    """Description of a fitted model in plain arrays: the equation, its terms and their uncertainty.

    logit p = intercept + Σ term scores (per bin of each feature, per cell of each pair) − weight offset.
    Includes the names of the input features after preprocessing and, for the weights wrapper, the offset of
    the prior correction.
    """
    clf = model.steps[-1][1]
    card = {'classifier': type(clf).__name__, 'offset': 0.0}
    if isinstance(clf, BalancedWeights):
        card['offset'] = float(clf.offset_)
        clf = clf.estimator_
        card['classifier'] = type(clf).__name__
    if len(model.steps) > 1:
        try:
            card['input_features'] = [str(c) for c in model.steps[0][1].get_feature_names_out()]
        except Exception:                                    # the card does not depend on this
            pass
    if isinstance(clf, RabMVClassifier):
        keys = ['intercept_', 'term_names_', 'term_scores_', 'term_scores_std_', 'bin_edges_', 'term_importances_',
                'best_rounds_', 'validation_history_', 'categorical_features_', 'interaction_pairs_',
                'pair_bin_edges_', 'feature_names_in_']
    else:                                                    # InterpretML EBM
        keys = ['intercept_', 'term_names_', 'term_features_', 'term_scores_', 'standard_deviations_', 'bins_',
                'feature_names_in_', 'feature_types_in_', 'bagged_scores_', 'bagged_intercept_', 'best_iteration_',
                'histogram_edges_', 'unique_val_counts_']
        try:
            card['term_importances'] = np.asarray(clf.term_importances())
        except Exception:
            pass
    for k in keys:
        if hasattr(clf, k):
            card[k.rstrip('_')] = _plain(getattr(clf, k))
    card['hyperparameters'] = {k: v for k, v in clf.get_params().items() if np.isscalar(v) or v is None}
    return card


def oversampling_diagnostics(fitted, X_fit, y_fit, tolerance=1e-9):
    """Aggregates of the synthetic cases created by SMOTE / ADASYN in the final fit (None without them).

    The sampler is re-run on the same preprocessed matrix with its fixed seed (identical cases to the fit). Per
    feature: share of synthetic cases whose value does not exist among the real cases (a value invented by
    interpolation), and, for RabMV, the bins that contain synthetic cases but no real case with the mean
    absolute score the model gives them.
    """
    steps = dict(fitted.steps)
    sampler = steps.get('bal')
    if sampler is None or sampler == 'passthrough':
        return None
    Xt = steps['prep'].transform(X_fit) if 'prep' in steps else X_fit
    Xt = np.asarray(Xt, dtype=float)
    Xr, _ = clone(sampler).fit_resample(Xt, np.asarray(y_fit))
    n = len(Xt)
    synth = Xr[n:]
    out = {'n_real': n, 'n_synthetic': len(synth), 'features': []}
    names = list(steps['prep'].get_feature_names_out()) if 'prep' in steps else [f'x{j}' for j in range(Xt.shape[1])]
    clf = steps['clf']
    clf = clf.estimator_ if isinstance(clf, BalancedWeights) else clf
    for j, name in enumerate(names):
        real = np.unique(Xt[:, j])
        pos = np.clip(np.searchsorted(real, synth[:, j]), 0, len(real) - 1)
        near = np.minimum(np.abs(real[pos] - synth[:, j]), np.abs(real[np.maximum(pos - 1, 0)] - synth[:, j]))
        row = {'feature': str(name), 'real_distinct_values': int(len(real)),
               'synthetic_off_support_share': float(np.mean(near > tolerance)) if len(synth) else 0.0}
        if isinstance(clf, RabMVClassifier) and j < len(clf.bin_edges_):
            edges = clf.bin_edges_[j]
            b_real = np.bincount(np.searchsorted(edges, Xt[:, j], side='right'), minlength=edges.size + 1)
            b_syn = np.bincount(np.searchsorted(edges, synth[:, j], side='right'), minlength=edges.size + 1)
            only = (b_syn > 0) & (b_real == 0)
            score = np.abs(clf.term_scores_[j])
            row.update(bins=int(edges.size + 1), bins_only_synthetic=int(only.sum()),
                       synthetic_in_those_bins=int(b_syn[only].sum()),
                       mean_abs_score_only_synthetic=float(score[only].mean()) if only.any() else 0.0,
                       mean_abs_score_other_bins=float(score[~only].mean()) if (~only).any() else 0.0)
        out['features'].append(row)
    return out


def prefix(balancing):
    """Prefix of the hyperparameters inside the pipeline."""
    return 'clf__base_estimator__' if balancing == 'weights' else 'clf__'


def with_prefix(params, balancing):
    """Add the pipeline prefix to a dictionary of hyperparameters."""
    pre = prefix(balancing)
    return {pre + k: v for k, v in params.items()}
