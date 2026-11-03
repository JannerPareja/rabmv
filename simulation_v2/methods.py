"""Compared methods (the «M» of ADEMP) and their performance measures.

Every method except the naive one receives **the same harmonized matrix**: the eligible concepts of RabMV
in their common resolution, with NaN where the concept was not measured (by the form) or left blank (by
the person). They differ only in how they treat NaN. Each strategy that is a preprocessing step is run on
**both engines** (RabMV and InterpretML's EBM), so that the effect of the strategy is not confounded with the
engine; the hyperparameters of each engine are fixed by ``simulation_v2.calibrate`` (seeds 1–20, no-difference
scenario, before the evaluation). Main effects only.

=====================  ======================================================================================
``rabmv``              RabMV with ``missing='marginalize'`` (the proposal)
``rabmv_impute``       RabMV, NaN imputed with the training median (numeric, ordinal) or mode (nominal, binary)
``rabmv_indicator``    RabMV, imputed as above plus a missing indicator per column with NaN in training
``rabmv_mice``         RabMV on m = 5 chained-equation imputations (outcome not used, so it is deployable;
                       Sperrin et al. 2020, Hoogland et al. 2020), probabilities averaged
``rabmv_pattern``      Pattern submodels (Mercaldo and Blume 2020): for each missingness pattern, a model fitted
                       on every training case with those concepts measured, using those concepts only
``drop``               RabMV refitted without the gap concept (reference)
``ebm_native``         InterpretML EBM with its own missing-value bin (its default)
``ebm_marginalize``    The same fitted EBM with RabMV's rule applied after the fit: the score of each missing
                       bin is replaced by the mean score of the measured cases of the term (the exploratory
                       ablation of phase 6); it separates the rule from fitting with it
``ebm_impute``, ``ebm_indicator``, ``ebm_mice``, ``ebm_pattern``: the same strategies with the EBM engine
``trees_native``       Gradient-boosted trees with native missing values (scikit-learn's HistGradientBoosting
                       learns the side of each split for NaN, as XGBoost's sparsity-aware split)
``naive``              RabMV on the matrix of an ETL that declares everything exact (baseline of v1)
``oracle``             The true logit (bound)
=====================  ======================================================================================
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from rabmv import RabMVClassifier

STRATEGIES = ('impute', 'indicator', 'mice', 'pattern')
METHODS = (('rabmv',) + tuple(f'rabmv_{s}' for s in STRATEGIES) + ('drop', 'ebm_native', 'ebm_marginalize')
           + tuple(f'ebm_{s}' for s in STRATEGIES) + ('trees_native', 'naive', 'oracle'))
NO_SHORTCUT = ('naive', 'oracle', 'drop')
N_BAGS = 8
MICE_M = 5
SHORTCUT_DRAWS = 5
SHAPE_ROWS = 2000
EPS = 1e-12
CALIBRATION_FILE = Path(__file__).resolve().parent / 'results' / 'calibration' / 'hyperparameters.json'
DEFAULT_HYPER = {'rabmv': dict(learning_rate=0.05, max_leaves=3, min_samples_leaf=10, max_bins=32),
                 'ebm': dict(learning_rate=0.05, min_samples_leaf=2, max_bins=1024)}


def hyperparameters():
    """Fixed hyperparameters of each engine (from the calibration file when it exists)."""
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text(encoding='utf-8'))['chosen']
    return DEFAULT_HYPER


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Engine:
    """An additive engine on a float matrix: ``'rabmv'`` (marginalizes NaN) or ``'ebm'`` (missing bin)."""

    def __init__(self, kind, columns, types, hyper=None):
        self.kind, self.columns, self.types = kind, list(columns), types
        self.hyper = dict(hyper or hyperparameters()[kind])

    def fit(self, X, y):
        X = np.asarray(X, float)
        cat = [i for i, c in enumerate(self.columns) if self.types.get(c) == 'nominal']
        if self.kind == 'rabmv':
            self.model = RabMVClassifier(missing='marginalize', categorical_features=cat, random_state=0, n_bags=N_BAGS,
                                         early_stopping_rounds=50, **self.hyper).fit(X, y)
        else:
            from interpret.glassbox import ExplainableBoostingClassifier
            ft = ['nominal' if self.types.get(c) == 'nominal' else 'continuous' for c in self.columns]
            self.model = ExplainableBoostingClassifier(outer_bags=N_BAGS, interactions=0, n_jobs=1, random_state=0,
                                                       early_stopping_rounds=50, feature_types=ft,
                                                       feature_names=self.columns, **self.hyper)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model.fit(pd.DataFrame(X, columns=self.columns), y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if self.kind == 'rabmv':
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(pd.DataFrame(X, columns=self.columns))[:, 1]


def marginalize_ebm(engine):
    """Copy of a fitted EBM engine whose missing bins score the mean of the measured bins of their term.

    InterpretML keeps, for each main term, the missing-value score in bin 0 and the unknown-category score in
    the last bin; ``bin_weights_`` holds the training weight of every bin. Adding a constant to a whole term
    cannot change the ranking, so this is RabMV's rule (NaN contributes the mean of the measured cases).
    """
    import copy
    out = copy.deepcopy(engine)
    m = out.model
    for t in range(len(m.term_features_)):
        s, w = np.asarray(m.term_scores_[t], float), np.asarray(m.bin_weights_[t], float)
        if s.ndim != 1 or w[1:-1].sum() <= 0:
            continue
        s = s.copy()
        s[0] = np.sum(w[1:-1] * s[1:-1]) / np.sum(w[1:-1])
        m.term_scores_[t] = s
    return out


class _Fill:
    """Median (numeric, ordinal) or mode (nominal, binary) of the training measured values."""

    def __init__(self, X, types):
        self.values = {}
        for c in X.columns:
            v = X[c].dropna()
            if v.empty:
                self.values[c] = 0.0
            elif types[c] in ('nominal', 'binary'):
                self.values[c] = float(v.mode().iloc[0])
            else:
                self.values[c] = float(v.median())

    def __call__(self, X):
        return X.fillna(self.values)


class Fitted:
    """Every method fitted on one training registry."""

    def __init__(self, m_train, X_naive, y, gap):
        X = m_train.X
        self.columns, self.types = list(X.columns), m_train.types
        self.gap = gap if gap in self.columns else None
        self.X_train, self.y_train = X, y
        self.fill = _Fill(X, self.types)
        self.ind_columns = [c for c in self.columns if X[c].isna().any()]
        self.ind_types = dict(self.types, **{f'{c}__missing': 'binary' for c in self.ind_columns})
        self.levels = {c: np.sort(X[c].dropna().unique()) for c in self.columns}
        self.measured_values = {c: X[c].dropna().to_numpy() for c in self.columns}
        self.models, self.patterns = {}, {}
        Xi = self._indicator(X)
        Xf = self.fill(X)
        self.imputers, imputed = [], []
        for k in range(MICE_M):
            imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=k)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                imp.fit(X)
                imputed.append(self._round(pd.DataFrame(imp.transform(X), columns=self.columns)))
            self.imputers.append(imp)
        for eng in ('rabmv', 'ebm'):
            if eng == 'rabmv':
                self.models['rabmv'] = Engine('rabmv', self.columns, self.types).fit(X, y)
            else:
                self.models['ebm_native'] = Engine('ebm', self.columns, self.types).fit(X, y)
                self.models['ebm_marginalize'] = marginalize_ebm(self.models['ebm_native'])
            self.models[f'{eng}_impute'] = Engine(eng, self.columns, self.types).fit(Xf, y)
            self.models[f'{eng}_indicator'] = Engine(eng, list(Xi.columns), self.ind_types).fit(Xi, y)
            self.models[f'{eng}_mice'] = [Engine(eng, self.columns, self.types).fit(Xk, y) for Xk in imputed]
        cat = [i for i, c in enumerate(self.columns) if self.types[c] == 'nominal']
        self.models['trees_native'] = HistGradientBoostingClassifier(
            categorical_features=[i in cat for i in range(len(self.columns))], random_state=0).fit(X.to_numpy(), y)
        if self.gap:
            self.drop_columns = [c for c in self.columns if c != self.gap]
            self.models['drop'] = Engine('rabmv', self.drop_columns, self.types).fit(X[self.drop_columns], y)
        h = hyperparameters()['rabmv']
        self.naive_columns = list(X_naive.columns)
        self.models['naive'] = RabMVClassifier(random_state=0, n_bags=N_BAGS, early_stopping_rounds=50,
                                               **h).fit(X_naive.to_numpy(), y)

    # -- helpers
    def _indicator(self, X):
        Xi = self.fill(X)
        for c in self.ind_columns:
            Xi[f'{c}__missing'] = X[c].isna().astype(float)
        return Xi

    def _round(self, Xk):
        """Imputed values back to the observed levels of each discrete concept."""
        for c in self.columns:
            lv = self.levels[c]
            if self.types[c] != 'numeric' and len(lv):
                v = Xk[c].to_numpy()
                Xk[c] = lv[np.abs(v[:, None] - lv[None, :]).argmin(1)]
        return Xk

    def _imputed(self, X):
        """The m imputations of a matrix, cached so that both engines use the same ones (and compute them once)."""
        key = hashlib.sha1(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()).hexdigest()
        if getattr(self, '_cache_key', None) != key:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._cache = [self._round(pd.DataFrame(imp.transform(X), columns=self.columns))
                               for imp in self.imputers]
            self._cache_key = key
        return self._cache

    def _pattern_model(self, eng, observed):
        key = (eng, tuple(observed))
        if key not in self.patterns:
            cols = [c for c, o in zip(self.columns, observed) if o]
            rows = self.X_train[cols].notna().all(axis=1).to_numpy()
            self.patterns[key] = (cols, Engine(eng, cols, self.types).fit(self.X_train.loc[rows, cols],
                                                                          self.y_train[rows]))
        return self.patterns[key]

    # -- prediction
    def predict(self, method, X, X_naive=None, true_logit=None):
        """Probability of the class for a harmonized matrix ``X`` (DataFrame with ``self.columns``)."""
        X = X[self.columns]
        if method in ('rabmv', 'ebm_native', 'ebm_marginalize'):
            return self.models[method].predict_proba(X)
        if method.endswith('_impute'):
            return self.models[method].predict_proba(self.fill(X))
        if method.endswith('_indicator'):
            return self.models[method].predict_proba(self._indicator(X))
        if method.endswith('_mice'):
            out = np.zeros(len(X))
            for Xk, model in zip(self._imputed(X), self.models[method]):
                out += model.predict_proba(Xk)
            return out / len(self.imputers)
        if method.endswith('_pattern'):
            eng = method.split('_')[0]
            obs = X.notna().to_numpy()
            out = np.empty(len(X))
            for pat in np.unique(obs, axis=0):
                rows = (obs == pat).all(axis=1)
                cols, model = self._pattern_model(eng, pat)
                out[rows] = model.predict_proba(X.loc[rows, cols])
            return out
        if method == 'trees_native':
            return self.models['trees_native'].predict_proba(X.to_numpy())[:, 1]
        if method == 'drop':
            return None if not self.gap else self.models['drop'].predict_proba(X[self.drop_columns])
        if method == 'naive':
            return self.models['naive'].predict_proba(X_naive[self.naive_columns].to_numpy())[:, 1]
        if method == 'oracle':
            return 1 / (1 + np.exp(-true_logit))
        raise ValueError(method)

    def shortcut(self, method, X, y, seed):
        """How much the model draws from the missingness itself, on one registry.

        Each NaN is replaced by a value drawn from the training measured values of its concept
        (``SHORTCUT_DRAWS`` draws) and the probability is averaged over the draws. Returns the mean absolute
        logit change in the cases with some NaN and the drop of AUC-PR when the missingness information is
        removed (positive: the model gains from knowing what is missing).
        """
        X = X[self.columns]
        miss = X.isna().to_numpy()
        rows = miss.any(axis=1)
        if not rows.any() or method in NO_SHORTCUT:
            return np.nan, np.nan
        rng = np.random.default_rng(seed + 60_000)
        p0 = self.predict(method, X)
        acc = np.zeros(len(X))
        for _ in range(SHORTCUT_DRAWS):
            Xd = X.copy()
            for j, c in enumerate(self.columns):
                if miss[:, j].any():
                    Xd.loc[miss[:, j], c] = rng.choice(self.measured_values[c], miss[:, j].sum())
            acc += self.predict(method, Xd)
        p1 = acc / SHORTCUT_DRAWS
        mean_abs = float(np.mean(np.abs(_logit(p0[rows]) - _logit(p1[rows]))))
        return mean_abs, float(average_precision_score(y, p0) - average_precision_score(y, p1))

    def shape(self, method, X, true_effect, seed):
        """RMSE of the partial dependence of the gap concept against its true effect (logit, centred).

        ``true_effect``: {code of the common resolution: mean true contribution}. The partial dependence is
        computed on ``SHAPE_ROWS`` cases of a registry where the concept is measured; both curves are centred
        with the frequencies of the codes. Returns (rmse, codes, estimate, truth).
        """
        if not self.gap or method in NO_SHORTCUT:
            return np.nan, None, None, None
        X = X[self.columns]
        rng = np.random.default_rng(seed + 70_000)
        sub = X.iloc[rng.choice(len(X), min(SHAPE_ROWS, len(X)), replace=False)].copy()
        codes = np.array(sorted(true_effect))
        freq = X[self.gap].value_counts(normalize=True).reindex(codes).fillna(0).to_numpy()
        est = []
        for v in codes:
            sub[self.gap] = v
            est.append(np.mean(_logit(self.predict(method, sub))))
        est = np.array(est)
        tru = np.array([true_effect[c] for c in codes])
        est -= np.sum(freq * est) / freq.sum()
        tru -= np.sum(freq * tru) / freq.sum()
        return float(np.sqrt(np.sum(freq * (est - tru) ** 2) / freq.sum())), codes, est, tru


def calibration(y, p):
    """Calibration intercept (calibration in the large, slope fixed at 1) and slope (logistic recalibration).

    The intercept solves Σ y = Σ expit(a + logit p), a monotone equation in a, by Brent's method on
    [−50, 50]: it always converges (an unguarded Newton step can diverge when the model is very badly
    calibrated). NaN if y has a single class.
    """
    from scipy.optimize import brentq
    from scipy.special import expit
    y = np.asarray(y, float)
    lp = _logit(p)
    if y.min() == y.max():
        return np.nan, np.nan

    def g(a):
        return y.sum() - expit(a + lp).sum()

    a = brentq(g, -50.0, 50.0, xtol=1e-10)
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(lp[:, None], y)
    return float(a), float(lr.coef_[0, 0])


def metrics(y, p, p_oracle):
    """AUC-PR, AUC-ROC, Brier, log-loss, excess log-loss over the oracle and calibration."""
    ll = log_loss(y, np.clip(p, EPS, 1 - EPS))
    a, b = calibration(y, p)
    return {'aucpr': float(average_precision_score(y, p)), 'aucroc': float(roc_auc_score(y, p)),
            'brier': float(brier_score_loss(y, p)), 'logloss': float(ll),
            'excess_logloss': float(ll - log_loss(y, np.clip(p_oracle, EPS, 1 - EPS))),
            'cal_intercept': a, 'cal_slope': b}
