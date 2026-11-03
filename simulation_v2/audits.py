"""Compared audits (objective O4): RabMV's checks, the battery of Shukla et al. (2026) and a simple test.

Shukla, Mohanty and McCaslin (2026), «Auditing measurement processes in clinical registries: detecting
label–environment confounding before model development», PMLR 340:1834–1864, implemented as written in
their sections 4 and 5:

* Test A, label-prevalence instability: χ² of independence between the label and each environment variable
  (here the form and the year), Cramér's V, 500 label permutations. FAIL if p < 0.002 and V > 0.1; CAUTION
  if p < 0.01 and V > 0.1.
* Test B, conditional predictive gain: 5-fold cross-validated AUROC of a logistic regression on the features
  against one on the features plus the environment (one-hot form and year); the environment columns are
  shuffled within the folds 200 times. FAIL if p < 0.01 and ΔAUROC > 0; CAUTION if p < 0.05.
* Test C, label–era monotonicity: fraction of eras (years) without negative labels; Poisson approximation of
  P(≥ k zero-negative eras) with the negative rate of the active eras. FAIL if p < 0.001; CAUTION if p < 0.01.
  Their negative label is the rarer, under-reported one; here it is the completed suicide.
* Overall: FAIL if ≥ 2 FAIL; CAUTION if ≥ 1 FAIL or ≥ 2 CAUTION; PASS otherwise. Test A counts once per
  environment variable.
* Attribution: AUROC iid (stratified 5-fold) minus the era-blocked AUROC (leave one year out; years with
  ≥ 2 negatives evaluable), the form-blocked AUROC (leave one form out; undefined when a held-out form has a
  single class, their corollary 4) and the residualized AUROC (logit f_X − logit g_E, both out of fold).

Deviations, declared: (1) the features are the harmonized eligible matrix with NaN imputed by the median
(their audit needs no schema; the imputation only makes the logistic model fit); (2) the attribution uses
the logistic model of test B instead of the study model, to bound the computation; (3) the blocked AUROC
pools the out-of-fold predictions of the evaluable blocks.

The simple baseline is Fisher's exact test (two-sided) between «the concept is NaN» and the class, per
concept, with alpha = 0.01: the most direct audit of a missingness shortcut.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from rabmv import harmonize, run_checks

PERM_A = 500
PERM_B = 200
FOLDS = 5
V_FLOOR = 0.1
ALPHA_MISSING = 0.01


def _codes(e):
    return np.unique(np.asarray(e).astype(str), return_inverse=True)[1]


def _chi2(y, e):
    """Pearson χ² (no continuity correction) and Cramér's V of a 0/1 label against environment codes 0…k−1."""
    y, e = np.asarray(y).astype(int), np.asarray(e)
    k = int(e.max()) + 1
    t = np.bincount(y * k + e, minlength=2 * k).reshape(2, k).astype(float)
    t = t[:, t.sum(0) > 0]
    t = t[t.sum(1) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, np.nan
    exp = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    chi2 = float(((t - exp) ** 2 / exp).sum())
    return chi2, np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))


def test_a(y, e, rng):
    """Permutation-calibrated χ² between the label and one environment variable."""
    e = _codes(e)
    chi2, v = _chi2(y, e)
    if np.isnan(chi2):
        return {'stat': np.nan, 'V': np.nan, 'p': np.nan, 'flag': 'PASS'}
    perm = np.array([_chi2(rng.permutation(y), e)[0] for _ in range(PERM_A)])
    p = (1 + np.sum(perm >= chi2)) / (1 + PERM_A)
    flag = 'FAIL' if (p < 0.002 and v > V_FLOOR) else 'CAUTION' if (p < 0.01 and v > V_FLOOR) else 'PASS'
    return {'stat': float(chi2), 'V': float(v), 'p': float(p), 'flag': flag}


def _lr():
    return LogisticRegression(solver='newton-cholesky', max_iter=200)


def _oof(X, y, folds):
    out = np.empty(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[te] = _lr().fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return out


def _env(form, year):
    E = pd.get_dummies(pd.DataFrame({'form': form, 'year': year}).astype(str), drop_first=True).astype(float)
    return E.to_numpy()


def test_b(X, y, E, rng, seed):
    """Conditional predictive gain of the environment, permutation-calibrated within folds."""
    if E.shape[1] == 0:
        return {'stat': np.nan, 'p': np.nan, 'flag': 'PASS'}
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, y))
    base = roc_auc_score(y, _oof(X, y, folds))
    obs = roc_auc_score(y, _oof(np.hstack([X, E]), y, folds)) - base
    perm = []
    for _ in range(PERM_B):
        Ep = E.copy()
        for tr, te in folds:                     # shuffle the environment within each fold
            Ep[tr] = Ep[rng.permutation(tr)]
            Ep[te] = Ep[rng.permutation(te)]
        perm.append(roc_auc_score(y, _oof(np.hstack([X, Ep]), y, folds)) - base)
    p = (1 + np.sum(np.array(perm) >= obs)) / (1 + PERM_B)
    flag = 'FAIL' if (p < 0.01 and obs > 0) else 'CAUTION' if p < 0.05 else 'PASS'
    return {'stat': float(obs), 'p': float(p), 'flag': flag, 'auroc_iid': float(base), 'folds': folds}


def test_c(y, year):
    """Fraction of years without negatives (completed suicides) and its Poisson tail."""
    years = np.unique(year)
    neg = np.array([np.sum((year == t) & (y == 1)) for t in years])
    n = np.array([np.sum(year == t) for t in years])
    k = int(np.sum(neg == 0))
    active = neg > 0
    rate = neg[active].sum() / n[active].sum() if active.any() else 0.0
    lam = float(np.sum((1 - rate) ** n))
    p = 1.0 if k == 0 else float(poisson.sf(k - 1, lam))
    flag = 'FAIL' if p < 0.001 else 'CAUTION' if p < 0.01 else 'PASS'
    return {'stat': k / len(years), 'p': p, 'flag': flag}


def overall(flags):
    fails, cautions = flags.count('FAIL'), flags.count('CAUTION')
    if fails >= 2:
        return 'FAIL'
    if fails >= 1 or cautions >= 2:
        return 'CAUTION'
    return 'PASS'


def _blocked(X, y, groups, min_neg):
    """Pooled out-of-fold AUROC leaving one group out; NaN if no evaluable block."""
    out = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if np.sum(y[te] == 1) < min_neg or np.sum(y[te] == 0) < 1 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[te] = _lr().fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    ok = ~np.isnan(out)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], out[ok]))


def shukla(m_train, y, data, seed):
    """Full battery and attribution on a training registry."""
    rng = np.random.default_rng(seed + 80_000)
    X = m_train.X.fillna(m_train.X.median()).fillna(0).to_numpy()
    form = data['form'].to_numpy()
    year = pd.to_datetime(data['date']).dt.year.to_numpy()
    a_form, a_year = test_a(y, form, rng), test_a(y, year, rng)
    E = _env(form, year)
    b = test_b(X, y, E, rng, seed)
    c = test_c(y, year)
    r = {'shukla_A_form_V': a_form['V'], 'shukla_A_form_p': a_form['p'], 'shukla_A_form': a_form['flag'],
         'shukla_A_year_V': a_year['V'], 'shukla_A_year_p': a_year['p'], 'shukla_A_year': a_year['flag'],
         'shukla_B_gain': b['stat'], 'shukla_B_p': b['p'], 'shukla_B': b['flag'],
         'shukla_C_score': c['stat'], 'shukla_C_p': c['p'], 'shukla_C': c['flag']}
    r['shukla_overall'] = overall([a_form['flag'], a_year['flag'], b['flag'], c['flag']])
    folds = b.get('folds') or list(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, y))
    iid = roc_auc_score(y, _oof(X, y, folds))
    r['attr_auroc_iid'] = float(iid)
    r['attr_auroc_year_blocked'] = _blocked(X, y, year, 2)
    r['attr_auroc_form_blocked'] = _blocked(X, y, form, 2) if len(np.unique(form)) > 1 else np.nan
    if E.shape[1]:
        lf, lg = _oof(X, y, folds), _oof(E, y, folds)
        res = np.log(lf / (1 - lf)) - np.log(lg / (1 - lg))
        r['attr_auroc_residualized'] = float(roc_auc_score(y, res))
    else:
        r['attr_auroc_residualized'] = np.nan
    return r


def missingness_test(m_train, y):
    """Concepts whose missingness is associated with the class (Fisher, two-sided, alpha 0.01)."""
    flagged = []
    for c in m_train.X.columns:
        miss = m_train.X[c].isna().to_numpy()
        if miss.any() and not miss.all():
            t = [[np.sum(miss & (y == 1)), np.sum(miss & (y == 0))],
                 [np.sum(~miss & (y == 1)), np.sum(~miss & (y == 0))]]
            if fisher_exact(t)[1] < ALPHA_MISSING:
                flagged.append(c)
    return flagged


def rabmv_checks(data, y, schema_declared):
    """Concepts with an alert of RabMV's checks (pairs of check B count for both concepts)."""
    a = run_checks(harmonize(data, schema_declared), pd.Series(y), schema_declared, data=data).alerts()
    out = set()
    for c in a['concept']:
        out.update(x.strip() for x in str(c).split('&'))
    return sorted(out)
