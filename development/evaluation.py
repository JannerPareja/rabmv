"""Threshold, metrics and calibration of one outer fold.

English translation of c1/evaluacion.py of Competencia 1 (fingerprint 212775606D48CDC3); the logic is the
same. The threshold is chosen by maximizing F1 on X_thresh (20 % of the outer training set, which the model
did not see when fitted) and applied to the test fold. The recalibration (Platt and isotonic) is also fitted
on X_thresh. The test fold is only used to measure.

FairLGBM in Competencia 1 used one threshold per sex (equal opportunity, Hardt, Price and Srebro, 2016):
on X_thresh the two thresholds that maximize F1 are chosen subject to the sensitivity (TPR) of women and
men differing by at most ``TPR_TOLERANCE``. The sensitivity by sex and its gap are reported for every model.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS = 1e-6
TPR_TOLERANCE = 0.05       # maximum gap in sensitivity between sexes allowed on X_thresh (per-sex thresholds)


def f1_threshold(y, p, default=0.5):
    """Threshold that maximizes F1 over the exact cuts of the precision-recall curve.

    Parameters
    ----------
    y, p : array-like
        Labels and probabilities of X_thresh.

    Returns
    -------
    float
    """
    y = np.asarray(y)
    if y.sum() == 0:
        return default
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(den), where=den > 0)
    return float(thr[int(np.argmax(f1))])


def equity_thresholds(y, p, sex, tolerance=TPR_TOLERANCE):
    """Thresholds per sex that maximize F1 with |TPR_women − TPR_men| ≤ tolerance.

    Every pair of cuts (observed values of p in each group of X_thresh) is tried. There is always a feasible
    solution (e.g. minimum thresholds: TPR = 1 in both groups).

    Returns
    -------
    dict
        {0: threshold for women, 1: threshold for men}
    """
    y, p, sex = np.asarray(y), np.asarray(p, float), np.asarray(sex).astype(int)
    table = {}
    for g in (0, 1):
        m = sex == g
        yg, pg = y[m], p[m]
        cuts = np.unique(np.concatenate([pg, [np.inf]]))
        tp = np.array([(pg[yg == 1] >= c).sum() for c in cuts])
        fp = np.array([(pg[yg == 0] >= c).sum() for c in cuts])
        npos = max(int(yg.sum()), 1)
        table[g] = (cuts, tp, fp, tp / npos)
    c0, tp0, fp0, r0 = table[0]
    c1, tp1, fp1, r1 = table[1]
    TP = tp0[:, None] + tp1[None, :]
    FP = fp0[:, None] + fp1[None, :]
    P = int(y.sum())
    f1 = np.where(TP > 0, 2 * TP / (2 * TP + FP + (P - TP)), 0.0)
    f1 = np.where(np.abs(r0[:, None] - r1[None, :]) <= tolerance + 1e-12, f1, -1.0)
    i, j = np.unravel_index(int(np.argmax(f1)), f1.shape)
    return {0: float(c0[i]), 1: float(c1[j])}


def decide(y_thresh, p_thresh, sex_thresh=None, by_sex=False):
    """Decision rule chosen on X_thresh: a single F1 threshold, or thresholds per sex."""
    if by_sex:
        return equity_thresholds(y_thresh, p_thresh, sex_thresh)
    return f1_threshold(y_thresh, p_thresh)


def _predict(p, threshold, sex=None):
    """Predicted class with a single threshold or with thresholds per sex (1 if p ≥ threshold)."""
    p = np.asarray(p, float)
    if isinstance(threshold, dict):
        u = np.where(np.asarray(sex).astype(int) == 1, threshold[1], threshold[0])
        return (p >= u).astype(int)
    return (p >= threshold).astype(int)


def _logit(p):
    """logit(p) = log(p / (1 − p)), with p clipped to [EPS, 1 − EPS] to avoid infinities."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p):
    """Calibration intercept (slope fixed at 1) and slope (logistic regression on logit p).

    Returns
    -------
    (float, float)
        Ideal: (0, 1). Negative intercept = overestimates risk; slope < 1 = probabilities too extreme.
    """
    y, lp = np.asarray(y, float), _logit(p)

    def nll(w):
        """Negative log-likelihood of the recalibration with intercept and slope."""
        z = w[0] + w[1] * lp
        return np.sum(np.logaddexp(0, z) - y * z)

    def nll0(a):
        """Negative log-likelihood with the slope fixed at 1 (intercept only)."""
        z = a[0] + lp
        return np.sum(np.logaddexp(0, z) - y * z)

    slope = minimize(nll, [0.0, 1.0], method='BFGS').x[1]
    intercept = minimize(nll0, [0.0], method='BFGS').x[0]
    return float(intercept), float(slope)


def ece(y, p, bins=10):
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    cuts = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, cuts[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def recalibrators(y_thresh, p_thresh):
    """Fit Platt (logistic on logit p) and isotonic regression on X_thresh.

    Returns
    -------
    dict of callables
        'platt' and 'isotonic': functions p → recalibrated p.
    """
    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(_logit(p_thresh).reshape(-1, 1), y_thresh)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p_thresh, y_thresh)
    return {'platt': lambda p: platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1],
            'isotonic': lambda p: iso.predict(np.asarray(p, float))}


def metrics(y, p, threshold, sex=None):
    """Discrimination, decision, calibration and equity (by sex) metrics on the test fold.

    Parameters
    ----------
    threshold : float or dict
        Single threshold or {0: women, 1: men}.
    sex : array-like, optional
        0 = woman, 1 = man; needed for thresholds per sex and equity metrics.

    Returns
    -------
    dict
    """
    y, p = np.asarray(y), np.asarray(p, float)
    pred = _predict(p, threshold, sex)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ci, cs = calibration_intercept_slope(y, p)
    return {'auc_pr': average_precision_score(y, p), 'auc_roc': roc_auc_score(y, p),
            'brier': brier_score_loss(y, p), 'ece': ece(y, p),
            'calib_intercept': ci, 'calib_slope': cs,
            'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
            'specificity': tn / (tn + fp) if tn + fp else np.nan,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn), **_equity(y, pred, sex)}


def _equity(y, pred, sex):
    """Sensitivity (TPR) by sex and its absolute gap."""
    if sex is None:
        return {}
    sex = np.asarray(sex).astype(int)
    tpr = {}
    for g, name in ((0, 'women'), (1, 'men')):
        m = (sex == g) & (y == 1)
        tpr[name] = float(pred[m].mean()) if m.any() else np.nan
    return {'tpr_women': tpr['women'], 'tpr_men': tpr['men'], 'tpr_gap': abs(tpr['women'] - tpr['men'])}


def evaluate_fold(y_thresh, p_thresh, y_test, p_test, sex_thresh=None, sex_test=None, by_sex=False):
    """Threshold on X_thresh and metrics on the test fold, with original and recalibrated probabilities.

    The decision (threshold) is taken with the original probabilities; Platt and isotonic are reported for
    calibration (Brier, ECE, intercept, slope).

    Returns
    -------
    threshold : float or dict
    rows : dict
        {'original': {...}, 'platt': {...}, 'isotonic': {...}}
    """
    u = decide(y_thresh, p_thresh, sex_thresh, by_sex)
    rows = {'original': metrics(y_test, p_test, u, sex_test)}
    for name, f in recalibrators(y_thresh, p_thresh).items():
        p_rec = f(p_test)
        u_rec = decide(y_thresh, f(p_thresh), sex_thresh, by_sex)
        rows[name] = metrics(y_test, p_rec, u_rec, sex_test)
    return u, rows
