"""Empirical audit checks: what the data suggest and the schema did not declare.

When the form coincides with the class, a measurement difference is confounded with a real effect. These
checks cannot prove it; they detect the **coarse patterns** that a real effect rarely produces and that a
form measuring differently does produce. Each alert is an invitation to review the mapping, not an
automatic conclusion.

Checks (those of the design):

``A``  Constant in one class: the concept takes a single value in all measured cases of one class and
       varies in the other (a field that the form does not ask, filled with a default value).
``B``  Single choice: a pair of binary concepts is never marked together in one class and is in the
       other (the form only allowed one option among them). It is tested pairwise so that other questions
       of the form do not hide the pattern; concepts constant in a class are excluded, because check A
       already reports that pattern.
``C``  Blank recorded as "No": the proportion of empty fields in the source data is very different
       between classes (only possible if the source data are passed).
``E``  Undeclared version change: within a form, a year in which the field is almost never filled while
       the other years are.

Checks A, B and C use Fisher's exact test (one-sided). The default threshold ``alpha = 0.01`` was set by
the calibration of the simulation (seeds 1–50, pre-specified rule: the largest alpha with per-check
specificity ≥ 0.995 and sufficient sensitivity with 180 events), not with real data. Reports contain
only counts (those from 1 to 4 suppressed); never individual rows.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .availability import MEASURED, NOT_MEASURED, read_dates
from .schema.normalization import normalize_value

CHECKS = ('A', 'B', 'C', 'E')
ALERT, NO_ALERT, NOT_EVALUABLE = 'alert', 'no alert', 'not evaluable'


@dataclass(frozen=True)
class Checks:
    """Result of the empirical checks.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per concept (or pair) and check: result (``'alert'``, ``'no alert'`` or
        ``'not evaluable'``), p-value, counts and detail.
    parameters : dict
    """

    table: pd.DataFrame
    parameters: dict

    def alerts(self):
        """Rows with an alert."""
        return self.table[self.table['result'] == ALERT].reset_index(drop=True)

    def audit(self, suppress_below=5):
        """Table for reporting, with counts from 1 to 4 suppressed.

        Parameters
        ----------
        suppress_below : int, default=5
            0 disables suppression (internal use only, never for sharing).
        """
        t = self.table.copy()
        if suppress_below:
            for col in [c for c in t.columns if c.startswith('n_')]:
                t[col] = t[col].map(lambda n: f'<{suppress_below}'
                                    if isinstance(n, (int, np.integer)) and 0 < n < suppress_below else n)
        return t


def _fisher_less(a_yes, a_no, b_yes, b_no):
    """One-sided p-value that the proportion of "yes" in A is lower than in B."""
    return float(fisher_exact([[a_yes, a_no], [b_yes, b_no]], alternative='less')[1])


def _readable(value):
    """Value for a report: 0.0 → '0', 'F' → 'F'."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _row(concept, check, result, p=np.nan, detail='', **counts):
    """One row of the checks table."""
    return {'concept': concept, 'check': check, 'result': result, 'p': p, 'detail': detail, **counts}


def run_checks(harmonized, y, schema, data=None, date_column='date', alpha=0.01, blank_difference=0.2,
               min_cases=10, low_completeness=0.05, high_completeness=0.5):
    """Apply checks A, B, C and E.

    Parameters
    ----------
    harmonized : rabmv.availability.Harmonized
    y : array-like
        Class of each case (exactly two classes), aligned with ``harmonized``.
    schema : rabmv.schema.Schema
    data : pandas.DataFrame, optional
        Source data (the same that were harmonized). Needed for C and E, which look at blanks as they
        come from the form; without them, those checks are "not evaluable".
    date_column : str
    alpha : float, default=0.01
        Significance threshold, set by the calibration of the simulation (seeds 1–50).
    blank_difference : float, default=0.2
        Minimum difference between classes in the proportion of blanks for a C alert.
    min_cases : int, default=10
        Minimum cases per class (or per year, in E) to evaluate.
    low_completeness, high_completeness : float
        E alert if a year has completeness ≤ ``low_completeness`` and the median of the other years of
        that form is ≥ ``high_completeness``.

    Returns
    -------
    Checks
    """
    index = harmonized.status.index
    y = pd.Series(np.asarray(y), index=index) if not isinstance(y, pd.Series) else y
    if not y.index.equals(index):
        raise ValueError('y must have the same index as the harmonized data')
    classes = sorted(y.dropna().unique().tolist(), key=str)
    if len(classes) != 2 or y.isna().any():
        raise ValueError('y must have exactly 2 classes and no empty values')
    if data is not None and not data.index.equals(index):
        raise ValueError('data must have the same index as the harmonized data')
    catalog = schema.catalog
    rows = []
    constant = set()                         # concepts constant in some class (reported by A, not B)

    # ------------------------------------------------------------------ A: constant in one class
    for cid in catalog.concepts:
        measured = harmonized.status[cid] == MEASURED
        v = harmonized.values[cid]
        by_class = [v[measured & (y == c)] for c in classes]
        if any(len(s) < min_cases for s in by_class):
            rows.append(_row(cid, 'A', NOT_EVALUABLE, detail='fewer measured cases than the minimum in some class'))
            continue
        best = None
        for k in (0, 1):
            s, other = by_class[k], by_class[1 - k]
            if s.nunique() != 1:
                continue
            constant.add(cid)
            value = s.iloc[0]
            p = _fisher_less(int((other == value).sum()), int((other != value).sum()), len(s), 0)
            if best is None or p < best[0]:
                best = (p, k, value, int((other != value).sum()), len(other))
        if best is None:
            rows.append(_row(cid, 'A', NO_ALERT, detail='the concept varies in both classes'))
        else:
            p, k, value, n_different, n_other = best
            res = ALERT if p < alpha else NO_ALERT
            rows.append(_row(cid, 'A', res, p, f'equals {_readable(value)} in every case of class {classes[k]}; '
                             f'in class {classes[1 - k]} it takes other values',
                             n_constant_class=len(by_class[k]), n_other_class_different=n_different))

    # ------------------------------------------------------------------ B: single choice (pairwise)
    # If a form only allows marking one option among several factors, those factors never appear
    # together in the cases of that class. Each pair of binaries is tested: among the cases with at least
    # one of the two marked, is the proportion with both 0 in one class and not in the other?
    binaries = [c for c, con in catalog.concepts.items() if con.type == 'binary' and c not in constant
                and all(((harmonized.status[c] == MEASURED) & (y == cl)).sum() >= min_cases for cl in classes)]
    if len(binaries) < 2:
        rows.append(_row('(binary concepts)', 'B', NOT_EVALUABLE,
                         detail='fewer than 2 binaries that vary and are measured in both classes'))
    for i, a in enumerate(binaries):
        for b in binaries[i + 1:]:
            both_measured = (harmonized.status[a] == MEASURED) & (harmonized.status[b] == MEASURED)
            va, vb = harmonized.values[a] == 1, harmonized.values[b] == 1
            counts = []
            for c in classes:
                some = both_measured & (y == c) & (va | vb)
                counts.append((int((some & va & vb).sum()), int((some & (va ^ vb)).sum())))
            best = None
            for k in (0, 1):
                (two_k, one_k), (two_o, one_o) = counts[k], counts[1 - k]
                if two_k == 0 and one_k >= min_cases:
                    p = _fisher_less(two_k, one_k, two_o, one_o)
                    if best is None or p < best[0]:
                        best = (p, k)
            name = f'{a} & {b}'
            n = dict(n_both_class_0=counts[0][0], n_both_class_1=counts[1][0])
            if best is None:
                evaluable = any(d + u >= min_cases for d, u in counts)
                rows.append(_row(name, 'B', NO_ALERT if evaluable else NOT_EVALUABLE,
                                 detail=('they appear together in both classes' if evaluable
                                         else 'fewer cases with either marked than the minimum'), **n))
            else:
                p, k = best
                rows.append(_row(name, 'B', ALERT if p < alpha else NO_ALERT, p,
                                 f'in class {classes[k]} they are never marked together', **n))

    # ------------------------------------------------------------------ C and E: blanks in the source
    if data is None:
        for cid in catalog.concepts:
            rows.append(_row(cid, 'C', NOT_EVALUABLE, detail='the source data were not passed'))
            rows.append(_row(cid, 'E', NOT_EVALUABLE, detail='the source data were not passed'))
    else:
        mappings = {(m.form, m.version): m for m in schema.mappings}
        key = list(zip(harmonized.form, harmonized.version))
        years = read_dates(data[date_column]).dt.year
        for cid in catalog.concepts:
            blank = pd.Series(np.nan, index=index)                 # 1 = empty in the source; NaN = not asked
            for (f, ver), m in mappings.items():
                mc = m.concepts.get(cid)
                rows_m = np.array([k == (f, ver) for k in key])
                if mc is None or not rows_m.any() or mc.field not in data.columns:
                    continue
                vals = data.loc[rows_m, mc.field].map(normalize_value)
                blank[rows_m] = vals.isna().astype(float)
            asked = harmonized.status[cid] != NOT_MEASURED
            # C: proportion of blanks by class
            n = [(int(((y == c) & asked & (blank == 1)).sum()),
                  int(((y == c) & asked & (blank == 0)).sum())) for c in classes]
            if any(a + b < min_cases for a, b in n):
                rows.append(_row(cid, 'C', NOT_EVALUABLE, detail='fewer asked cases than the minimum in some class'))
            else:
                prop = [a / (a + b) for a, b in n]
                k = int(np.argmax(prop))
                p = _fisher_less(n[1 - k][0], n[1 - k][1], n[k][0], n[k][1])
                alert = p < alpha and prop[k] - prop[1 - k] >= blank_difference
                rows.append(_row(cid, 'C', ALERT if alert else NO_ALERT, p,
                                 f'blanks in the source: class {classes[0]} {prop[0]:.0%}, '
                                 f'class {classes[1]} {prop[1]:.0%}',
                                 n_blanks_class_0=n[0][0], n_blanks_class_1=n[1][0]))
            # E: completeness by year within each form
            worst = None
            for f in sorted(set(harmonized.form)):
                in_f = (harmonized.form == f) & asked & blank.notna()
                comp = {}
                for a in sorted(years[in_f].dropna().unique()):
                    sel = in_f & (years == a)
                    if sel.sum() >= min_cases:
                        comp[int(a)] = float(1 - blank[sel].mean())
                if len(comp) < 2:
                    continue
                for a, c_a in comp.items():
                    rest = np.median([v for b, v in comp.items() if b != a])
                    if c_a <= low_completeness and rest >= high_completeness and (worst is None or c_a < worst[2]):
                        worst = (f, a, c_a, rest)
            if worst is None:
                rows.append(_row(cid, 'E', NO_ALERT, detail='no year with a sharp drop in completeness'))
            else:
                f, a, c_a, rest = worst
                rows.append(_row(cid, 'E', ALERT, detail=f'form {f}, year {a}: completeness {c_a:.0%} '
                                 f'versus a median of {rest:.0%} in the other years'))

    table = pd.DataFrame(rows)
    return Checks(table=table, parameters=dict(alpha=alpha, blank_difference=blank_difference, min_cases=min_cases,
                                               low_completeness=low_completeness,
                                               high_completeness=high_completeness))
