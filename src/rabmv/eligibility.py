"""Eligibility rule, common resolution and audit table (layer 4).

A concept enters the prediction only if it was measured comparably in the sources of **both classes**:

1. There are measured cases of both classes (at least ``min_per_class`` in each).
2. Every form version that measures it has ``exact`` or ``aggregated`` equivalence. In ``'sensitivity'``
   mode ``partial`` is also admitted; the main result uses ``'strict'`` mode.
3. There is a common resolution with at least two groups: the finest one that every form can express
   (the union of their groupings). If merging them puts everything into one group, the concept no
   longer distinguishes anything and is excluded.

The matrix passed to the core contains, for each eligible concept, its value in the common resolution
and a mask: ``True`` if the case has the concept measured. Unmeasured cases and, by default, missing
values have mask ``False`` and are marginalized (they contribute 0 to that shape function).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .availability import COMPLETE, MEASURED, MISSING, NOT_MEASURED

MODES = ('strict', 'sensitivity')
ADMITTED = {'strict': ('exact', 'aggregated'), 'sensitivity': ('exact', 'aggregated', 'partial')}
MISSING_OPTIONS = ('marginalize', 'category')


def common_resolution(concept, names):
    """Groups of the finest resolution that all the given resolutions can express.

    It is the union of the partitions: two codes end up together if some form does not distinguish them.

    Parameters
    ----------
    concept : rabmv.schema.Concept
    names : iterable of str
        Resolutions used by the forms (``'complete'`` = every code separately).

    Returns
    -------
    tuple of tuple
        Groups of codes, in the order of the catalog codes.
    """
    codes = list(concept.codes)
    parent = {c: c for c in codes}

    def root(c):
        """Representative of the group of ``c``."""
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for name in set(names) - {COMPLETE}:
        for group in concept.resolutions[name]:
            for c in group[1:]:
                parent[root(c)] = root(group[0])
    groups, seen = [], {}
    for c in codes:
        r = root(c)
        if r not in seen:
            seen[r] = len(groups)
            groups.append([])
        groups[seen[r]].append(c)
    return tuple(tuple(g) for g in groups)


@dataclass(frozen=True)
class EligibleMatrix:
    """What the core receives: values in the common resolution and a measurement mask.

    Attributes
    ----------
    X : pandas.DataFrame
        Cases × eligible concepts (float). Ordinal and aggregated: group index (0, 1, …, in order);
        nominal: category index; binary: 0/1; numeric: the value. NaN where there is no measurement.
    mask : pandas.DataFrame
        Cases × eligible concepts (bool): True if the case has the concept measured.
    types : dict
        Concept → catalog type.
    groups : dict
        Concept → code groups of the common resolution (empty for numeric concepts).
    """

    X: pd.DataFrame
    mask: pd.DataFrame
    types: dict
    groups: dict

    def categorical_indices(self):
        """Positions of the nominal columns, for the ``categorical_features`` hyperparameter of RabMV."""
        return [i for i, c in enumerate(self.X.columns) if self.types[c] == 'nominal']


@dataclass(frozen=True)
class Eligibility:
    """Result of the eligibility rule.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per concept: type, equivalences per class, measurement counts per class, common
        resolution, whether it is eligible and the reason.
    eligible : list of str
    groups : dict
        Eligible concept → groups of its common resolution.
    types : dict
    mode : str
    missing : str
    """

    table: pd.DataFrame
    eligible: list
    groups: dict
    types: dict
    mode: str
    missing: str

    def audit(self, suppress_below=5):
        """Audit table for reporting (counts from 1 to 4 suppressed).

        Parameters
        ----------
        suppress_below : int, default=5
            0 disables suppression (internal use only, never for sharing).

        Returns
        -------
        pandas.DataFrame
        """
        t = self.table.copy()
        if suppress_below:
            for col in [c for c in t.columns if c.startswith(('measured_', 'missing_', 'not_measured_'))]:
                t[col] = t[col].map(lambda n: f'<{suppress_below}' if 0 < n < suppress_below else n)
        return t

    def matrix(self, harmonized):
        """Values of the eligible concepts in their common resolution, with their mask.

        Parameters
        ----------
        harmonized : rabmv.availability.Harmonized

        Returns
        -------
        EligibleMatrix
        """
        X, M = {}, {}
        for c in self.eligible:
            values = harmonized.values[c]
            status = harmonized.status[c]
            measured = status == MEASURED
            if self.missing == 'category':
                measured = measured | (status == MISSING)
            kind = self.types[c]
            if kind == 'numeric':
                x = pd.to_numeric(values, errors='coerce').astype(float)
            else:
                index = {code: i for i, g in enumerate(self.groups[c]) for code in g}
                if self.missing == 'category':
                    missing_index = float(len(self.groups[c]))
                    x = pd.Series([index.get(v, np.nan) if s == MEASURED
                                   else (missing_index if s == MISSING else np.nan)
                                   for v, s in zip(values, status)], index=values.index, dtype=float)
                else:
                    x = values.map(lambda v, ix=index: ix.get(v, np.nan)).astype(float)
            X[c] = x.where(measured)
            M[c] = measured.astype(bool)
        return EligibleMatrix(X=pd.DataFrame(X, index=harmonized.values.index),
                              mask=pd.DataFrame(M, index=harmonized.values.index),
                              types={c: self.types[c] for c in self.eligible},
                              groups={c: self.groups[c] for c in self.eligible})


def evaluate_eligibility(harmonized, y, catalog, mode='strict', min_per_class=10, missing='marginalize'):
    """Apply the eligibility rule to every concept of the catalog.

    Parameters
    ----------
    harmonized : rabmv.availability.Harmonized
    y : array-like
        Class of each case, aligned with ``harmonized`` (exactly two classes).
    catalog : rabmv.schema.Catalog
    mode : {'strict', 'sensitivity'}
    min_per_class : int, default=10
        Minimum measured cases in each class for a concept to be eligible.
    missing : {'marginalize', 'category'}
        What to do with a field that was asked but came empty: marginalize it (default) or treat it as
        one more category (sensitivity analysis only; not applicable to numeric concepts).

    Returns
    -------
    Eligibility
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if missing not in MISSING_OPTIONS:
        raise ValueError("missing must be 'marginalize' or 'category'")
    if int(min_per_class) < 1:
        raise ValueError('min_per_class must be ≥ 1')
    index = harmonized.status.index
    y = pd.Series(np.asarray(y), index=index) if not isinstance(y, pd.Series) else y
    if not y.index.equals(index):
        raise ValueError('y must have the same index as the harmonized data')
    if y.isna().any():
        raise ValueError('y has empty values')
    classes = sorted(y.unique().tolist(), key=str)
    if len(classes) != 2:
        raise ValueError(f'exactly 2 classes are needed in y (there are {len(classes)})')
    admitted = ADMITTED[mode]
    source = harmonized.form.astype(str) + ' ' + harmonized.version.astype(str)

    rows, eligible, groups, types = [], [], {}, {}
    for cid, concept in catalog.concepts.items():
        status = harmonized.status[cid]
        eq = harmonized.equivalence[cid]
        res = harmonized.resolution[cid]
        row = {'concept': cid, 'type': concept.type}
        reasons = []
        used = set()
        for k, cls in enumerate(classes):
            of_class = y == cls
            measured = of_class & (status != NOT_MEASURED)
            versions = {}
            for f, e in zip(source[measured], eq[measured]):
                versions[f] = e
            not_measuring = sorted(set(source[of_class & (status == NOT_MEASURED)]) - set(versions))
            description = [f'{f}: {e}' for f, e in sorted(versions.items())]
            description += [f'{f}: not measured' for f in not_measuring]
            row[f'equivalence_class_{cls}'] = '; '.join(description) or 'not measured'
            n_meas = int((of_class & (status == MEASURED)).sum())
            row[f'measured_class_{cls}'] = n_meas
            row[f'missing_class_{cls}'] = int((of_class & (status == MISSING)).sum())
            row[f'not_measured_class_{cls}'] = int((of_class & (status == NOT_MEASURED)).sum())
            if not versions:
                reasons.append(f'not measured in class {cls}')
                continue
            bad = sorted(f for f, e in versions.items() if e not in admitted)
            if bad:
                reasons.append(f'equivalence not admitted in {mode} mode: '
                               + ', '.join(f'{f} ({versions[f]})' for f in bad))
            if n_meas < int(min_per_class):
                reasons.append(f'fewer than {min_per_class} measured cases in class {cls}')
            used |= {r for r in res[measured] if isinstance(r, str)}
        common = None
        if not reasons:
            if concept.type == 'numeric':
                common = ()
                row['common_resolution'] = COMPLETE
            else:
                common = common_resolution(concept, used)
                if len(common) < 2:
                    reasons.append('merging the resolutions of the forms leaves the concept in a single group')
                name = COMPLETE if len(common) == len(concept.codes) else (
                    next(iter(used - {COMPLETE})) if len(used - {COMPLETE}) == 1 else 'union')
                row['common_resolution'] = f'{name} ({len(common)} groups)'
        else:
            row['common_resolution'] = None
        if missing == 'category' and concept.type == 'numeric' and not reasons:
            n_missing = sum(row[f'missing_class_{c}'] for c in classes)
            if n_missing:
                reasons.append('missing = "category" does not apply to numeric concepts')
        row['eligible'] = not reasons
        row['reason'] = 'meets the rule' if not reasons else '; '.join(reasons)
        rows.append(row)
        types[cid] = concept.type
        if not reasons:
            eligible.append(cid)
            groups[cid] = common
    return Eligibility(table=pd.DataFrame(rows), eligible=eligible, groups=groups, types=types,
                       mode=mode, missing=missing)
