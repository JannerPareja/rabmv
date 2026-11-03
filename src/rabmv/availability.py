"""Harmonization and per-case availability matrix (layer 3).

Translates the data as they come from each form into catalog concepts and marks, case by case and
concept by concept, whether the concept was measured, measured but left without data, or not measured
because the form in force for that case does not ask for it.

Statuses per case and concept:

``measured``
    The form asks for it and there is a value (a blank declared as "No" counts as measured with value 0).
``missing``
    The form asks for it, but the field came empty and the mapping declares ``blank = "missing"``.
``not_measured``
    The version of the form in force for that case does not have the concept.

No data error is reported with rows: only with counts and with unrecognized category values, so that
individual cases are never exposed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import SchemaError
from .schema.normalization import normalize_value

MEASURED, MISSING, NOT_MEASURED = 'measured', 'missing', 'not_measured'
STATUSES = (MEASURED, MISSING, NOT_MEASURED)
COMPLETE = 'complete'


@dataclass(frozen=True)
class Harmonized:
    """Data translated into concepts, with their availability.

    Attributes
    ----------
    values : pandas.DataFrame
        Cases × concepts with canonical codes; NaN when the concept was not measured or is missing.
    status : pandas.DataFrame
        Cases × concepts with ``'measured'``, ``'missing'`` or ``'not_measured'``.
    equivalence : pandas.DataFrame
        Cases × concepts with the equivalence declared in the mapping in force (``None`` if not measured).
    resolution : pandas.DataFrame
        Cases × concepts with the mapping resolution (``'complete'`` or the name of the grouping;
        ``None`` if not measured).
    form, version : pandas.Series
        Form and version in force for each case.
    """

    values: pd.DataFrame
    status: pd.DataFrame
    equivalence: pd.DataFrame
    resolution: pd.DataFrame
    form: pd.Series
    version: pd.Series

    def summary(self, suppress_below=5):
        """Availability counts by form, version and concept (no individual rows).

        Parameters
        ----------
        suppress_below : int, default=5
            Counts between 1 and this value minus 1 are shown as ``'<5'`` (or the given threshold).
            0 disables suppression (internal use only, never for sharing).

        Returns
        -------
        pandas.DataFrame
            One row per form, version and concept, with the equivalence and the count of each status.
        """
        rows = []
        groups = pd.DataFrame({'form': self.form, 'version': self.version})
        for (form, version), idx in groups.groupby(['form', 'version'], sort=True).groups.items():
            for c in self.status.columns:
                st = self.status.loc[idx, c]
                eq = self.equivalence.loc[idx, c].dropna()
                rows.append({'form': form, 'version': version, 'concept': c,
                             'equivalence': eq.iloc[0] if len(eq) else 'not_measured',
                             'resolution': (self.resolution.loc[idx, c].dropna().iloc[0]
                                            if len(eq) else None),
                             'cases': len(idx), **{s: int((st == s).sum()) for s in STATUSES}})
        table = pd.DataFrame(rows)
        if suppress_below:
            for col in ('cases',) + STATUSES:
                table[col] = table[col].map(lambda n: f'<{suppress_below}' if 0 < n < suppress_below else n)
        return table


def read_dates(series):
    """Dates by day: date-typed columns or ISO 8601 text (YYYY-MM-DD, with or without time).

    It never guesses the order of day and month: anything that is neither a date nor ISO becomes NaT
    (the caller reports it as an invalid date).

    Parameters
    ----------
    series : pandas.Series

    Returns
    -------
    pandas.Series of dtype datetime64, normalized to the day
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        dates = series
    else:
        dates = pd.to_datetime(series, format='ISO8601', errors='coerce')
    return dates.dt.normalize()


def harmonize(data, schema, form_column='form', date_column='date'):
    """Translate the source data into concepts and build the availability matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per case, with the columns named by the mappings, the form and the date of the case.
    schema : rabmv.schema.Schema
    form_column, date_column : str
        Columns with the form identifier and the date that decides the version in force.

    Returns
    -------
    Harmonized

    Raises
    ------
    SchemaError
        If columns are missing, dates are invalid, cases have no version in force, category values are
        not recognized by the mapping or numbers are out of range. All problems are reported at once,
        with counts and without individual rows.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas.DataFrame')
    problems = []
    for col in (form_column, date_column):
        if col not in data.columns:
            problems.append(f'column «{col}» is missing from the data')
    if problems:
        raise SchemaError(problems, 'data')

    if not data.index.is_unique:
        raise SchemaError(['the data index has repeated values: each case must have a unique index'], 'data')
    index = data.index
    forms = data[form_column].astype('string')
    dates = read_dates(data[date_column])
    n_bad = int(dates.isna().sum())
    if n_bad:
        problems.append(f'{n_bad} case(s) with an empty or invalid date in «{date_column}» (date-typed columns '
                        'or text in YYYY-MM-DD format are accepted)')

    version = pd.Series(pd.NA, index=index, dtype='string')
    assigned = []                                    # (mapping, cases in which it is in force)
    for form_name in sorted(set(forms.dropna())):
        mappings = [m for m in schema.mappings if m.form == form_name]
        of_form = forms == form_name
        if not mappings:
            problems.append(f'{int(of_form.sum())} case(s) of form «{form_name}», which is not in the schema')
            continue
        for m in mappings:
            in_force = of_form.fillna(False).astype(bool) & (dates >= pd.Timestamp(m.valid_from))
            if m.valid_to is not None:
                in_force &= dates <= pd.Timestamp(m.valid_to)
            in_force = in_force.fillna(False).astype(bool)
            if in_force.any():
                assigned.append((m, in_force))
                version[in_force] = m.version
        without = of_form.fillna(False).astype(bool) & dates.notna() & version.isna()
        if without.any():
            problems.append(f'{int(without.sum())} case(s) of form «{form_name}» with a date on which no '
                            'version of the mapping was in force')
    n_no_form = int(forms.isna().sum())
    if n_no_form:
        problems.append(f'{n_no_form} case(s) without a form in «{form_column}»')

    catalog = schema.catalog
    values, status, equivalence, resolution = {}, {}, {}, {}
    for cid, concept in catalog.concepts.items():
        dtype = 'object' if concept.type == 'nominal' else 'float64'
        values[cid] = pd.Series(np.nan, index=index, dtype=dtype)
        status[cid] = pd.Series(NOT_MEASURED, index=index, dtype='object')
        equivalence[cid] = pd.Series(None, index=index, dtype='object')
        resolution[cid] = pd.Series(None, index=index, dtype='object')

    for m, rows in assigned:
        for cid, mc in m.concepts.items():
            where = f'form «{m.form}» {m.version}, concept «{cid}» (field «{mc.field}»)'
            if mc.field not in data.columns:
                problems.append(f'{where}: the column does not exist in the data')
                continue
            concept = catalog[cid]
            raw = data.loc[rows, mc.field]
            norm = raw.map(normalize_value)
            empty = norm.isna()
            if concept.type == 'numeric':
                code = pd.to_numeric(raw.where(~empty), errors='coerce')
                non_numeric = int((~empty & code.isna()).sum())
                if non_numeric:
                    problems.append(f'{where}: {non_numeric} non-numeric value(s)')
                out = pd.Series(False, index=code.index)
                if concept.minimum is not None:
                    out |= code < concept.minimum
                if concept.maximum is not None:
                    out |= code > concept.maximum
                if int(out.sum()):
                    problems.append(f'{where}: {int(out.sum())} value(s) outside the range '
                                    f'[{concept.minimum}, {concept.maximum}]')
            else:
                code = norm.map(lambda v, r=mc.recoding: r.get(v) if v is not None else None)
                unknown = ~empty & code.isna()
                if unknown.any():
                    if mc.others == 'no':
                        code = code.where(~unknown, 0)
                    else:
                        count = norm[unknown].value_counts()
                        listing = ', '.join(f'«{v}» ({n})' for v, n in count.head(10).items())
                        extra = f' and {len(count) - 10} more' if len(count) > 10 else ''
                        problems.append(f'{where}: values the recoding does not recognize: {listing}{extra}')
            if mc.blank == 'no':
                code = code.where(~empty, 0)
                st = pd.Series(MEASURED, index=code.index, dtype='object')
            else:
                st = pd.Series(np.where(empty, MISSING, MEASURED), index=code.index, dtype='object')
            if concept.type != 'nominal':
                code = pd.to_numeric(code, errors='coerce').astype('float64')
            else:
                code = code.astype('object')
            values[cid].loc[rows] = code.where(st == MEASURED)
            status[cid].loc[rows] = st
            equivalence[cid].loc[rows] = mc.equivalence
            resolution[cid].loc[rows] = mc.resolution or COMPLETE

    if problems:
        raise SchemaError(problems, 'data')
    return Harmonized(values=pd.DataFrame(values), status=pd.DataFrame(status),
                      equivalence=pd.DataFrame(equivalence), resolution=pd.DataFrame(resolution),
                      form=forms.astype('object'), version=version.astype('object'))
