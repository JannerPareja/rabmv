"""Loading of the modelling table, main window, input of each variant and outer splits."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from rabmv import Schema, evaluate_eligibility, harmonize

from . import config as C


class DataError(Exception):
    """Problem with the input data; the message only contains counts, never rows."""


def sha256(path):
    """SHA-256 fingerprint (hexadecimal) of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass
class Data:
    """Everything the tasks need, loaded once per process.

    Attributes
    ----------
    X : pandas.DataFrame
        Core with the Competencia 1 encoding (input of EBM and RabMV).
    X_common_education : pandas.DataFrame
        ``X`` with education replaced by the registry's (common resolution, NaN = missing or not measured):
        input of the diagnostic variant.
    X_registry : pandas.DataFrame
        Eligible concepts in their common resolution, with NaN where there is no measurement (registry-aware
        RabMV).
    categorical : list of int
        Positions of the nominal columns of ``X_registry``.
    y : numpy.ndarray
    sex : numpy.ndarray
        0 = woman, 1 = man (for the equity metrics).
    fingerprint : str
    year : numpy.ndarray
        Year of notification (temporal validation of phase 6).
    """

    X: pd.DataFrame
    X_common_education: pd.DataFrame
    X_registry: pd.DataFrame
    categorical: list
    y: np.ndarray
    sex: np.ndarray
    fingerprint: str
    year: np.ndarray = None

    def input(self, variant):
        """Input matrix of a variant."""
        return {'RabMV_registry': self.X_registry, 'EBM_registry': self.X_registry,
                'RabMV_common_education': self.X_common_education}.get(variant, self.X)


def load(data_path, schema_folder, check_fingerprint=True):
    """Read datos_modelado.csv, check its fingerprint, apply the window and build both inputs.

    Parameters
    ----------
    data_path : str or pathlib.Path
        datos_modelado.csv of etl.py version 2.
    schema_folder : str or pathlib.Path
        Folder with ``catalog.toml`` and ``mappings/`` (Valledupar schema).
    check_fingerprint : bool
        False only in the tests with synthetic data.

    Returns
    -------
    Data

    Raises
    ------
    DataError
    """
    data_path, schema_folder = Path(data_path), Path(schema_folder)
    fingerprint = sha256(data_path)
    if check_fingerprint and fingerprint != C.DATA_SHA256:
        raise DataError(f'the fingerprint of {data_path.name} ({fingerprint[:16].upper()}) is not that of etl.py '
                        f'version 2 ({C.DATA_SHA256[:16].upper()})')
    # Inferred types: codes saved as 2.0 arrive as numbers (the mapping reads them as 2).
    df = pd.read_csv(data_path, dtype={'item': str, 'fecha_not': str, 'tip_ss': str})
    missing = [c for c in C.CORE + ['item', 'fecha_not', C.TARGET] if c not in df.columns]
    if missing:
        raise DataError(f'missing columns: {missing}')
    year = pd.to_datetime(df['fecha_not'], format='ISO8601', errors='coerce').dt.year
    if year.isna().any():
        raise DataError(f'{int(year.isna().sum())} case(s) with an invalid fecha_not')
    inside = year.between(*C.WINDOW)
    df = df[inside].set_index('item')
    year = year[inside].to_numpy().astype(int)
    y = df[C.TARGET].to_numpy().astype(int)
    if check_fingerprint and (len(df) != C.EXPECTED_EVENTS['cases'] or y.sum() != C.EXPECTED_EVENTS['completed']):
        raise DataError(f'window {C.WINDOW}: {len(df)} cases and {int(y.sum())} completed suicides '
                        f'(expected {C.EXPECTED_EVENTS})')

    X = df[C.CORE].copy()

    # Registry-aware input: the same harmonization as the phase-4 audit
    schema = Schema.from_folder(schema_folder / 'catalog.toml', schema_folder / 'mappings')
    source = df.copy()
    source['form'] = [C.FORM_BY_CLASS[v] for v in y]
    source['date'] = source['fecha_not']
    harmonized = harmonize(source, schema, form_column='form', date_column='date')
    elig = evaluate_eligibility(harmonized, pd.Series(y, index=df.index), schema.catalog, mode='strict')
    if set(elig.eligible) != set(C.CONCEPTS):
        raise DataError(f'the eligible concepts {sorted(elig.eligible)} are not the pre-specified ones '
                        f'{sorted(C.CONCEPTS)}')
    matrix = elig.matrix(harmonized)
    X_registry = matrix.X[C.CONCEPTS]
    categorical = [i for i, c in enumerate(X_registry.columns) if matrix.types[c] == 'nominal']
    if not X_registry.index.equals(X.index):
        raise DataError('the registry-aware matrix is not aligned with the modelling table')
    X_common = X.copy()
    X_common['escolaridad'] = X_registry['escolaridad'].to_numpy(dtype=float)
    return Data(X=X, X_common_education=X_common, X_registry=X_registry, categorical=categorical, y=y,
                sex=X[C.SENSITIVE_COLUMN].to_numpy().astype(int), fingerprint=fingerprint, year=year)


def splits(y):
    """Outer folds of the 5×5 (RepeatedStratifiedKFold with the study seed).

    The 5 repetitions are generated so that folds 0–4 are exactly repetition 1, which will later be used
    together with repetitions 2–5.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
    """
    cv = RepeatedStratifiedKFold(n_splits=C.OUTER, n_repeats=C.REPETITIONS, random_state=C.SEED)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y)]
