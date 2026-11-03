"""Tests of harmonization and the per-case availability matrix (layer 3).

They use a small synthetic table with the two example forms (356 in two versions and 105). The values are
made up; there are no real data.
"""
import tomllib

import numpy as np
import pandas as pd
import pytest

from rabmv.availability import MEASURED, MISSING, NOT_MEASURED, harmonize
from rabmv.schema import Mapping, Schema, SchemaError
from tests.test_schema import MAPPING_105, with_change

# The «schema» fixture (356 V02 of 2016, 356 V03 and 105 V09) is in tests/conftest.py.


def table():
    """Six synthetic cases: three of form 356 (2016, 2018, 2019) and three of form 105 (2021, 2021, 2022)."""
    return pd.DataFrame({
        'form': ['356', '356', '356', '105_cesar', '105_cesar', '105_cesar'],
        'date': ['2016-12-31 15:00', '2018-05-02', '2019-07-10', '2021-01-15', '2021-03-03', '2022-11-30'],
        'age': [30, 25, 41, 45, 52.0, 19],
        'sex': ['F', 'M', 'F', 'Male', 'MALE', 'Female'],
        'school_level': [np.nan, 3.0, np.nan, np.nan, np.nan, np.nan],
        'education': [None, None, None, 'Secondary', 'Primary', None],
        'depression': [1, 2, 2, np.nan, np.nan, np.nan],
        'partner_conflict': [2, 1, 2, np.nan, np.nan, np.nan],
        'vulnerability_factor': [None, None, None, 'Depression or psychiatric disorder', 'Substance use', None],
    }, index=[101, 102, 103, 201, 202, 203])


def text(series):
    """List of a text column with empty values as None (pandas may store them as NaN)."""
    return [v if isinstance(v, str) else None for v in series]


def test_version_in_force_per_case(schema):
    """Each case takes the version in force on its date; a time within the last day is still in force."""
    a = harmonize(table(), schema)
    assert a.version.tolist() == ['V02', 'V03', 'V03', 'V09', 'V09', 'V09']
    assert a.form.tolist() == ['356', '356', '356', '105_cesar', '105_cesar', '105_cesar']


def test_education_statuses(schema):
    """Not measured in 2016, measured, missing and measured at aggregated resolution in form 105."""
    a = harmonize(table(), schema)
    assert a.status['education'].tolist() == [NOT_MEASURED, MEASURED, MISSING, MEASURED, MEASURED, MISSING]
    v = a.values['education']
    assert v[102] == 3 and v[201] == 3 and v[202] == 2
    assert v[[101, 103, 203]].isna().all()
    assert text(a.resolution['education']) == [None, 'complete', 'complete',
                                               'five_levels', 'five_levels', 'five_levels']
    assert text(a.equivalence['education']) == [None, 'exact', 'exact', 'aggregated', 'aggregated', 'aggregated']


def test_concepts_not_measured_by_the_form(schema):
    """Form 105 does not measure partner_conflict and form 356 does not measure psychiatric_history."""
    a = harmonize(table(), schema)
    assert (a.status.loc[[201, 202, 203], 'partner_conflict'] == NOT_MEASURED).all()
    assert (a.status.loc[[101, 102, 103], 'psychiatric_history'] == NOT_MEASURED).all()
    assert a.values.loc[[201, 202, 203], 'partner_conflict'].isna().all()


def test_single_choice_with_others_equal_to_no(schema):
    """Vulnerability factor: the chosen option gives 1; another option or a blank gives 0 (measured)."""
    a = harmonize(table(), schema)
    for c in ('depression', 'psychiatric_history'):
        assert a.values.loc[[201, 202, 203], c].tolist() == [1.0, 0.0, 0.0]
        assert (a.status.loc[[201, 202, 203], c] == MEASURED).all()
    assert a.values.loc[[101, 102, 103], 'depression'].tolist() == [1.0, 0.0, 0.0]


def test_recoding_ignores_case_and_accents(schema):
    """«MALE», «Male» and «M» reach the same canonical code."""
    a = harmonize(table(), schema)
    assert a.values['sex'].tolist() == ['F', 'M', 'F', 'M', 'M', 'F']
    assert a.values['age'].tolist() == [30, 25, 41, 45, 52, 19]


def test_summary_suppresses_small_counts(schema):
    """The summary aggregates by form, version and concept, and suppresses counts from 1 to 4."""
    a = harmonize(table(), schema)
    r = a.summary()
    row = r[(r.form == '105_cesar') & (r.concept == 'education')].iloc[0]
    assert row.equivalence == 'aggregated' and row.resolution == 'five_levels'
    assert row.measured == '<5' and row.missing == '<5' and row.not_measured == 0
    r0 = a.summary(suppress_below=0)
    row = r0[(r0.form == '105_cesar') & (r0.concept == 'education')].iloc[0]
    assert (row.measured, row.missing, row.not_measured, row.cases) == (2, 1, 0, 3)
    row = r0[(r0.form == '105_cesar') & (r0.concept == 'partner_conflict')].iloc[0]
    assert row.equivalence == 'not_measured' and row.not_measured == 3


def test_data_errors_are_reported_together_and_without_rows(schema):
    """Unrecognized categories, dates without a version, foreign forms and invalid numbers: all together."""
    d = table()
    d['age'] = d['age'].astype(object)
    d.loc[102, 'sex'] = 'X'
    d.loc[203, 'date'] = '2019-01-01'             # form 105 starts in March 2020
    d.loc[103, 'age'] = 150                       # out of range
    d.loc[101, 'age'] = 'abc'                     # non-numeric
    d.loc[999] = d.loc[201]
    d.loc[999, 'form'] = '777'                    # form that is not in the schema
    d.loc[998] = d.loc[201]
    d.loc[998, 'date'] = 'not a date'
    with pytest.raises(SchemaError) as e:
        harmonize(d, schema)
    message = str(e.value)
    assert '«x» (1)' in message
    assert 'no version of the mapping was in force' in message
    assert '1 value(s) outside the range' in message
    assert '1 non-numeric value(s)' in message
    assert 'form «777», which is not in the schema' in message
    assert 'empty or invalid date' in message
    for individual_value in ('150', 'abc', '2019-01-01'):
        assert individual_value not in message      # values of a case are never shown


def test_missing_column(schema):
    """If the mapping names a column that is not in the data, it says which one."""
    with pytest.raises(SchemaError, match='field «vulnerability_factor»\\): the column does not exist'):
        harmonize(table().drop(columns='vulnerability_factor'), schema)
    with pytest.raises(SchemaError, match='column «date» is missing'):
        harmonize(table().drop(columns='date'), schema)


def test_repeated_index(schema):
    """Each case must have a unique index."""
    d = table()
    d.index = [1, 1, 2, 3, 4, 5]
    with pytest.raises(SchemaError, match='index'):
        harmonize(d, schema)


def test_others_defaults_to_error(schema):
    """Without «others = "no"», an unlisted option is an error, not a silent "No"."""
    text_105 = with_change(MAPPING_105, 'others = "no"\n', '')
    cat = schema.catalog
    mappings = [m for m in schema.mappings if m.form != '105_cesar']
    mappings.append(Mapping.from_dict(tomllib.loads(text_105), cat))
    with pytest.raises(SchemaError, match='«substance use» \\(1\\)'):
        harmonize(table(), Schema(cat, mappings))


@pytest.mark.parametrize('convert', [
    lambda s: pd.to_datetime(s, format='ISO8601'),
    lambda s: pd.to_datetime(s, format='ISO8601').dt.date,
], ids=['datetime64', 'date_objects'])
def test_date_typed_columns(schema, convert):
    """Also accepts date-typed columns (for example, read from Excel) and date objects."""
    d = table()
    d['date'] = convert(d['date'])
    assert harmonize(d, schema).version.tolist() == ['V02', 'V03', 'V03', 'V09', 'V09', 'V09']


def test_does_not_guess_day_and_month_order(schema):
    """A date like 03/04/2021 is ambiguous (3 April or 4 March?): it is rejected, not guessed."""
    d = table()
    d.loc[201, 'date'] = '03/04/2021'
    with pytest.raises(SchemaError, match='1 case\\(s\\) with an empty or invalid date'):
        harmonize(d, schema)


def test_rejects_what_is_not_a_dataframe(schema):
    """The input must be a pandas table."""
    with pytest.raises(TypeError):
        harmonize(table().to_numpy(), schema)
