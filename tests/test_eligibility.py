"""Tests of the eligibility rule, the common resolution and the audit (layer 4).

Scenario S1 of the design (software test): with the correct schema, the eligibility of each concept is
the expected one. The two-form example (356 for attempts and 105 for completed suicides) is used,
repeated to have enough cases; the values are made up.
"""
import numpy as np
import pandas as pd
import pytest

from rabmv import RabMVClassifier
from rabmv.availability import harmonize
from rabmv.eligibility import common_resolution, evaluate_eligibility
from tests.test_availability import table


def large_data(times=5):
    """The example table repeated: 15 attempts (356) and 15 completed suicides (105), with a unique index."""
    d = pd.concat([table()] * times, ignore_index=True)
    y = (d['form'] == '105_cesar').astype(int)
    return d, y


def result(schema, mode='strict', minimum=3, missing='marginalize'):
    """Harmonize the example and apply the rule."""
    d, y = large_data()
    a = harmonize(d, schema)
    return a, y, evaluate_eligibility(a, y, schema.catalog, mode=mode, min_per_class=minimum, missing=missing)


def row(e, concept):
    """Row of the eligibility table for a concept."""
    return e.table.set_index('concept').loc[concept]


# ----------------------------------------------------------------------------- common resolution
def test_common_resolution_is_the_union_of_groupings(schema):
    """Complete with complete = every code; with five levels = the five groups."""
    edu = schema.catalog['education']
    assert common_resolution(edu, {'complete'}) == tuple((c,) for c in edu.codes)
    assert common_resolution(edu, {'complete', 'five_levels'}) == edu.resolutions['five_levels']


def test_common_resolution_merges_crossing_groupings(schema):
    """If two forms group differently, the codes that some form does not distinguish end up together."""
    edu = schema.catalog['education']
    other = {**edu.resolutions, 'other': ((0, 1), (2, 3), (4, 5, 6, 7, 8, 9, 10, 11, 12))}
    concept = type(edu)(id=edu.id, type=edu.type, definition=edu.definition, codes=edu.codes, resolutions=other)
    assert common_resolution(concept, {'five_levels', 'other'}) == ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),)


# ----------------------------------------------------------------------------- scenario S1
def test_expected_eligibility_in_strict_mode(schema):
    """Age, sex and education enter; partial concepts and those not measured in one class stay out."""
    _, _, e = result(schema)
    assert e.eligible == ['age', 'sex', 'education']
    assert 'equivalence not admitted in strict mode: 105_cesar V09 (partial)' in row(e, 'depression').reason
    assert 'not measured in class 0' in row(e, 'psychiatric_history').reason
    assert 'not measured in class 1' in row(e, 'partner_conflict').reason
    assert row(e, 'age').common_resolution == 'complete'
    assert row(e, 'education').common_resolution == 'five_levels (5 groups)'
    assert e.groups['education'] == schema.catalog['education'].resolutions['five_levels']


def test_sensitivity_mode_admits_partial_concepts(schema):
    """In sensitivity mode, depression (exact in 356, partial in 105) also enters."""
    _, _, e = result(schema, mode='sensitivity')
    assert e.eligible == ['age', 'sex', 'education', 'depression']
    assert 'psychiatric_history' not in e.eligible and 'partner_conflict' not in e.eligible


def test_the_table_describes_each_version(schema):
    """The audit says which version measured what, with which equivalence, and counts the statuses."""
    _, _, e = result(schema)
    f = row(e, 'education')
    assert f.equivalence_class_0 == '356 V03: exact; 356 V02: not measured'
    assert f.equivalence_class_1 == '105_cesar V09: aggregated'
    assert (f.measured_class_0, f.missing_class_0, f.not_measured_class_0) == (5, 5, 5)
    assert (f.measured_class_1, f.missing_class_1, f.not_measured_class_1) == (10, 5, 0)


def test_minimum_measured_cases_per_class(schema):
    """With the minimum above the measured cases, the concept stays out and the reason is given."""
    _, _, e = result(schema, minimum=6)
    assert 'education' not in e.eligible
    assert 'fewer than 6 measured cases in class 0' in row(e, 'education').reason


def test_audit_suppresses_small_counts(schema):
    """Counts from 1 to 4 are not shown in the reporting table."""
    _, _, e = result(schema)
    aud = e.audit().set_index('concept')
    assert aud.loc['education', 'measured_class_0'] == 5
    assert aud.loc['psychiatric_history', 'measured_class_1'] == 15
    d, y = large_data(times=1)
    e1 = evaluate_eligibility(harmonize(d, schema), y, schema.catalog, min_per_class=1)
    assert e1.audit().set_index('concept').loc['education', 'measured_class_0'] == '<5'


# ----------------------------------------------------------------------------- matrix and mask
def test_matrix_in_common_resolution_and_mask(schema):
    """Education becomes a group index; not measured and missing stay out of the mask."""
    a, _, e = result(schema)
    m = e.matrix(a)
    assert list(m.X.columns) == ['age', 'sex', 'education'] == list(m.mask.columns)
    first = m.X.iloc[:6]
    assert first['education'].tolist()[1] == 2.0             # code 3 → group [3, 4, 5, 6] → index 2
    assert first['education'].tolist()[3:5] == [2.0, 1.0]    # secondary → 2; primary → 1
    assert m.mask['education'].iloc[:6].tolist() == [False, True, False, True, True, False]
    assert m.X['education'][~m.mask['education']].isna().all()
    assert m.X['sex'].iloc[:6].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]   # F → 0, M → 1
    assert m.X['age'].iloc[:6].tolist() == [30.0, 25.0, 41.0, 45.0, 52.0, 19.0]
    assert m.mask['age'].all()
    assert m.types == {'age': 'numeric', 'sex': 'nominal', 'education': 'ordinal'}


def test_missing_as_category_only_in_sensitivity(schema):
    """With missing = 'category', a blank is one more group (index 5) and enters the mask."""
    a, _, e = result(schema, missing='category')
    m = e.matrix(a)
    assert m.X['education'].iloc[:6].tolist()[2] == 5.0
    assert m.mask['education'].iloc[:6].tolist() == [False, True, True, True, True, True]


# ----------------------------------------------------------------------------- invalid inputs
def test_invalid_inputs(schema):
    """Invalid classes, indices and parameters are rejected with a clear message."""
    d, y = large_data()
    a = harmonize(d, schema)
    cat = schema.catalog
    with pytest.raises(ValueError, match='exactly 2 classes'):
        evaluate_eligibility(a, np.zeros(len(y)), cat)
    with pytest.raises(ValueError, match='same index'):
        evaluate_eligibility(a, y.reset_index(drop=True).set_axis(range(100, 100 + len(y))), cat)
    with pytest.raises(ValueError, match='mode'):
        evaluate_eligibility(a, y, cat, mode='lax')
    with pytest.raises(ValueError, match='missing'):
        evaluate_eligibility(a, y, cat, missing='impute')
    with pytest.raises(ValueError, match='empty values'):
        evaluate_eligibility(a, y.astype(float).where(y.index != 0), cat)


def test_the_matrix_feeds_the_core(schema):
    """Integration: the eligible matrix enters RabMV with marginalization and nominal features as categorical."""
    a, y, e = result(schema)
    m = e.matrix(a)
    assert m.categorical_indices() == [1]                    # sex is nominal
    model = RabMVClassifier(missing='marginalize', categorical_features=m.categorical_indices(), min_samples_leaf=2,
                            n_bags=2, max_rounds=20, validation_size=0.0).fit(m.X.to_numpy(), y.to_numpy())
    c = model.contributions(m.X.to_numpy())
    not_measured = ~m.mask['education'].to_numpy()
    assert (c[not_measured, 2] == 0).all()
    np.testing.assert_allclose(model.decision_function(m.X.to_numpy()), model.intercept_ + c.sum(1), atol=1e-12)
