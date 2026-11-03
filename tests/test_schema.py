"""Tests of the registry-aware schema: catalog (layer 1) and mappings per form and version (layer 2).

They use example TOML files written to a temporary folder; no data.
"""
import datetime as dt
import math

import pytest

from rabmv.schema import Catalog, Mapping, Schema, SchemaError, normalize_value

CATALOG = '''
[catalog]
name = "Suicidal behaviour (example)"
version = "2026.1"

[concepts.age]
definition = "Age in completed years"
type = "numeric"
minimum = 0
maximum = 120

[concepts.sex]
definition = "Biological sex"
type = "nominal"
categories = ["F", "M"]

[concepts.education]
definition = "Highest level of education (national dictionary codes; 0 = none)"
type = "ordinal"
levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

[concepts.education.resolutions]
five_levels = [[0], [1, 2], [3, 4, 5, 6], [7, 8], [9, 10, 11, 12]]

[concepts.depression]
definition = "Depressive disorder"
type = "binary"

[concepts.psychiatric_history]
definition = "History of psychiatric disorder"
type = "binary"

[concepts.partner_conflict]
definition = "Conflict with partner or former partner"
type = "binary"
labels = { 1 = "Yes", 0 = "No" }
'''

MAPPING_356 = '''
form = "356"
version = "V03"
valid_from = 2017-01-01

[concepts.age]
equivalence = "exact"
blank = "missing"

[concepts.sex]
equivalence = "exact"
blank = "missing"
recoding = { F = "F", M = "M" }

[concepts.education]
field = "school_level"
equivalence = "exact"
blank = "missing"
recoding = { 1 = 1, 2 = 2, 3 = 3, 5 = 5, 7 = 7, 8 = 8, 9 = 9, 10 = 10, 11 = 11, 12 = 12, 13 = 0 }

[concepts.depression]
equivalence = "exact"
response = "multiple"
blank = "no"
recoding = { 1 = 1, 2 = 0 }

[concepts.partner_conflict]
equivalence = "exact"
response = "multiple"
blank = "no"
recoding = { 1 = 1, 2 = 0 }
'''

MAPPING_105 = '''
form = "105_cesar"
version = "V09"
valid_from = 2020-03-06
note = "Complementary data specific to the Cesar health secretariat"

[concepts.age]
equivalence = "exact"
blank = "missing"

[concepts.sex]
equivalence = "exact"
blank = "missing"
recoding = { Female = "F", Male = "M" }

[concepts.education]
equivalence = "aggregated"
resolution = "five_levels"
blank = "missing"
recoding = { Primary = 2, Secondary = 3, Technical = 7, University = 9, "Out of school" = 0 }

[concepts.depression]
field = "vulnerability_factor"
equivalence = "partial"
response = "single"
blank = "no"
others = "no"
recoding = { "Depression or Psychiatric Disorder" = 1 }

[concepts.psychiatric_history]
field = "vulnerability_factor"
equivalence = "partial"
response = "single"
blank = "no"
others = "no"
recoding = { "Depression or Psychiatric Disorder" = 1 }
'''


def write(tmp_path, name, text):
    """Write a test TOML file and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return path


@pytest.fixture
def catalog(tmp_path):
    """Valid example catalog."""
    return Catalog.from_toml(write(tmp_path, 'catalog.toml', CATALOG))


def with_change(base, old, new):
    """Copy of a TOML text with one replacement (to build errors)."""
    assert old in base, old
    return base.replace(old, new, 1)


def error_of(function, *args):
    """Run and return the SchemaError it must raise."""
    with pytest.raises(SchemaError) as e:
        function(*args)
    return e.value


# ----------------------------------------------------------------------------- normalization
@pytest.mark.parametrize('value, expected', [
    (1.0, '1'), (2, '2'), (' Básica  Primaria ', 'basica primaria'), ('SÍ', 'si'),
    (None, None), (float('nan'), None), ('   ', None), (2.5, '2.5'),
])
def test_normalize_value(value, expected):
    """Case, accents, spaces, integers written as decimals and empty values."""
    assert normalize_value(value) == expected


# ----------------------------------------------------------------------------- valid catalog
def test_valid_catalog(catalog):
    """Reads types, codes, range, labels and resolutions."""
    assert catalog.name == 'Suicidal behaviour (example)' and catalog.version == '2026.1'
    assert set(catalog.concepts) == {'age', 'sex', 'education', 'depression', 'psychiatric_history',
                                     'partner_conflict'}
    assert catalog['age'].type == 'numeric' and catalog['age'].maximum == 120
    assert catalog['sex'].codes == ('F', 'M')
    assert catalog['depression'].codes == (0, 1)
    assert catalog['partner_conflict'].labels == {1: 'Yes', 0: 'No'}
    groups = catalog['education'].groups('five_levels')
    assert groups[0] == 0 and groups[2] == 1 and groups[5] == 2 and groups[8] == 3 and groups[12] == 4
    assert 'education' in catalog and 'mechanism' not in catalog


# ----------------------------------------------------------------------------- invalid catalog
@pytest.mark.parametrize('old, new, message', [
    ('type = "numeric"', 'type = "continuous"', '«type» must be one of'),
    ('definition = "Biological sex"\n', '', 'missing «definition»'),
    ('[concepts.sex]', '[concepts.Sex]', 'lower case'),
    ('levels = [0, 1, 2,', 'levels = [0, 0, 2,', 'repeated codes'),
    ('five_levels = [[0], [1, 2],', 'five_levels = [[0], [2],', 'codes without a group: [1]'),
    ('five_levels = [[0], [1, 2],', 'five_levels = [[0, 1], [1, 2],', 'codes in more than one group'),
    ('five_levels = [[0], [1, 2],', 'five_levels = [[0], [1, 2, 99],', 'do not exist in the concept'),
    ('minimum = 0\nmaximum = 120', 'minimum = 120\nmaximum = 0', '«minimum» must be less'),
    ('minimum = 0\n', 'minimum = 0\nlevels = [1, 2]\n', 'a numeric concept does not take «levels»'),
    ('definition = "Age in completed years"', 'defintion = "Age in completed years"', 'unknown key «defintion»'),
    ('categories = ["F", "M"]', 'categories = ["F", 1]', 'all integers or all text'),
    ('labels = { 1 = "Yes", 0 = "No" }', 'labels = { 1 = "Yes", 7 = "No" }', 'the label «7»'),
    ('type = "binary"\n\n[concepts.psychiatric_history]',
     'type = "binary"\nlevels = [1, 2]\n\n[concepts.psychiatric_history]',
     'a binary concept does not take «levels»'),
    ('version = "2026.1"', 'version = 2026', 'missing [catalog] version'),
    ('five_levels = [[0], [1, 2], [3, 4, 5, 6]', 'five_levels = [[0], [1, 3], [2, 4, 5, 6]', 'is not contiguous'),
])
def test_invalid_catalog(tmp_path, old, new, message):
    """Each typical error when writing a catalog is reported with a clear message."""
    e = error_of(Catalog.from_toml, write(tmp_path, 'c.toml', with_change(CATALOG, old, new)))
    assert any(message in p for p in e.problems), e.problems


def test_catalog_collects_every_problem(tmp_path):
    """A file with several errors reports all of them at once, not just the first."""
    text = with_change(CATALOG, 'type = "numeric"', 'type = "continuous"')
    text = with_change(text, 'definition = "Biological sex"\n', '')
    text = with_change(text, 'levels = [0, 1, 2,', 'levels = [0, 0, 2,')
    e = error_of(Catalog.from_toml, write(tmp_path, 'c.toml', text))
    assert len(e.problems) >= 3
    assert 'c.toml' in str(e)


def test_badly_written_toml(tmp_path):
    """A TOML syntax error is also reported as a SchemaError."""
    e = error_of(Catalog.from_toml, write(tmp_path, 'c.toml', '[catalog\nname = "x"'))
    assert 'not valid TOML' in e.problems[0]


def test_empty_catalog_and_unknown_section(tmp_path):
    """Without concepts and with sections that do not exist."""
    e = error_of(Catalog.from_toml, write(tmp_path, 'c.toml', '[catalog]\nname="x"\nversion="1"\n[other]\na=1'))
    assert any('has no concepts' in p for p in e.problems)
    assert any('unknown section [other]' in p for p in e.problems)


# ----------------------------------------------------------------------------- valid mappings
def test_valid_mapping(tmp_path, catalog):
    """Reads the form, its dates, the equivalence of each concept and the normalized recoding."""
    m = Mapping.from_toml(write(tmp_path, '105.toml', MAPPING_105), catalog)
    assert (m.form, m.version, m.valid_from, m.valid_to) == ('105_cesar', 'V09', dt.date(2020, 3, 6), None)
    edu = m.concepts['education']
    assert edu.equivalence == 'aggregated' and edu.resolution == 'five_levels' and edu.blank == 'missing'
    assert edu.field == 'education'
    assert edu.recoding == {'primary': 2, 'secondary': 3, 'technical': 7, 'university': 9, 'out of school': 0}
    assert m.concepts['sex'].recoding == {'female': 'F', 'male': 'M'}
    assert m.concepts['age'].recoding == {}
    assert 'Cesar' in m.note


def test_one_field_can_feed_several_concepts(tmp_path, catalog):
    """The vulnerability factor of form 105 feeds two concepts with single choice."""
    m = Mapping.from_toml(write(tmp_path, '105.toml', MAPPING_105), catalog)
    for c in ('depression', 'psychiatric_history'):
        assert m.concepts[c].field == 'vulnerability_factor'
        assert m.concepts[c].response == 'single'
        assert m.concepts[c].recoding == {'depression or psychiatric disorder': 1}
        assert m.concepts[c].others == 'no'
    assert m.concepts['sex'].others == 'error'


def test_concepts_not_measured(tmp_path, catalog):
    """What the mapping does not mention is not measured by that form."""
    m105 = Mapping.from_toml(write(tmp_path, '105.toml', MAPPING_105), catalog)
    m356 = Mapping.from_toml(write(tmp_path, '356.toml', MAPPING_356), catalog)
    assert m105.not_measured(catalog) == ['partner_conflict']
    assert m356.not_measured(catalog) == ['psychiatric_history']


def test_validity_includes_the_limits(tmp_path, catalog):
    """«valid_from» and «valid_to» are included; outside them the version is not in force."""
    text = with_change(MAPPING_356, 'valid_from = 2017-01-01', 'valid_from = 2017-01-01\nvalid_to = 2019-12-31')
    m = Mapping.from_toml(write(tmp_path, 'm.toml', text), catalog)
    assert m.in_force(dt.date(2017, 1, 1)) and m.in_force(dt.date(2019, 12, 31))
    assert not m.in_force(dt.date(2016, 12, 31)) and not m.in_force(dt.date(2020, 1, 1))


# ----------------------------------------------------------------------------- invalid mappings
_DEPRESSION_105 = ('field = "vulnerability_factor"\nequivalence = "partial"\nresponse = "single"\nblank = "no"\n'
                   'others = "no"\nrecoding = { "Depression or Psychiatric Disorder" = 1 }\n\n'
                   '[concepts.psychiatric_history]')


@pytest.mark.parametrize('old, new, message', [
    ('[concepts.depression]', '[concepts.sadness]', 'concept «sadness»: does not exist in the catalog'),
    ('resolution = "five_levels"\n', '', 'requires «resolution»'),
    ('resolution = "five_levels"', 'resolution = "three_levels"', 'the resolution «three_levels» does not exist'),
    ('equivalence = "exact"\nblank = "missing"\nrecoding = { Female',
     'equivalence = "exact"\nresolution = "five_levels"\nblank = "missing"\nrecoding = { Female',
     'only applies to an aggregated equivalence'),
    ('equivalence = "aggregated"', 'equivalence = "not_measured"', '«equivalence» must be one of'),
    ('University = 9', 'University = 99', '«University» = 99 is not a code'),
    ('[concepts.age]\nequivalence = "exact"\nblank = "missing"\n',
     '[concepts.age]\nequivalence = "exact"\nblank = "missing"\nrecoding = { 1 = 1 }\n',
     'a numeric concept does not take «recoding»'),
    ('recoding = { Female = "F", Male = "M" }\n', '', 'missing «recoding»'),
    ('valid_from = 2020-03-06', 'valid_from = "2020-03-06"', '«valid_from» must be an unquoted date'),
    ('valid_from = 2020-03-06', 'valid_from = 2020-03-06\nvalid_to = 2019-01-01', '«valid_to» is earlier than'),
    (_DEPRESSION_105, _DEPRESSION_105.replace('response = "single"', 'response = "one"'), '«response» must be one of'),
    ('"Out of school" = 0', '"Out of school" = 0, "PRIMARY" = 3', 'appears twice with different codes'),
    ('form = "105_cesar"', 'form = "105_cesar"\ndate = 2020-01-01', 'unknown key «date»'),
    (_DEPRESSION_105, _DEPRESSION_105.replace('blank = "no"\n', ''), '«blank» is required'),
    ('[concepts.sex]\nequivalence = "exact"\nblank = "missing"',
     '[concepts.sex]\nequivalence = "exact"\nblank = "no"', 'blank = "no" only applies to binary concepts'),
    ('[concepts.sex]\nequivalence = "exact"\nblank = "missing"',
     '[concepts.sex]\nequivalence = "exact"\nblank = "missing"\nothers = "no"',
     'others = "no" only applies to binary concepts'),
    ('[concepts.sex]\nequivalence = "exact"\nblank = "missing"',
     '[concepts.sex]\nequivalence = "exact"\nblank = "missing"\nothers = "ignore"',
     '«others» must be one of'),
])
def test_invalid_mapping(tmp_path, catalog, old, new, message):
    """Each typical error when writing a mapping is reported with a clear message."""
    text = with_change(MAPPING_105, old, new)
    e = error_of(Mapping.from_toml, write(tmp_path, 'm.toml', text), catalog)
    assert any(message in p for p in e.problems), e.problems


def test_mapping_without_concepts(tmp_path, catalog):
    """A mapping must measure at least one concept."""
    e = error_of(Mapping.from_toml, write(tmp_path, 'm.toml', 'form="x"\nversion="1"\nvalid_from=2016-01-01'),
                 catalog)
    assert any('does not measure any concept' in p for p in e.problems)


# ----------------------------------------------------------------------------- schema
def test_schema_from_folder_and_version_in_force(tmp_path):
    """Reads catalog and mappings; returns the version of each form in force on a date."""
    catalog_path = write(tmp_path, 'catalog.toml', CATALOG)
    folder = tmp_path / 'mappings'
    folder.mkdir()
    v02 = with_change(MAPPING_356, 'version = "V03"', 'version = "V02"')
    write(folder, '356_v02.toml', with_change(v02, 'valid_from = 2017-01-01',
                                              'valid_from = 2016-01-01\nvalid_to = 2016-12-31'))
    write(folder, '356_v03.toml', MAPPING_356)
    write(folder, '105.toml', MAPPING_105)
    schema = Schema.from_folder(catalog_path, folder)
    assert schema.forms() == ['105_cesar', '356']
    assert schema.mapping_for('356', dt.date(2016, 6, 1)).version == 'V02'
    assert schema.mapping_for('356', dt.date(2021, 6, 1)).version == 'V03'
    with pytest.raises(KeyError, match='no version'):
        schema.mapping_for('105_cesar', dt.date(2019, 1, 1))
    with pytest.raises(KeyError, match='is not in the schema'):
        schema.mapping_for('999', dt.date(2019, 1, 1))


def test_schema_rejects_overlapping_and_repeated_versions(tmp_path, catalog):
    """Two versions of a form cannot be in force at the same time nor repeat."""
    a = Mapping.from_toml(write(tmp_path, 'a.toml', MAPPING_356), catalog)
    b = Mapping.from_toml(write(tmp_path, 'b.toml', with_change(MAPPING_356, 'version = "V03"', 'version = "V04"')),
                          catalog)
    e = error_of(Schema, catalog, [a, b])
    assert any('overlap in time' in p for p in e.problems)
    e = error_of(Schema, catalog, [a, a])
    assert any('two mappings of version' in p for p in e.problems)


def test_schema_collects_problems_of_several_files(tmp_path):
    """If several mappings have errors, the report includes all of them."""
    catalog_path = write(tmp_path, 'catalog.toml', CATALOG)
    folder = tmp_path / 'mappings'
    folder.mkdir()
    write(folder, 'bad1.toml', with_change(MAPPING_105, 'University = 9', 'University = 99'))
    write(folder, 'bad2.toml', with_change(MAPPING_356, 'valid_from = 2017-01-01', 'valid_from = "2017"'))
    e = error_of(Schema.from_folder, catalog_path, folder)
    assert len(e.problems) == 2
    assert 'bad1.toml' in str(e) and 'bad2.toml' in str(e)


def test_nan_is_not_a_code():
    """Safeguard: a NaN from the source is never confused with a value."""
    assert normalize_value(math.nan) is None
