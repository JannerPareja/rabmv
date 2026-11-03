"""Fixtures shared by the schema, availability and eligibility tests."""
import tomllib

import pytest

from rabmv.schema import Catalog, Mapping, Schema
from tests.test_schema import CATALOG, MAPPING_105, MAPPING_356, with_change

# 2016 version of form 356: without education (like the real form of that year)
MAPPING_356_V02 = with_change(with_change(MAPPING_356, 'version = "V03"', 'version = "V02"'),
                              'valid_from = 2017-01-01', 'valid_from = 2016-01-01\nvalid_to = 2016-12-31')
_EDU = MAPPING_356_V02.index('[concepts.education]')
MAPPING_356_V02 = MAPPING_356_V02[:_EDU] + MAPPING_356_V02[MAPPING_356_V02.index('[concepts.depression]'):]


@pytest.fixture
def schema():
    """Example schema: 356 V02 (2016), 356 V03 (from 2017) and 105 V09 (from March 2020)."""
    cat = Catalog.from_dict(tomllib.loads(CATALOG))
    mappings = [Mapping.from_dict(tomllib.loads(t), cat, source=n)
                for n, t in (('356_v02', MAPPING_356_V02), ('356_v03', MAPPING_356), ('105', MAPPING_105))]
    return Schema(cat, mappings)
