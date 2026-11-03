"""Simulation v2 of RabMV (plan in ``simulation_v2/PLAN.md``).

Speed only, no change of any output: the simulation harmonizes about a quarter of a million rows per task,
and ``rabmv``'s text normalization of source values (a pure function) dominated the time. Inside the
simulation it is memoized by (type, value); ``tests/test_simulation_v2.py`` checks that the harmonized
data are identical with and without the memo.
"""
import rabmv.availability as _availability
import rabmv.checks as _checks
from rabmv.schema.normalization import normalize_value as _normalize_value

_MEMO = {}


def _memo_normalize(value):
    key = (type(value), value)
    try:
        return _MEMO[key]
    except (KeyError, TypeError):
        out = _normalize_value(value)
        try:
            if len(_MEMO) < 1_000_000:
                _MEMO[key] = out
        except TypeError:
            pass
        return out


_availability.normalize_value = _memo_normalize
_checks.normalize_value = _memo_normalize
