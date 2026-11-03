"""Mapping of one form in one version: how each field translates to a catalog concept (layer 2).

One TOML file per form and version. Catalog concepts that the mapping does not mention are considered
**not measured** by that form. Example::

    form = "105_cesar"
    version = "V09"
    valid_from = 2020-03-06
    valid_to = 2022-12-31          # optional: without «valid_to», still in force

    [concepts.education]
    field = "education"            # column in the source data (by default, the concept id)
    equivalence = "aggregated"     # exact | aggregated | partial
    resolution = "five_levels"     # required if aggregated: a catalog resolution
    blank = "missing"              # what an empty field means: missing | no
    recoding = { primary = 2, secondary = 3, technical = 7, university = 9, none = 0 }

    [concepts.depression]
    field = "vulnerability_factor"
    equivalence = "partial"
    response = "single"            # single | multiple (how the factors were recorded)
    blank = "no"
    others = "no"                  # field values not listed: error (default) | no
    recoding = { "depression or psychiatric disorder" = 1 }

In ``recoding``, the keys are the values as they come in the source (compared without case or accents)
and the values are canonical codes of the catalog. The same field can feed several concepts (like the
vulnerability factor of form 105); then ``others = "no"`` declares that a value of another option means
"No" for this concept. ``blank = "no"`` and ``others = "no"`` only apply to binary concepts.
"""
import datetime as dt
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SchemaError
from .normalization import normalize_value

EQUIVALENCES = ('exact', 'aggregated', 'partial')
RESPONSES = ('single', 'multiple')
BLANKS = ('missing', 'no')
OTHERS = ('error', 'no')
_MAPPING_KEYS = {'form', 'version', 'valid_from', 'valid_to', 'note', 'concepts'}
_CONCEPT_KEYS = {'field', 'equivalence', 'resolution', 'response', 'blank', 'others', 'recoding', 'note'}


@dataclass(frozen=True)
class ConceptMapping:
    """How a form measures a concept.

    Attributes
    ----------
    concept, field, equivalence, blank : str
    resolution : str or None
        Catalog resolution (aggregated equivalence only).
    response : str or None
        ``'single'`` or ``'multiple'`` (how the options were recorded on the form).
    recoding : dict
        Normalized source value → canonical code (empty for numeric concepts).
    others : str
        ``'error'`` or ``'no'``: meaning of a field value that is not in the recoding.
    note : str
    """

    concept: str
    field: str
    equivalence: str
    blank: str
    resolution: str | None = None
    response: str | None = None
    recoding: dict = field(default_factory=dict)
    others: str = 'error'
    note: str = ''


@dataclass(frozen=True)
class Mapping:
    """Validated mapping of a form in one version.

    Attributes
    ----------
    form, version : str
    valid_from : datetime.date
    valid_to : datetime.date or None
    concepts : dict
        Concept identifier → ConceptMapping.
    source : str
    note : str
    """

    form: str
    version: str
    valid_from: dt.date
    valid_to: dt.date | None
    concepts: dict
    source: str = ''
    note: str = ''

    def in_force(self, date):
        """True if the version was in force on ``date`` (limits included)."""
        return self.valid_from <= date and (self.valid_to is None or date <= self.valid_to)

    def not_measured(self, catalog):
        """Catalog concepts that this form does not measure."""
        return [c for c in catalog.concepts if c not in self.concepts]

    @classmethod
    def from_dict(cls, d, catalog, source='mapping'):
        """Validate an already-read mapping (dict) against the catalog and build it.

        Raises
        ------
        SchemaError
            With every problem found.
        """
        problems = []
        for k in sorted(set(d) - _MAPPING_KEYS):
            problems.append(f'unknown key «{k}» (valid: {", ".join(sorted(_MAPPING_KEYS))})')
        form, version = d.get('form'), d.get('version')
        for k, v in (('form', form), ('version', version)):
            if not isinstance(v, str) or not v.strip():
                problems.append(f'missing «{k}» (text)')
        valid_from, valid_to = d.get('valid_from'), d.get('valid_to')
        if not isinstance(valid_from, dt.date) or isinstance(valid_from, dt.datetime):
            problems.append('«valid_from» must be an unquoted date, for example valid_from = 2016-01-01')
        if valid_to is not None and (not isinstance(valid_to, dt.date) or isinstance(valid_to, dt.datetime)):
            problems.append('«valid_to» must be an unquoted date, for example valid_to = 2022-12-31')
        if isinstance(valid_from, dt.date) and isinstance(valid_to, dt.date) and valid_to < valid_from:
            problems.append('«valid_to» is earlier than «valid_from»')
        raw = d.get('concepts') or {}
        if not raw:
            problems.append('the mapping does not measure any concept ([concepts.<id>])')
        concepts = {}
        for cid, sec in raw.items():
            m = _validate_concept(cid, sec, catalog, problems)
            if m is not None:
                concepts[cid] = m
        if problems:
            raise SchemaError(problems, source)
        return cls(form=form.strip(), version=version.strip(), valid_from=valid_from, valid_to=valid_to,
                   concepts=concepts, source=source, note=str(d.get('note', '')))

    @classmethod
    def from_toml(cls, path, catalog):
        """Read and validate a mapping from a TOML file.

        Raises
        ------
        SchemaError
        """
        path = Path(path)
        try:
            d = tomllib.loads(path.read_text(encoding='utf-8'))
        except tomllib.TOMLDecodeError as e:
            raise SchemaError([f'the file is not valid TOML: {e}'], str(path)) from e
        return cls.from_dict(d, catalog, source=str(path))


def _validate_concept(cid, sec, catalog, problems):
    """Validate the section of a concept in the mapping; return ConceptMapping or None."""
    p0 = len(problems)
    where = f'concept «{cid}»'
    if cid not in catalog:
        problems.append(f'{where}: does not exist in the catalog')
        return None
    if not isinstance(sec, dict):
        problems.append(f'{where}: must be a section [concepts.{cid}]')
        return None
    concept = catalog[cid]
    for k in sorted(set(sec) - _CONCEPT_KEYS):
        problems.append(f'{where}: unknown key «{k}» (valid: {", ".join(sorted(_CONCEPT_KEYS))})')
    column = sec.get('field', cid)
    if not isinstance(column, str) or not column.strip():
        problems.append(f'{where}: «field» must be the name of a column')
    equivalence = sec.get('equivalence')
    if equivalence not in EQUIVALENCES:
        problems.append(f'{where}: «equivalence» must be one of {", ".join(EQUIVALENCES)} '
                        f'(got {equivalence!r}); a concept that is not measured is simply left out')
    resolution = sec.get('resolution')
    if equivalence == 'aggregated':
        if resolution is None:
            problems.append(f'{where}: an aggregated equivalence requires «resolution»')
        elif resolution not in concept.resolutions:
            available = ', '.join(concept.resolutions) or 'none'
            problems.append(f'{where}: the resolution «{resolution}» does not exist in the catalog '
                            f'(available: {available})')
    elif resolution is not None:
        problems.append(f'{where}: «resolution» only applies to an aggregated equivalence')
    response = sec.get('response')
    if response is not None and response not in RESPONSES:
        problems.append(f'{where}: «response» must be one of {", ".join(RESPONSES)}')
    blank = sec.get('blank')
    if blank not in BLANKS:
        problems.append(f'{where}: «blank» is required and must be one of {", ".join(BLANKS)} '
                        '(what an empty field means on this form)')
    elif blank == 'no' and concept.type != 'binary':
        problems.append(f'{where}: blank = "no" only applies to binary concepts; use "missing"')
    others = sec.get('others', 'error')
    if others not in OTHERS:
        problems.append(f'{where}: «others» must be one of {", ".join(OTHERS)}')
    elif others == 'no' and concept.type != 'binary':
        problems.append(f'{where}: others = "no" only applies to binary concepts')

    recoding = {}
    raw = sec.get('recoding')
    if concept.type == 'numeric':
        if raw is not None:
            problems.append(f'{where}: a numeric concept does not take «recoding»')
    elif not isinstance(raw, dict) or not raw:
        problems.append(f'{where}: missing «recoding» (source value = canonical code)')
    else:
        for original, target in raw.items():
            key = normalize_value(original)
            if key is None:
                problems.append(f'{where}: the recoding has an empty source value')
                continue
            if key in recoding and recoding[key] != target:
                problems.append(f'{where}: «{original}» appears twice with different codes')
            if target not in concept.codes:
                problems.append(f'{where}: «{original}» = {target!r} is not a code of the concept '
                                f'(valid: {list(concept.codes)})')
            recoding[key] = target
    if len(problems) > p0:
        return None
    return ConceptMapping(concept=cid, field=column.strip(), equivalence=equivalence, blank=blank,
                          resolution=resolution, response=response, recoding=recoding, others=others,
                          note=str(sec.get('note', '')))
