"""Concept catalog: what is measured, with which type and which canonical codes (layer 1).

The catalog is the common language: the mappings of each form translate their fields into these
concepts. It is written in TOML. Example::

    [catalog]
    name = "Suicidal behaviour"
    version = "2026.1"

    [concepts.age]
    definition = "Age in completed years"
    type = "numeric"
    minimum = 0
    maximum = 120

    [concepts.education]
    definition = "Highest level of education attained (codes of the national dictionary)"
    type = "ordinal"
    levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    [concepts.education.resolutions]
    five_levels = [[0], [1, 2], [3, 4, 5, 6], [7, 8], [9, 10, 11, 12]]

Types: ``numeric`` (with optional ``minimum`` and ``maximum``), ``ordinal`` (``levels`` in order),
``nominal`` (``categories``) and ``binary`` (canonical codes 1 = present, 0 = absent). A resolution is a
grouping of the codes (each code in exactly one group) that allows comparing forms that measure the
concept in less detail; in an ordinal concept, each group gathers consecutive levels.
"""
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SchemaError

TYPES = ('numeric', 'ordinal', 'nominal', 'binary')
BINARY_CODES = (0, 1)
_ID = re.compile(r'^[a-z][a-z0-9_]*$')
_CONCEPT_KEYS = {'definition', 'type', 'minimum', 'maximum', 'levels', 'categories', 'labels', 'resolutions'}


@dataclass(frozen=True)
class Concept:
    """A concept of the catalog.

    Attributes
    ----------
    id : str
    type : str
    definition : str
    codes : tuple
        Valid canonical codes (in order for ordinal concepts); empty for numeric concepts.
    minimum, maximum : float or None
        Valid range of a numeric concept.
    labels : dict
        Canonical code → readable text (optional).
    resolutions : dict
        Name → tuple of groups (each group is a tuple of codes).
    """

    id: str
    type: str
    definition: str
    codes: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    labels: dict = field(default_factory=dict)
    resolutions: dict = field(default_factory=dict)

    def groups(self, resolution):
        """Canonical code → index of its group in a resolution.

        Parameters
        ----------
        resolution : str

        Returns
        -------
        dict
        """
        return {c: i for i, group in enumerate(self.resolutions[resolution]) for c in group}


def _is_integer(v):
    """True if ``v`` is a real integer (not a boolean)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_concept(cid, d, problems):
    """Validate the section of a concept and return the Concept (or None on serious errors)."""
    p0 = len(problems)
    where = f'concept «{cid}»'
    if not _ID.match(cid):
        problems.append(f'{where}: the identifier must be lower case, without accents or spaces '
                        '(letters, digits and _; it starts with a letter)')
    if not isinstance(d, dict):
        problems.append(f'{where}: must be a section [concepts.{cid}]')
        return None
    for k in sorted(set(d) - _CONCEPT_KEYS):
        problems.append(f'{where}: unknown key «{k}» (valid: {", ".join(sorted(_CONCEPT_KEYS))})')
    definition = d.get('definition')
    if not isinstance(definition, str) or not definition.strip():
        problems.append(f'{where}: missing «definition» (text explaining what it measures)')
    kind = d.get('type')
    if kind not in TYPES:
        problems.append(f'{where}: «type» must be one of {", ".join(TYPES)} (got {kind!r})')
        return None

    codes, minimum, maximum = (), d.get('minimum'), d.get('maximum')
    if kind == 'numeric':
        for k in ('levels', 'categories', 'resolutions', 'labels'):
            if k in d:
                problems.append(f'{where}: a numeric concept does not take «{k}»')
        for k, v in (('minimum', minimum), ('maximum', maximum)):
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                problems.append(f'{where}: «{k}» must be a number')
        if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)) and minimum >= maximum:
            problems.append(f'{where}: «minimum» must be less than «maximum»')
    else:
        for k in ('minimum', 'maximum'):
            if k in d:
                problems.append(f'{where}: «{k}» only applies to numeric concepts')
        if kind == 'binary':
            for k in ('levels', 'categories'):
                if k in d:
                    problems.append(f'{where}: a binary concept does not take «{k}» '
                                    '(its canonical codes are 1 = present and 0 = absent)')
            codes = BINARY_CODES
        else:
            key = 'levels' if kind == 'ordinal' else 'categories'
            other = 'categories' if kind == 'ordinal' else 'levels'
            if other in d:
                problems.append(f'{where}: a {kind} concept uses «{key}», not «{other}»')
            values = d.get(key)
            if not isinstance(values, list) or len(values) < 2:
                problems.append(f'{where}: «{key}» must be a list with at least 2 codes')
            else:
                if kind == 'ordinal' and not all(_is_integer(v) for v in values):
                    problems.append(f'{where}: the levels of an ordinal concept must be integers')
                elif kind == 'nominal' and not (all(_is_integer(v) for v in values)
                                                or all(isinstance(v, str) for v in values)):
                    problems.append(f'{where}: the categories must be all integers or all text')
                if len(set(values)) != len(values):
                    problems.append(f'{where}: «{key}» has repeated codes')
                codes = tuple(values)

    labels = {}
    raw = d.get('labels') or {}
    if not isinstance(raw, dict):
        problems.append(f'{where}: «labels» must be a table code = "text"')
        raw = {}
    for k, v in raw.items():
        code = int(k) if k.lstrip('-').isdigit() and codes and _is_integer(codes[0]) else k
        if codes and code not in codes:
            problems.append(f'{where}: the label «{k}» does not match any code')
        labels[code] = v

    resolutions = {}
    res = d.get('resolutions') or {}
    if res and kind not in ('ordinal', 'nominal'):
        problems.append(f'{where}: resolutions only apply to ordinal or nominal concepts')
    elif not isinstance(res, dict):
        problems.append(f'{where}: «resolutions» must be a section of name = list of groups')
    else:
        for name, groups in res.items():
            if not _ID.match(name):
                problems.append(f'{where}: the resolution name «{name}» must be lower case, without spaces')
            if not isinstance(groups, list) or not all(isinstance(g, list) and g for g in groups):
                problems.append(f'{where}, resolution «{name}»: must be a list of non-empty groups')
                continue
            flat = [c for g in groups for c in g]
            repeated = sorted({c for c in flat if flat.count(c) > 1}, key=str)
            missing = [c for c in codes if c not in flat]
            extra = sorted({c for c in flat if c not in codes}, key=str)
            if repeated:
                problems.append(f'{where}, resolution «{name}»: codes in more than one group: {repeated}')
            if missing:
                problems.append(f'{where}, resolution «{name}»: codes without a group: {missing}')
            if extra:
                problems.append(f'{where}, resolution «{name}»: codes that do not exist in the concept: {extra}')
            if len(groups) < 2:
                problems.append(f'{where}, resolution «{name}»: needs at least 2 groups')
            if kind == 'ordinal' and not (repeated or missing or extra):
                position = {c: i for i, c in enumerate(codes)}
                for g in groups:
                    pos = sorted(position[c] for c in g)
                    if pos != list(range(pos[0], pos[-1] + 1)):
                        problems.append(f'{where}, resolution «{name}»: the group {g} is not contiguous; '
                                        'in an ordinal concept each group must gather consecutive levels')
            resolutions[name] = tuple(tuple(g) for g in groups)

    if len(problems) > p0:
        return None
    return Concept(id=cid, type=kind, definition=definition.strip(), codes=codes, minimum=minimum,
                   maximum=maximum, labels=labels, resolutions=resolutions)


@dataclass(frozen=True)
class Catalog:
    """Validated concept catalog.

    Attributes
    ----------
    name : str
    version : str
    concepts : dict
        Identifier → Concept.
    """

    name: str
    version: str
    concepts: dict

    @classmethod
    def from_dict(cls, d, source='catalog'):
        """Validate an already-read catalog (dict) and build it.

        Raises
        ------
        SchemaError
            With every problem found.
        """
        problems = []
        for k in sorted(set(d) - {'catalog', 'concepts'}):
            problems.append(f'unknown section [{k}] (valid: [catalog] and [concepts.<id>])')
        head = d.get('catalog') or {}
        name, version = head.get('name'), head.get('version')
        if not isinstance(name, str) or not name.strip():
            problems.append('missing [catalog] name')
        if not isinstance(version, str) or not version.strip():
            problems.append('missing [catalog] version (text, for example "2026.1")')
        raw = d.get('concepts') or {}
        if not raw:
            problems.append('the catalog has no concepts ([concepts.<id>])')
        concepts = {}
        for cid, sec in raw.items():
            c = _validate_concept(cid, sec, problems)
            if c is not None:
                concepts[cid] = c
        if problems:
            raise SchemaError(problems, source)
        return cls(name=name.strip(), version=version.strip(), concepts=concepts)

    @classmethod
    def from_toml(cls, path):
        """Read and validate a catalog from a TOML file.

        Raises
        ------
        SchemaError
            If the file is not valid TOML or has content problems.
        """
        path = Path(path)
        try:
            d = tomllib.loads(path.read_text(encoding='utf-8'))
        except tomllib.TOMLDecodeError as e:
            raise SchemaError([f'the file is not valid TOML: {e}'], str(path)) from e
        return cls.from_dict(d, source=str(path))

    def __getitem__(self, cid):
        """Concept by identifier."""
        return self.concepts[cid]

    def __contains__(self, cid):
        """True if the concept exists."""
        return cid in self.concepts
