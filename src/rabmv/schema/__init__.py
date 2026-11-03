"""Registry-aware schema: concept catalog (layer 1) and mappings per form and version (layer 2).

``Schema`` gathers a catalog with the mappings of every form and version of a study, and checks that the
versions of the same form do not overlap in time.
"""
from pathlib import Path

from .catalog import TYPES, Catalog, Concept
from .errors import SchemaError
from .mapping import BLANKS, EQUIVALENCES, OTHERS, RESPONSES, ConceptMapping, Mapping
from .normalization import normalize_value


class Schema:
    """Catalog plus the mappings of every form and version.

    Parameters
    ----------
    catalog : Catalog
    mappings : iterable of Mapping

    Raises
    ------
    SchemaError
        If a form-version pair repeats or two versions of a form overlap in time.
    """

    def __init__(self, catalog, mappings):
        """Validate the set and store the mappings sorted by form and date."""
        self.catalog = catalog
        self.mappings = sorted(mappings, key=lambda m: (m.form, m.valid_from))
        problems, seen = [], set()
        for m in self.mappings:
            if (m.form, m.version) in seen:
                problems.append(f'form «{m.form}» has two mappings of version «{m.version}»')
            seen.add((m.form, m.version))
        for a, b in zip(self.mappings, self.mappings[1:]):
            if a.form == b.form and (a.valid_to is None or b.valid_from <= a.valid_to):
                problems.append(f'form «{a.form}»: versions «{a.version}» and «{b.version}» overlap in time')
        if not self.mappings:
            problems.append('the schema has no mappings')
        if problems:
            raise SchemaError(problems, 'schema')

    @classmethod
    def from_folder(cls, catalog_path, mappings_folder):
        """Read the catalog and every ``*.toml`` of a mappings folder.

        Collects the problems of all files before failing.
        """
        catalog = Catalog.from_toml(catalog_path)
        mappings, problems = [], []
        for path in sorted(Path(mappings_folder).glob('*.toml')):
            try:
                mappings.append(Mapping.from_toml(path, catalog))
            except SchemaError as e:
                problems.append(str(e))
        if problems:
            raise SchemaError(problems, f'folder {mappings_folder}')
        return cls(catalog, mappings)

    def forms(self):
        """Identifiers of the forms in the schema."""
        return sorted({m.form for m in self.mappings})

    def mapping_for(self, form, date):
        """Mapping of a form in force on a date.

        Raises
        ------
        KeyError
            If the form does not exist or no version was in force on that date.
        """
        candidates = [m for m in self.mappings if m.form == form]
        if not candidates:
            raise KeyError(f'form «{form}» is not in the schema')
        for m in candidates:
            if m.in_force(date):
                return m
        raise KeyError(f'no version of form «{form}» was in force on {date}')


__all__ = ['BLANKS', 'EQUIVALENCES', 'OTHERS', 'RESPONSES', 'TYPES', 'Catalog', 'Concept', 'ConceptMapping',
           'Mapping', 'Schema', 'SchemaError', 'normalize_value']
