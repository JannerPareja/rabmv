"""Validation error of the schema (catalog and mappings)."""


class SchemaError(ValueError):
    """One or more problems in a catalog or a mapping.

    Collects every problem found, not just the first one, so that whoever writes the file can fix them
    all at once.

    Parameters
    ----------
    problems : list of str
        Plain-language description of each problem.
    source : str, optional
        File or name of the validated object.
    """

    def __init__(self, problems, source=None):
        """Store the problems and build the message."""
        self.problems = list(problems)
        self.source = source
        n = len(self.problems)
        header = f'{n} problem(s) in {source}:' if source else f'{n} problem(s):'
        super().__init__('\n'.join([header] + [f'  - {p}' for p in self.problems]))
