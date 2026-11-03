"""Normalization of values as they come from each form, to compare them with the mappings."""
import math
import unicodedata


def normalize_value(value):
    """Turn a source value into comparable text.

    Lower case, no accents and single spaces. An integer written as a decimal (``1.0``) becomes ``'1'``.
    Empty values (``None``, NaN or blank text) return ``None``.

    Parameters
    ----------
    value : object

    Returns
    -------
    str or None
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            value = int(value)
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(text.lower().split())
    return text or None
