"""RabMV: Registry-Aware Boosting Machine of Valledupar.

An explainable boosting machine (a generalized additive model fitted by cyclic Newton boosting with an
optimal piecewise step) that knows the structure of the registry: which concepts each form measures,
in which version and how comparably. Package of the method, separate from any study that applies it.

- ``rabmv.RabMVClassifier``: the estimator (scikit-learn API). Without its registry-aware options it is
  bit-for-bit identical to EBMP v1.
- ``rabmv.schema``: concept catalog and mappings per form and version (TOML).
- ``rabmv.harmonize``: translation of the source data into concepts and per-case availability.
- ``rabmv.evaluate_eligibility``: eligibility rule, common resolution, matrix with mask and audit table.
- ``rabmv.run_checks``: empirical audit checks A, B, C and E.
- ``rabmv.drop_explained_pairs``: optional refinement of check B (single-choice pairs explained by A or C).
"""
from .availability import Harmonized, harmonize
from .checks import Checks, run_checks
from .eligibility import Eligibility, EligibleMatrix, evaluate_eligibility
from .estimator import RabMVClassifier
from .refine import drop_explained_pairs
from .schema import Catalog, Mapping, Schema, SchemaError

__all__ = ['Catalog', 'Checks', 'Eligibility', 'EligibleMatrix', 'Harmonized', 'Mapping', 'RabMVClassifier',
           'Schema', 'SchemaError', 'drop_explained_pairs', 'evaluate_eligibility', 'harmonize', 'run_checks']
__version__ = '0.1.0'
