# RabMV

**Registry-Aware Boosting Machine of Valledupar**: an explainable boosting machine (a generalized additive
model fitted by cyclic Newton boosting) that knows the structure of the public-health registry its data come
from — which concepts each surveillance form measures, in which version and how comparably.

## The problem it solves

In many surveillance registries each outcome is recorded with a different instrument. In Colombia, for
example, suicide attempts are notified with SIVIGILA form 356 and completed suicides come from vital
statistics and municipal instruments. When the form coincides with the class, a model can learn **the
form instead of the person**: a question that one form does not ask, a single-choice field, a blank recorded
as "No" or a field that disappears in one year become perfect predictors of the class. The model then looks
excellent in the registry where it was trained and fails where the registry is organized differently.

## What RabMV does

1. **Declares the registry** in TOML files: a catalog of concepts and one mapping per form and version, with
   the equivalence of each field (`exact`, `aggregated`, `partial`), the meaning of a blank and the validity
   dates ([schema](schema.md)).
2. **Harmonizes** the source data into concepts and marks, case by case, whether each concept was measured,
   left blank or not measured by the form in force.
3. **Audits** what the data suggest and the schema did not declare (checks A, B, C, E).
4. **Admits** a concept only if both classes measured it comparably, at the finest common resolution.
5. **Fits** an additive model in which what a form did not measure contributes 0, with purified pairwise
   interactions so that this 0 is the exact conditional average of each pair ([theory](theory/purification.md)).

## Where to go next

| Page | Content |
|---|---|
| [Installation](installation.md) | `pip install rabmv` and optional extras |
| [Quickstart](quickstart.md) | The five steps on the public synthetic registry |
| [Schema reference](schema.md) | Catalog and mapping files |
| [Method](method.md) | Model, marginalization, eligibility and checks |
| [Theory](theory/purification.md) | Proofs for purified marginalization |
| [Validation](validation.md) | The two frozen simulations and what they showed |
| [Reproducibility](reproducibility.md) | How to rerun every result |
| [API](api.md) | Public functions and classes |
| [Authors and citation](about.md) | Authors, how to cite, funding and acknowledgements |
