# Method

## Layers

| Layer | Function | Output |
|---|---|---|
| 1. Catalog | `rabmv.Catalog` | Concepts, types, codes, resolutions |
| 2. Mappings | `rabmv.Mapping` | How each form and version measures each concept |
| 3. Availability | `rabmv.harmonize` | Values in catalog codes and a status per case and concept: `measured`, `missing` or `not_measured` |
| 4. Eligibility | `rabmv.evaluate_eligibility` | Which concepts enter, at which common resolution, with an audit table |
| 5. Model | `rabmv.RabMVClassifier` | Additive logit with marginalization of what was not measured |
| Audit | `rabmv.run_checks`, `rabmv.drop_explained_pairs` | Alerts of undeclared differential measurement |

## Eligibility rule

A concept enters only if it was measured comparably in the sources of **both classes**:

1. There are at least `min_per_class` measured cases in each class.
2. Every form version that measures it has `exact` or `aggregated` equivalence. Mode `sensitivity` also
   admits `partial`; the main analyses use `strict`.
3. A common resolution with at least two groups exists: the finest one that every form can express. If
   merging the forms' groupings leaves a single group, the concept is excluded.

## Model

$$
\eta(x) = \beta_0 + \sum_j f_j(x_j) + \sum_{(a,b)\in\mathcal P} f_{ab}(x_a,x_b), \qquad
P(y=1\mid x)=\sigma(\eta(x)).
$$

Each shape function is a step function on quantile bins, fitted by cyclic Newton boosting: at each round and
for each term, the optimal piecewise-constant Newton step with at most `max_leaves` pieces is found by dynamic
programming over the bins, on bagged samples with early stopping. Each category of a nominal concept
(`categorical_features`) gets its own Newton step, without grouping "neighbouring" categories, so the result
does not depend on how categories were numbered. Optional final smoothing (`smoothing`) is a Whittaker smoother
of each ordered function (nominal functions are not smoothed).

**Marginalization (`missing='marginalize'`).**
- A case with a concept not measured (NaN in the eligible matrix) contributes no gradient or hessian to that
  concept's function.
- Bins and centring use only measured cases.
- At prediction time the concept contributes 0, its mean over the measured cases.
- Pairs are fitted on cases with both concepts measured, and each pair table is purified (rows and columns
  with weighted mean 0, mass moved to the main effects), so a pair with an unmeasured concept also
  contributes exactly its conditional average, 0 ([proofs](theory/purification.md)).

**Explanations.** `contributions(X)` returns the exact additive contribution of every term, and
`decision_function(X) = intercept_ + contributions(X).sum(1)`.

## Audit checks

The checks detect coarse patterns that a real effect rarely produces and that a form measuring differently
does produce. Each alert invites a review of the mapping; it is not an automatic conclusion. Reports contain
counts only, with counts from 1 to 4 suppressed.

| Check | Pattern |
|---|---|
| A | Constant in one class and variable in the other (a question not asked, filled with "No") |
| B | Two binary concepts never marked together in one class and marked together in the other (single choice) |
| C | Very different proportion of blanks between classes in the source (blanks recorded as "No") |
| E | Within a form, a year in which the field is almost never filled while the other years are (undeclared version change) |

A, B and C use one-sided Fisher exact tests with `alpha = 0.01`, calibrated by simulation v1 before the checks
were applied to real data.

**Refinement of B.** `rabmv.drop_explained_pairs` withdraws a B alert when one concept of the pair already has
an A or C alert, because then the pair only looks single-choice through that concept. It is optional and was
validated after simulation v2 (see [validation](validation.md)).

## Limits

- Marginalization averages the logit, not the probability.
- A missing main effect receives its unconditional mean, not its mean given the other concepts.
- The strict eligibility rule gives up partially comparable concepts even when the form is unrelated to the
  class.
- Simulation v2 quantifies all three.
