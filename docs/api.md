# API

Reference of rabmv 0.1.0, generated from the docstrings by `docs/gen_api.py`.

| Name | Summary |
|---|---|
| [`Catalog`](#catalog) | Validated concept catalog. |
| [`Checks`](#checks) | Result of the empirical checks. |
| [`Eligibility`](#eligibility) | Result of the eligibility rule. |
| [`EligibleMatrix`](#eligiblematrix) | What the core receives: values in the common resolution and a measurement mask. |
| [`Harmonized`](#harmonized) | Data translated into concepts, with their availability. |
| [`Mapping`](#mapping) | Validated mapping of a form in one version. |
| [`RabMVClassifier`](#rabmvclassifier) | Registry-Aware Boosting Machine of Valledupar (main effects and, optionally, pairs). |
| [`Schema`](#schema) | Catalog plus the mappings of every form and version. |
| [`SchemaError`](#schemaerror) | One or more problems in a catalog or a mapping. |
| [`drop_explained_pairs`](#drop_explained_pairs) | Turn into "no alert" the B alerts whose pair contains a concept with an A or C alert. |
| [`evaluate_eligibility`](#evaluate_eligibility) | Apply the eligibility rule to every concept of the catalog. |
| [`harmonize`](#harmonize) | Translate the source data into concepts and build the availability matrix. |
| [`run_checks`](#run_checks) | Apply checks A, B, C and E. |

## Catalog

*class* `rabmv.Catalog`

```python
Catalog(name: str, version: str, concepts: dict) -> None
```

Validated concept catalog.

**Attributes**

- **`name`** (*str*)
- **`version`** (*str*)
- **`concepts`** (*dict*): Identifier → Concept.

### Catalog.from_toml

```python
from_toml(path)
```

Read and validate a catalog from a TOML file.

**Raises**

- **`SchemaError`**: If the file is not valid TOML or has content problems.

## Checks

*class* `rabmv.Checks`

```python
Checks(table: pandas.DataFrame, parameters: dict) -> None
```

Result of the empirical checks.

**Attributes**

- **`table`** (*pandas.DataFrame*): One row per concept (or pair) and check: result (`'alert'`, `'no alert'` or `'not evaluable'`), p-value, counts and detail.
- **`parameters`** (*dict*)

### Checks.alerts

```python
alerts(self)
```

Rows with an alert.

### Checks.audit

```python
audit(self, suppress_below=5)
```

Table for reporting, with counts from 1 to 4 suppressed.

**Parameters**

- **`suppress_below`** (*int, default=5*): 0 disables suppression (internal use only, never for sharing).

## Eligibility

*class* `rabmv.Eligibility`

```python
Eligibility(table: pandas.DataFrame, eligible: list, groups: dict, types: dict, mode: str, missing: str) -> None
```

Result of the eligibility rule.

**Attributes**

- **`table`** (*pandas.DataFrame*): One row per concept: type, equivalences per class, measurement counts per class, common resolution, whether it is eligible and the reason.
- **`eligible`** (*list of str*)
- **`groups`** (*dict*): Eligible concept → groups of its common resolution.
- **`types`** (*dict*)
- **`mode`** (*str*)
- **`missing`** (*str*)

### Eligibility.matrix

```python
matrix(self, harmonized)
```

Values of the eligible concepts in their common resolution, with their mask.

**Parameters**

- **`harmonized`** (*rabmv.availability.Harmonized*)

**Returns**

- **`EligibleMatrix`**

### Eligibility.audit

```python
audit(self, suppress_below=5)
```

Audit table for reporting (counts from 1 to 4 suppressed).

**Parameters**

- **`suppress_below`** (*int, default=5*): 0 disables suppression (internal use only, never for sharing).

**Returns**

- **`pandas.DataFrame`**

## EligibleMatrix

*class* `rabmv.EligibleMatrix`

```python
EligibleMatrix(X: pandas.DataFrame, mask: pandas.DataFrame, types: dict, groups: dict) -> None
```

What the core receives: values in the common resolution and a measurement mask.

**Attributes**

- **`X`** (*pandas.DataFrame*): Cases × eligible concepts (float). Ordinal and aggregated: group index (0, 1, …, in order); nominal: category index; binary: 0/1; numeric: the value. NaN where there is no measurement.
- **`mask`** (*pandas.DataFrame*): Cases × eligible concepts (bool): True if the case has the concept measured.
- **`types`** (*dict*): Concept → catalog type.
- **`groups`** (*dict*): Concept → code groups of the common resolution (empty for numeric concepts).

## Harmonized

*class* `rabmv.Harmonized`

```python
Harmonized(values: pandas.DataFrame, status: pandas.DataFrame, equivalence: pandas.DataFrame, resolution: pandas.DataFrame, form: pandas.Series, version: pandas.Series) -> None
```

Data translated into concepts, with their availability.

**Attributes**

- **`values`** (*pandas.DataFrame*): Cases × concepts with canonical codes; NaN when the concept was not measured or is missing.
- **`status`** (*pandas.DataFrame*): Cases × concepts with `'measured'`, `'missing'` or `'not_measured'`.
- **`equivalence`** (*pandas.DataFrame*): Cases × concepts with the equivalence declared in the mapping in force (`None` if not measured).
- **`resolution`** (*pandas.DataFrame*): Cases × concepts with the mapping resolution (`'complete'` or the name of the grouping; `None` if not measured).
- **`form, version`** (*pandas.Series*): Form and version in force for each case.

### Harmonized.summary

```python
summary(self, suppress_below=5)
```

Availability counts by form, version and concept (no individual rows).

**Parameters**

- **`suppress_below`** (*int, default=5*): Counts between 1 and this value minus 1 are shown as `'<5'` (or the given threshold). 0 disables suppression (internal use only, never for sharing).

**Returns**

- **`pandas.DataFrame`**: One row per form, version and concept, with the equivalence and the count of each status.

## Mapping

*class* `rabmv.Mapping`

```python
Mapping(form: str, version: str, valid_from: datetime.date, valid_to: datetime.date | None, concepts: dict, source: str = '', note: str = '') -> None
```

Validated mapping of a form in one version.

**Attributes**

- **`form, version`** (*str*)
- **`valid_from`** (*datetime.date*)
- **`valid_to`** (*datetime.date or None*)
- **`concepts`** (*dict*): Concept identifier → ConceptMapping.
- **`source`** (*str*)
- **`note`** (*str*)

### Mapping.from_toml

```python
from_toml(path, catalog)
```

Read and validate a mapping from a TOML file.

**Raises**

- **`SchemaError`**

## RabMVClassifier

*class* `rabmv.RabMVClassifier`

```python
RabMVClassifier(max_bins=32, learning_rate=0.05, max_leaves=3, min_samples_leaf=10, reg_lambda=1.0, max_rounds=2000, early_stopping_rounds=50, early_stopping_tolerance=1e-05, validation_size=0.15, n_bags=8, interactions=0, interaction_bins=8, random_state=42, split='optimal', newton=True, smoothing=0.0, missing='reject', categorical_features=None)
```

Registry-Aware Boosting Machine of Valledupar (main effects and, optionally, pairs).

**Parameters**

- **`max_bins`** (*int, default=32*): Maximum bins per feature (training quantiles).
- **`learning_rate`** (*float, default=0.05*): Fraction of the Newton step applied at each update, in (0, 1].
- **`max_leaves`** (*int, default=3*): Maximum contiguous segments of each shape-function update.
- **`min_samples_leaf`** (*int, default=10*): Minimum training cases per segment.
- **`reg_lambda`** (*float, default=1.0*): L2 regularization of each segment value.
- **`max_rounds`** (*int, default=2000*): Maximum rounds (each round visits all features).
- **`early_stopping_rounds`** (*int, default=50*): Rounds without improvement of the validation loss after which a bag stops.
- **`early_stopping_tolerance`** (*float, default=1e-5*): Minimum improvement of the validation loss (mean log loss) that counts as an improvement.
- **`validation_size`** (*float, default=0.15*): Stratified fraction of each bag held out for early stopping (0 = no early stopping).
- **`n_bags`** (*int, default=8*): Bags (different validation splits); shape functions are averaged.
- **`interactions`** (*int, default=0*): Pairs of features modelled after the main effects (0 = main effects only).
- **`interaction_bins`** (*int, default=8*): Bins per feature in the pair tables.
- **`random_state`** (*int, default=42*): Seed of the validation splits.
- **`split`** (*{'optimal', 'greedy'}, default='optimal'*): How the segments of each update are chosen: optimal partition by dynamic programming (RabMV) or best-first greedy tree (standard EBM). 'greedy' exists only for the ablation study.
- **`newton`** (*bool, default=True*): True: Newton step (gradient / hessian). False: gradient step (gradient / case weight), only for the ablation study.
- **`smoothing`** (*float, default=0.0*): Strength κ of the final smoothing of each ordered shape function (more than 2 bins): neighbouring bins borrow information (`smooth_curve`), which reduces variance when each bin has few events. The intercept is then refitted so that the mean training probability still equals the observed proportion. 0 = EBMP v1.
- **`missing`** (*{'reject', 'marginalize'}, default='reject'*): What to do with a `NaN` in X. `'reject'` (as v1) raises an error. `'marginalize'` (registry-aware method) treats it as "concept not measured for this case": the case contributes no gradient or hessian to that feature's function, cuts and centring use only measured cases, and at prediction time the feature contributes 0 (the average). Without any NaN and without interactions, both values produce exactly the same model as v1. With interactions, each pair is fitted on the cases that have both concepts measured, its grid is nested in the main-effect grids and its table is purified (rows and columns with weighted mean 0, mass moved to the main effects); an unmeasured concept makes the pair contribute 0, which is then the exact average of the pair over that concept given the other one.
- **`categorical_features`** (*sequence of int or None, default=None*): Indices of nominal columns (unordered categories, such as the health insurance scheme). Each category gets its own Newton step, without grouping "neighbouring" categories, because their order is arbitrary. They are not smoothed. With None (default) all columns are ordered, as v1.

**Attributes**

- **`intercept_`** (*float*)
- **`term_names_`** (*list of str*): Name of each term (feature or pair "a & b").
- **`term_scores_`** (*list of numpy.ndarray*): Value of each term per bin (or per cell for pairs), centred.
- **`term_scores_std_`** (*list of numpy.ndarray*): Standard deviation across bags of each function.
- **`bin_edges_`** (*list of numpy.ndarray*): Cuts of each feature.
- **`term_importances_`** (*numpy.ndarray*): Weighted mean of |f(x)| of each term on the training data.
- **`best_rounds_`** (*numpy.ndarray*): Round with the best validation loss for each bag (main effects).
- **`validation_history_`** (*numpy.ndarray of shape (rounds, n_bags)*): Validation loss per round and bag (main effects).
- **`categorical_features_`** (*list of int*)
- **`interaction_pairs_`** (*list of (int, int)*)

### RabMVClassifier.fit

```python
fit(self, X, y, sample_weight=None)
```

Fit the model.

**Parameters**

- **`X`** (*array-like of shape (n, p)*): Numeric features.
- **`y`** (*array-like of shape (n,)*): Binary class.
- **`sample_weight`** (*array-like of shape (n,), optional*): Sample weights (e.g. class_weight='balanced').

**Returns**

- **`RabMVClassifier`**: The fitted estimator.

**Raises**

- **`ValueError`**: On infinite values, NaN (except with `missing='marginalize'`), a column without any measured value, if `y` does not have exactly two classes, or if a hyperparameter is invalid.

### RabMVClassifier.predict_proba

```python
predict_proba(self, X)
```

Probability of each class.

**Parameters**

- **`X`** (*array-like of shape (n, p)*)

**Returns**

- **`numpy.ndarray of shape (n, 2)`**: Column 1: probability of `classes_[1]`.

### RabMVClassifier.predict

```python
predict(self, X)
```

Predicted class with threshold 0.5 (a study usually chooses its own threshold).

**Parameters**

- **`X`** (*array-like of shape (n, p)*)

**Returns**

- **`numpy.ndarray of shape (n,)`**

### RabMVClassifier.decision_function

```python
decision_function(self, X)
```

Predicted logit.

**Parameters**

- **`X`** (*array-like of shape (n, p)*)

**Returns**

- **`numpy.ndarray of shape (n,)`**

### RabMVClassifier.contributions

```python
contributions(self, X)
```

Exact contribution of each term to the logit of each case.

Because the model is additive, `decision_function(X) = intercept_ + contributions(X).sum(1)`:
it is an exact local explanation (equivalent to SHAP for an additive model with centred terms)
and needs no background of cases.

**Parameters**

- **`X`** (*array-like of shape (n, p)*)

**Returns**

- **`numpy.ndarray of shape (n, n_terms)`**

## Schema

*class* `rabmv.Schema`

```python
Schema(catalog, mappings)
```

Catalog plus the mappings of every form and version.

**Parameters**

- **`catalog`** (*Catalog*)
- **`mappings`** (*iterable of Mapping*)

**Raises**

- **`SchemaError`**: If a form-version pair repeats or two versions of a form overlap in time.

### Schema.from_folder

```python
from_folder(catalog_path, mappings_folder)
```

Read the catalog and every `*.toml` of a mappings folder.

Collects the problems of all files before failing.

### Schema.mapping_for

```python
mapping_for(self, form, date)
```

Mapping of a form in force on a date.

**Raises**

- **`KeyError`**: If the form does not exist or no version was in force on that date.

## SchemaError

*class* `rabmv.SchemaError`

```python
SchemaError(problems, source=None)
```

One or more problems in a catalog or a mapping.

Collects every problem found, not just the first one, so that whoever writes the file can fix them
all at once.

**Parameters**

- **`problems`** (*list of str*): Plain-language description of each problem.
- **`source`** (*str, optional*): File or name of the validated object.

## drop_explained_pairs

*function* `rabmv.drop_explained_pairs`

```python
drop_explained_pairs(checks, explaining=('A', 'C'))
```

Turn into "no alert" the B alerts whose pair contains a concept with an A or C alert.

**Parameters**

- **`checks`** (*rabmv.Checks*): Output of `rabmv.run_checks`.
- **`explaining`** (*tuple of str, default=('A', 'C')*): Checks whose alert explains a single-choice pattern.

**Returns**

- **`rabmv.Checks`**: A copy; the changed rows keep their p-value and say why in `detail`.

## evaluate_eligibility

*function* `rabmv.evaluate_eligibility`

```python
evaluate_eligibility(harmonized, y, catalog, mode='strict', min_per_class=10, missing='marginalize')
```

Apply the eligibility rule to every concept of the catalog.

**Parameters**

- **`harmonized`** (*rabmv.availability.Harmonized*)
- **`y`** (*array-like*): Class of each case, aligned with `harmonized` (exactly two classes).
- **`catalog`** (*rabmv.schema.Catalog*)
- **`mode`** (*{'strict', 'sensitivity'}*)
- **`min_per_class`** (*int, default=10*): Minimum measured cases in each class for a concept to be eligible.
- **`missing`** (*{'marginalize', 'category'}*): What to do with a field that was asked but came empty: marginalize it (default) or treat it as one more category (sensitivity analysis only; not applicable to numeric concepts).

**Returns**

- **`Eligibility`**

## harmonize

*function* `rabmv.harmonize`

```python
harmonize(data, schema, form_column='form', date_column='date')
```

Translate the source data into concepts and build the availability matrix.

**Parameters**

- **`data`** (*pandas.DataFrame*): One row per case, with the columns named by the mappings, the form and the date of the case.
- **`schema`** (*rabmv.schema.Schema*)
- **`form_column, date_column`** (*str*): Columns with the form identifier and the date that decides the version in force.

**Returns**

- **`Harmonized`**

**Raises**

- **`SchemaError`**: If columns are missing, dates are invalid, cases have no version in force, category values are not recognized by the mapping or numbers are out of range. All problems are reported at once, with counts and without individual rows.

## run_checks

*function* `rabmv.run_checks`

```python
run_checks(harmonized, y, schema, data=None, date_column='date', alpha=0.01, blank_difference=0.2, min_cases=10, low_completeness=0.05, high_completeness=0.5)
```

Apply checks A, B, C and E.

**Parameters**

- **`harmonized`** (*rabmv.availability.Harmonized*)
- **`y`** (*array-like*): Class of each case (exactly two classes), aligned with `harmonized`.
- **`schema`** (*rabmv.schema.Schema*)
- **`data`** (*pandas.DataFrame, optional*): Source data (the same that were harmonized). Needed for C and E, which look at blanks as they come from the form; without them, those checks are "not evaluable".
- **`date_column`** (*str*)
- **`alpha`** (*float, default=0.01*): Significance threshold, set by the calibration of the simulation (seeds 1–50).
- **`blank_difference`** (*float, default=0.2*): Minimum difference between classes in the proportion of blanks for a C alert.
- **`min_cases`** (*int, default=10*): Minimum cases per class (or per year, in E) to evaluate.
- **`low_completeness, high_completeness`** (*float*): E alert if a year has completeness ≤ `low_completeness` and the median of the other years of that form is ≥ `high_completeness`.

**Returns**

- **`Checks`**
