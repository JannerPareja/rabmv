# Schema reference

A schema has two layers, both written in TOML: one **catalog** of concepts and one **mapping** per form and
version. Load them with `rabmv.Schema.from_folder(catalog_path, mappings_folder)`, which reads every `*.toml`
of the folder and reports every problem of every file before failing. File keys are in English; concept
identifiers and source values are those of each registry and may be in any language.

## Catalog

```toml
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
```

**Identifiers.** Concept identifiers are lower case, without accents or spaces.

**Concept keys:**

| Key | Applies to | Meaning |
|---|---|---|
| `definition` | all | What the concept measures (required) |
| `type` | all | `numeric`, `ordinal`, `nominal` or `binary` |
| `minimum`, `maximum` | numeric | Plausible range |
| `levels` | ordinal | Integer codes in order |
| `categories` | nominal | Codes (all integers or all text) |
| `labels` | ordinal, nominal | Table `code = "text"` for reports |
| `resolutions` | ordinal, nominal | Named groupings of the codes; every code in exactly one group, at least two groups; groups of an ordinal concept are contiguous |

A binary concept has the canonical codes 1 (present) and 0 (absent).

## Mapping (one file per form and version)

```toml
form = "105_cesar"
version = "V09"
valid_from = 2020-03-06
valid_to = 2022-12-31          # optional: without valid_to, still in force

[concepts.education]
field = "education"            # source column (default: the concept id)
equivalence = "aggregated"     # exact | aggregated | partial
resolution = "five_levels"     # required when aggregated
blank = "missing"              # what an empty field means: missing | no
recoding = { primary = 2, secondary = 3, technical = 7, university = 9, none = 0 }

[concepts.depression]
field = "vulnerability_factor"
equivalence = "partial"
response = "single"            # single | multiple
blank = "no"
others = "no"                  # values of other options mean "No" for this concept
recoding = { "depression or psychiatric disorder" = 1 }
```

**Not measured.** A catalog concept that the mapping does not mention is **not measured** by that form and
version.

**Concept keys:**

| Key | Meaning |
|---|---|
| `field` | Source column |
| `equivalence` | `exact` (same codes), `aggregated` (a coarser catalog resolution; requires `resolution`), `partial` (the form measures something narrower or differently, for example a single-choice question) |
| `blank` | Required. `missing` (an empty field is a missing value) or `no` (an empty field means "No"; binary only) |
| `response` | `single` or `multiple` for factor questions |
| `others` | `error` (default: an unlisted value is an error) or `no` (binary only: another option means "No") |
| `recoding` | Source value → canonical code. Values are compared without case or accents; `1.0` equals `1` |
| `note` | Free text |

The form in force for a case is chosen by `form` and the date in `date` (`valid_from` ≤ date ≤ `valid_to`).
Data errors are reported with counts and unrecognized values only, never with rows.
