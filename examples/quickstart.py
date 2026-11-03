"""RabMV in five steps on the public synthetic registry (examples/synthetic_registry).

The registry is synthetic (simulation v2, condition ``E_pi100_education``, seed 1): attempts are recorded with
form A and completed suicides with form B, as in Valledupar. Form B does not ask f1 or f2, asks f3–f5 as a
single choice, leaves f6 blank and records education in 5 levels; form A did not ask education in 2016.

Run from the repository root:

    python examples/quickstart.py
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import rabmv

HERE = Path(__file__).resolve().parent / 'synthetic_registry'


def main():
    data = pd.read_csv(HERE / 'registry.csv', dtype=str, keep_default_na=False, na_values=[''])
    y = (data['outcome'] == 'completed').astype(int)

    # 1. The schema: what each form measures, in which version and how comparably
    schema = rabmv.Schema.from_folder(HERE / 'catalog.toml', HERE / 'mappings')

    # 2. Per-case availability: measured, missing or not measured by the form in force
    harmonized = rabmv.harmonize(data, schema)
    print(harmonized.summary().head(10).to_string(), '\n')

    # 3. Audit: what the data suggest that a naive schema (everything exact) did not declare
    naive = rabmv.Schema.from_folder(HERE / 'catalog.toml', HERE / 'mappings_undeclared')
    checks = rabmv.run_checks(rabmv.harmonize(data, naive), y, naive, data=data)
    refined = rabmv.drop_explained_pairs(checks)
    print('Alerts against the naive schema:')
    print(refined.alerts()[['concept', 'check', 'detail']].to_string(index=False), '\n')

    # 4. Eligibility: a concept enters only if both classes measured it comparably
    eligibility = rabmv.evaluate_eligibility(harmonized, y, schema.catalog)
    print(eligibility.audit()[['concept', 'eligible', 'reason']].to_string(index=False), '\n')
    matrix = eligibility.matrix(harmonized)

    # 5. The model: what a form did not measure contributes 0 (marginalization)
    X_tr, X_te, y_tr, y_te = train_test_split(matrix.X, y, test_size=0.3, stratify=y, random_state=0)
    model = rabmv.RabMVClassifier(missing='marginalize', categorical_features=matrix.categorical_indices(),
                                  random_state=0).fit(X_tr, y_tr.to_numpy())
    p = model.predict_proba(X_te)[:, 1]
    print(f'Held-out AUC-PR: {average_precision_score(y_te, p):.3f} (prevalence {y_te.mean():.3f})')
    importance = pd.Series(model.term_importances_, index=model.term_names_).sort_values(ascending=False)
    print('Term importances (mean |contribution|):')
    print(importance.round(3).to_string())


if __name__ == '__main__':
    main()
