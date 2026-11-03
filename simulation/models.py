"""Simulation scenarios with models: S2 (transport) and S5 (concept not measured in another secretariat).

Compared methods (the «M» of ADEMP), with fixed hyperparameters to bound the computation:

* **Strict RabMV**: eligible concepts in their common resolution, ``missing='marginalize'`` and nominal
  features as categorical.
* **Naive RabMV**: every concept as an ETL that declares everything exact would leave it (what v1 did):
  the questions not asked stay at "No" and missing education is one more category.
* **Naive trees**: scikit-learn's ``HistGradientBoostingClassifier`` with the same naive matrix.

S2: trained on the Valledupar registry and evaluated on new events recorded (a) as in Valledupar and (b) by
a secretariat that records completed suicides with the complete form. Estimands: AUC-PR in each case and
its drop. Descriptive.

S5: the strict model is trained on the Valledupar registry and evaluated in a secretariat with a complete
form except for one core concept. Main concept: occupation (the largest effect in the core after sex and
age, OR ≈ 2.2); secondary: education (almost null effect). Two variants:

* S5a: the concept is not measured in ANY case. Marginalizing or imputing any constant adds the same to
  every logit, so the AUC-PR is identical by construction; it is decided by equivalence with retraining
  without the concept (±0.01) and the global calibration is reported.
* S5b: the concept is missing only in the 2016–2017 cases (a version change). It is decided by
  non-inferiority against retraining and against imputing the mode (margin 0.01).

This split came from the 3-seed software trial, before freezing the protocol: with the concept absent in
every case, the AUC-PR comparison with the mode is degenerate.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from rabmv import RabMVClassifier, evaluate_eligibility, harmonize
from simulation.generator import population, record, schema, subsample

HYPER = dict(learning_rate=0.05, max_leaves=3, min_samples_leaf=10, n_bags=8, early_stopping_rounds=50)
S5_CONCEPTS = ('occupation', 'education')            # main, secondary


def naive_matrix(harmonized):
    """Every concept as numbers; an empty value becomes −1 (one more category, like "no information")."""
    X = {}
    for c in harmonized.values.columns:
        v = harmonized.values[c]
        if v.dtype == object:
            codes = {k: i for i, k in enumerate(sorted(v.dropna().unique(), key=str))}
            v = v.map(codes)
        X[c] = pd.to_numeric(v, errors='coerce').fillna(-1.0)
    return pd.DataFrame(X)


def _strict_rabmv(matrix):
    """RabMV with marginalization and categorical features for an EligibleMatrix."""
    return RabMVClassifier(missing='marginalize', categorical_features=matrix.categorical_indices(), random_state=0,
                           **HYPER)


def one_seed(seed, events):
    """S2 and S5 for one seed: a dictionary of AUC-PR values and differences."""
    s_ok, s_ex, s_null = schema('correct'), schema('all_exact'), schema('null')
    pop = subsample(population(seed), events, seed)
    y = pop['y'].to_numpy()
    d = record(pop, 'valledupar', seed)
    a = harmonize(d, s_ok)
    elig = evaluate_eligibility(a, pop['y'], s_ok.catalog)
    m_tr = elig.matrix(a)
    strict = _strict_rabmv(m_tr).fit(m_tr.X.to_numpy(), y)
    X_naive = naive_matrix(harmonize(d, s_ex))
    naive = RabMVClassifier(random_state=0, **HYPER).fit(X_naive.to_numpy(), y)
    trees = HistGradientBoostingClassifier(random_state=0).fit(X_naive.to_numpy(), y)

    # new events from the same population (another seed), with the same number of completed suicides
    test = subsample(population(seed + 5000), events, seed + 5000)
    yt = test['y'].to_numpy()
    r = {'seed': seed, 'events': events, 'completed_train': int(y.sum()), 'completed_test': int(yt.sum())}
    for name, registry, naive_schema, strict_schema in (('valledupar', 'valledupar', s_ex, s_ok),
                                                       ('transport', 'transport', s_null, s_null)):
        dt = record(test, registry, seed + 5000)
        mt = elig.matrix(harmonize(dt, strict_schema))
        Xn = naive_matrix(harmonize(dt, naive_schema))[X_naive.columns]
        r[f'aucpr_strict_{name}'] = average_precision_score(yt, strict.predict_proba(mt.X.to_numpy())[:, 1])
        r[f'aucpr_naive_{name}'] = average_precision_score(yt, naive.predict_proba(Xn.to_numpy())[:, 1])
        r[f'aucpr_trees_{name}'] = average_precision_score(yt, trees.predict_proba(Xn.to_numpy())[:, 1])
    for m in ('strict', 'naive', 'trees'):
        r[f'drop_{m}'] = r[f'aucpr_{m}_valledupar'] - r[f'aucpr_{m}_transport']

    # S5: the other secretariat does not measure a core concept, in every case (a) or in 2016–2017 (b)
    dt = record(test, 'null', seed + 5000)
    mt = elig.matrix(harmonize(dt, s_null))
    prev = float(yt.mean())
    r['test_prevalence'] = prev
    partial = test['year'].to_numpy() <= 2017
    r['fraction_s5b_without_concept'] = float(partial.mean())
    for concept in S5_CONCEPTS:
        j = list(mt.X.columns).index(concept)
        cols = [c for c in m_tr.X.columns if c != concept]
        cat_without = [cols.index(c) for c in cols if m_tr.types[c] == 'nominal']
        retrained = RabMVClassifier(missing='marginalize', categorical_features=cat_without, random_state=0,
                                    **HYPER).fit(m_tr.X[cols].to_numpy(), y)
        p_retrain = retrained.predict_proba(mt.X[cols].to_numpy())[:, 1]
        mode = pd.Series(m_tr.X[concept]).mode().iloc[0]
        for variant, absent in (('a', np.ones(len(yt), bool)), ('b', partial)):
            X_marg = mt.X.to_numpy().copy()
            X_marg[absent, j] = np.nan
            X_mode = mt.X.to_numpy().copy()
            X_mode[absent, j] = mode
            p_marg = strict.predict_proba(X_marg)[:, 1]
            p_mode = strict.predict_proba(X_mode)[:, 1]
            pre = f's5{variant}_{concept}'
            for name, pr in (('marginalize', p_marg), ('retrain', p_retrain), ('mode', p_mode)):
                r[f'aucpr_{pre}_{name}'] = average_precision_score(yt, pr)
                r[f'calib_{pre}_{name}'] = float(pr.mean()) - prev         # global calibration (mean − prevalence)
            r[f'diff_{pre}_marg_vs_retrain'] = r[f'aucpr_{pre}_marginalize'] - r[f'aucpr_{pre}_retrain']
            r[f'diff_{pre}_marg_vs_mode'] = r[f'aucpr_{pre}_marginalize'] - r[f'aucpr_{pre}_mode']
    return r
