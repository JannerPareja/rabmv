"""Figures, model equations and tables from the saved checkpoints, without training again.

Everything here is computed from the task files of the work folder (design version 3: each holds the fitted
model, its model card and the probabilities of every partition). The outputs are aggregates: cut points,
scores of each bin, curves and binned calibration (bins with fewer than ``MIN_BIN`` cases are not drawn); no
per-case value leaves the work folder.

Labels: with ``--schema <folder>`` and a ``labels_en.toml`` in that folder, every variable, category and group
of a common resolution is written with its English label (figures, equations and term tables); the internal
name stays in brackets for traceability. Without it the internal names are used.

Outputs in ``development/results/<level>/report/``:

* ``models/<variant>__<balancing>__<optimizer>/fold_XX_equation.txt`` — the equation of the fitted model,
  logit p = β₀ + Σ f_j(x_j) [+ Σ f_jk(x_j, x_k)] − offset, with the importance of each term;
* ``.../fold_XX_terms.csv`` — every bin of every term: limits (in the original units when the preprocessing
  allows it), score, standard deviation across bags;
* ``.../fold_XX_shape_functions.png`` and ``fold_XX_pairs.png`` — shape functions with ±2 standard
  deviations, and the pair tables;
* ``pr_curves.png``, ``calibration.png`` — per combination, pooled over folds;
* ``convergence.png`` — best inner AUC-PR so far against the evaluation number, per optimizer;
* ``auc_pr_by_combination.png`` — AUC-PR of each fold, per combination;
* ``oversampling.csv`` — the SMOTE / ADASYN diagnostics of every task.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MIN_BIN = 5                 # minimum cases per calibration bin drawn (counts of 1–4 are never shown)
PROGRESS_EVERY = 20         # models between progress lines of the report


# ----------------------------------------------------------------------------- English labels
class Labels:
    """English names of variables, categories and resolution groups (``labels_en.toml`` of the study).

    Parameters
    ----------
    schema_folder : str or pathlib.Path or None
        Folder with ``catalog.toml``, ``mappings/`` and ``labels_en.toml``. None: internal names.
    """

    def __init__(self, schema_folder=None):
        self.variables, self.resolutions, self.groups = {}, {}, {}
        if schema_folder is None:
            return
        import tomllib
        folder = Path(schema_folder)
        with open(folder / 'labels_en.toml', 'rb') as f:
            data = tomllib.load(f)
        self.variables = data.get('variables', {})
        self.resolutions = data.get('resolutions', {})
        from rabmv import Schema
        from rabmv.eligibility import common_resolution
        schema = Schema.from_folder(folder / 'catalog.toml', folder / 'mappings')
        for cid, concept in schema.catalog.concepts.items():
            used = {'complete'} | {m.concepts[cid].resolution for m in schema.mappings
                                   if cid in m.concepts and m.concepts[cid].resolution}
            if concept.codes:
                self.groups[cid] = (common_resolution(concept, used), sorted(used - {'complete'}))

    def _variable(self, feature):
        """(variable, category code or None) of an input feature name."""
        if '__' not in feature:
            return feature, None
        group, column = feature.split('__', 1)
        if group == 'cat':
            for var in sorted(self.variables, key=len, reverse=True):
                if column.startswith(var + '_'):
                    return var, column[len(var) + 1:]
        return column, None

    def _category(self, var, code):
        cats = self.variables.get(var, {}).get('categories', {})
        for key in (str(code), str(code).rstrip('0').rstrip('.') if '.' in str(code) else str(code)):
            if key in cats:
                return cats[key]
        return None

    def name(self, feature):
        """English label of an input feature or of a pair «a & b»."""
        if ' & ' in feature:
            return ' × '.join(self.name(f) for f in feature.split(' & '))
        var, code = self._variable(feature)
        base = self.variables.get(var, {}).get('name')
        if base is None:
            return feature
        if code is not None:
            return f'{base}: {self._category(var, code) or code}'
        return base

    def bin(self, feature, lower, upper, categories=None):
        """English label of a bin: the category or resolution group whose code lies in [lower, upper)."""
        var, code = self._variable(feature)
        if code is not None:                                    # one-hot column: 0 / 1
            return '' if not np.isfinite(lower) and not np.isfinite(upper) else (
                'yes' if lower <= 1 < upper or lower > 0.5 else 'no')
        if categories is not None:                              # EBM nominal bin: its category value
            values = [categories]
        else:
            if var in self.groups:
                n = len(self.groups[var][0])
                values = [i for i in range(n) if lower <= i < upper] if self._registry_feature(feature) else None
            else:
                values = None
            if values is None:
                cats = self.variables.get(var, {}).get('categories', {})
                numeric = [k for k in cats if k.lstrip('-').isdigit()]
                if not numeric and var in self.groups:             # ordinal without labels: its codes
                    codes = [c for g in self.groups[var][0] for c in g if isinstance(c, (int, float))]
                    return ', '.join(str(c) for c in codes if lower <= c < upper)
                values = [int(k) for k in numeric if lower <= int(k) < upper]
                return ', '.join(cats[str(v)] for v in values) if values else ''
        return ', '.join(filter(None, (self._group(var, v) for v in values)))

    def _registry_feature(self, feature):
        return '__' not in feature

    def _group(self, var, index):
        """Label of group ``index`` of the common resolution of a registry concept."""
        try:
            index = int(float(index))
        except (TypeError, ValueError):
            return str(index)
        groups, used = self.groups.get(var, ((), []))
        if index >= len(groups):
            return str(index)
        for name in used:
            labels = self.resolutions.get(var, {}).get(name)
            if labels and len(labels) == len(groups):
                return labels[index]
        return ' / '.join(filter(None, (self._category(var, c) or str(c) for c in groups[index])))


# ----------------------------------------------------------------------------- terms of a model card
def _main_cuts_ebm(card, feature):
    """Cut points of an EBM feature for main effects (continuous) or None (nominal)."""
    b = card['bins'][feature][0]
    return None if isinstance(b, dict) else np.asarray(b, float)


def terms(card):
    """Terms of a fitted model as plain records.

    Returns
    -------
    list of dict
        ``name``, ``features`` (indices), ``cuts`` (list, one array per feature; None for a nominal EBM
        feature), ``scores`` and ``std`` (arrays over bins, or 2-D over cells for pairs), ``importance``.
    """
    out = []
    if card['classifier'] == 'RabMVClassifier':
        p = len(card['bin_edges'])
        imp = card.get('term_importances', np.full(len(card['term_scores']), np.nan))
        for t, name in enumerate(card['term_names']):
            if t < p:
                out.append(dict(name=name, features=(t,), cuts=[np.asarray(card['bin_edges'][t])],
                                scores=np.asarray(card['term_scores'][t]), std=np.asarray(card['term_scores_std'][t]),
                                importance=float(imp[t])))
            else:
                a, b = card['interaction_pairs'][t - p]
                ea, eb = np.asarray(card['pair_bin_edges'][a]), np.asarray(card['pair_bin_edges'][b])
                shape = (ea.size + 1, eb.size + 1)
                out.append(dict(name=name, features=(a, b), cuts=[ea, eb],
                                scores=np.asarray(card['term_scores'][t]).reshape(shape),
                                std=np.asarray(card['term_scores_std'][t]).reshape(shape), importance=float(imp[t])))
        return out
    imp = card.get('term_importances', np.full(len(card['term_scores']), np.nan))
    for t, (name, feats) in enumerate(zip(card['term_names'], card['term_features'])):
        s = np.asarray(card['term_scores'][t])
        sd = np.asarray(card.get('standard_deviations', [np.zeros_like(s)] * len(card['term_scores']))[t])
        inner = tuple(slice(1, -1) for _ in feats)           # drop the "missing" and "unknown" bins of EBM
        cuts, cats = [], []
        for f in feats:
            levels = card['bins'][f]
            b = levels[min(len(feats) - 1, len(levels) - 1)]
            cuts.append(None if isinstance(b, dict) else np.asarray(b, float))
            cats.append([k for k, _ in sorted(b.items(), key=lambda kv: kv[1])] if isinstance(b, dict) else None)
        out.append(dict(name=name, features=tuple(feats), cuts=cuts, categories=cats, scores=s[inner], std=sd[inner],
                        importance=float(imp[t]), missing_score=float(np.ravel(s)[0]) if len(feats) == 1 else None))
    return out


def _original_units(model, feature_name):
    """(mean, scale) to undo the StandardScaler of a feature of the Competencia 1 matrix, or None."""
    try:
        ct = model.steps[0][1]
        group, column = feature_name.split('__', 1)
        tr = ct.named_transformers_[group]
        scaler = tr if hasattr(tr, 'scale_') else tr.steps[-1][1]
        cols = list(tr.get_feature_names_out()) if hasattr(tr, 'get_feature_names_out') else None
        j = cols.index(column) if cols and column in cols else list(ct.get_feature_names_out()).index(feature_name)
        return float(scaler.mean_[j]), float(scaler.scale_[j])
    except Exception:
        return None


def terms_table(task, labels=None):
    """One row per bin (or cell) of every term, with limits in original units when possible and, with
    ``labels``, the English name of the term and of each bin."""
    labels = labels or Labels()
    card, model = task['model_card'], task.get('model')
    names = card.get('input_features')
    names = list(card.get('feature_names_in', [])) if names is None else names
    rows = [dict(term='(intercept)', term_en='Intercept', bin='', bin_en='', lower=np.nan, upper=np.nan,
                 score=float(np.ravel(card['intercept'])[0]), std=np.nan)]
    if card.get('offset'):
        rows.append(dict(term='(prior correction of the weights)', term_en='Prior correction of the class weights',
                         bin='', bin_en='', lower=np.nan, upper=np.nan, score=-float(card['offset']), std=np.nan))
    for t in terms(card):
        if len(t['features']) == 1:
            cuts = t['cuts'][0]
            fname = str(names[t['features'][0]]) if len(names) > t['features'][0] else t['name']
            unit = _original_units(model, fname) if model is not None else None
            lim = None if cuts is None else np.concatenate([[-np.inf], cuts, [np.inf]])
            if lim is not None and unit:
                lim = lim * unit[1] + unit[0]
            cats = (t.get('categories') or [None])[0]
            for k, (s, sd) in enumerate(zip(t['scores'], t['std'])):
                lo, hi = (np.nan, np.nan) if lim is None else (lim[k], lim[k + 1])
                rows.append(dict(term=t['name'], term_en=labels.name(fname), bin=str(k),
                                 bin_en=labels.bin(fname, lo, hi, cats[k] if cats and k < len(cats) else None),
                                 lower=lo, upper=hi, score=float(s), std=float(sd)))
            if t.get('missing_score') is not None:
                rows.append(dict(term=t['name'], term_en=labels.name(fname), bin='missing',
                                 bin_en='Missing (value learned by EBM)', lower=np.nan, upper=np.nan,
                                 score=t['missing_score'], std=np.nan))
        else:
            pname = ' & '.join(str(names[f]) if len(names) > f else str(f) for f in t['features'])
            for (i, j), s in np.ndenumerate(t['scores']):
                rows.append(dict(term=t['name'], term_en=labels.name(pname), bin=f'{i},{j}', bin_en='', lower=np.nan,
                                 upper=np.nan, score=float(s), std=float(t['std'][i, j])))
    return pd.DataFrame(rows)


def equation_text(task, labels=None):
    """The fitted model written as an equation, with its terms ordered by importance (English labels, the
    internal name in brackets)."""
    labels = labels or Labels()
    card = task['model_card']
    names = card.get('input_features')
    names = list(card.get('feature_names_in', [])) if names is None else names
    ts = sorted(terms(card), key=lambda t: -np.nan_to_num(t['importance']))
    b0 = float(np.ravel(card['intercept'])[0])
    lines = [f"{task['variant']} · {task['balancing']} · {task['optimizer']} · fold {task['fold']}",
             f"hyperparameters: {task.get('best_params')}", '',
             'logit p(x) = β₀ + Σ_j f_j(x_j)' + (' + Σ_(j,k) f_jk(x_j, x_k)' if any(len(t['features']) == 2 for t in ts)
                                                else '') + (' − log(w₁/w₀)' if card.get('offset') else ''),
             f'β₀ = {b0:.6f}']
    if card.get('offset'):
        lines.append(f"log(w₁/w₀) = {card['offset']:.6f}   (prior correction of the class weights)")
    lines += ['', 'Terms (importance = mean |f| on the training data); each f is a step function whose values per '
              'bin are in the _terms.csv file:']
    def label(t):
        raw = ' & '.join(str(names[f]) if len(names) > f else str(f) for f in t['features'])
        return f"{labels.name(raw)} [{t['name']}]"
    lines += [f"  {label(t):<70s} importance {t['importance']:.4f}" for t in ts]
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _numeric_edges(sub):
    """Bin edges in original units for a continuous term (more than 12 bins, or bins without category labels).

    The open ends (−∞, +∞) are drawn one median bin width beyond the outermost cut. None for categorical terms.
    """
    lo, hi = sub.lower.to_numpy(float), sub.upper.to_numpy(float)
    if len(sub) < 2 or not np.isfinite(np.concatenate([lo, hi])).any():
        return None
    if len(sub) <= 12 and (sub.bin_en != '').all():
        return None
    edges = np.concatenate([lo, hi[-1:]])
    finite = edges[np.isfinite(edges)]
    if finite.size < 2:
        return None
    width = float(np.median(np.diff(finite))) or 1.0
    if not np.isfinite(edges[0]):
        edges[0] = finite[0] - width
    if not np.isfinite(edges[-1]):
        edges[-1] = finite[-1] + width
    return edges if np.all(np.diff(edges) > 0) else None


def fig_shape_functions(task, path, labels=None):
    """Shape functions of the main effects with ±2 standard deviations across bags."""
    plt = _plt()
    tab = terms_table(task, labels)
    tab = tab[tab.bin != 'missing']
    mains = [t for t in terms(task['model_card']) if len(t['features']) == 1]
    if not mains:
        return
    cols = 4
    rows = int(np.ceil(len(mains) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax, t in zip(axes.ravel(), mains):
        sub = tab[tab.term == t['name']]
        edges = _numeric_edges(sub)
        if edges is not None:                                   # continuous: steps over the original units
            ax.stairs(sub.score.to_numpy(), edges, color='#1f4e79', baseline=None)
            ax.stairs((sub.score + 2 * sub['std']).to_numpy(), edges, baseline=(sub.score - 2 * sub['std']).to_numpy(),
                      fill=True, alpha=0.25, color='#2e75b6', linewidth=0)
        else:                                                   # categories / few bins: one step per bin
            x = np.arange(len(sub))
            ax.step(x, sub.score, where='mid', color='#1f4e79')
            ax.fill_between(x, sub.score - 2 * sub['std'], sub.score + 2 * sub['std'], step='mid', alpha=0.25,
                            color='#2e75b6', linewidth=0)
            if (sub.bin_en != '').all():
                ax.set_xticks(x)
                ax.set_xticklabels(list(sub.bin_en), fontsize=6, rotation=60, ha='right')
        ax.axhline(0, color='grey', linewidth=0.6)
        ax.set_title(sub.term_en.iloc[0] if len(sub) else t['name'], fontsize=9)
        ax.set_ylabel('logit contribution', fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(mains):]:
        ax.axis('off')
    fig.suptitle(f"{task['variant']} · {task['balancing']} · {task['optimizer']} · fold {task['fold']}: "
                 'shape functions f_j (logit) ± 2 SD across bags', fontsize=10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pairs(task, path, labels=None):
    """Pair tables (logit) of the interactions, if any."""
    labels = labels or Labels()
    card = task['model_card']
    names = card.get('input_features')
    names = list(card.get('feature_names_in', [])) if names is None else names
    pairs = [t for t in terms(card) if len(t['features']) == 2]
    if not pairs:
        return
    plt = _plt()
    cols = min(4, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    for ax, t in zip(axes.ravel(), pairs):
        m = np.abs(t['scores']).max() or 1
        im = ax.imshow(t['scores'], cmap='RdBu_r', vmin=-m, vmax=m, aspect='auto', origin='lower')
        a, b = (str(names[f]) if len(names) > f else str(f) for f in t['features'])
        ax.set_title(labels.name(f'{a} & {b}'), fontsize=8)
        ax.set_ylabel(labels.name(a), fontsize=7)
        ax.set_xlabel(labels.name(b), fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grid(plt, n, height, cols=4):
    """Figure with ``n`` panels in rows of ``cols``; unused panels are hidden."""
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, min(cols, n), figsize=(4 * min(cols, n), height * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    return fig, axes


def _combinations(res):
    by = {}
    for r in res:
        by.setdefault((r['variant'], r['balancing'], r['optimizer']), []).append(r)
    return by


def fig_pr_and_calibration(res, folder):
    """Precision-recall curves and binned calibration, pooled over folds, per combination."""
    from sklearn.metrics import precision_recall_curve
    plt = _plt()
    by = _combinations([r for r in res if r.get('p_test') is not None])
    for kind in ('pr', 'calibration'):
        groups = sorted({(v, b) for v, b, _ in by})
        fig, axes = _grid(plt, len(groups), 3.8)
        for ax, (v, b) in zip(axes.ravel(), groups):
            for (v2, b2, o), rs in sorted(by.items()):
                if (v2, b2) != (v, b):
                    continue
                y = np.concatenate([r['y_test'] for r in rs])
                p = np.concatenate([r['p_test'] for r in rs])
                if kind == 'pr':
                    prec, rec, _ = precision_recall_curve(y, p)
                    ax.plot(rec, prec, label=o, linewidth=1)
                else:
                    q = np.quantile(p, np.linspace(0, 1, 11))
                    idx = np.clip(np.searchsorted(q[1:-1], p, side='right'), 0, 9)
                    pts = [(p[idx == k].mean(), y[idx == k].mean()) for k in range(10) if (idx == k).sum() >= MIN_BIN]
                    if pts:
                        ax.plot(*zip(*pts), marker='o', markersize=3, label=o, linewidth=1)
            if kind == 'calibration':
                ax.plot([0, 1], [0, 1], color='grey', linewidth=0.6)
                ax.set_xlabel('mean predicted')
                ax.set_ylabel('observed proportion')
            else:
                ax.set_xlabel('recall')
                ax.set_ylabel('precision')
            ax.set_title(f'{v} · {b}', fontsize=9)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(folder) / ('pr_curves.png' if kind == 'pr' else 'calibration.png'), dpi=150)
        plt.close(fig)


def fig_convergence(res, folder):
    """Best inner AUC-PR so far against the evaluation number (mean over folds), per optimizer."""
    plt = _plt()
    by = _combinations(res)
    groups = sorted({(v, b) for v, b, _ in by})
    fig, axes = _grid(plt, len(groups), 3.5)
    for ax, (v, b) in zip(axes.ravel(), groups):
        for (v2, b2, o), rs in sorted(by.items()):
            if (v2, b2) != (v, b):
                continue
            curves = []
            for r in rs:
                s = np.array([h['score'] if np.isfinite(h['score']) else -np.inf for h in r['history']])
                curves.append(np.maximum.accumulate(s))
            n = min(len(c) for c in curves)
            ax.plot(np.arange(1, n + 1), np.mean([c[:n] for c in curves], 0), label=o, linewidth=1)
        ax.set_title(f'{v} · {b}', fontsize=9)
        ax.set_xlabel('evaluation')
        ax.set_ylabel('best inner AUC-PR')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(folder) / 'convergence.png', dpi=150)
    plt.close(fig)


def fig_auc_by_combination(folds, folder):
    """AUC-PR of every fold (points) and its mean (bar), per combination."""
    plt = _plt()
    folds = folds.assign(combination=folds.variant + ' · ' + folds.balancing + ' · ' + folds.optimizer)
    order = list(dict.fromkeys(folds.combination))
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(order)), 4.5))
    means = folds.groupby('combination', sort=False).auc_pr.mean().reindex(order)
    ax.bar(range(len(order)), means, color='#d9e2f3', edgecolor='#1f4e79')
    for i, c in enumerate(order):
        v = folds.loc[folds.combination == c, 'auc_pr']
        ax.scatter(np.full(len(v), i), v, s=8, color='#1f4e79')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_ylabel('AUC-PR (test fold)')
    fig.tight_layout()
    fig.savefig(Path(folder) / 'auc_pr_by_combination.png', dpi=150)
    plt.close(fig)


def oversampling_table(res):
    """The SMOTE / ADASYN diagnostics of every task, one row per task and feature."""
    rows = []
    for r in res:
        o = r.get('oversampling')
        if not o:
            continue
        for f in o['features']:
            rows.append(dict(variant=r['variant'], balancing=r['balancing'], optimizer=r['optimizer'], fold=r['fold'],
                             n_real=o['n_real'], n_synthetic=o['n_synthetic'], **f))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- entry point
def build(res, folds, folder, labels=None, progress=False):
    """Write every figure and table of the report into ``folder``. Returns the number of models described.

    With ``progress``, a line every ``PROGRESS_EVERY`` models and before the figures of the whole level.
    """
    import time
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n, t0, total = 0, time.perf_counter(), sum(1 for r in res if r.get('model_card'))
    for r in res:
        if not r.get('model_card'):
            continue
        sub = folder / 'models' / f"{r['variant']}__{r['balancing']}__{r['optimizer']}"
        sub.mkdir(parents=True, exist_ok=True)
        stem = f"fold_{r['fold']:02d}"
        (sub / f'{stem}_equation.txt').write_text(equation_text(r, labels), encoding='utf-8')
        terms_table(r, labels).to_csv(sub / f'{stem}_terms.csv', index=False, lineterminator='\n')
        fig_shape_functions(r, sub / f'{stem}_shape_functions.png', labels)
        fig_pairs(r, sub / f'{stem}_pairs.png', labels)
        n += 1
        if progress and (n % PROGRESS_EVERY == 0 or n == total):
            print(f'[report] {(time.perf_counter() - t0) / 60:5.1f} min  {n}/{total} models described', flush=True)
    if progress:
        print('[report] figures of the whole level (curves, convergence, AUC-PR per fold)', flush=True)
    fig_pr_and_calibration(res, folder)
    fig_convergence(res, folder)
    fig_auc_by_combination(folds, folder)
    ov = oversampling_table(res)
    if not ov.empty:
        ov.to_csv(folder / 'oversampling.csv', index=False, lineterminator='\n')
    return n

