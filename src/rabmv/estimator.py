"""RabMV estimator: an explainable boosting machine written from scratch.

Reorganization of EBMP v1 (``c1/ebm_propio.py``, fingerprint 0396E530B408B66A) without behavioural
changes; the regression test checks it against the frozen copy of v1.

RabMV (Registry-Aware Boosting Machine of Valledupar) is a generalized additive model for binary
classification,

.. math::

    \\operatorname{logit} \\hat p(x) = \\beta_0 + \\sum_{j=1}^{p} f_j(x_j)
                                   \\; [+ \\sum_{(j,k) \\in \\mathcal{I}} f_{jk}(x_j, x_k)],

in which each shape function :math:`f_j` is piecewise constant over the bins (quantiles) of its
feature and is learned by **cyclic Newton boosting**: each round visits the features one by one and
adds to each :math:`f_j` a small step that reduces the weighted logistic loss.

What distinguishes it from a standard EBM (InterpretML), besides being written only with numpy:

1. **Optimal step by dynamic programming.** The standard EBM fits a small tree greedily at each step.
   RabMV chooses, among *all* partitions of the ordered bins into at most ``max_leaves`` contiguous
   segments, the one that maximizes the Newton gain :math:`\\sum_s G_s^2 / (H_s + \\lambda)`, exactly
   (Bellman equation, O(L·B²) per step). Each segment value is the regularized Newton step
   :math:`-G_s / (H_s + \\lambda)`.
2. **Vectorized bags.** The ``n_bags`` bags (each with its own early-stopping split) are trained at
   once as columns of a single matrix: the cost per step is one pass over the data for all bags.
3. **Exact explanation.** Because the model is additive, the contribution of each feature to the
   prediction of a case is exactly :math:`f_j(x_j)` (centred): no sampled SHAP and no background of
   real cases are needed (method ``contributions``).
4. **Uncertainty of each shape function**: the standard deviation across bags.
5. **Registry awareness** (``missing='marginalize'``): a concept that the registry form did not measure
   for a case contributes 0, and is excluded from that term's gradient, hessian, cuts and centring.
   Pairs are allowed: their grid is nested in the main-effect grids and each pair table is purified
   (functional ANOVA, ``core.purification``), so that giving 0 to a pair with an unmeasured concept is the
   exact average of that pair over the unmeasured concept given the measured one.

Design (documented decisions):

* Main effects by default (``interactions=0``): with few positives, each interaction is a
  two-dimensional table estimated with very few events per cell; it adds variance, pair selection is
  unstable across folds and the equation loses its feature-by-feature reading. Interactions exist as an
  option for the ablation study.
* Ablation study: each component of the step can be removed separately without touching the rest of
  the algorithm: ``split='greedy'`` replaces dynamic programming with the standard EBM greedy tree and
  ``newton=False`` replaces the Newton step with a gradient step. Other components are removed with
  parameters: ``n_bags=1`` (no bags), ``validation_size=0`` (no early stopping), ``reg_lambda=0``
  (no regularization); ``interactions>0`` adds pairs.
* EBMP v2 (``smoothing > 0``): with few events, each segment of a shape function is noisy. The final
  smoothing makes neighbouring bins borrow information (Whittaker smoother) and the intercept is
  refitted to preserve calibration. In a simulation with the structure of the Valledupar study (2412
  cases, 12 structurally zero factors, very different prevalence by sex) its AUC-PR gain grew as events
  decreased (+0.007 with 242 positives, +0.014 to +0.027 with 60). The strength is chosen by internal
  validation; 0 reproduces EBMP v1 exactly.
* Bins from training quantiles (like the standard EBM); a feature with few distinct values (binary,
  one-hot, ordinal) uses its values directly.
* Sample weights (``sample_weight``) in the gradient and hessian: this makes it work with
  class_weight='balanced' through a wrapper, and with SMOTE/ADASYN.
* Identifiability: at the end each :math:`f_j` is centred (weighted mean 0 on the training data) and its
  mean is moved to the intercept.

Properties:

* For each feature, the chosen partition is optimal among all partitions into at most L contiguous
  segments (Bellman's principle of optimality).
* The Newton step per segment minimizes the quadratic approximation of the loss in that segment; with a
  small ``learning_rate`` and :math:`\\lambda > 0` the training loss decreases at each step (block
  descent on a convex function).
* Complexity: training O(R · p · (n·K + K·L·B²)); prediction O(n · p · log B).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from .core.binning import quantile_cuts
from .core.numeric import LOGIT_MAX, log_loss, sigmoid
from .core.purification import coarse_of_fine, nested_cuts, purify
from .core.segmentation import SPLIT_STRATEGIES, binary_step
from .core.smoothing import smooth_curve

MISSING_OPTIONS = ('reject', 'marginalize')


class RabMVClassifier(ClassifierMixin, BaseEstimator):
    """Registry-Aware Boosting Machine of Valledupar (main effects and, optionally, pairs).

    Parameters
    ----------
    max_bins : int, default=32
        Maximum bins per feature (training quantiles).
    learning_rate : float, default=0.05
        Fraction of the Newton step applied at each update, in (0, 1].
    max_leaves : int, default=3
        Maximum contiguous segments of each shape-function update.
    min_samples_leaf : int, default=10
        Minimum training cases per segment.
    reg_lambda : float, default=1.0
        L2 regularization of each segment value.
    max_rounds : int, default=2000
        Maximum rounds (each round visits all features).
    early_stopping_rounds : int, default=50
        Rounds without improvement of the validation loss after which a bag stops.
    early_stopping_tolerance : float, default=1e-5
        Minimum improvement of the validation loss (mean log loss) that counts as an improvement.
    validation_size : float, default=0.15
        Stratified fraction of each bag held out for early stopping (0 = no early stopping).
    n_bags : int, default=8
        Bags (different validation splits); shape functions are averaged.
    interactions : int, default=0
        Pairs of features modelled after the main effects (0 = main effects only).
    interaction_bins : int, default=8
        Bins per feature in the pair tables.
    random_state : int, default=42
        Seed of the validation splits.
    split : {'optimal', 'greedy'}, default='optimal'
        How the segments of each update are chosen: optimal partition by dynamic programming (RabMV)
        or best-first greedy tree (standard EBM). 'greedy' exists only for the ablation study.
    newton : bool, default=True
        True: Newton step (gradient / hessian). False: gradient step (gradient / case weight), only
        for the ablation study.
    smoothing : float, default=0.0
        Strength κ of the final smoothing of each ordered shape function (more than 2 bins):
        neighbouring bins borrow information (``smooth_curve``), which reduces variance when each bin
        has few events. The intercept is then refitted so that the mean training probability still
        equals the observed proportion. 0 = EBMP v1.
    missing : {'reject', 'marginalize'}, default='reject'
        What to do with a ``NaN`` in X. ``'reject'`` (as v1) raises an error. ``'marginalize'``
        (registry-aware method) treats it as "concept not measured for this case": the case contributes
        no gradient or hessian to that feature's function, cuts and centring use only measured cases,
        and at prediction time the feature contributes 0 (the average). Without any NaN and without
        interactions, both values produce exactly the same model as v1. With interactions, each pair is
        fitted on the cases that have both concepts measured, its grid is nested in the main-effect grids
        and its table is purified (rows and columns with weighted mean 0, mass moved to the main
        effects); an unmeasured concept makes the pair contribute 0, which is then the exact average of
        the pair over that concept given the other one.
    categorical_features : sequence of int or None, default=None
        Indices of nominal columns (unordered categories, such as the health insurance scheme). Each
        category gets its own Newton step, without grouping "neighbouring" categories, because their
        order is arbitrary. They are not smoothed. With None (default) all columns are ordered, as v1.

    Attributes
    ----------
    intercept_ : float
    term_names_ : list of str
        Name of each term (feature or pair "a & b").
    term_scores_ : list of numpy.ndarray
        Value of each term per bin (or per cell for pairs), centred.
    term_scores_std_ : list of numpy.ndarray
        Standard deviation across bags of each function.
    bin_edges_ : list of numpy.ndarray
        Cuts of each feature.
    term_importances_ : numpy.ndarray
        Weighted mean of |f(x)| of each term on the training data.
    best_rounds_ : numpy.ndarray
        Round with the best validation loss for each bag (main effects).
    validation_history_ : numpy.ndarray of shape (rounds, n_bags)
        Validation loss per round and bag (main effects).
    categorical_features_ : list of int
    interaction_pairs_ : list of (int, int)
    """

    def __init__(self, max_bins=32, learning_rate=0.05, max_leaves=3, min_samples_leaf=10,
                 reg_lambda=1.0, max_rounds=2000, early_stopping_rounds=50, early_stopping_tolerance=1e-5,
                 validation_size=0.15, n_bags=8, interactions=0, interaction_bins=8, random_state=42,
                 split='optimal', newton=True, smoothing=0.0, missing='reject', categorical_features=None):
        """Store the hyperparameters as given (scikit-learn convention: they are validated in fit)."""
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.validation_size = validation_size
        self.n_bags = n_bags
        self.interactions = interactions
        self.interaction_bins = interaction_bins
        self.random_state = random_state
        self.split = split
        self.newton = newton
        self.smoothing = smoothing
        self.missing = missing
        self.categorical_features = categorical_features

    def __sklearn_tags__(self):
        """scikit-learn tags: binary classifier; accepts NaN only when missing values are marginalized."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.allow_nan = self.missing == 'marginalize'
        return tags

    def _marginalizes(self):
        """True if NaN is treated as a concept that was not measured."""
        return self.missing == 'marginalize'

    # ------------------------------------------------------------------ validation
    def _validate_hyperparameters(self):
        """Check the hyperparameters and raise ValueError with a clear message if any is invalid."""
        rules = [
            (isinstance(self.max_bins, (int, np.integer)) and self.max_bins >= 2, 'max_bins must be an integer ≥ 2'),
            (0 < float(self.learning_rate) <= 1, 'learning_rate must be in (0, 1]'),
            (isinstance(self.max_leaves, (int, np.integer)) and self.max_leaves >= 1,
             'max_leaves must be an integer ≥ 1'),
            (int(self.min_samples_leaf) >= 1, 'min_samples_leaf must be ≥ 1'),
            (float(self.reg_lambda) >= 0, 'reg_lambda must be ≥ 0'),
            (int(self.max_rounds) >= 1, 'max_rounds must be ≥ 1'),
            (int(self.early_stopping_rounds) >= 1, 'early_stopping_rounds must be ≥ 1'),
            (float(self.early_stopping_tolerance) >= 0, 'early_stopping_tolerance must be ≥ 0'),
            (0 <= float(self.validation_size) < 0.5, 'validation_size must be in [0, 0.5)'),
            (int(self.n_bags) >= 1, 'n_bags must be ≥ 1'),
            (int(self.interactions) >= 0, 'interactions must be ≥ 0'),
            (int(self.interaction_bins) >= 2, 'interaction_bins must be ≥ 2'),
            (self.split in SPLIT_STRATEGIES, f'split must be one of {sorted(SPLIT_STRATEGIES)}'),
            (isinstance(self.newton, (bool, np.bool_)), 'newton must be True or False'),
            (float(self.smoothing) >= 0 and np.isfinite(float(self.smoothing)), 'smoothing must be ≥ 0'),
            (self.missing in MISSING_OPTIONS, "missing must be 'reject' or 'marginalize'"),
        ]
        for ok, message in rules:
            if not ok:
                raise ValueError(message)

    def _bag_splits(self, y, rng):
        """Training and validation masks of each bag (stratified by class).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Boolean matrices (n, K): training and validation.
        """
        n, K = len(y), int(self.n_bags)
        val = np.zeros((n, K), bool)
        if self.validation_size > 0:
            for k in range(K):
                for c in (0, 1):
                    idx = np.flatnonzero((y == c) & self._has_weight)
                    m = int(round(self.validation_size * len(idx)))
                    if 0 < m < len(idx):
                        val[rng.choice(idx, m, replace=False), k] = True
        return ~val, val

    # ------------------------------------------------------------------ core
    def _boosting(self, terms, n_cells, ordered, y, w, train, val, Z0, measured=None):
        """Cyclic Newton boosting of a set of terms, with K bags at once.

        Parameters
        ----------
        terms : list of numpy.ndarray of shape (n,)
            Bin (or cell) index of each case in each term.
        n_cells : list of int
            Number of bins (or cells) of each term.
        ordered : list of bool
            True if the bins are ordered (optimal segmentation is used); False for nominal features and
            pair tables (Newton step per cell).
        y : numpy.ndarray of shape (n,)
        w : numpy.ndarray of shape (n,)
            Sample weights.
        train, val : numpy.ndarray of shape (n, K)
            Training and validation masks of each bag.
        Z0 : numpy.ndarray of shape (n, K)
            Initial logits (intercept and, in the pair stage, the main effects).
        measured : list, optional
            Per term, ``None`` if every case has it measured, or a boolean array (n,) with the measured
            cases. A case that was not measured contributes no gradient, hessian or count to that term and
            does not receive its update (marginalization). With ``None`` the arithmetic is that of v1.

        Returns
        -------
        (list of numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Functions per term of shape (K, cells), final logits (n, K), best round per bag and
            validation-loss history.
        """
        K, n = Z0.shape[1], Z0.shape[0]
        Wt_all, Wv = w[:, None] * train, w[:, None] * val
        Nt_all = train * (w[:, None] > 0)                        # cases (with weight) per bag
        has_val = bool(val.any())
        F = [np.zeros((K, b)) for b in n_cells]
        Z = Z0.copy()
        best = np.full(K, np.inf)
        best_F = [f.copy() for f in F]
        best_round = np.zeros(K, int)
        no_improvement = np.zeros(K, int)
        active = np.ones(K, bool)
        history = []
        lr, lam = float(self.learning_rate), float(self.reg_lambda)
        msl, L = int(self.min_samples_leaf), int(self.max_leaves)
        tol = float(self.early_stopping_tolerance)
        newton = bool(self.newton)
        segment = SPLIT_STRATEGIES[self.split]
        measured = [None] * len(n_cells) if measured is None else measured
        masks = [None if m is None else m[:, None] for m in measured]   # (n, 1) to multiply by bag

        def prepare(act):
            """Working arrays with the active bags only (stopped bags are no longer recomputed)."""
            k = len(act)
            col = np.arange(k)
            idx = [(bins[:, None] + B * col[None, :]).ravel() for bins, B in zip(terms, n_cells)]
            cnt = [np.bincount(ix, weights=(Nt_all[:, act] if m is None else Nt_all[:, act] * m).ravel(),
                               minlength=k * B).reshape(k, B)
                   for ix, B, m in zip(idx, n_cells, masks)]
            return k, idx, cnt, Wt_all[:, act]

        act = np.flatnonzero(active)
        k, indices, counts, Wt = prepare(act)
        Zw = Z[:, act].copy()
        for rnd in range(1, int(self.max_rounds) + 1):
            for t, B in enumerate(n_cells):
                p = sigmoid(Zw)
                Wm = Wt if masks[t] is None else Wt * masks[t]          # without unmeasured cases
                g = ((p - y[:, None]) * Wm).ravel()
                h = (p * (1 - p) * Wm).ravel() if newton else Wm.ravel()   # Newton or gradient
                G = np.bincount(indices[t], weights=g, minlength=k * B).reshape(k, B)
                H = np.bincount(indices[t], weights=h, minlength=k * B).reshape(k, B)
                N = counts[t]
                if not ordered[t]:                               # pair table / nominal: Newton per cell
                    step = np.where(N >= msl, -G / (H + lam + 1e-12), 0.0)
                elif B == 2:                                     # binary feature: closed form
                    step = binary_step(G, H, N, L, lam, msl)
                else:                                            # optimal segmentation (Bellman)
                    step = segment(G, H, N, L, lam, msl)
                F[t][act] += lr * step
                if masks[t] is None:
                    Zw += lr * step.ravel()[indices[t]].reshape(n, k)
                else:                                            # unmeasured cases receive no update
                    Zw += lr * step.ravel()[indices[t]].reshape(n, k) * masks[t]
                np.clip(Zw, -LOGIT_MAX, LOGIT_MAX, out=Zw)
            Z[:, act] = Zw
            if not has_val:
                best_F, best_round[:] = [f.copy() for f in F], rnd
                continue
            loss = np.full(K, np.inf)
            loss[act] = log_loss(y, Zw, Wv[:, act])
            history.append(np.where(active, loss, np.nan))
            improved = active & (loss < best - tol)              # minimum improvement (avoids useless rounds)
            if improved.any():
                best[improved] = loss[improved]
                best_round[improved] = rnd
                for t in range(len(F)):
                    best_F[t][improved] = F[t][improved]
            no_improvement = np.where(improved, 0, no_improvement + 1)
            new_active = active & (no_improvement < int(self.early_stopping_rounds))
            if not new_active.any():
                break
            if (new_active != active).any():                     # some bag stopped: fewer columns
                active = new_active
                act = np.flatnonzero(active)
                k, indices, counts, Wt = prepare(act)
                Zw = Z[:, act].copy()
        Zf = Z0.copy()
        rows = np.arange(K)
        for t, bins in enumerate(terms):
            if masks[t] is None:
                Zf += best_F[t][rows[None, :], bins[:, None]]
            else:
                Zf += best_F[t][rows[None, :], bins[:, None]] * masks[t]
        return best_F, Zf, best_round, np.array(history)

    def _select_pairs(self, Xb, y, w, Z, both=None):
        """Choose the ``interactions`` pairs with the largest Newton gain on the residual.

        ``both`` maps each pair to the mask of cases with both concepts measured (None = all); only those
        cases enter the gain of that pair.

        Returns
        -------
        list of (int, int)
        """
        p = sigmoid(Z.mean(1))
        g, h = (p - y) * w, p * (1 - p) * w
        cand = []
        nb = [int(self.pair_bin_edges_[j].size + 1) for j in range(Xb.shape[1])]
        for a in range(Xb.shape[1]):
            for b in range(a + 1, Xb.shape[1]):
                if nb[a] < 2 or nb[b] < 2:
                    continue
                cell = self._pair_cell(a, b)
                m = None if both is None else both[a, b]
                G = np.bincount(cell, weights=g if m is None else g * m, minlength=nb[a] * nb[b])
                H = np.bincount(cell, weights=h if m is None else h * m, minlength=nb[a] * nb[b])
                cand.append(((G ** 2 / (H + self.reg_lambda + 1e-12)).sum(), a, b))
        cand.sort(reverse=True)
        return [(a, b) for _, a, b in cand[:int(self.interactions)]]

    def _pair_cell(self, a, b, Xb=None):
        """Cell index of the (a, b) pair table on the coarse grid of each case."""
        Xb = self._Xb_pair if Xb is None else Xb
        nb_b = int(self.pair_bin_edges_[b].size + 1)
        return Xb[:, a] * nb_b + Xb[:, b]

    # ------------------------------------------------------------------ scikit-learn API
    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Numeric features.
        y : array-like of shape (n,)
            Binary class.
        sample_weight : array-like of shape (n,), optional
            Sample weights (e.g. class_weight='balanced').

        Returns
        -------
        RabMVClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            On infinite values, NaN (except with ``missing='marginalize'``), a column without any measured
            value, if ``y`` does not have exactly two classes, or if a hyperparameter is invalid.
        """
        self._validate_hyperparameters()
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_min_samples=2,
                             ensure_all_finite='allow-nan' if self._marginalizes() else True)
        check_classification_targets(y)                          # rejects continuous targets
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('Only binary classification is supported. RabMV needs exactly 2 classes '
                             f'in y (got {len(self.classes_)}).')
        y01 = (y == self.classes_[1]).astype(float)
        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError('sample_weight must be non-negative and not all zero')
        w = w * (len(w) / w.sum())                                # mean-1 scale (does not change the optimum)
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape

        # measured cases of each feature (None = all; then the arithmetic is exactly that of v1)
        M = ~np.isnan(X)
        measured = [None if M[:, j].all() else M[:, j] for j in range(p)]
        empty = [j for j in range(p) if not M[:, j].any()]
        if empty:
            raise ValueError(f'columns {empty} have no measured value')

        # nominal columns: each category is its own bin and neighbouring categories are not grouped
        categorical = set() if self.categorical_features is None else self.categorical_features
        if (isinstance(categorical, (str, bytes)) or not all(isinstance(j, (int, np.integer)) for j in categorical)
                or any(not 0 <= j < p for j in categorical) or len(set(categorical)) != len(list(categorical))):
            raise ValueError(f'categorical_features must be a list of distinct column indices between 0 and {p - 1}')
        categorical = {int(j) for j in categorical}
        for j in sorted(categorical):
            values = X[measured[j] if measured[j] is not None else slice(None), j]
            if np.unique(values).size > int(self.max_bins):
                raise ValueError(f'categorical column {j} has more categories than max_bins ({self.max_bins})')
        self.categorical_features_ = sorted(categorical)

        # bins of each feature (measured cases only)
        self.bin_edges_ = [quantile_cuts(X[:, j], int(self.max_bins), w) if m is None
                           else quantile_cuts(X[m, j], int(self.max_bins), w[m])
                           for j, m in enumerate(measured)]
        Xb = np.column_stack([np.searchsorted(c, X[:, j], side='right') if m is None
                              else np.where(m, np.searchsorted(c, np.where(m, X[:, j], 0.0), side='right'), 0)
                              for j, (c, m) in enumerate(zip(self.bin_edges_, measured))])
        n_bins = [c.size + 1 for c in self.bin_edges_]
        term_weights = [w if m is None else w * m for m in measured]   # for centring and importances

        # bags and initial intercept (weighted log-odds of each bag)
        self._has_weight = w > 0
        train, val = self._bag_splits(y01, rng)
        del self._has_weight
        Wt = w[:, None] * train
        rate = np.clip((Wt * y01[:, None]).sum(0) / Wt.sum(0), 1e-6, 1 - 1e-6)
        b0 = np.log(rate / (1 - rate))
        Z0 = np.repeat(b0[None, :], n, 0)

        # stage 1: main effects
        F, Z, self.best_rounds_, self.validation_history_ = self._boosting(
            [Xb[:, j] for j in range(p)], n_bins, [j not in categorical for j in range(p)], y01, w, train, val,
            Z0, measured)
        terms, scores, std = list(self._feature_names(p)), [], []
        for f in F:
            scores.append(f.mean(0))
            std.append(f.std(0, ddof=1) if f.shape[0] > 1 else np.zeros(f.shape[1]))
        self.interaction_pairs_ = []

        # stage 2 (optional): pairs on the residual of the main effects
        pair_measured, Fp = [], []
        if int(self.interactions) > 0 and p >= 2:
            if self._marginalizes():                 # registry-aware: grid nested in the main grid (purification)
                self.pair_bin_edges_ = [
                    nested_cuts(quantile_cuts(X[:, j] if m is None else X[m, j], int(self.interaction_bins),
                                              w if m is None else w[m]), self.bin_edges_[j])
                    for j, m in enumerate(measured)]
            else:                                    # v1
                self.pair_bin_edges_ = [quantile_cuts(X[:, j], int(self.interaction_bins), w) for j in range(p)]
            self._Xb_pair = self._bin_pairs(X)
            both = {}
            for a in range(p):
                for b in range(a + 1, p):
                    ma, mb = measured[a], measured[b]
                    both[a, b] = None if ma is None and mb is None else (
                        (np.ones(n, bool) if ma is None else ma) & (np.ones(n, bool) if mb is None else mb))
            self.interaction_pairs_ = self._select_pairs(Xb, y01, w, Z, both)
            if self.interaction_pairs_:
                cells = [self._pair_cell(a, b) for a, b in self.interaction_pairs_]
                n_c = [(self.pair_bin_edges_[a].size + 1) * (self.pair_bin_edges_[b].size + 1)
                       for a, b in self.interaction_pairs_]
                pair_measured = [both[a, b] for a, b in self.interaction_pairs_]
                Fp, Z, _, _ = self._boosting(cells, n_c, [False] * len(cells), y01, w, train, val, Z,
                                             pair_measured if any(m is not None for m in pair_measured) else None)
                names = self._feature_names(p)
                for (a, b), f in zip(self.interaction_pairs_, Fp):
                    terms.append(f'{names[a]} & {names[b]}')
                    scores.append(f.mean(0))
                    std.append(f.std(0, ddof=1) if f.shape[0] > 1 else np.zeros(f.shape[1]))
            del self._Xb_pair

        # EBMP v2: final smoothing of the ordered functions (pairs are not smoothed)
        kappa = float(self.smoothing)
        if kappa > 0:
            for j in range(p):
                if n_bins[j] > 2 and j not in categorical:       # nominal features have no neighbours
                    W = np.bincount(Xb[:, j], weights=term_weights[j], minlength=n_bins[j])
                    scores[j] = smooth_curve(scores[j], W, kappa)

        indices = [Xb[:, j] for j in range(p)] + ([self._pair_cell(a, b, self._bin_pairs(X))
                                                   for a, b in self.interaction_pairs_]
                                                  if self.interaction_pairs_ else [])
        pair_weights = [w if m is None else w * m for m in pair_measured]

        # registry-aware pairs: purification (functional ANOVA); the prediction of every cell is unchanged
        if self._marginalizes() and self.interaction_pairs_:
            self._purify_pairs(F, Fp, scores, std, indices, pair_weights, p)

        # identifiability: each term with weighted mean 0; its mean goes to the intercept
        self.intercept_ = float(b0.mean())
        imp = []
        weights = term_weights + pair_weights                    # centred on the cases where the term is measured
        masks = list(measured) + list(pair_measured)
        for t, idx in enumerate(indices):
            mean = float(np.average(scores[t][idx], weights=weights[t]))   # centred on measured cases
            scores[t] = scores[t] - mean
            self.intercept_ += mean
            imp.append(float(np.average(np.abs(scores[t][idx]), weights=weights[t])))
        if kappa > 0:                       # smoothing changes the mean of p: refit the intercept
            z = self.intercept_ + sum(scores[t][idx] if masks[t] is None
                                      else scores[t][idx] * masks[t] for t, idx in enumerate(indices))
            c = 0.0
            for _ in range(25):             # one-dimensional Newton: Σ w (σ(z + c) − y) = 0
                q = sigmoid(np.clip(z + c, -LOGIT_MAX, LOGIT_MAX))
                g1, h1 = float(np.sum(w * (q - y01))), float(np.sum(w * q * (1 - q)))
                if h1 <= 0 or abs(g1) < 1e-10:
                    break
                c -= g1 / h1
            self.intercept_ += c
        self.term_names_, self.term_scores_, self.term_scores_std_ = terms, scores, std
        self.term_importances_ = np.array(imp)
        return self

    def _bin_pairs(self, X):
        """Coarse-grid indices of the pairs for a matrix X (a NaN gets bin 0; its pair is masked)."""
        return np.column_stack([np.searchsorted(c, np.nan_to_num(X[:, j], nan=-np.inf), side='right')
                                for j, c in enumerate(self.pair_bin_edges_)])

    def _purify_pairs(self, F, Fp, scores, std, indices, pair_weights, p):
        """Purify every pair table and move its row and column means to the main effects.

        The mean table (the one used for prediction, after smoothing of the main effects) is purified with
        the cell weights of the cases that have both concepts measured; each bag table is purified the same
        way to recompute the standard deviations across bags (purification is a linear projection, so the
        mean of the purified bags equals the purified mean). Predictions of cases with both concepts measured
        do not change.
        """
        K = Fp[0].shape[0]
        extra = [np.zeros((K, F[j].shape[1])) for j in range(p)]     # per-bag mass moved to each main effect
        for t, (a, b) in enumerate(self.interaction_pairs_):
            na, nb = self.pair_bin_edges_[a].size + 1, self.pair_bin_edges_[b].size + 1
            W = np.bincount(indices[p + t], weights=pair_weights[t], minlength=na * nb).reshape(na, nb)
            map_a = coarse_of_fine(self.bin_edges_[a], self.pair_bin_edges_[a])
            map_b = coarse_of_fine(self.bin_edges_[b], self.pair_bin_edges_[b])
            pure, r, c = purify(scores[p + t].reshape(na, nb), W)
            scores[p + t] = pure.ravel()
            scores[a] = scores[a] + r[map_a]
            scores[b] = scores[b] + c[map_b]
            bags = np.empty_like(Fp[t])
            for k in range(K):
                pk, rk, ck = purify(Fp[t][k].reshape(na, nb), W)
                bags[k] = pk.ravel()
                extra[a][k] += rk[map_a]
                extra[b][k] += ck[map_b]
            std[p + t] = bags.std(0, ddof=1) if K > 1 else np.zeros(bags.shape[1])
        for j in {j for pair in self.interaction_pairs_ for j in pair}:
            g = F[j] + extra[j]
            std[j] = g.std(0, ddof=1) if K > 1 else np.zeros(g.shape[1])

    def _feature_names(self, p):
        """Feature names (the input ones if known)."""
        names = getattr(self, 'feature_names_in_', None)
        return [str(v) for v in names] if names is not None else [f'x{j}' for j in range(p)]

    def contributions(self, X):
        """Exact contribution of each term to the logit of each case.

        Because the model is additive, ``decision_function(X) = intercept_ + contributions(X).sum(1)``:
        it is an exact local explanation (equivalent to SHAP for an additive model with centred terms)
        and needs no background of cases.

        Parameters
        ----------
        X : array-like of shape (n, p)

        Returns
        -------
        numpy.ndarray of shape (n, n_terms)
        """
        check_is_fitted(self, 'term_scores_')
        X = validate_data(self, X, dtype=np.float64, reset=False,
                          ensure_all_finite='allow-nan' if self._marginalizes() else True)
        p = X.shape[1]
        cols = [self.term_scores_[j][np.searchsorted(self.bin_edges_[j], X[:, j], side='right')] for j in range(p)]
        if self._marginalizes():                                # an unmeasured concept contributes 0
            cols = [np.where(np.isnan(X[:, j]), 0.0, c) for j, c in enumerate(cols)]
        if self.interaction_pairs_:
            Xp = self._bin_pairs(X)
            for t, (a, b) in enumerate(self.interaction_pairs_):
                c = self.term_scores_[p + t][self._pair_cell(a, b, Xp)]
                if self._marginalizes():                        # a pair with an unmeasured concept contributes 0
                    c = np.where(np.isnan(X[:, a]) | np.isnan(X[:, b]), 0.0, c)
                cols.append(c)
        return np.column_stack(cols)

    def decision_function(self, X):
        """Predicted logit.

        Parameters
        ----------
        X : array-like of shape (n, p)

        Returns
        -------
        numpy.ndarray of shape (n,)
        """
        check_is_fitted(self, 'term_scores_')
        return self.intercept_ + self.contributions(X).sum(1)

    def predict_proba(self, X):
        """Probability of each class.

        Parameters
        ----------
        X : array-like of shape (n, p)

        Returns
        -------
        numpy.ndarray of shape (n, 2)
            Column 1: probability of ``classes_[1]``.
        """
        check_is_fitted(self, 'term_scores_')
        p1 = sigmoid(np.clip(self.decision_function(X), -LOGIT_MAX, LOGIT_MAX))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """Predicted class with threshold 0.5 (a study usually chooses its own threshold).

        Parameters
        ----------
        X : array-like of shape (n, p)

        Returns
        -------
        numpy.ndarray of shape (n,)
        """
        check_is_fitted(self, 'term_scores_')
        return self.classes_[(self.decision_function(X) > 0).astype(int)]   # logit > 0 ⇔ p > 0.5

    def shape_functions(self):
        """Readable table of the main-effect shape functions.

        Returns
        -------
        list of dict
            Per feature: name, bins (lower and upper limits), value of f, standard deviation across bags
            and importance.
        """
        check_is_fitted(self, 'term_scores_')
        rows = []
        for j, c in enumerate(self.bin_edges_):
            lim = np.concatenate([[-np.inf], c, [np.inf]])
            rows.append({'term': self.term_names_[j], 'lower': lim[:-1].tolist(), 'upper': lim[1:].tolist(),
                         'score': self.term_scores_[j].tolist(), 'std': self.term_scores_std_[j].tolist(),
                         'importance': float(self.term_importances_[j])})
        return rows
