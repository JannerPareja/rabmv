# Marginalization with purified interactions: statements and proofs

This note proves the properties that RabMV relies on when a concept was not measured and the model has
pairwise interactions (`missing='marginalize'`, `interactions > 0`). The code is in
`src/rabmv/core/purification.py` and in `RabMVClassifier._purify_pairs`. Each statement is checked
numerically in `tests/test_purification.py` and `tests/test_purification_theory.py`.

## 1. Setting

The fitted model is an additive logit

$$
\eta(x) = \beta_0 + \sum_{j} f_j(x_j) + \sum_{(a,b)\in\mathcal P} f_{ab}(x_a, x_b),
$$

**Main effects.** Each $f_j$ is a step function on the main grid of feature $j$.

**Pair terms.** For each selected pair $(a,b)\in\mathcal P$, $f_{ab}$ is a table on a coarse grid. Its row
index is $i\in\{1,\dots,n_a\}$, the coarse bin of $x_a$; its column index is $k\in\{1,\dots,n_b\}$, the coarse
bin of $x_b$. The coarse cuts are a subset of the main cuts (`nested_cuts`), so every main bin lies inside
exactly one coarse bin (map $\kappa_j$, `coarse_of_fine`).

**Weights.** $W_{ik}\ge 0$ is the total training weight of the cases with **both** concepts measured that fall
in cell $(i,k)$. Write $\Omega=\{(i,k): W_{ik}>0\}$ for the support, $w_{i\cdot}=\sum_k W_{ik}$ for the row
weights and $w_{\cdot k}=\sum_i W_{ik}$ for the column weights.

**The rule of RabMV.** When $x_b$ was not measured for a case, the model gives $f_b$ and every $f_{ab}$ that
involves $b$ the value 0.

**Notation.** For tables $X, Y$ on the grid, $\langle X, Y\rangle_W=\sum_{i,k} W_{ik}X_{ik}Y_{ik}$. This is
an inner product on $\mathbb R^{\Omega}$ (the restriction of the tables to the support) and a semi-inner
product on the whole grid. The subspaces are:

- $\mathcal R = \{X: X_{ik} = r_i\}$, row-constant tables;
- $\mathcal C = \{X: X_{ik} = c_k\}$, column-constant tables;
- $\mathcal S = \mathcal R + \mathcal C$, additive tables.

## 2. Definition

A table $P$ is **purified** with respect to $W$ when every row and every column has weighted mean zero:

$$
\sum_k W_{ik}P_{ik}=0 \;\;\forall i, \qquad \sum_i W_{ik}P_{ik}=0 \;\;\forall k. \tag{1}
$$

## 3. Results

### Theorem 1 (purification is the $W$-orthogonal residual onto additive tables)

For every table $T$ there are vectors $r, c$ such that $P = T - r\mathbf 1^\top - \mathbf 1 c^\top$ is
purified. On the support, $P|_\Omega$ is unique and equals $T - \Pi_{\mathcal S}T$, where $\Pi_{\mathcal S}$ is
the $W$-orthogonal projection onto $\mathcal S$; equivalently, $(r,c)$ solve the weighted least-squares problem

$$
\min_{r,c}\; \sum_{(i,k)\in\Omega} W_{ik}\,(T_{ik} - r_i - c_k)^2 .
$$

The vectors $r, c$ themselves are unique up to the constants of the connected components of the bipartite
graph $G$ whose rows and columns are joined by an edge when $W_{ik}>0$. Within each component, one constant
can be added to the $r_i$ and subtracted from the $c_k$.

*Proof.* The problem is a finite-dimensional least-squares problem, so a minimizer exists. Its normal
equations are

$$
\frac{\partial}{\partial r_i}: \;\sum_k W_{ik}(T_{ik}-r_i-c_k)=0, \qquad
\frac{\partial}{\partial c_k}: \;\sum_i W_{ik}(T_{ik}-r_i-c_k)=0,
$$

which say exactly that $P$ satisfies (1). Conversely, if $P = T - r\mathbf 1^\top - \mathbf 1 c^\top$
satisfies (1), then $\langle P, S\rangle_W = 0$ for every $S\in\mathcal S$, because
$\langle P, r'\mathbf 1^\top + \mathbf 1 c'^\top\rangle_W = \sum_i r'_i\sum_k W_{ik}P_{ik} + \sum_k
c'_k\sum_i W_{ik}P_{ik} = 0$. So $P$ is the orthogonal residual, and an orthogonal residual is unique in an
inner-product space; that space is $\mathbb R^\Omega$.

For the second claim, two additive tables agree on $\Omega$ if and only if their difference
$d_{ik} = \delta r_i + \delta c_k$ vanishes on every edge of $G$. That holds if and only if $\delta r_i =
-\delta c_k = \text{const}$ on each connected component. Rows and columns outside the support ($w_{i\cdot}=0$
or $w_{\cdot k}=0$) are isolated vertices, and the algorithm sets their $r_i$ or $c_k$ to 0. $\square$

### Theorem 2 (the algorithm converges to that projection)

`purify` alternates two steps:
- row centring: $T \leftarrow T - \bar r\,\mathbf 1^\top$, with $\bar r_i = \sum_k W_{ik}T_{ik}/w_{i\cdot}$;
- column centring, defined in the same way on the columns.

Each step is the $W$-orthogonal projection onto $\mathcal R^\perp$ (respectively $\mathcal C^\perp$), and the
iterates converge, on $\Omega$, to the projection onto $\mathcal R^\perp\cap\mathcal C^\perp = \mathcal
S^\perp$, that is, to $P$ of Theorem 1. The convergence is geometric:

$$
\|T^{(t)}-P\|_W \le \cos^{2t-1}\theta_F\,\|T^{(0)}-P\|_W ,
$$

where $\theta_F$ is the Friedrichs angle between $\mathcal R^\perp$ and $\mathcal C^\perp$.

*Proof.* Subtracting the weighted row means is the orthogonal projection onto the tables with zero weighted
row means: the residual has zero weighted row means and the removed part lies in $\mathcal R$. Zero weighted
row means is exactly orthogonality to $\mathcal R$, so the set of such tables is $\mathcal R^\perp$; the same
holds for the columns.

By von Neumann's alternating projection theorem, the iterates converge to the projection onto the
intersection $\mathcal R^\perp\cap\mathcal C^\perp=(\mathcal R+\mathcal C)^\perp$. In finite dimension the two
subspaces are closed and their sum is closed, so the rate is linear, governed by the cosine of the Friedrichs
angle (Deutsch, *Best Approximation in Inner Product Spaces*, 2001, ch. 9).

The accumulated corrections $r^{(t)}, c^{(t)}$ are partial sums of geometrically decreasing terms, so they
converge too. The stopping rule in the code (largest correction below `tol`) is reached in finitely many
iterations. $\square$

### Lemma 3 (invariance of every prediction)

Purification moves mass from the pair to the main effects:
- $f_a(u) \mathrel{+}= r_{\kappa_a(u)}$ for every main bin $u$ of $a$;
- $f_b(v) \mathrel{+}= c_{\kappa_b(v)}$ for every main bin $v$ of $b$;
- $f_{ab}\leftarrow P$.

With these moves, the logit of **every** input (inside or outside the support, measured or not) for which
both concepts are measured is unchanged.

*Proof.* For a case with main bins $(u, v)$ and coarse cell $(i,k)=(\kappa_a(u),\kappa_b(v))$, the sum
$f_a(u)+f_b(v)+f_{ab}(i,k)$ changes by $r_i + c_k - (r_i + c_k)=0$.

Nesting is what makes the move exact. $r_i$ is defined per coarse bin, and every main bin $u$ lies in exactly
one coarse bin, so adding $r_{\kappa_a(u)}$ to $f_a(u)$ reproduces the removed mass for every $x_a$. With
non-nested grids, a main bin could straddle two coarse bins, and no step function on the main grid could
absorb $r$ exactly.

The final centring of each term on its measured cases moves constants to $\beta_0$ and does not change any
logit either. $\square$

### Theorem 4 (zero is the exact conditional average of a purified pair)

Let $x_b$ be unmeasured for a case whose $x_a$ falls in coarse bin $i$. Under the empirical distribution of
the training cases with both concepts measured,

$$
\mathbb E_W\!\left[f_{ab}(x_a,x_b)\mid \kappa_a(x_a)=i\right]
 = \frac{\sum_k W_{ik}P_{ik}}{w_{i\cdot}} = 0 .
$$

The same holds with the roles of $a$ and $b$ exchanged. So giving the pair the value 0 is the exact
conditional marginalization of the pair term, whichever of the two concepts is missing.

*Proof.* Immediate from the row (respectively column) condition in (1). It needs $w_{i\cdot}>0$, that is, at
least one training case with both measured in that row. Otherwise the conditional average is undefined and 0
is a convention. $\square$

### Proposition 5 (why purification is necessary)

Without purification, giving 0 to a pair is **not** a marginalization: it depends on an arbitrary split of
the logit between $f_a$ and $f_{ab}$.

*Proof by example.* Let the fitted pair table be $T_{ik} = g_i$, which depends on $x_a$ only. This happens
when boosting assigns part of the effect of $x_a$ to the pair. The logit is identical whether $g$ sits in
$f_a$ or in $f_{ab}$, but dropping the unpurified pair when $x_b$ is missing removes $g$, the effect of
$x_a$, which was measured.

Theorem 1 gives $P = 0$ and $r = g$ (up to the component constant). After purification, $g$ lives in $f_a$
and the marginalized prediction keeps it. In general, two additive decompositions with the same logit give
different "pair = 0" predictions unless the pair is fixed by an identifiability condition. Condition (1) is
that condition: it is the weighted functional-ANOVA decomposition (Hooker 2007; Lengerich et al. 2020) on the
empirical distribution of complete cases, and Theorem 1 shows it is unique on the support. $\square$

### Proposition 6 (what the rule does and does not average)

For a case with $x_b$ unmeasured, RabMV returns

$$
\hat\eta = \beta_0 + \sum_{j\ne b} f_j(x_j) + \sum_{(a',b')\not\ni b} f_{a'b'} + 0 + 0 .
$$

The exact conditional expectation of the logit given the measured concepts would add
$\mathbb E[f_b(x_b)\mid x_{-b}] + \sum_{a}\mathbb E[f_{ab}\mid x_{-b}]$. Compared with it:

1. **The pair terms.** Their part is exact given the coarse bin of the partner (Theorem 4). It is exact
   given all measured concepts only if, within that bin, $x_b$ does not depend on the other measured
   concepts.
2. **The main effect $f_b$.** It is centred on the measured cases, so 0 is its **unconditional** mean. It
   equals $\mathbb E[f_b(x_b)\mid x_{-b}]$ only when $f_b(x_b)$ is mean-independent of the measured
   concepts.
3. **The scale.** The averaging is on the logit, not the probability: by Jensen's inequality,
   $\sigma(\mathbb E[\eta]) \ne \mathbb E[\sigma(\eta)]$ in general. This is the same convention as the
   additive explanations of EBM and of SHAP for additive models.

*Proof.* (1) is Theorem 4 restricted to its conditioning set. (2) follows from the centring of $f_b$ with
the weights of the measured cases. (3) holds because $\sigma$ is strictly convex on $(-\infty,0)$ and
strictly concave on $(0,\infty)$. $\square$

These limits are deliberate. The rule does not impute $x_b$ from the other concepts, because in a registry
the other concepts are distributed differently in each form, and that is exactly where an imputation model
would learn the form. They are reported as such in the article. Simulation v2 measured their cost: where
missingness is unrelated to the class (MAR), the rule loses nothing against chained-equation imputation
(criterion C2).

### Corollary 7 (bags and standard deviations)

Purification is linear in $T$ for a fixed $W$, because $P = (I-\Pi_{\mathcal S})T$. So the mean over bags of
the purified bag tables equals the purified mean table. That is why `_purify_pairs` purifies each bag with
the same $W$ to recompute the standard deviations: the point estimate is consistent with them. $\square$

## 4. Numerical verification

Each statement is checked in the tests:

| Statement | Test |
|---|---|
| (1) holds and cells are invariant (Thm 1, Lemma 3) | `test_purify_rows_and_columns_have_weighted_mean_zero_and_cells_are_invariant` |
| Equals the weighted least-squares residual, computed independently (Thm 1) | `test_purify_equals_the_weighted_least_squares_residual` |
| Unique on the support with a disconnected support; constants free per component (Thm 1) | `test_purified_table_is_unique_on_a_disconnected_support` |
| Geometric convergence within the iteration limit (Thm 2) | `test_purify_converges_geometrically` |
| Nested grids make the move exact (Lemma 3) | `test_nested_cuts_are_a_subset_and_each_fine_bin_is_inside_one_coarse_bin` |
| Zero is the conditional average in the fitted model (Thm 4) | `test_zero_for_an_unmeasured_concept_is_the_conditional_average_of_the_pair` |
| A pair carrying a main effect: dropping it unpurified loses the effect (Prop 5) | `test_dropping_an_unpurified_pair_loses_a_main_effect` |
| Complete cases unchanged by purification in the fitted model (Lemma 3) | `test_purification_does_not_change_predictions_of_complete_cases` |
| Main effect: 0 is the unconditional, not the conditional, mean under dependence (Prop 6) | `test_marginal_main_effect_is_the_unconditional_mean` |
| Mean of purified bags equals purified mean (Cor 7) | `test_purify_commutes_with_the_mean_over_bags` |

## References

- Deutsch, F. (2001). *Best Approximation in Inner Product Spaces*. Springer.
- Hooker, G. (2007). Generalized functional ANOVA diagnostics for high-dimensional functions of dependent
  variables. *Journal of Computational and Graphical Statistics*, 16(3), 709–732.
- Lengerich, B., Tan, S., Chang, C.-H., Hooker, G., and Caruana, R. (2020). Purifying interaction effects
  with the functional ANOVA: an efficient algorithm for recovering identifiable additive models. *AISTATS*,
  PMLR 108, 2402–2412.
- von Neumann, J. (1950). *Functional Operators, Vol. II*. Princeton University Press (alternating
  projections, first published in 1933 lecture notes).
