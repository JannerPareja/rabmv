"""Refinement of the audit checks: single-choice pairs explained by another alert.

Found after simulation v2 (criterion C4, reported as not met; see ``simulation_v2/posthoc_c4.py``): check B
("in one class two binary concepts are never marked together") also fires when one of the two concepts is
almost never marked in that class for another reason that the checks already report:

* check A: the concept is constant in one class (a question the form does not ask, filled with "No");
* check C: its blanks were recorded as "No" in one class.

Then any concept paired with it looks like part of a single-choice question, and a concept that both forms
measure the same way is wrongly named. ``drop_explained_pairs`` removes those B alerts. It is a separate
function, off by default, so that ``run_checks`` keeps the exact behaviour that the frozen simulations
evaluated; it was validated afterwards on the 200 replicates of simulation v2 with the form equal to the class
(localization 200/200, without losing any detection), and it is reported as a post-hoc refinement.
"""
from dataclasses import replace

from .checks import ALERT, NO_ALERT, Checks


def _concepts(name):
    return {x.strip() for x in str(name).split('&')}


def drop_explained_pairs(checks, explaining=('A', 'C')):
    """Turn into "no alert" the B alerts whose pair contains a concept with an A or C alert.

    Parameters
    ----------
    checks : rabmv.Checks
        Output of ``rabmv.run_checks``.
    explaining : tuple of str, default=('A', 'C')
        Checks whose alert explains a single-choice pattern.

    Returns
    -------
    rabmv.Checks
        A copy; the changed rows keep their p-value and say why in ``detail``.
    """
    if not isinstance(checks, Checks):
        raise TypeError('checks must be the output of rabmv.run_checks')
    t = checks.table.copy()
    flagged = set(t.loc[(t['check'].isin(explaining)) & (t['result'] == ALERT), 'concept'])
    rows = (t['check'] == 'B') & (t['result'] == ALERT) & t['concept'].map(lambda c: bool(_concepts(c) & flagged))
    for i in t.index[rows]:
        cause = sorted(_concepts(t.at[i, 'concept']) & flagged)
        t.at[i, 'result'] = NO_ALERT
        t.at[i, 'detail'] = (f"{t.at[i, 'detail']}; explained by the alert of {', '.join(cause)} "
                             f"({'/'.join(explaining)}), not reported")
    params = dict(checks.parameters, refined=f"drop_explained_pairs(explaining={tuple(explaining)})")
    return replace(checks, table=t, parameters=params)
