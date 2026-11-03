"""Tests of the simulation generator (phase 3): each scenario produces what it says it produces."""
import numpy as np

from rabmv import evaluate_eligibility, harmonize, run_checks
from simulation.calibrate import EXPECTED_ELIGIBLE
from simulation.generator import population, record, schema, subsample


def test_prevalence_and_size():
    """About 2,350 events with a completed-suicide prevalence close to 7.6 %."""
    prev = np.mean([population(s)['y'].mean() for s in range(1, 11)])
    assert 0.065 < prev < 0.09
    assert subsample(population(1), 60, 1)['y'].sum() == 60


def test_correct_schema_gives_the_expected_eligibility():
    """S1: with the schema that tells the truth, the core and the equally measured factors enter."""
    pop = population(2)
    s = schema('correct')
    a = harmonize(record(pop, 'valledupar', 2), s)
    assert evaluate_eligibility(a, pop['y'], s.catalog).eligible == EXPECTED_ELIGIBLE


def test_all_exact_schema_triggers_the_checks():
    """S3: declaring everything exact makes A (f1, f2), B (f3–f5), C (f6) and E (education) appear."""
    pop = population(3)
    d = record(pop, 'valledupar', 3)
    s = schema('all_exact')
    al = {(r.check, r.concept) for r in run_checks(harmonize(d, s), pop['y'], s, data=d).alerts().itertuples()}
    assert {('A', 'f1'), ('A', 'f2'), ('C', 'f6'), ('E', 'education')} <= al
    assert any(t == 'B' and c in {'f3 & f4', 'f3 & f5', 'f4 & f5'} for t, c in al)


def test_null_scenario_without_false_alarms():
    """S4: if the forms measure the same way, there are no alerts."""
    pop = population(4)
    d = record(pop, 'null', 4)
    s = schema('null')
    assert len(run_checks(harmonize(d, s), pop['y'], s, data=d).alerts()) == 0


def test_transport_records_everyone_with_the_complete_form():
    """In transport nobody uses form B and there is no 2016 education gap."""
    d = record(population(5), 'transport', 5)
    assert (d['form'] == 'A').all()
    assert (d.loc[d['date'].str.startswith('2016'), 'education'].notna()).all()
