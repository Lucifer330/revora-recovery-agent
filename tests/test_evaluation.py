from revora.evaluation import evaluate, render_report
from revora.failure_lab import run_failure_lab
from revora.schemas import Split


def test_failure_lab_all_pass():
    results = run_failure_lab()
    failed = [r.name for r in results if not r.passed]
    assert failed == [], failed


def test_evaluation_metrics_are_internally_consistent():
    report = evaluate(split=Split.HELD_OUT)
    assert report.payments_evaluated > 0
    for run in report.runs.values():
        m = run.metrics
        assert m.recovered_payments <= m.total_payments
        assert m.revenue_recovered_paise <= m.revenue_at_risk_paise
        expected_rate = m.recovered_payments / m.total_payments
        assert abs(m.recovery_rate - expected_rate) < 1e-12
        assert m.policy_violations == 0
        assert m.duplicate_actions == 0
        assert m.verification_failures == 0
        recovered_from_states = sum(
            1 for state in run.final_states.values() if state.value == "RECOVERED"
        )
        assert recovered_from_states == m.recovered_payments


def test_evaluation_reproducible():
    a = evaluate(split=Split.HELD_OUT)
    b = evaluate(split=Split.HELD_OUT)
    assert render_report(a) == render_report(b)
    assert a.runs["revora"].metrics.revenue_recovered_paise == b.runs["revora"].metrics.revenue_recovered_paise


def test_held_out_not_empty_and_not_full_dataset():
    report = evaluate(split=Split.HELD_OUT)
    assert 0 < report.payments_evaluated < report.dataset_size
