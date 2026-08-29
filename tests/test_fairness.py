"""Fairness regressions: independent environment, equal safety, executable baselines."""

from __future__ import annotations

import ast
from pathlib import Path

from revora.baselines import decide_simple_retry, decide_static_rules
from revora.dataset import generate_dataset
from revora.decision import decide as decide_revora
from revora.diagnosis import diagnose
from revora.environment import recovery_probability
from revora.evaluation import evaluate
from revora.orchestrator import run_cycle
from revora.policy import PolicyContext, SHARED_RETRY_LIMIT, evaluate_safety
from revora.schemas import (
    CustomerHistory,
    FailedPayment,
    FailureCategory,
    PaymentMethodType,
    RecoveryAction,
    Split,
)
from revora.scoring import ACTION_FIT, WEIGHT_AMOUNT, WEIGHT_CATEGORY
from revora.simulator import RecoverySimulator

SRC = Path(__file__).resolve().parents[1] / "src" / "revora"
DECISION_STACK = (
    "baselines.py",
    "decision.py",
    "decision_ranked.py",
    "knowledge.py",
    "scoring.py",
)


def _pay(**kwargs) -> FailedPayment:
    data = dict(
        payment_id="pay_fair_1",
        event_id="evt_fair_1",
        amount_paise=49900,
        currency="INR",
        failure_code="authentication_failed",
        payment_method=PaymentMethodType.CARD,
        attempt_count=1,
        previous_recovery_attempts=0,
        hours_since_failure=2.0,
        customer=CustomerHistory(4, 2, 10),
        created_at_epoch_s=1,
        merchant_id="m1",
        split=Split.HELD_OUT,
    )
    data.update(kwargs)
    return FailedPayment(**data)


def test_decision_stack_does_not_import_environment():
    for name in DECISION_STACK:
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "environment" not in node.module, name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "environment" not in alias.name, name


def test_execute_has_no_strategy_id_parameter():
    import inspect

    from revora.simulator import RecoverySimulator

    sig = inspect.signature(RecoverySimulator.execute)
    assert "system_id" not in sig.parameters
    assert "strategy" not in sig.parameters


def test_same_payment_action_attempt_same_probability():
    p = _pay()
    cat = diagnose(p).category
    a = recovery_probability(p, cat, RecoveryAction.RETRY_LATER, 1)
    b = recovery_probability(p, cat, RecoveryAction.RETRY_LATER, 1)
    assert a == b


def test_probability_ignores_who_asked_by_being_pure_function():
    p = _pay(failure_code="insufficient_funds")
    cat = diagnose(p).category
    p_retry = recovery_probability(p, cat, RecoveryAction.RETRY_LATER, 1)
    p_nudge = recovery_probability(p, cat, RecoveryAction.RECOVERY_NUDGE, 1)
    assert 0.01 <= p_retry <= 0.88
    assert 0.01 <= p_nudge <= 0.88


def test_simple_retry_executes_on_auth_failure():
    payment = _pay(payment_id="pay_auth_sr", event_id="evt_auth_sr")
    sim = RecoverySimulator()
    decision, audit = run_cycle(payment, decide_simple_retry, sim)
    assert decision.action == RecoveryAction.RETRY_LATER
    assert audit.executed is True
    assert decision.policy.rule_id != "eligibility"


def test_static_rules_executes_auth_retry():
    payment = _pay(payment_id="pay_auth_st", event_id="evt_auth_st")
    sim = RecoverySimulator()
    decision, audit = run_cycle(payment, decide_static_rules, sim)
    assert decision.action == RecoveryAction.RETRY_LATER
    assert audit.executed is True
    assert "eligibility" not in decision.policy.reason.lower()


def test_baseline_audits_do_not_use_eligibility_rule():
    ds = generate_dataset(size=40, seed=7)
    held = ds.by_split(Split.HELD_OUT) or ds.payments[:8]
    for decide in (decide_simple_retry, decide_static_rules):
        sim = RecoverySimulator()
        for p in held[:8]:
            _d, audit = run_cycle(p, decide, sim)
            assert audit.policy_reason.find("per Revora knowledge") == -1


def test_shared_retry_limit_same_constant():
    p = _pay(failure_code="gateway_timeout", previous_recovery_attempts=SHARED_RETRY_LIMIT)
    d = diagnose(p)
    ctx = PolicyContext(0, 999.0, None, False)
    safety = evaluate_safety(p, d.category, RecoveryAction.RETRY_LATER, ctx)
    assert safety.allowed is False
    assert safety.rule_id == "retry_limit"


def test_split_not_referenced_in_decision_stack():
    for name in DECISION_STACK:
        source = (SRC / name).read_text(encoding="utf-8")
        assert "payment.split" not in source
        assert "p.split" not in source


def test_scoring_weights_unchanged_by_evaluation():
    before = (WEIGHT_CATEGORY, WEIGHT_AMOUNT, dict(ACTION_FIT))
    evaluate(split=Split.HELD_OUT)
    after = (WEIGHT_CATEGORY, WEIGHT_AMOUNT, dict(ACTION_FIT))
    assert before == after


def test_same_action_same_outcome_across_isolated_sims():
    payment = _pay(
        payment_id="pay_iso",
        event_id="evt_iso_a",
        failure_code="gateway_timeout",
    )
    cat = diagnose(payment).category
    outcomes = []
    for event in ("evt_iso_a", "evt_iso_b"):
        sim = RecoverySimulator()
        p = payment.evolve(event_id=event)
        sim.execute(p, cat, RecoveryAction.RETRY_LATER, allowed=True)
        outcomes.append(sim.get(p.payment_id).state)
    assert outcomes[0] == outcomes[1]


def test_metrics_function_applied_to_all_strategies():
    report = evaluate(split=Split.HELD_OUT)
    assert set(report.runs) == {
        "simple_retry",
        "static_rules",
        "revora",
        "revora_c2",
        "revora_c2_heuristic",
        "revora_c2_ai",
    }
    for run in report.runs.values():
        assert run.metrics.policy_violations == 0
        assert run.metrics.duplicate_actions == 0
        assert run.metrics.verification_failures == 0
        assert run.by_category
        assert sum(s.n for s in run.by_category) == report.payments_evaluated
