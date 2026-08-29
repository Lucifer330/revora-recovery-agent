"""C2 mechanism-aware ledger: POLICY CHANGE, not AI."""

from pathlib import Path

from revora.decision import decide, decide_c2
from revora.diagnosis import diagnose
from revora.evaluation import evaluate
from revora.orchestrator import run_cycle
from revora.policy import (
    MAX_GLOBAL_RECOVERY_ATTEMPTS,
    PolicyContext,
    evaluate_revora_c2_policy,
    evaluate_revora_policy,
    evaluate_safety,
    family_attempts_for_c2,
)
from revora.schemas import (
    CustomerHistory,
    FailedPayment,
    PaymentMethodType,
    RecoveryAction,
    Split,
)
from revora.simulator import RecoverySimulator

ENV_PATH = Path(__file__).resolve().parents[1] / "src" / "revora" / "environment.py"


def _expired(**kwargs) -> FailedPayment:
    data = dict(
        payment_id="pay_c2_exp",
        event_id="evt_c2_exp",
        amount_paise=99900,
        currency="INR",
        failure_code="card_expired",
        payment_method=PaymentMethodType.CARD,
        attempt_count=2,
        previous_recovery_attempts=2,
        hours_since_failure=3.0,
        customer=CustomerHistory(5, 1, 8),
        created_at_epoch_s=1,
        merchant_id="m1",
        split=Split.HELD_OUT,
    )
    data.update(kwargs)
    return FailedPayment(**data)


def test_c2_does_not_count_retry_history_as_method_update():
    payment = _expired(previous_recovery_attempts=2)
    ctx = PolicyContext(0, 999.0, None, False, ())
    d = diagnose(payment)
    v1 = evaluate_revora_policy(payment, d.category, RecoveryAction.PAYMENT_METHOD_UPDATE, ctx)
    c2 = evaluate_revora_c2_policy(payment, d.category, RecoveryAction.PAYMENT_METHOD_UPDATE, ctx)
    assert v1.allowed is False
    assert v1.rule_id == "stopping_rule"
    assert c2.allowed is True
    assert family_attempts_for_c2(payment, RecoveryAction.PAYMENT_METHOD_UPDATE, ctx) == 0
    assert family_attempts_for_c2(payment, RecoveryAction.RETRY_LATER, ctx) == 2


def test_c2_counts_same_family_update_attempts():
    payment = _expired(previous_recovery_attempts=0)
    ctx = PolicyContext(
        recovery_attempts_this_run=2,
        hours_since_last_recovery=9.0,
        last_action=RecoveryAction.PAYMENT_METHOD_UPDATE,
        event_already_processed=False,
        attempts_by_family=(("PAYMENT_METHOD_UPDATE", 2),),
    )
    d = diagnose(payment)
    denied = evaluate_revora_c2_policy(
        payment, d.category, RecoveryAction.PAYMENT_METHOD_UPDATE, ctx
    )
    assert denied.allowed is False
    assert denied.rule_id == "stopping_rule"


def test_global_safety_cap_still_applies_to_c2():
    payment = _expired(previous_recovery_attempts=MAX_GLOBAL_RECOVERY_ATTEMPTS)
    ctx = PolicyContext(0, 999.0, None, False, ())
    d = diagnose(payment)
    safety = evaluate_safety(payment, d.category, RecoveryAction.PAYMENT_METHOD_UPDATE, ctx)
    c2 = evaluate_revora_c2_policy(payment, d.category, RecoveryAction.PAYMENT_METHOD_UPDATE, ctx)
    assert safety.allowed is False
    assert safety.rule_id == "global_attempt_cap"
    assert c2.allowed is False
    assert c2.rule_id == "global_attempt_cap"


def test_c2_duplicate_protection():
    payment = _expired(previous_recovery_attempts=0, payment_id="pay_c2_dup", event_id="evt_c2_dup")
    sim = RecoverySimulator()
    _d1, a1 = run_cycle(payment, decide_c2, sim)
    _d2, a2 = run_cycle(payment, decide_c2, sim, replay_same_event=True)
    assert a1.executed is True
    assert a2.executed is False


def test_v1_still_stops_on_pooled_history():
    payment = _expired(previous_recovery_attempts=2, payment_id="pay_v1", event_id="evt_v1")
    decision = decide(payment, PolicyContext(0, 999.0, None, False))
    assert decision.action == RecoveryAction.STOP_AND_ESCALATE


def test_c2_allows_method_update_despite_pooled_retry_history():
    payment = _expired(previous_recovery_attempts=2, payment_id="pay_c2", event_id="evt_c2")
    decision = decide_c2(payment, PolicyContext(0, 999.0, None, False))
    assert decision.action == RecoveryAction.PAYMENT_METHOD_UPDATE
    assert decision.allowed is True


def test_v1_metrics_reproducible_with_c2_arm_present():
    a = evaluate(split=Split.HELD_OUT)
    b = evaluate(split=Split.HELD_OUT)
    assert a.runs["revora"].metrics.revenue_recovered_paise == b.runs["revora"].metrics.revenue_recovered_paise
    assert a.runs["revora"].metrics.recovery_rate == b.runs["revora"].metrics.recovery_rate
    assert "revora_c2" in a.runs


def test_environment_module_untouched_marker():
    text = ENV_PATH.read_text(encoding="utf-8")
    assert "EXPIRED_SAME_INSTRUMENT_RETRY_P = 0.02" in text
    assert "NOT be imported by scoring, knowledge, decision, or baselines" in text
    assert "do not depend on which strategy" in text.lower()
