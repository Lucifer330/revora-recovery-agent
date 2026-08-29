from revora.decision import decide
from revora.orchestrator import run_cycle
from revora.policy import PolicyContext
from revora.schemas import (
    CustomerHistory,
    FailedPayment,
    PaymentMethodType,
    PaymentState,
    RecoveryAction,
    Split,
)
from revora.simulator import InjectedFault, RecoverySimulator
from revora.verification import verify_outcome


def _pay(**kwargs) -> FailedPayment:
    data = dict(
        payment_id="pay_sim_1",
        event_id="evt_sim_1",
        amount_paise=49900,
        currency="INR",
        failure_code="gateway_timeout",
        payment_method=PaymentMethodType.CARD,
        attempt_count=1,
        previous_recovery_attempts=0,
        hours_since_failure=1.0,
        customer=CustomerHistory(4, 1, 7),
        created_at_epoch_s=1,
        merchant_id="m1",
        split=Split.DEV,
    )
    data.update(kwargs)
    return FailedPayment(**data)


def test_execution_success_is_not_automatic_recovery():
    payment = _pay(payment_id="pay_timeout", event_id="evt_timeout")
    sim = RecoverySimulator()
    _d, audit = run_cycle(
        payment,
        decide,
        sim,
        fault=InjectedFault(timeout_on_execute=True),
    )
    assert audit.recorded_recovered is False
    assert sim.get(payment.payment_id).state != PaymentState.RECOVERED


def test_verify_requires_recovered_state():
    payment = _pay()
    sim = RecoverySimulator()
    sim.seed_failed(payment)
    sim.get(payment.payment_id).last_execution_ok = True
    result = verify_outcome(sim, payment.payment_id, execution_accepted=True)
    assert result.recovered is False
    assert result.verified_state == PaymentState.FAILED


def test_duplicate_webhook_does_not_double_execute():
    payment = _pay(payment_id="pay_dup", event_id="evt_dup")
    sim = RecoverySimulator()
    _d1, a1 = run_cycle(payment, decide, sim)
    _d2, a2 = run_cycle(payment, decide, sim, replay_same_event=True)
    assert a1.executed is True
    assert a2.executed is False


def test_policy_gate_is_authoritative():
    payment = _pay(
        payment_id="pay_rep",
        event_id="evt_rep",
        failure_code="repeated_decline",
        attempt_count=6,
        previous_recovery_attempts=4,
    )
    decision = decide(payment, PolicyContext(0, 999.0, None, False))
    assert decision.allowed is True
    assert decision.action == RecoveryAction.STOP_AND_ESCALATE
    assert decision.policy.rule_id in {"stop_always_allowed", "fallback_escalate", "allow"}
