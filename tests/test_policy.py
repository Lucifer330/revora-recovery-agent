from revora.diagnosis import diagnose
from revora.policy import PolicyContext, evaluate_policy
from revora.schemas import (
    CustomerHistory,
    FailedPayment,
    FailureCategory,
    PaymentMethodType,
    RecoveryAction,
    Split,
)


def _pay(**kwargs) -> FailedPayment:
    data = dict(
        payment_id="pay_1",
        event_id="evt_1",
        amount_paise=19900,
        currency="INR",
        failure_code="gateway_timeout",
        payment_method=PaymentMethodType.CARD,
        attempt_count=1,
        previous_recovery_attempts=0,
        hours_since_failure=1.0,
        customer=CustomerHistory(2, 1, 9),
        created_at_epoch_s=1,
        merchant_id="m1",
        split=Split.TRAIN,
    )
    data.update(kwargs)
    return FailedPayment(**data)


CTX = PolicyContext(0, 999.0, None, False)


def test_expired_card_blocked_for_revora_not_shared_safety():
    p = _pay(failure_code="card_expired")
    d = diagnose(p)
    revora = evaluate_policy(p, d.category, RecoveryAction.RETRY_LATER, CTX)
    assert revora.allowed is False
    assert revora.rule_id in {"forbidden_pair", "eligibility"}
    from revora.policy import evaluate_safety

    safety = evaluate_safety(p, d.category, RecoveryAction.RETRY_LATER, CTX)
    assert safety.allowed is True



def test_retry_limit():
    p = _pay(failure_code="gateway_timeout", previous_recovery_attempts=2)
    d = diagnose(p)
    assert d.category == FailureCategory.TRANSIENT_NETWORK
    decision = evaluate_policy(
        p,
        d.category,
        RecoveryAction.RETRY_LATER,
        PolicyContext(1, 999.0, RecoveryAction.ALTERNATIVE_PATH, False),
    )
    assert decision.allowed is False
    assert decision.rule_id in {"retry_limit", "stopping_rule", "global_attempt_cap"}


def test_duplicate_event_blocked():
    p = _pay()
    d = diagnose(p)
    decision = evaluate_policy(
        p,
        d.category,
        RecoveryAction.RETRY_LATER,
        PolicyContext(0, 999.0, None, True),
    )
    assert decision.allowed is False
    assert decision.rule_id == "idempotency_duplicate_event"


def test_cooldown_blocks_immediate_retry():
    p = _pay(failure_code="gateway_timeout")
    d = diagnose(p)
    decision = evaluate_policy(
        p,
        d.category,
        RecoveryAction.RETRY_LATER,
        PolicyContext(1, 0.1, RecoveryAction.RETRY_LATER, False),
    )
    assert decision.allowed is False
    assert decision.rule_id == "cooldown"


def test_stop_always_allowed():
    p = _pay()
    d = diagnose(p)
    decision = evaluate_policy(p, d.category, RecoveryAction.STOP_AND_ESCALATE, CTX)
    assert decision.allowed is True


def test_missing_context_rejected():
    decision = evaluate_policy(None, None, RecoveryAction.RETRY_LATER, CTX)
    assert decision.allowed is False
    assert decision.rule_id == "require_context"
