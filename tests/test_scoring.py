from revora.diagnosis import diagnose
from revora.schemas import CustomerHistory, FailedPayment, PaymentMethodType, Split
from revora.scoring import WEIGHT_AMOUNT, WEIGHT_ATTEMPTS, WEIGHT_CATEGORY, WEIGHT_HISTORY, WEIGHT_PRIOR_RECOVERY, WEIGHT_RECENCY, score_payment


def _pay(**kwargs) -> FailedPayment:
    data = dict(
        payment_id="pay_1",
        event_id="evt_1",
        amount_paise=99900,
        currency="INR",
        failure_code="gateway_timeout",
        payment_method=PaymentMethodType.CARD,
        attempt_count=1,
        previous_recovery_attempts=0,
        hours_since_failure=1.0,
        customer=CustomerHistory(10, 0, 3),
        created_at_epoch_s=1,
        merchant_id="m1",
        split=Split.TRAIN,
    )
    data.update(kwargs)
    return FailedPayment(**data)


def test_weights_sum_to_one():
    assert abs(
        WEIGHT_CATEGORY
        + WEIGHT_ATTEMPTS
        + WEIGHT_PRIOR_RECOVERY
        + WEIGHT_HISTORY
        + WEIGHT_RECENCY
        + WEIGHT_AMOUNT
        - 1.0
    ) < 1e-9


def test_score_is_deterministic_and_bounded():
    p = _pay()
    category = diagnose(p).category
    a = score_payment(p, category)
    b = score_payment(p, category)
    assert a.value == b.value
    assert 0.0 <= a.value <= 1.0
    assert abs(sum(c.weighted_value for c in a.components) - a.value) < 1e-3


def test_more_attempts_lowers_or_equals_score():
    fresh = _pay(attempt_count=1, previous_recovery_attempts=0)
    tired = _pay(attempt_count=4, previous_recovery_attempts=2)
    cat = diagnose(fresh).category
    # Use same category for fair comparison (tired might override to repeated)
    s1 = score_payment(fresh, cat)
    s2 = score_payment(tired, cat)
    assert s2.value < s1.value
