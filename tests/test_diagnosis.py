from revora.diagnosis import FAILURE_CODE_MAP, DiagnosisError, diagnose
from revora.schemas import CustomerHistory, FailedPayment, FailureCategory, PaymentMethodType, Split


def _pay(**kwargs) -> FailedPayment:
    data = dict(
        payment_id="pay_1",
        event_id="evt_1",
        amount_paise=10000,
        currency="INR",
        failure_code="gateway_timeout",
        payment_method=PaymentMethodType.UPI,
        attempt_count=1,
        previous_recovery_attempts=0,
        hours_since_failure=2.0,
        customer=CustomerHistory(3, 1, 4),
        created_at_epoch_s=1,
        merchant_id="m1",
        split=Split.TRAIN,
    )
    data.update(kwargs)
    return FailedPayment(**data)


def test_maps_known_codes():
    for code, category in FAILURE_CODE_MAP.items():
        d = diagnose(_pay(failure_code=code, attempt_count=1, previous_recovery_attempts=0))
        if code in {"repeated_decline", "max_attempts_issuer"}:
            assert d.category == FailureCategory.REPEATED_FAILURE
        else:
            assert d.category == category


def test_repeated_override_from_attempt_count():
    d = diagnose(_pay(failure_code="gateway_timeout", attempt_count=5, previous_recovery_attempts=0))
    assert d.category == FailureCategory.REPEATED_FAILURE


def test_missing_and_invalid_context():
    try:
        diagnose(None)
        assert False
    except DiagnosisError:
        pass
    try:
        diagnose(_pay(amount_paise=0))
        assert False
    except DiagnosisError:
        pass
