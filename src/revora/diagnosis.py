"""Map synthetic gateway failure codes to a failure category.

Diagnosis uses only payment context (code, attempt counts). It does not
read simulator outcomes.
"""

from __future__ import annotations

from revora.schemas import Diagnosis, FailedPayment, FailureCategory

FAILURE_CODE_MAP: dict[str, FailureCategory] = {
    "gateway_timeout": FailureCategory.TRANSIENT_NETWORK,
    "network_error": FailureCategory.TRANSIENT_NETWORK,
    "issuer_unavailable": FailureCategory.TRANSIENT_NETWORK,
    "insufficient_funds": FailureCategory.INSUFFICIENT_FUNDS,
    "bank_decline_funds": FailureCategory.INSUFFICIENT_FUNDS,
    "card_expired": FailureCategory.EXPIRED_PAYMENT_METHOD,
    "invalid_card": FailureCategory.EXPIRED_PAYMENT_METHOD,
    "token_revoked": FailureCategory.EXPIRED_PAYMENT_METHOD,
    "authentication_failed": FailureCategory.AUTHENTICATION_FAILURE,
    "3ds_failed": FailureCategory.AUTHENTICATION_FAILURE,
    "upi_pin_failed": FailureCategory.AUTHENTICATION_FAILURE,
    "repeated_decline": FailureCategory.REPEATED_FAILURE,
    "max_attempts_issuer": FailureCategory.REPEATED_FAILURE,
    "unknown_decline": FailureCategory.UNKNOWN,
    "do_not_honor": FailureCategory.UNKNOWN,
}

# Repeated-failure override: context that already shows exhaustion.
REPEATED_ATTEMPT_THRESHOLD = 5
REPEATED_RECOVERY_THRESHOLD = 3


class DiagnosisError(ValueError):
    """Missing or invalid payment context."""


def diagnose(payment: FailedPayment | None) -> Diagnosis:
    if payment is None:
        raise DiagnosisError("Missing payment context")
    if not payment.payment_id or not payment.event_id:
        raise DiagnosisError("Invalid payment context: payment_id and event_id are required")
    if payment.amount_paise <= 0:
        raise DiagnosisError("Invalid payment context: amount must be positive")
    if payment.attempt_count < 1:
        raise DiagnosisError("Invalid payment context: attempt_count must be >= 1")

    reasons: list[str] = []
    mapped = FAILURE_CODE_MAP.get(payment.failure_code)
    if mapped is None:
        category = FailureCategory.UNKNOWN
        reasons.append(f"Unmapped failure_code '{payment.failure_code}' treated as UNKNOWN")
        confidence = 0.35
    else:
        category = mapped
        reasons.append(f"failure_code '{payment.failure_code}' maps to {category.value}")
        confidence = 0.9

    if (
        payment.attempt_count >= REPEATED_ATTEMPT_THRESHOLD
        or payment.previous_recovery_attempts >= REPEATED_RECOVERY_THRESHOLD
    ):
        if category != FailureCategory.REPEATED_FAILURE:
            reasons.append(
                "Override to REPEATED_FAILURE because attempt_count="
                f"{payment.attempt_count} or previous_recovery_attempts="
                f"{payment.previous_recovery_attempts} exceeds exhaustion thresholds"
            )
            category = FailureCategory.REPEATED_FAILURE
            confidence = min(confidence, 0.8)

    return Diagnosis(
        category=category,
        failure_code=payment.failure_code,
        reasons=tuple(reasons),
        confidence=confidence,
    )
