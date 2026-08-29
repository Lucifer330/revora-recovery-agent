"""Independent synthetic outcome environment.

This module is the experimental world model. It is NOT calibrated to Razorpay
traffic and must NOT be imported by scoring, knowledge, decision, or baselines.

Outcomes depend on observable payment context and the chosen action only.
They do not depend on which strategy selected the action.

SHARED SYNTHETIC ASSUMPTION (not real-world evidence):
    Retrying the same expired/revoked instrument (EXPIRED + RETRY_LATER) almost
    never recovers. This is treated as a physical constraint of the toy world,
    available to any strategy that chooses that pair.

All other coefficients are experimental knobs for the lab, chosen a priori
without reference to Revora's ranking table or evaluation targets.
"""

from __future__ import annotations

from revora.schemas import FailedPayment, FailureCategory, PaymentMethodType, RecoveryAction

# --- Shared synthetic assumption ---
EXPIRED_SAME_INSTRUMENT_RETRY_P = 0.02

# Category intercepts: baseline recoverability of the failure class, action-agnostic.
_CATEGORY_INTERCEPT: dict[FailureCategory, float] = {
    FailureCategory.TRANSIENT_NETWORK: 0.40,
    FailureCategory.INSUFFICIENT_FUNDS: 0.30,
    FailureCategory.EXPIRED_PAYMENT_METHOD: 0.22,
    FailureCategory.AUTHENTICATION_FAILURE: 0.27,
    FailureCategory.REPEATED_FAILURE: 0.11,
    FailureCategory.UNKNOWN: 0.18,
}

# Modest action deltas. Overlapping on purpose so no action is a unique jackpot.
_ACTION_DELTA: dict[RecoveryAction, dict[FailureCategory, float]] = {
    RecoveryAction.RETRY_LATER: {
        FailureCategory.TRANSIENT_NETWORK: 0.16,
        FailureCategory.INSUFFICIENT_FUNDS: -0.06,
        FailureCategory.EXPIRED_PAYMENT_METHOD: 0.0,
        FailureCategory.AUTHENTICATION_FAILURE: 0.06,
        FailureCategory.REPEATED_FAILURE: -0.02,
        FailureCategory.UNKNOWN: 0.03,
    },
    RecoveryAction.RECOVERY_NUDGE: {
        FailureCategory.TRANSIENT_NETWORK: -0.04,
        FailureCategory.INSUFFICIENT_FUNDS: 0.08,
        FailureCategory.EXPIRED_PAYMENT_METHOD: -0.10,
        FailureCategory.AUTHENTICATION_FAILURE: -0.03,
        FailureCategory.REPEATED_FAILURE: 0.01,
        FailureCategory.UNKNOWN: 0.05,
    },
    RecoveryAction.PAYMENT_METHOD_UPDATE: {
        FailureCategory.TRANSIENT_NETWORK: -0.08,
        FailureCategory.INSUFFICIENT_FUNDS: -0.04,
        FailureCategory.EXPIRED_PAYMENT_METHOD: 0.28,
        FailureCategory.AUTHENTICATION_FAILURE: 0.07,
        FailureCategory.REPEATED_FAILURE: 0.03,
        FailureCategory.UNKNOWN: 0.01,
    },
    RecoveryAction.ALTERNATIVE_PATH: {
        FailureCategory.TRANSIENT_NETWORK: 0.04,
        FailureCategory.INSUFFICIENT_FUNDS: 0.01,
        FailureCategory.EXPIRED_PAYMENT_METHOD: 0.18,
        FailureCategory.AUTHENTICATION_FAILURE: 0.09,
        FailureCategory.REPEATED_FAILURE: 0.05,
        FailureCategory.UNKNOWN: 0.02,
    },
    RecoveryAction.STOP_AND_ESCALATE: {
        FailureCategory.TRANSIENT_NETWORK: 0.0,
        FailureCategory.INSUFFICIENT_FUNDS: 0.0,
        FailureCategory.EXPIRED_PAYMENT_METHOD: 0.0,
        FailureCategory.AUTHENTICATION_FAILURE: 0.0,
        FailureCategory.REPEATED_FAILURE: 0.0,
        FailureCategory.UNKNOWN: 0.0,
    },
}

# Hours until the episode can observe an outcome / schedule a follow-up.
_RESOLUTION_HOURS: dict[RecoveryAction, float] = {
    RecoveryAction.RETRY_LATER: 1.0,
    RecoveryAction.RECOVERY_NUDGE: 6.0,
    RecoveryAction.PAYMENT_METHOD_UPDATE: 3.0,
    RecoveryAction.ALTERNATIVE_PATH: 2.0,
    RecoveryAction.STOP_AND_ESCALATE: 0.0,
}


def _clip(p: float, lo: float = 0.0, hi: float = 0.90) -> float:
    return max(lo, min(hi, p))


def resolution_hours(action: RecoveryAction) -> float:
    """Wall-clock hours the toy world takes to resolve this action."""
    return _RESOLUTION_HOURS[action]


def recovery_probability(
    payment: FailedPayment,
    category: FailureCategory,
    action: RecoveryAction,
    recovery_attempts_this_run: int,
) -> float:
    """P(recover) from observables + action. No strategy identifier."""
    if action == RecoveryAction.STOP_AND_ESCALATE:
        return 0.0

    if category == FailureCategory.EXPIRED_PAYMENT_METHOD and action == RecoveryAction.RETRY_LATER:
        return EXPIRED_SAME_INSTRUMENT_RETRY_P

    intercept = _CATEGORY_INTERCEPT[category]
    delta = _ACTION_DELTA[action][category]

    # Liquidity proxy from observed history (available at decision time too).
    history = 0.12 * (payment.customer.observed_success_rate - 0.5)

    # Time since original failure: delayed contact can help funds, hurts stale retries.
    hours = payment.hours_since_failure
    if category == FailureCategory.INSUFFICIENT_FUNDS:
        time_term = 0.08 * min(1.0, hours / 72.0)
    else:
        time_term = -0.04 * min(1.0, hours / 120.0)

    attempts = payment.previous_recovery_attempts + recovery_attempts_this_run
    attempt_term = -0.05 * max(0, attempts)

    method_term = 0.0
    if (
        payment.payment_method == PaymentMethodType.UPI
        and category == FailureCategory.AUTHENTICATION_FAILURE
        and action == RecoveryAction.RETRY_LATER
    ):
        method_term = 0.04

    return _clip(intercept + delta + history + time_term + attempt_term + method_term, lo=0.01, hi=0.88)
