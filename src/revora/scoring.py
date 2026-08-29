"""Explainable recoverability / expected-value heuristic.

This is a documented weighted formula. It is not machine learning.
Weights were chosen a priori (not fit on the held-out set).

The score ranks candidate actions; the deterministic policy engine still
has final authority to allow or reject execution.
"""

from __future__ import annotations

from revora.schemas import (
    FailedPayment,
    FailureCategory,
    RecoverabilityScore,
    RecoveryAction,
    ScoreComponent,
)

# --- Documented weights (must sum to 1.0) ---
WEIGHT_CATEGORY = 0.40
WEIGHT_ATTEMPTS = 0.15
WEIGHT_PRIOR_RECOVERY = 0.15
WEIGHT_HISTORY = 0.15
WEIGHT_RECENCY = 0.10
WEIGHT_AMOUNT = 0.05

CATEGORY_PRIOR: dict[FailureCategory, float] = {
    FailureCategory.TRANSIENT_NETWORK: 0.85,
    FailureCategory.INSUFFICIENT_FUNDS: 0.55,
    FailureCategory.EXPIRED_PAYMENT_METHOD: 0.70,
    FailureCategory.AUTHENTICATION_FAILURE: 0.50,
    FailureCategory.REPEATED_FAILURE: 0.12,
    FailureCategory.UNKNOWN: 0.28,
}

# How well an action typically fits a category. Used only for ranking.
ACTION_FIT: dict[tuple[FailureCategory, RecoveryAction], float] = {
    (FailureCategory.TRANSIENT_NETWORK, RecoveryAction.RETRY_LATER): 1.0,
    (FailureCategory.TRANSIENT_NETWORK, RecoveryAction.ALTERNATIVE_PATH): 0.55,
    (FailureCategory.INSUFFICIENT_FUNDS, RecoveryAction.RECOVERY_NUDGE): 1.0,
    (FailureCategory.INSUFFICIENT_FUNDS, RecoveryAction.RETRY_LATER): 0.35,
    (FailureCategory.EXPIRED_PAYMENT_METHOD, RecoveryAction.PAYMENT_METHOD_UPDATE): 1.0,
    (FailureCategory.EXPIRED_PAYMENT_METHOD, RecoveryAction.ALTERNATIVE_PATH): 0.70,
    (FailureCategory.AUTHENTICATION_FAILURE, RecoveryAction.ALTERNATIVE_PATH): 1.0,
    (FailureCategory.AUTHENTICATION_FAILURE, RecoveryAction.PAYMENT_METHOD_UPDATE): 0.75,
    (FailureCategory.REPEATED_FAILURE, RecoveryAction.STOP_AND_ESCALATE): 1.0,
    (FailureCategory.REPEATED_FAILURE, RecoveryAction.ALTERNATIVE_PATH): 0.25,
    (FailureCategory.UNKNOWN, RecoveryAction.RECOVERY_NUDGE): 0.80,
    (FailureCategory.UNKNOWN, RecoveryAction.STOP_AND_ESCALATE): 0.90,
}

FORMULA = (
    "score = clip(w_cat*category_prior + w_att*attempt_factor + "
    "w_prev*prior_recovery_factor + w_hist*customer_success_rate + "
    "w_rec*recency_factor + w_amt*amount_factor, 0, 1)"
)

ASSUMPTIONS = (
    "Weights were set a priori from payment-ops intuition, not trained on this dataset.",
    "Category priors reflect typical recoverability, not observed simulator outcomes.",
    "Customer history is a noisy proxy for willingness/ability to pay.",
    "Amount has a small weight: ticket size is a weak recoverability signal.",
)

LIMITATIONS = (
    "Not calibrated to real Razorpay traffic; synthetic only.",
    "Linear combination cannot capture interaction effects well.",
    "Does not observe bank-side retry windows or mandate data.",
    "Must not be treated as a probability from a fitted model.",
    "Policy may reject a high-scoring action; score is not authority.",
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def attempt_factor(attempt_count: int) -> float:
    return _clip01(1.0 - 0.22 * max(0, attempt_count - 1))


def prior_recovery_factor(previous_recovery_attempts: int) -> float:
    return _clip01(1.0 - 0.28 * max(0, previous_recovery_attempts))


def recency_factor(hours_since_failure: float) -> float:
    # Half-life-ish decay over ~3 days.
    return _clip01(1.0 / (1.0 + hours_since_failure / 72.0))


def amount_factor(amount_paise: int) -> float:
    # Mild preference for mid-ticket (very large tickets often need more care).
    rupees = amount_paise / 100.0
    if rupees <= 0:
        return 0.0
    if rupees < 500:
        return 0.75
    if rupees < 5000:
        return 0.90
    if rupees < 20000:
        return 0.70
    return 0.45


def score_payment(payment: FailedPayment, category: FailureCategory) -> RecoverabilityScore:
    cat = CATEGORY_PRIOR[category]
    att = attempt_factor(payment.attempt_count)
    prev = prior_recovery_factor(payment.previous_recovery_attempts)
    hist = _clip01(payment.customer.observed_success_rate)
    rec = recency_factor(payment.hours_since_failure)
    amt = amount_factor(payment.amount_paise)

    parts = (
        ScoreComponent("category_prior", cat, WEIGHT_CATEGORY, WEIGHT_CATEGORY * cat, f"{category.value}"),
        ScoreComponent("attempt_factor", att, WEIGHT_ATTEMPTS, WEIGHT_ATTEMPTS * att, f"attempt_count={payment.attempt_count}"),
        ScoreComponent(
            "prior_recovery_factor",
            prev,
            WEIGHT_PRIOR_RECOVERY,
            WEIGHT_PRIOR_RECOVERY * prev,
            f"previous_recovery_attempts={payment.previous_recovery_attempts}",
        ),
        ScoreComponent(
            "customer_success_rate",
            hist,
            WEIGHT_HISTORY,
            WEIGHT_HISTORY * hist,
            f"successes={payment.customer.successful_payments} failures={payment.customer.failed_payments}",
        ),
        ScoreComponent(
            "recency_factor",
            rec,
            WEIGHT_RECENCY,
            WEIGHT_RECENCY * rec,
            f"hours_since_failure={payment.hours_since_failure}",
        ),
        ScoreComponent("amount_factor", amt, WEIGHT_AMOUNT, WEIGHT_AMOUNT * amt, f"amount_paise={payment.amount_paise}"),
    )
    value = _clip01(sum(p.weighted_value for p in parts))
    return RecoverabilityScore(
        value=round(value, 4),
        components=parts,
        formula=FORMULA,
        assumptions=ASSUMPTIONS,
        limitations=LIMITATIONS,
    )


def expected_value_paise(score: RecoverabilityScore, amount_paise: int, action: RecoveryAction, category: FailureCategory) -> float:
    """Heuristic EV for ranking only. Not a forecast used as ground truth."""
    fit = ACTION_FIT.get((category, action), 0.15)
    if action == RecoveryAction.STOP_AND_ESCALATE:
        return 0.0
    return score.value * fit * amount_paise


def rank_candidates(
    payment: FailedPayment,
    category: FailureCategory,
    score: RecoverabilityScore,
    actions: tuple[RecoveryAction, ...],
) -> list[tuple[RecoveryAction, float, str]]:
    ranked: list[tuple[RecoveryAction, float, str]] = []
    for action in actions:
        ev = expected_value_paise(score, payment.amount_paise, action, category)
        fit = ACTION_FIT.get((category, action), 0.15)
        note = f"EV={ev:.0f} paise; score={score.value:.4f}; action_fit={fit:.2f}"
        ranked.append((action, ev, note))
    ranked.sort(key=lambda row: (-row[1], row[0].value))
    return ranked
