"""Frozen deterministic AI-shaped ranker for evaluation.

Same I/O schema as a future LLM adapter. No live API. Rules written a priori
from ops intuition, not from environment.py or held-out metrics.

Differences vs HeuristicRanker (intentional, so ablation can disagree):
- INSUFFICIENT_FUNDS: if hours_since_failure < 12 and RETRY_LATER eligible,
  prefer RETRY_LATER (customer may still have a window); else NUDGE.
- AUTHENTICATION_FAILURE: UPI prefers RETRY_LATER if eligible, else ALTERNATIVE_PATH.
  Card/other: PAYMENT_METHOD_UPDATE before ALTERNATIVE_PATH if both eligible.
- EXPIRED: same as heuristic (update then alt path).
- Other categories: same category order as heuristic (usually skipped by skip table).
"""

from __future__ import annotations

from revora.policy import PolicyContext
from revora.rankers.schema import Recommendation
from revora.rankers.validate import parse_recommendation
from revora.schemas import FailedPayment, FailureCategory, PaymentMethodType, RecoveryAction

RANKER_ID = "frozen_ai_v1"


def _draft(
    payment: FailedPayment,
    category: FailureCategory,
    eligible: tuple[RecoveryAction, ...],
) -> dict[str, object]:
    eligible_set = set(eligible)

    def pick(*order: RecoveryAction) -> list[str]:
        ranked = [a.value for a in order if a in eligible_set]
        for extra in sorted(eligible_set, key=lambda a: a.value):
            if extra.value not in ranked:
                ranked.append(extra.value)
        if not ranked:
            ranked = [RecoveryAction.STOP_AND_ESCALATE.value]
        return ranked

    if category == FailureCategory.INSUFFICIENT_FUNDS:
        if payment.hours_since_failure < 12 and RecoveryAction.RETRY_LATER in eligible_set:
            ranked = pick(
                RecoveryAction.RETRY_LATER,
                RecoveryAction.RECOVERY_NUDGE,
                RecoveryAction.STOP_AND_ESCALATE,
            )
            code = "FROZEN_FUNDS_EARLY_RETRY"
        else:
            ranked = pick(
                RecoveryAction.RECOVERY_NUDGE,
                RecoveryAction.RETRY_LATER,
                RecoveryAction.STOP_AND_ESCALATE,
            )
            code = "FROZEN_FUNDS_NUDGE"
    elif category == FailureCategory.AUTHENTICATION_FAILURE:
        if payment.payment_method == PaymentMethodType.UPI:
            ranked = pick(
                RecoveryAction.RETRY_LATER,
                RecoveryAction.ALTERNATIVE_PATH,
                RecoveryAction.PAYMENT_METHOD_UPDATE,
                RecoveryAction.STOP_AND_ESCALATE,
            )
            code = "FROZEN_AUTH_UPI_RETRY"
        else:
            ranked = pick(
                RecoveryAction.PAYMENT_METHOD_UPDATE,
                RecoveryAction.ALTERNATIVE_PATH,
                RecoveryAction.STOP_AND_ESCALATE,
            )
            code = "FROZEN_AUTH_INSTRUMENT_UPDATE"
    elif category == FailureCategory.EXPIRED_PAYMENT_METHOD:
        ranked = pick(
            RecoveryAction.PAYMENT_METHOD_UPDATE,
            RecoveryAction.ALTERNATIVE_PATH,
            RecoveryAction.STOP_AND_ESCALATE,
        )
        code = "FROZEN_EXPIRED_UPDATE"
    else:
        ranked = pick(
            RecoveryAction.RETRY_LATER,
            RecoveryAction.ALTERNATIVE_PATH,
            RecoveryAction.RECOVERY_NUDGE,
            RecoveryAction.STOP_AND_ESCALATE,
        )
        code = "FROZEN_DEFAULT"

    return {
        "recommended_action": ranked[0],
        "ranked_actions": ranked,
        "suggested_delay_hours": 0,
        "abstain": False,
        "confidence": 0.7,
        "reason_codes": [code],
        "rationale": f"Frozen AI adapter rule {code}",
    }


class FrozenAIRanker:
    """Deterministic stand-in. Swap later for Grok/OpenAI without touching the gate."""

    ranker_id = RANKER_ID

    def rank(
        self,
        payment: FailedPayment,
        category: FailureCategory,
        eligible: tuple[RecoveryAction, ...],
        ctx: PolicyContext,
    ) -> Recommendation:
        _ = ctx
        raw = _draft(payment, category, eligible)
        return parse_recommendation(raw, eligible)


class FutureLLMRanker:
    """Placeholder for a live model. Evaluation must not instantiate this path."""

    ranker_id = "live_llm_disabled"

    def rank(self, *args: object, **kwargs: object) -> Recommendation:
        raise RuntimeError("Live LLM ranker is disabled for evaluation")
