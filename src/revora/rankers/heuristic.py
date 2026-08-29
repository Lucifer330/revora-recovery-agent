"""Deterministic heuristic ranker (C2 + Heuristic arm).

A priori ranking among C2-eligible actions only. Written before evaluation.
Does not import environment, simulator, or held-out outcomes.

Priority (first eligible wins):

TRANSIENT_NETWORK: RETRY_LATER, ALTERNATIVE_PATH, RECOVERY_NUDGE, STOP
INSUFFICIENT_FUNDS: RECOVERY_NUDGE, RETRY_LATER, ALTERNATIVE_PATH, STOP
  (nudge first regardless of hours — NSF playbook: do not hammer immediately)
EXPIRED_PAYMENT_METHOD: PAYMENT_METHOD_UPDATE, ALTERNATIVE_PATH, STOP
AUTHENTICATION_FAILURE: ALTERNATIVE_PATH, PAYMENT_METHOD_UPDATE, STOP
REPEATED_FAILURE: STOP, ALTERNATIVE_PATH
UNKNOWN: STOP, RECOVERY_NUDGE, RETRY_LATER

Tie-break: action name ascending if two share the same slot (should not happen).

Delay is recorded as 0. This pass does not advance the episode clock from delay
so C2 vs C2+heuristic differs only by action choice, not extra wait.
"""

from __future__ import annotations

from revora.policy import PolicyContext
from revora.rankers.schema import Recommendation
from revora.schemas import FailedPayment, FailureCategory, RecoveryAction

RANKER_ID = "heuristic_v1"

_PRIORITY: dict[FailureCategory, tuple[RecoveryAction, ...]] = {
    FailureCategory.TRANSIENT_NETWORK: (
        RecoveryAction.RETRY_LATER,
        RecoveryAction.ALTERNATIVE_PATH,
        RecoveryAction.RECOVERY_NUDGE,
        RecoveryAction.STOP_AND_ESCALATE,
    ),
    FailureCategory.INSUFFICIENT_FUNDS: (
        RecoveryAction.RECOVERY_NUDGE,
        RecoveryAction.RETRY_LATER,
        RecoveryAction.ALTERNATIVE_PATH,
        RecoveryAction.STOP_AND_ESCALATE,
    ),
    FailureCategory.EXPIRED_PAYMENT_METHOD: (
        RecoveryAction.PAYMENT_METHOD_UPDATE,
        RecoveryAction.ALTERNATIVE_PATH,
        RecoveryAction.STOP_AND_ESCALATE,
    ),
    FailureCategory.AUTHENTICATION_FAILURE: (
        RecoveryAction.ALTERNATIVE_PATH,
        RecoveryAction.PAYMENT_METHOD_UPDATE,
        RecoveryAction.STOP_AND_ESCALATE,
    ),
    FailureCategory.REPEATED_FAILURE: (
        RecoveryAction.STOP_AND_ESCALATE,
        RecoveryAction.ALTERNATIVE_PATH,
    ),
    FailureCategory.UNKNOWN: (
        RecoveryAction.STOP_AND_ESCALATE,
        RecoveryAction.RECOVERY_NUDGE,
        RecoveryAction.RETRY_LATER,
    ),
}


class HeuristicRanker:
    ranker_id = RANKER_ID

    def rank(
        self,
        payment: FailedPayment,
        category: FailureCategory,
        eligible: tuple[RecoveryAction, ...],
        ctx: PolicyContext,
    ) -> Recommendation:
        _ = (payment, ctx)
        eligible_set = set(eligible)
        ranked: list[RecoveryAction] = []
        for action in _PRIORITY[category]:
            if action in eligible_set and action not in ranked:
                ranked.append(action)
        for action in sorted(eligible_set, key=lambda a: a.value):
            if action not in ranked:
                ranked.append(action)
        if not ranked:
            ranked = [RecoveryAction.STOP_AND_ESCALATE]
        chosen = ranked[0]
        return Recommendation(
            recommended_action=chosen,
            ranked_actions=tuple(ranked),
            suggested_delay_hours=0,
            abstain=False,
            confidence=1.0,
            reason_codes=("HEURISTIC_CATEGORY_PRIORITY",),
            rationale=f"Heuristic priority for {category.value}: {chosen.value}",
        )
