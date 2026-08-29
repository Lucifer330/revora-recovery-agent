"""Baseline recovery systems.

Baseline A — Simple Retry: always RETRY_LATER (shared safety only).
Baseline B — Static rules: category → one action, no scoring, shared safety only.

Neither baseline is filtered through Revora's knowledge-base allow-list.
"""

from __future__ import annotations

from revora.diagnosis import diagnose
from revora.policy import PolicyContext, evaluate_safety, first_allowed
from revora.schemas import FailedPayment, FailureCategory, RecoveryAction, SystemDecision

SIMPLE_RETRY_ID = "simple_retry"
STATIC_RULES_ID = "static_rules"

# Naive merchant playbook: retry almost everything; update expired methods.
STATIC_MAP: dict[FailureCategory, RecoveryAction] = {
    FailureCategory.TRANSIENT_NETWORK: RecoveryAction.RETRY_LATER,
    FailureCategory.INSUFFICIENT_FUNDS: RecoveryAction.RETRY_LATER,
    FailureCategory.EXPIRED_PAYMENT_METHOD: RecoveryAction.PAYMENT_METHOD_UPDATE,
    FailureCategory.AUTHENTICATION_FAILURE: RecoveryAction.RETRY_LATER,
    FailureCategory.REPEATED_FAILURE: RecoveryAction.RETRY_LATER,
    FailureCategory.UNKNOWN: RecoveryAction.RETRY_LATER,
}


def decide_simple_retry(payment: FailedPayment, ctx: PolicyContext) -> SystemDecision:
    diagnosis = diagnose(payment)
    recommended = RecoveryAction.RETRY_LATER
    policy = evaluate_safety(payment, diagnosis.category, recommended, ctx)
    if not policy.allowed:
        policy = first_allowed(
            payment,
            diagnosis.category,
            [RecoveryAction.STOP_AND_ESCALATE],
            ctx,
            revora_rules=False,
        )
    return SystemDecision(
        system_id=SIMPLE_RETRY_ID,
        action=policy.action,
        allowed=policy.allowed,
        policy=policy,
        diagnosis=diagnosis,
        candidates=(),
        score=None,
        ranking_notes=("Always recommends RETRY_LATER",),
    )


def decide_static_rules(payment: FailedPayment, ctx: PolicyContext) -> SystemDecision:
    diagnosis = diagnose(payment)
    recommended = STATIC_MAP[diagnosis.category]
    policy = evaluate_safety(payment, diagnosis.category, recommended, ctx)
    if not policy.allowed:
        policy = first_allowed(
            payment,
            diagnosis.category,
            [RecoveryAction.STOP_AND_ESCALATE],
            ctx,
            revora_rules=False,
        )
    return SystemDecision(
        system_id=STATIC_RULES_ID,
        action=policy.action,
        allowed=policy.allowed,
        policy=policy,
        diagnosis=diagnosis,
        candidates=(),
        score=None,
        ranking_notes=(f"Static map: {diagnosis.category.value} -> {recommended.value}",),
    )
