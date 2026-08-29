"""Policy gates.

Shared safety applies to every strategy (idempotency, retry cap, cooldown,
global stop). Revora-specific eligibility uses the curated knowledge base and
must not be applied to baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

from revora.knowledge import KnowledgeEntry, lookup_policies
from revora.schemas import (
    FailedPayment,
    FailureCategory,
    PaymentMethodType,
    PolicyDecision,
    RecoveryAction,
)

MAX_GLOBAL_RECOVERY_ATTEMPTS = 4
SHARED_RETRY_LIMIT = 3
SHARED_RETRY_COOLDOWN_HOURS = 2.0

# Revora product rules only — not the shared processor gate.
REVORA_FORBIDDEN: frozenset[tuple[FailureCategory, RecoveryAction]] = frozenset(
    {
        (FailureCategory.EXPIRED_PAYMENT_METHOD, RecoveryAction.RETRY_LATER),
        (FailureCategory.REPEATED_FAILURE, RecoveryAction.RETRY_LATER),
    }
)


@dataclass(frozen=True)
class PolicyContext:
    """Runtime facts the gate needs. Comes from the payment + simulator ledger."""

    recovery_attempts_this_run: int
    hours_since_last_recovery: float
    last_action: RecoveryAction | None
    event_already_processed: bool
    # Per-action-family counts this episode. Empty for callers that only need v1.
    attempts_by_family: tuple[tuple[str, int], ...] = ()


class PolicyError(ValueError):
    pass


def _entry_for(
    category: FailureCategory,
    method: PaymentMethodType,
) -> KnowledgeEntry | None:
    matched = lookup_policies(category, method)
    return matched[0] if matched else None


def evaluate_safety(
    payment: FailedPayment | None,
    category: FailureCategory | None,
    action: RecoveryAction,
    ctx: PolicyContext,
) -> PolicyDecision:
    """Strategy-agnostic safety. No knowledge-base allow-list."""
    if payment is None or category is None:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason="Missing or invalid payment context",
            rule_id="require_context",
        )

    if ctx.event_already_processed:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason="Duplicate webhook/event_id already processed (idempotency)",
            rule_id="idempotency_duplicate_event",
        )

    if action not in RecoveryAction:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason="Invalid recovery action",
            rule_id="invalid_action",
        )

    if action == RecoveryAction.STOP_AND_ESCALATE:
        return PolicyDecision(
            action=action,
            allowed=True,
            reason="Stop-and-escalate is always permitted",
            rule_id="stop_always_allowed",
        )

    total_attempts = payment.previous_recovery_attempts + ctx.recovery_attempts_this_run
    if total_attempts >= MAX_GLOBAL_RECOVERY_ATTEMPTS:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason=f"Global recovery attempt cap {MAX_GLOBAL_RECOVERY_ATTEMPTS} reached",
            rule_id="global_attempt_cap",
        )

    if action == RecoveryAction.RETRY_LATER:
        retries_used = payment.previous_recovery_attempts + ctx.recovery_attempts_this_run
        if retries_used >= SHARED_RETRY_LIMIT:
            return PolicyDecision(
                action=action,
                allowed=False,
                reason="Maximum retry limit reached",
                rule_id="retry_limit",
            )
        if (
            ctx.last_action == RecoveryAction.RETRY_LATER
            and ctx.hours_since_last_recovery < SHARED_RETRY_COOLDOWN_HOURS
        ):
            return PolicyDecision(
                action=action,
                allowed=False,
                reason=(
                    f"Cooldown active: {ctx.hours_since_last_recovery:.2f}h "
                    f"< {SHARED_RETRY_COOLDOWN_HOURS}h"
                ),
                rule_id="cooldown",
            )

    return PolicyDecision(
        action=action,
        allowed=True,
        reason="Action permitted by shared safety (retry, cooldown, caps, idempotency)",
        rule_id="allow",
    )


def evaluate_revora_eligibility(
    payment: FailedPayment,
    category: FailureCategory,
    action: RecoveryAction,
    *,
    attempts_this_run: int = 0,
) -> PolicyDecision:
    """Revora-only allow-list and product forbids. Not used by baselines."""
    if (category, action) in REVORA_FORBIDDEN:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason=f"{action.value} is forbidden for {category.value} under Revora policy",
            rule_id="forbidden_pair",
        )
    if action == RecoveryAction.STOP_AND_ESCALATE:
        return PolicyDecision(
            action=action,
            allowed=True,
            reason="Stop-and-escalate is always permitted",
            rule_id="stop_always_allowed",
        )
    entry = _entry_for(category, payment.payment_method)
    if entry and action not in entry.candidate_actions:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason=(
                f"{action.value} is not eligible for {category.value} "
                f"per Revora knowledge '{entry.id}'"
            ),
            rule_id="eligibility",
        )
    if entry:
        total = payment.previous_recovery_attempts + attempts_this_run
        if total >= entry.stop_after_recovery_attempts:
            return PolicyDecision(
                action=action,
                allowed=False,
                reason=(
                    f"Revora stopping rule: recovery attempts {total} "
                    f">= {entry.stop_after_recovery_attempts}"
                ),
                rule_id="stopping_rule",
            )
    return PolicyDecision(
        action=action,
        allowed=True,
        reason="Revora eligibility passed",
        rule_id="revora_eligible",
    )


def family_attempts_for_c2(
    payment: FailedPayment,
    action: RecoveryAction,
    ctx: PolicyContext,
) -> int:
    """C2 ledger: dataset history is treated as RETRY_LATER-family only.

    Unlabeled previous_recovery_attempts do not exhaust PAYMENT_METHOD_UPDATE
    or other non-retry mechanisms. This-episode counts are per action value.
    """
    this_episode = dict(ctx.attempts_by_family).get(action.value, 0)
    if action == RecoveryAction.RETRY_LATER:
        return payment.previous_recovery_attempts + this_episode
    return this_episode


def evaluate_revora_eligibility_c2(
    payment: FailedPayment,
    category: FailureCategory,
    action: RecoveryAction,
    ctx: PolicyContext,
) -> PolicyDecision:
    """Revora C2 eligibility: stop_after is per mechanism family, not pooled.

    This is a POLICY CHANGE, not an AI improvement.
    """
    if (category, action) in REVORA_FORBIDDEN:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason=f"{action.value} is forbidden for {category.value} under Revora policy",
            rule_id="forbidden_pair",
        )
    if action == RecoveryAction.STOP_AND_ESCALATE:
        return PolicyDecision(
            action=action,
            allowed=True,
            reason="Stop-and-escalate is always permitted",
            rule_id="stop_always_allowed",
        )
    entry = _entry_for(category, payment.payment_method)
    if entry and action not in entry.candidate_actions:
        return PolicyDecision(
            action=action,
            allowed=False,
            reason=(
                f"{action.value} is not eligible for {category.value} "
                f"per Revora knowledge '{entry.id}'"
            ),
            rule_id="eligibility",
        )
    if entry:
        family_total = family_attempts_for_c2(payment, action, ctx)
        if family_total >= entry.stop_after_recovery_attempts:
            return PolicyDecision(
                action=action,
                allowed=False,
                reason=(
                    f"C2 stopping rule: {action.value} attempts {family_total} "
                    f">= {entry.stop_after_recovery_attempts} (mechanism ledger)"
                ),
                rule_id="stopping_rule",
            )
    return PolicyDecision(
        action=action,
        allowed=True,
        reason="Revora C2 eligibility passed (mechanism-aware ledger)",
        rule_id="revora_eligible",
    )


def c2_eligible_actions(
    payment: FailedPayment,
    category: FailureCategory,
    actions: tuple[RecoveryAction, ...] | list[RecoveryAction],
    ctx: PolicyContext,
) -> tuple[RecoveryAction, ...]:
    """C2 eligibility only (ledger/allow-list). Does not apply shared safety."""
    out: list[RecoveryAction] = []
    for action in actions:
        decision = evaluate_revora_eligibility_c2(payment, category, action, ctx)
        if decision.allowed:
            out.append(action)
    return tuple(out)


def evaluate_revora_c2_policy(
    payment: FailedPayment | None,
    category: FailureCategory | None,
    action: RecoveryAction,
    ctx: PolicyContext,
) -> PolicyDecision:
    safety = evaluate_safety(payment, category, action, ctx)
    if not safety.allowed:
        return safety
    if payment is None or category is None:
        return safety
    elig = evaluate_revora_eligibility_c2(payment, category, action, ctx)
    if not elig.allowed:
        return elig
    return PolicyDecision(
        action=action,
        allowed=True,
        reason="Revora C2 eligibility and shared safety both allow this action",
        rule_id="allow",
    )


def evaluate_revora_policy(
    payment: FailedPayment | None,
    category: FailureCategory | None,
    action: RecoveryAction,
    ctx: PolicyContext,
) -> PolicyDecision:
    safety = evaluate_safety(payment, category, action, ctx)
    if not safety.allowed:
        return safety
    if payment is None or category is None:
        return safety
    elig = evaluate_revora_eligibility(
        payment, category, action, attempts_this_run=ctx.recovery_attempts_this_run
    )
    if not elig.allowed:
        return elig
    return PolicyDecision(
        action=action,
        allowed=True,
        reason="Revora eligibility and shared safety both allow this action",
        rule_id="allow",
    )


def evaluate_policy(
    payment: FailedPayment | None,
    category: FailureCategory | None,
    action: RecoveryAction,
    ctx: PolicyContext,
) -> PolicyDecision:
    """Revora combined gate. Kept for Failure Lab and existing unit tests."""
    return evaluate_revora_policy(payment, category, action, ctx)


def first_allowed(
    payment: FailedPayment,
    category: FailureCategory,
    ranked_actions: list[RecoveryAction],
    ctx: PolicyContext,
    *,
    revora_rules: bool = True,
    c2_ledger: bool = False,
) -> PolicyDecision:
    if c2_ledger:
        gate = evaluate_revora_c2_policy
    elif revora_rules:
        gate = evaluate_revora_policy
    else:
        gate = evaluate_safety
    last = PolicyDecision(
        action=RecoveryAction.STOP_AND_ESCALATE,
        allowed=True,
        reason="No ranked action passed the policy gate; escalate",
        rule_id="fallback_escalate",
    )
    for action in ranked_actions:
        decision = gate(payment, category, action, ctx)
        if decision.allowed:
            return decision
        last = decision
    stop = evaluate_safety(payment, category, RecoveryAction.STOP_AND_ESCALATE, ctx)
    if stop.allowed:
        return PolicyDecision(
            action=RecoveryAction.STOP_AND_ESCALATE,
            allowed=True,
            reason=f"All candidates rejected (last: {last.reason}); escalating",
            rule_id="fallback_escalate",
        )
    return last
